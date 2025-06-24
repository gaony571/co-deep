import json 
import stanza
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import numpy as np
import torch
import datetime
import time
import os
import msvcrt
import sys
import locale
import requests

#huggingface
headers = {"Authorization": f"Bearer "}
additional_info_API_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mixtral-8x7B-Instruct-v0.1/v1/chat/completions"


#stanza 한국어 다운로드
stanza.download('ko')

#모델 로드
chunking_model = stanza.Pipeline(lang='ko', processors='tokenize,mwt,pos,lemma,depparse')
embedding_model = SentenceTransformer("BM-K/KoSimCSE-roberta")
ner_tokenizer = AutoTokenizer.from_pretrained("kykim/bert-kor-base-ner")
ner_model = AutoModelForTokenClassification.from_pretrained("kykim/bert-kor-base-ner")

#문장 청크 추출 함수
def extract_chunks(sentence):
    id_to_text = {word.id: word.text for word in sentence.words}
    chunks = []
    current_chunk = []

    for word in sentence.words:
        current_chunk.append(word.text)
        if word.deprel in ['conj', 'advcl', 'acl', 'parataxis', 'ccomp', 'xcomp']:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

#ner 객체 추출 함수
def extract_person_entities(tokens, tags):
    person_entities = []
    current_entity = []

    for token, tag in zip(tokens, tags):
        if tag == "B-PER":
            if current_entity:
                person_entities.append("".join(current_entity))
                current_entity = []
            current_entity.append(token)
        elif tag == "I-PER":
            if current_entity:
                current_entity.append(token)
        else:
            if current_entity:
                person_entities.append("".join(current_entity))
                current_entity = []

    if current_entity:
        person_entities.append("".join(current_entity))

    return person_entities

#객체 유사도 검사 함수
def is_similar(entity, entity_list, threshold=0.85):
    if not entity_list:
        return False
    entity_emb = embedding_model.encode(entity, convert_to_tensor=True)
    list_embs = embedding_model.encode(entity_list, convert_to_tensor=True)
    max_score = float(util.cos_sim(entity_emb, list_embs).max())
    return max_score >= threshold

#객체 중복 제거 함수
def deduplicate_entities(per_entities, threshold=0.85):
    unique_entities = []
    for entity in per_entities:
        if not is_similar(entity, unique_entities, threshold):
            unique_entities.append(entity)
    return unique_entities

#객체 목록 업데이트 함수
def add_new_people(per_entities, people, threshold=0.85):
    for entity in per_entities:
        if not is_similar(entity, people, threshold):
            people.append(entity)
    return people

#dfs 탐색 함수
def dfs_tom_tree(node, text, data, path=None):
    if path is None:
        path = []

    current_agent = node.get("agent")
    path.append(current_agent)

    agent_text = reconstruct_agent_utterance(current_agent, text)

    agent_data = data.get("agent", {}).get(current_agent, {})
    current_agent_episodes = [ep["summary"] for ep in agent_data.get("episodes", []) if "summary" in ep]

    all_texts = current_agent_episodes + [agent_text]
    embeddings = embedding_model.encode(all_texts)

    target_embedding = embeddings[-1]

    episode_embeddings = embeddings[:-1]

    similarities = cosine_similarity([target_embedding], episode_embeddings)[0]

    top_indices = np.argsort(similarities)[::-1][:3]

    top_episodes = [current_agent_episodes[i] for i in top_indices]

    episodes_str = "\n".join(f"- {e}" for e in top_episodes)

    agent_traits_list = data["agent"][current_agent]["traits"]

    agent_traits = ", ".join(agent_traits_list)

    result = summarize_and_infer(current_agent, agent_traits, agent_text, episodes_str)

    parsed = json.loads(result)
    result_traits = parsed["traits"]
    result_summary = parsed["summary"]
    result_inference = parsed["inference"]

    result_episode = {
        "summary": result_summary,
        "inference": result_inference,
    }

    data["agent"][current_agent]["traits"].append(result_traits)
    data["agent"][current_agent]["episodes"].append(result_episode)

    for child in node.get("thinks_about", []):      
        dfs_tom_tree(child, text, data, list(path))

#추가 정보 필요 판단 LLM 호출
def needs_additional_info(user_utterance: str):
    messages = [
        {"role": "system", "content": (
            "너는 상담심리 전문가야. 내담자의 발화를 읽고, 모호하거나, 불분명하거나, 암시적인 진술이 있는 지 확인해.\n"
            "만약 있다면 내담자가 말한 진술 내용의 일부 또는 전체 내용을 반복하면서 “〜（이）라는 것은 〜라는 뜻인가요?”라는 질문 형태로 반응해.\n"
            "이 경우 다른 부가적인 설명이나 예시없이 오직 질문만을 답해야해.\n"
            "만약 없다면 정확하게 1이라고 출력해. 이 경우 어떤 다른 설명이나 예시도 절대 출력하지 마."
        )},
        {"role": "user", "content": user_utterance}
    ]

    payload = {
        "messages": messages,
        "temperature": 0.2,
        "max_new_tokens": 32,
        "return_full_text": False
        
    }

    response = requests.post(additional_info_API_URL, headers=headers, json=payload)
    result = response.json()
    output_text = result['choices'][0]['message']['content']

    if output_text == "1":
        return False
    else:
        print(output_text)
        return True

#발화 정제 LLM 호출
def refine_utterance(utterance: str, memory: str):
    messages = [
        {"role": "system", "content": (
            "너는 심리상담 도우미야. 내담자의 발화를 다음 기준에 따라 정제해줘:\n"
            "- 과거 발화를 기반으로 대명사를 명확한 지칭어로 바꿔\n"
            "- 생략된 객체를 모두 복원하고 발화의 주체가 생략된 경우 내담자를 추가해.\n"
            "- 모든 시점을 상담자 시점으로 통일해\n"
            "- 1인칭 표현은 '내담자'로 바꿔\n"
            "- 인용 표현은 간접인용으로 변환하고, 사용자의 감정 표현은 유지해\n"
            "- 문장들을 자연스럽게 통합해서 하나의 문장으로 만들어\n"
            "- 출력은 반드시 정제된 문장 하나만 해야 해. 예시나 이유, 설명은 절대 포함하지 마."
        )},
        {"role": "user", "content": "###발화 \n" + utterance + "\n\n" + "###과거 발화 \n" + memory}
    ]

    payload = {
        "messages": messages,
            "temperature": 0.2,
            "max_new_tokens": 64,
            "return_full_text": False
        
    }

    response = requests.post(additional_info_API_URL, headers=headers, json=payload)
    result = response.json()
    refined_text = result['choices'][0]['message']['content']

    return refined_text

#agent 발화 재구성 LLM 호출
def reconstruct_agent_utterance(agent: str, utterance: str):
    messages = [
        {"role": "system", "content": (
            "너는 발화 재구성 도우미야. 입력된 발화를 다음 기준에 따라 재구성해줘:\n"
            "- 발화를 제시된 에이전트를 주어로 재구성해\n"
            "- 반드시 모든 인물의 호칭을 정확히 유지해\n"
            "- 발화 내의 여러 정보 중 에이전트의 시점에서 알 수 없는 정보는 포함되어서는 절대 안돼\n"
            "- 출력은 반드시 정제된 문장 하나만 해야 해. 예시나 이유, 설명은 절대 포함하지 마.\n"
            "- 결과를 반드시 한국어로 출력해야해."
        )},
        {"role": "user", "content": f"###발화 \n{utterance}\n\n###에이전트 이름 \n{agent}"}
    ]

    payload = {
        "messages": messages,
        
            "temperature": 0.2,
            "max_new_tokens": 128,
            "return_full_text": False
        
    }

    response = requests.post(additional_info_API_URL, headers=headers, json=payload)
    result = response.json()
    reconstructed_text = result['choices'][0]['message']['content']

    return reconstructed_text

#요약 및 추론 LLM 호출
def summarize_and_infer(agent: str, traits: str, utterance: str, episodes: str):
    messages = [
        {"role": "system", "content": (
            "너는 요약 및 추론 도우미야. 주어진 발화와 에이전트의 특성, 과거 에피소드를 바탕으로 다음을 수행해:\n"
            "- 출력은 반드시 JSON 형식으로 해야 해. JSON 키는 traits, summary, inference로 하고, 각각의 값은 다음과 같아:\n"
            " - traits에는 제시된 에이전트의 과거 에피소드를 종합하여 5단어 이내로 특징을 서술해\n"
            " - summary에는 발화의 중요한 감정, 사건, 생각의 흐름을 한 문장으로 요약하되, 설명은 절대 포함되어서는 안돼.\n"
            " - inference에는 발화와 과거 에피소드, traits를 바탕으로 제시된 에이전트가 어떤 감정을 느낄 지 한 단어로 추론해.\n"
            "- traits, summary, inference는 반드시 모두 포함되어야 해. 하나라도 빠지면 안돼.\n"
            "예시나 이유, 설명은 절대 포함하지 마."
            "반드시 아래의 출력 형식을 따라야해."
            "형식: {\"traits\": \"\", \"summary\": \"\", \"inference\": \"\"}"
        )},
        {"role": "user", "content": f"###발화 \n{utterance}\n\n###에이전트 이름 \n{agent}\n###특성\n{traits}\n###과거 에피소드 \n{episodes}"}
    ]   

    payload = {
        "messages": messages,
        
            "temperature": 0.2,
            "max_new_tokens": 128,
            "return_full_text": False
        
    }

    response = requests.post(additional_info_API_URL, headers=headers, json=payload)
    result = response.json()
    output_text = result['choices'][0]['message']['content']

    return output_text

#tom 트리 생성 LLM 호출(수정필요)
def create_tom_tree(utterance: str, agents: str):
    messages = [
        {"role": "system", "content": (
            "당신은 사람 간의 사고 구조를 추론하는 인지 심리 전문가입니다. 사용자가 입력한 발화를 바탕으로 인물 간의 인지 관계를 계층적 트리(JSON) 구조로 생성해야 합니다.\n"
            "트리 구조는 다음과 같은 형식을 따라야 합니다:\n"
            "- 트리의 최상위는 항상 '내담자'입니다.\n"
            "- 각 인물은 'agent'로 표기하며, 해당 인물이 인지하는 다른 인물은 'thinks_about'로 재귀적으로 표현합니다.\n"
            f"- 등장인물은 반드시 주어진 목록 안에서만 선택합니다: {agents}\n"
            "- 결과는  JSON 형식으로 출력해야 하며, 반드시 'root' 키를 포함해야 합니다.\n"
            "- 절대 다른 형식이나 예시, 설명을 포함하지 마세요.\n\n"
            "###예시:\n"
            "사용자 입력: 나는 친구가 나를 걱정하고 있다는 걸 느꼈고, 엄마는 아빠 때문에 스트레스를 많이 받아 보여.\n"
            "출력: {\"root\":{\"agent\":\"나\",\"thinks_about\":[{\"agent\":\"어머니\",\"thinks_about\":[{\"agent\":\"아버지\"}]},{\"agent\":\"친구\",\"thinks_about\":[{\"agent\":\"나\"}]}]}}"
        )},
        {"role": "user", "content": utterance}
    ]

    payload = {
        "messages": messages,
        
            "temperature": 0.2,
            "max_new_tokens": 256,
            "return_full_text": False
        
    }

    response = requests.post(additional_info_API_URL, headers=headers, json=payload)
    result = response.json()
    tom_tree = result['choices'][0]['message']['content']

    return tom_tree

#콘솔 UTF-8로 전환
if locale.getpreferredencoding().lower() != 'UTF-8':
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding='utf-8')

#에크만 감정 분류 임베딩
ekman_emotions = {
    "기쁨": "나는 너무 행복하고 웃음이 나왔다.",
    "슬픔": "마음이 너무 아프고 눈물이 났다.",
    "분노": "짜증나고 화가 났다.",
    "공포": "무섭고 두려운 느낌이 들었다.",
    "놀람": "깜짝 놀라고 충격을 받았다.",
    "혐오": "불쾌하고 역겨운 기분이 들었다."
}
emotion_labels = list(ekman_emotions.keys())
emotion_texts = list(ekman_emotions.values())
emotion_embeddings = embedding_model.encode(emotion_texts)

#상담 메모리 JSON 생성
schema = {
    "memory": [],
    "agent": {
        "내담자": {
            "traits": [],
            "episodes": []
        }
    }
}
json_file = "memory.json"

#JSON 메모리 초기화
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(schema, f, ensure_ascii=False, indent=4)

#상담 시작
while True:
    buf = []
    input_text = ""
    integrated_text = ""
    last_input_time = time.time()
    time_checker = 10
    Q_count = 0
    people = ["내담자"]

    #입력 및 종료 확인
    while True:
        if msvcrt.kbhit():
            input_text = msvcrt.getwch()
            if input_text == "\r":
                integrated_text += "".join(buf) + " "
                buf.clear()
            elif input_text == "\b":
                if buf:
                    buf.pop()
                    print("\b \b", end='', flush=True)
                last_input_time = time.time()
                continue
            buf.append(input_text)
            print(input_text, end='', flush=True)
            last_input_time = time.time()

        if time.time() - last_input_time > time_checker:
            if Q_count == 0:
                if needs_additional_info(integrated_text) == True:
                    integrated_text += " "
                    Q_count = 1
                    continue
                else:
                    break
            else:
                break

    Q_count = 0

    if integrated_text == "종료":
        break

    #메모리 파싱
    with open(json_file, "r", encoding="utf-8") as f:
       data = json.load(f)

    #발화 정제
    refined_text = refine_utterance(integrated_text, data["memory"][-1]["event"] if data["memory"] else "")

    #발화 청크 분할
    doc = chunking_model(refined_text)

    chunks = []
    for sentence in doc.sentences:
        chunks.extend(extract_chunks(sentence))


    #감정 분류
    chunk_embeddings = embedding_model.encode(chunks)
    predicted_emotions = []
    for chunk_emb in chunk_embeddings:
        sims = cosine_similarity([chunk_emb], emotion_embeddings)[0]
        best_match_idx = np.argmax(sims)
        predicted_emotion = emotion_labels[best_match_idx]
        predicted_emotions.append(predicted_emotion)

    #감정 집계
    emotion_counts = Counter(predicted_emotions)

    #대표 감정 추출
    if emotion_counts:
        max_count = max(emotion_counts.values())
        dominant_emotions = [emotion for emotion, count in emotion_counts.items() if count == max_count]
        dominant_emotion_str = ','.join(dominant_emotions)
        emotion_counts = Counter({emotion: emotion_counts[emotion] for emotion in dominant_emotions})


    #ner 진행
    ner_tokens = ner_tokenizer(refined_text, return_tensors="pt", truncation=True, is_split_into_words=False)

    with torch.no_grad():
        output = ner_model(**ner_tokens)
    logits = output.logits
    predictions = torch.argmax(logits, dim=2)

    label_list = ner_model.config.id2label
    tokens_decoded = ner_tokenizer.convert_ids_to_tokens(ner_tokens['input_ids'][0])
    predicted_labels = [label_list[p.item()] for p in predictions[0]]

    #NER 객체 추출
    per_entities = extract_person_entities(tokens_decoded, predicted_labels)

    #객체 중복 제거
    now_entities = deduplicate_entities(per_entities)

    #agent 목록 업데이트
    people = add_new_people(now_entities, people)

    #메모리 업데이트
    for person in people:
        if person not in data["agent"]:
            data["agent"][person] = {
                "traits": [],
                "episodes": []
            }


    now_entities_list = ", ".join(now_entities)

    #현재 발화와 객체 목록을 기반으로 TOM 트리 생성
    tree_text = create_tom_tree(refined_text, now_entities_list)

    #tom 트리 파싱
    tree = json.loads(tree_text)
    
    #dfs 탐색
    dfs_tom_tree(tree["root"], refined_text, data)

    #상담모델 호출 -> answer

    answer = ""

    print(answer)

    #메모리 업데이트
    data_update = {
        "timestamp" : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event" : refined_text,
        "emotions" : dominant_emotion_str,
        "answer" : answer
    }
    data["memory"].append(data_update)

    #메모리 JSON 파일 저장
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
