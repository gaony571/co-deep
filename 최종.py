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
import openai

# OpenAI API 키 설정
openai.api_key = ""

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

    #LLM으로 text 재구성 -> agent_text

    agent_data = data.get("agent", {}).get(current_agent, {})
    current_agent_episodes = [ep["summary"] for ep in agent_data.get("episodes", []) if "summary" in ep]

    agent_text = ""
    all_texts = current_agent_episodes + [agent_text]
    embeddings = embedding_model.encode(all_texts)

    target_embedding = embeddings[-1]

    episode_embeddings = embeddings[:-1]

    similarities = cosine_similarity([target_embedding], episode_embeddings)[0]

    top_indices = np.argsort(similarities)[::-1][:3]
    top_episodes = [current_agent_episodes[i] for i in top_indices]

    #top_episodes와 agent_text로 LLM호출 -> "traits" {"summary": summary, "inference": inference}

    result_traits = ""
    result = {
        "summary": "",
        "inference": ""
    }

    data["agent"][current_agent]["traits"].update(result_traits)
    data["agent"][current_agent]["episodes"].append(result)

    for child in node.get("thinks_about", []):
        dfs_tom_tree(child, text, data, list(path))

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
        "나": {
            "traits": {},
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
    people = ["나"]

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
                break

    if integrated_text == "종료":
        break

    #종결성 확인 모델 호출 및 확인 필요
    #상담 모델 호출 후 필요 시 재질문
    #발화 정제 모델 호출, 모델 결과 -> refined_text

    #발화 청크 분할
    refined_text = ""

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

    #메모리 파싱
    with open(json_file, "r", encoding="utf-8") as f:
       data = json.load(f)

    #메모리 업데이트
    for person in people:
        if person not in data["agent"]:
            data["agent"][person] = {
                "traits": {},
                "episodes": []
            }

    #LLM 호출 agent tree 생성 people 이름을 차용
    tree = {}

    #dfs 탐색
    dfs_tom_tree(tree["root"], refined_text, data)

    #상담모델 호출 -> answer

    answer = ""

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
