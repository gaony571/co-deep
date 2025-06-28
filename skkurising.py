import msvcrt
import datetime
import time
import json
import requests
import stanza
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import numpy as np
import locale
import os
import sys
import torch

headers = {"Authorization": f"Bearer "}
additional_info_API_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mixtral-8x7B-Instruct-v0.1/v1/chat/completions"

chunking_model = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

stanza.download('en')

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

def refine_utterance(utterance: str):
    messages = [
        {"role": "system", "content": (
            "You are a psychotherapy assistant that refines client utterances according to strict clarity and attribution guidelines.\n\n"
            "You must retain the original first-person point of view in the utterance, even after replacing pronouns with explicit references.\n\n"
            "You must change all passive expressions to active expressions.\n\n"
            "You must follow all of the following rules without exception:\n\n"
            "1. Always replace all pronouns (including: them, they, their, he, she, it, you, I, me, my, this, that, etc.) with explicit references like 'the client' or the named person.\n"
            "2. Never leave any pronoun unreplaced, even if the sentence feels natural.\n"
            "3. If the subject of a sentence is omitted, explicitly add 'the client' as the subject.\n"
            "4. Restore any omitted objects in the sentence using reasonable context.\n"
            "5. Unify all points of view to the client's perspective.\n"
            "6. Replace all first-person pronouns (I, me, my) with 'the client.'\n"
            "7. Convert quotations to indirect speech, but retain the client's emotional tone.\n"
            "8. Do not interpret or summarize the utterance. Do not change or shorten any meaning.\n"
            "9. Return only the final refined sentence, without explanation or additional text.\n\n"
            "### Example:\n"
            "Input: 'I told them I was hurt, but they didn’t care.'\n"
            "Output: 'The client told the client's friends that the client was hurt, but the client's friends did not care.'"
        )},
        {"role": "user", "content": "###utterance \n" + utterance + "\n"}
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

def extract_entities_and_people(text):
    doc = chunking_model(text)
    named_people = set()

    for ent in doc.ents:
        if ent.type == "PERSON":
            named_people.add(ent.text)

    for sent in doc.sentences:
        for word in sent.words:
            token = word.text.lower()
            if token in common_person_nouns:
                named_people.add(token)

    return list(named_people)

def deduplicate_entities_with_embeddings(per_entities, threshold=0.85):
    unique_entities = []
    unique_embeddings = []

    for entity in per_entities:
        entity_emb = embedding_model.encode(entity, convert_to_tensor=True)
        if not unique_embeddings:
            unique_entities.append(entity)
            unique_embeddings.append(entity_emb)
            continue

        sims = util.cos_sim(entity_emb, torch.stack(unique_embeddings))[0]
        max_sim = torch.max(sims).item()
        if max_sim < threshold:
            unique_entities.append(entity)
            unique_embeddings.append(entity_emb)

    return unique_entities, torch.stack(unique_embeddings)

def extract_connections(filtered_deps, person_list):
    connections = []
    person_list = [p.lower() for p in person_list]  
    token_map = {item["word"].lower(): item for item in filtered_deps}

    for token in filtered_deps:
        dep = token["deprel"]
        word = token["word"].lower()
        head = token["head"].lower()

        if dep in {"nsubj", "nsubj:pass"} and word in person_list:
            subj = word
            for t in filtered_deps:
                obj_word = t["word"].lower()
                if t["deprel"] in {"obj", "iobj", "obl"} and obj_word in person_list and obj_word != subj:
                    connections.append((subj, obj_word))

        if dep == "acl:relcl" and head in person_list:
            for t in filtered_deps:
                obj_word = t["word"].lower()
                if t["deprel"] in {"obj", "iobj", "obl"} and obj_word in person_list:
                    connections.append((head, obj_word))

        if dep in {"xcomp", "ccomp", "advcl"}:
            for t in filtered_deps:
                obj_word = t["word"].lower()
                if t["deprel"] in {"obj", "obl", "iobj"} and obj_word in person_list:
                    connections.append(("client", obj_word))

        if dep == "conj":
            head_word = head
            for t in filtered_deps:
                if t["deprel"] in {"nsubj", "nsubj:pass"} and t["head"].lower() == head_word and t["word"].lower() in person_list:
                    subj = t["word"].lower()
                    for obj_token in filtered_deps:
                        obj_word = obj_token["word"].lower()
                        if obj_token["deprel"] in {"obj", "iobj", "obl"} and obj_word in person_list:
                            connections.append((subj, obj_word))

        if dep in {"obj", "iobj"} and word == "client":
            verb = head
            for t in filtered_deps:
                subj_word = t["word"].lower()
                if t["deprel"] in {"nsubj"} and t["head"].lower() == verb and subj_word in person_list:
                    connections.append(("client", subj_word))

    return list(set(connections))  

def insert_thought(node, thinker, target):
    if thinker == target:
        return
    if node["agent"] == thinker:
        if "thinks_about" not in node:
            node["thinks_about"] = []
        if not any(child["agent"] == target for child in node["thinks_about"]):
            node["thinks_about"].append({"agent": target})
    else:
        for child in node.get("thinks_about", []):
            insert_thought(child, thinker, target)

def dfs_tom_tree(node, text, data, path=None):
    if path is None:
        path = []

    current_agent = node.get("agent")
    path.append(current_agent)

    agent_text = reconstruct_agent_utterance(current_agent, text)

    agent_data = data.get("agent", {}).get(current_agent, {})
    current_agent_episodes = [
        ep["summary"] for ep in agent_data.get("episodes", []) if "summary" in ep
    ]

    all_texts = current_agent_episodes + [agent_text]
    embeddings = embedding_model.encode(all_texts)

    target_embedding = embeddings[-1]
    episode_embeddings = embeddings[:-1]

    if len(episode_embeddings) > 0:
        similarities = cosine_similarity([target_embedding], episode_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:min(3, len(similarities))]
        top_episodes = [current_agent_episodes[i] for i in top_indices]
    else:
        top_episodes = []

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

def reconstruct_agent_utterance(agent: str, utterance: str):
    messages = [
        {"role": "system", "content": (
            "You are a speech reconstruction assistant. Your task is to rewrite the input utterance strictly from the perspective of a given agent. Follow all rules below without exception:\n"
            "1. The rewritten utterance must be in natural English and must be something the given agent could realistically say, based only on what the agent can directly observe, do, or know.\n"
            "2. All reconstructed sentences must have the given agent as the grammatical subject.\n"
            "3. You must keep all names exactly as provided. Do not replace names with pronouns or other references.\n"
            "4. You must exclude any information the agent cannot know firsthand, including:\n"
            " - another person's internal state, feelings, or thoughts\n"
            " - motivations, intentions, or desires of others\n"
            " - effects or consequences that the agent did not witness\n"
            "5. Do not include any explanations, justifications, reasoning, or commentary."
            "6. Do not use first-person pronouns (e.g., \"I\", \"we\") unless the given agent is explicitly a first-person subject.\n"
            "7. Return only the final reconstructed utterance. Do not return explanations or reasoning.\n"
            "Your output must be strictly limited to observable, agent-specific facts.\n\n"
            "Example:\n"
            "agent: \'friends\"\n"
            "Utterance: Client's friend hate client. So client is in sad.\n"
            "Output: friend hate client."
            )},
        {"role": "user", "content": f"###utterance: \n{utterance}\n\n###agent: \n{agent}"}
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

def summarize_and_infer(agent: str, traits: str, utterance: str, episodes: str):
    messages = [
        {"role": "system", "content": (
            "You are a summary and inference assistant. Based on the given utterance, the agent's traits, and past episodes, perform the following tasks.:\n"
            "- The output must be in JSON format. The JSON keys are traits, summary, and inference, and their values are as follows.:\n"
            " - traits describes the characteristics of the agent in up to five words, summarizing its past episodes.\n"
            " - The summary should summarize the important emotions, events, and thoughts of the speech in one sentence from the agent's perspective, but explanations should never be included.\n"
            " - Inference involves inferring what emotions an agent will feel based on speech, past episodes, and traits, and expressing them in a single word. It must be based on traits.\n"
            "- Traits, summary, and inference must all be included. Not one can be omitted.\n"
            "- Do not include examples, reasons, or explanations.\n"
            "You must follow the output format below..\n"
            "format: {\"traits\": \"\", \"summary\": \"\", \"inference\": \"\"}"
        )},
        {"role": "user", "content": f"###utterance: \n{utterance}\n\n###agent: \n{agent}\n###traits:\n{traits}\n###past episodes: \n{episodes}"}
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

def counseling_model(utterance: str, past_utterance: str, agent_memory: str):
    messages = [
        {
            "role": "system", 
            "content": (
                "You are a psychological counselor. Based on the client’s past dialogue and the current utterance analysis, "
                "your task is to generate an appropriate counseling response to the client’s current utterance.\n\n"
                "First, do not print out explanations, examples, notes, etc.\n"
                "Generate responses that keep the conversation going with the client as much as possible.\n"
                "Avoid long responses of more than six sentences unless necessary.\n"
                "If you believe the analysis of the current utterance is inaccurate or inconsistent, you may ignore it.\n\n"
                "You can use following counseling technique. But use ONLY when you think it is absolutely necessary, and in most cases, generate normal conversation.:\n\n"

                "1. Paraphrasing\n"
                "- Definition: Rephrasing the client’s statements (situations, events, subjects, thoughts) in the counselor’s own words.\n"
                "- When to use: When the client’s statements are vague or lengthy; to help organize their thoughts.\n"
                "- Caution: Avoid mechanical repetition. Do not omit emotional content.\n"
                "- Example:\n"
                "- Client: “I wanted to become a doctor, but I’m not sure anymore.”\n"
                "- Counselor: “It was a long-held dream, but recently you've been considering other options.”\n\n"

                "2. Reflecting Feeling\n"
                "- Definition: Empathically reflecting the emotion expressed in the client’s statement.\n"
                "- When to use: To help the client recognize or deepen their emotional expression.\n"
                "- Caution: Avoid exaggeration or distortion. Do not insert the counselor’s emotions.\n"
                "- Example:\n"
                "- Client: “I want to be perfect, but I always feel like I’m not enough.”\n"
                "- Counselor: “You’re feeling disappointed in yourself for not meeting your own expectations.”\n\n"

                "3. Summarizing\n"
                "- Definition: Concisely organizing two or more of the client’s statements into a summary of themes and core points.\n"
                "- When to use: Mid-session transitions, closing, or after lengthy sharing.\n"
                "- Caution: Avoid sounding mechanical. Ensure contextual relevance.\n"
                "- Example:\n"
                "- Counselor: “To summarize what you’ve shared, you’re struggling with seeking approval from your parents, anxiety about your future, and low self-esteem.”\n\n"

                "4. Confrontation\n"
                "- Definition: Gently pointing out contradictions between the client’s words, behaviors, or emotions.\n"
                "- When to use: When avoidance or recurring negative patterns appear.\n"
                "- Caution: Do not use before trust is established. Avoid sounding coercive.\n"
                "- Example:\n"
                "- Client: “I’m fine.”\n"
                "- Counselor: “You said you're fine, but I noticed tears. Are you really feeling okay?”\n\n"

                "5. Interpretation\n"
                "- Definition: Tentatively offering a hypothesis about the meaning of the client’s behavior, emotion, or relationships.\n"
                "- When to use: To help the client gain insight into recurring patterns.\n"
                "- Caution: Do not present as fact. Consider the client’s readiness to accept the interpretation.\n"
                "- Example:\n"
                "- Counselor: “You keep bringing up your anger toward your father—perhaps it shows a strong desire for his approval.”\n\n"

                "6. Information-Giving\n"
                "- Definition: Providing objective facts or professional information to help resolve the client’s issue.\n"
                "- When to use: When confusion arises from lack of information, or to support decision-making.\n"
                "- Caution: Avoid sounding authoritative or overexplaining.\n"
                "- Example:\n"
                "- Counselor: “The diagnostic criteria for depression generally include major symptoms lasting at least two weeks.”\n\n"

                "7. Self-Disclosure\n"
                "- Definition: Sharing personal experiences or emotions to build rapport with the client.\n"
                "- When to use: To foster trust, model behavior, or offer comfort.\n"
                "- Caution: Ensure the focus remains on the client. Keep disclosures relevant.\n"
                "- Example:\n"
                "- Counselor: “I experienced a similar conflict during my college years, so I can relate to your confusion.”\n\n"

                "8. Immediacy\n"
                "- Definition: Directly commenting on the interaction happening between the counselor and client in the moment.\n"
                "- When to use: When the conversation feels blocked, or emotions arise in-session.\n"
                "- Caution: Use neutral language. Avoid emotionally charged wording.\n"
                "- Example:\n"
                "- Counselor: “You seem hesitant while talking. What are you feeling right now?”\n"
            )
        },
        {
            "role": "user",
            "content": (
                f"Past dialogue: {past_utterance}\n"
                f"Current utterance: {utterance}\n"
                f"Current utterance analysis: {agent_memory}"
            )
        }
    ]

    payload = {
        "messages": messages,
        "temperature": 0.2,
        "max_new_tokens": 256,
        "return_full_text": False
    }

    response = requests.post(additional_info_API_URL, headers=headers, json=payload)
    result = response.json()
    counsel = result['choices'][0]['message']['content']

    return counsel


if locale.getpreferredencoding().lower() != 'UTF-8':
    os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding='utf-8')

#에크만 감정 분류 임베딩
ekman_emotions = {
    "Joy": 	"I felt incredibly happy and couldn't stop smiling.",
    "Sadness": "My heart felt heavy, and I began to cry.",
    "Anger": "I was furious and couldn't hold back my anger.",
    "Fear": "I was terrified and felt my heart racing.",
    "Surprise": "I was completely shocked and speechless.",
    "Disgust": "I felt disgusted and sick to my stomach."
}
emotion_labels = list(ekman_emotions.keys())
emotion_texts = list(ekman_emotions.values())
emotion_embeddings = embedding_model.encode(emotion_texts)

common_person_nouns = {
    "friend", "friends", "classmate", "classmates", "roommate", "roommates",
    "teacher", "teachers", "tutor", "professor", "professors",
    "mom", "mother", "dad", "father", "parents", "stepmom", "stepdad", "stepparents",
    "sibling", "siblings", "sister", "brother", "boyfriend", "girlfriend", "partner", "ex",
    "spouse", "husband", "wife", "fiancé", "fiancée",
    "child", "children", "kid", "kids", "son", "daughter",
    "boss", "manager", "colleague", "coworker", "teammate",
    "client", "customer", "patient",
    "therapist", "counselor", "psychologist", "psychiatrist", "doctor", "nurse",
    "stranger", "neighbor", "mentor", "coach", "student", "pupil",
    "enemy", "guy", "girl", "man", "woman", "people", "person",
    "someone", "somebody", "anyone", "anybody", "everyone", "everybody"
}

important_relations = {
    "nsubj", "nsubj:pass",
    "obj", "iobj", "obl",
    "xcomp", "ccomp", "advcl",
    "conj", "acl:relcl",
    "root"
}

schema = {
    "memory": [],
    "agent": {
        "client": {
            "traits": [],
            "episodes": []
        }
    }
}
json_file = "memory.json"

with open(json_file, "w", encoding="utf-8") as f:
    json.dump(schema, f, ensure_ascii=False, indent=4)

print("Hello, I'm skkurising counseling bot! Do you need help? Chat with me!\n")

while True:
    buf = []
    input_text = ""
    integrated_text = ""
    last_input_time = time.time()
    time_checker = 15
    people = ["client"]

    while True:
        if msvcrt.kbhit():
            input_text = msvcrt.getwch()
            if input_text == "\r":
                line = "".join(buf).strip()
                if line == "end":
                    sys.exit(0)
                integrated_text += "".join(buf) + "\n"
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

    print("\n입력: "+ integrated_text + "\n")

    refined_text = refine_utterance(integrated_text).lower()

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    doc = chunking_model(refined_text)

    chunks = []
    for sentence in doc.sentences:
        chunks.extend(extract_chunks(sentence))


    chunk_embeddings = embedding_model.encode(chunks)
    predicted_emotions = []
    for chunk_emb in chunk_embeddings:
        sims = cosine_similarity([chunk_emb], emotion_embeddings)[0]
        best_match_idx = np.argmax(sims)
        predicted_emotion = emotion_labels[best_match_idx]
        predicted_emotions.append(predicted_emotion)

    emotion_counts = Counter(predicted_emotions)

    if emotion_counts:
        max_count = max(emotion_counts.values())
        dominant_emotions = [emotion for emotion, count in emotion_counts.items() if count == max_count]
        dominant_emotion_str = ','.join(dominant_emotions)
        emotion_counts = Counter({emotion: emotion_counts[emotion] for emotion in dominant_emotions})


    current_per = extract_entities_and_people(refined_text)

    current_per, current_per_embedding = deduplicate_entities_with_embeddings(current_per)

    current_per = [p.lower() for p in current_per]

    people_embeddings = embedding_model.encode(people, convert_to_tensor=True)

    for agent in current_per:
        agent_emb = embedding_model.encode(agent, convert_to_tensor=True)

        sims = util.cos_sim(agent_emb, people_embeddings)[0]
        max_sim = torch.max(sims).item()

        if max_sim < 0.85:
            people.append(agent)

    for person in people:
            if person not in data["agent"]:
                data["agent"][person] = {
                    "traits": [],
                    "episodes": []
                }


    tree_doc = chunking_model(refined_text)
    tree = {"root": {"agent": "client", "thinks_about": []}}

    for sent in tree_doc.sentences:
        sent_text = sent.text
        person_mentions = [p for p in current_per if p in sent_text]
        if len(person_mentions) < 2:
            continue

        parsed = chunking_model(sent_text)
        filtered_tokens = []
        for s in parsed.sentences:
            for w in s.words:
                word_lower = w.text.lower()
                if word_lower in current_per or w.deprel in important_relations:
                    head_text = s.words[w.head - 1].text if w.head > 0 else "ROOT"
                    filtered_tokens.append({
                        "word": w.text,
                        "head": head_text,
                        "deprel": w.deprel
                    })

        pairs = extract_connections(filtered_tokens, current_per)
        for subj, obj in pairs:
            insert_thought(tree["root"], subj, obj)

    print(tree)

    dfs_tom_tree(tree["root"], refined_text, data)

    past = ""
    for item in data["memory"]:
        event = item.get("event", "")
        answer = item.get("answer", "")
        past += f"내담자: {event}\n응답: {answer}\n\n"

    agent_text = ""

    for agent_name in current_per:  
        episodes = data["agent"].get(agent_name, {}).get("episodes", [])
        if not episodes:
            agent_text += f"[{agent_name}]\n에피소드가 없습니다.\n\n"
            continue

        latest = episodes[-1]
        summary = latest.get("summary", "")
        inference = latest.get("inference", "")
        agent_text += f"[{agent_name}]\n요약: {summary}\n추론: {inference}\n\n"


    answer = counseling_model(refined_text, past, agent_text)

    print("answer: " + answer)

    data_update = {
        "timestamp" : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event" : refined_text,
        "emotions" : dominant_emotion_str,
        "answer" : answer
    }
    data["memory"].append(data_update)

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
