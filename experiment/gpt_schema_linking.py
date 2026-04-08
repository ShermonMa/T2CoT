import logging
import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import config
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
output_prompt = False
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import utils
import json

REL_META_DICT   = {}   
REL_CYPHER_DICT = {}  

def load_schema_structured(file_path):
    keep_empty = True
    REL_META_DICT.clear()
    REL_CYPHER_DICT.clear()

    with open(file_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    node = []
    relations = []
    for label, props in schema.get("nodes", {}).items():
        if keep_empty and not props:
            node.append(f"{label}")
        for prop in props:
            node.append(f"{label}.{prop}")

    for rel_type, info in schema.get("relationships", {}).items():
        props      = info.get("properties", [])
        from_label = info.get("from")
        to_label   = info.get("to")

        REL_META_DICT[rel_type]   = {"from": from_label, "to": to_label, "properties": props}
        REL_CYPHER_DICT[rel_type] = f"(:{from_label})-[:{rel_type}]->(:{to_label})"

        if keep_empty or props:
            relations.append(f"{rel_type}")      # 空骨架
        for prop in props:
            relations.append(f"{rel_type}")

    return node, relations

MODEL_PATH = config.SBERT_MODEL_PATH
DEVICE = "cuda"  
# 模拟 Neo4j 数据库 Schema
SCHEMA_ELEMENTS = []
# 相似度阈值
THRESHOLD = config.SBERT_THRESHOLD
def extract_phrases(question: str):
    stop_words = {'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'for', 'to', 'with', 'by', 'is', 'are', 'was', 'were'}
    words = re.findall(r'\b\w+\b', question.lower())
    return [w for w in words if w not in stop_words]

def perform_schema_linking(question: str, schema_elements: list, model: SentenceTransformer,
                           threshold: float = 0.5, topK : int = 3):
    phrases = extract_phrases(question)
    # print(phrases)
    schema_embeddings = model.encode(schema_elements)      # (n_schema, 384)
    # print("Schema elements:", schema_elements)
    phrase_embeddings = model.encode(phrases)              # (n_phrases, 384)
    sim_matrix = cosine_similarity(phrase_embeddings, schema_embeddings)  # (n_phrases, n_schema)
    linked = {}
    for i, phrase in enumerate(phrases):
        sorted_indices = np.argsort(sim_matrix[i])[::-1] 
        top_k_matches = []
        for idx in sorted_indices[:topK]: 
            score = sim_matrix[i][idx]
            matched_schema = schema_elements[idx]
            if score >= threshold:
                top_k_matches.append({
                    "schema": matched_schema,
                    "score": float(score)
                })
            else:
                break 
        linked[phrase] = top_k_matches
    return get_schema_list(linked)

def get_schema_list(linked_result: dict) -> list:
    schema_set = set()
    for phrase, matches in linked_result.items():
        for match in matches:
            schema_set.add(match['schema'])  # 只取 schema 字段
    return list(schema_set)

def control(QUESTION,topK):
    if not os.path.exists(MODEL_PATH):
        logger.warning("SBERT 模型不存在")
        raise FileNotFoundError(f"模型路径不存在: {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH)
    model.to(DEVICE)
    nodes = perform_schema_linking(QUESTION, SCHEMA_ELEMENTS[0], model, THRESHOLD,topK=topK)
    relations = perform_schema_linking(QUESTION, SCHEMA_ELEMENTS[1], model, THRESHOLD,topK=topK)
    return nodes,relations

GPT35_SYSTEM = (
    "You are CypherAgent, an AI that only translates natural-language questions into Neo4j Cypher queries.\n"
    "Let's think step by step.\n"
    "- Identify the nodes, node properties, and relationships in the question.\n"
    "- Match the extracted nodes and their properties to the corresponding schema elements.\n"
    "- Match the extracted relationships to the corresponding schema elements.\n"
    "- If a required field is missing, infer the most likely property name or relationship type "
    "based on common naming conventions and context.\n"
    "- Return **only** the query string, no explanations, no markdown, no greetings."
)
GPT35_SYSTEM_NO_COT = (
    "You are CypherAgent, an AI that only translates natural-language questions into Neo4j Cypher queries.\n"
    "- Return **only** the query string, no explanations, no markdown, no greetings."
)
def _build_gpt35_prompt(question: str, schema_text: str,use_cot = True) -> list:
    user_prompt = (
        f"your text to cypher question: {question}\n"
        f"The following are the available fields in the database:\n{schema_text}\n"
        "Please generate a Cypher query based on these fields with no explanations."
    )
    if use_cot:
        return [
            {"role": "system", "content": GPT35_SYSTEM},
            {"role": "user", "content": user_prompt}
        ]
    else:
        return [
            {"role": "system", "content": GPT35_SYSTEM_NO_COT},
            {"role": "user", "content": user_prompt}
        ]
def run_schema_linking(question: str,schema="",is_schema_solid: bool = False,schema_file="",use_cot=True) -> list:
    if is_schema_solid:
        schema_text = schema
    else:
        global SCHEMA_ELEMENTS
        SCHEMA_ELEMENTS = load_schema_structured(schema_file)
        nodes_schemas, relations_schemas = control(question, topK=5)
        node_part = "Node properties:\n" + "\n".join(
            f"  {i+1}. {s}" for i, s in enumerate(nodes_schemas)
        )
        rel_part = "The Relationships:\n" + "\n".join(
            f"  {i+1}. {REL_CYPHER_DICT[s]}" for i, s in enumerate(relations_schemas)
        )
        schema_text = f"{node_part}\n{rel_part}"
    prompt = _build_gpt35_prompt(question, schema_text,use_cot=use_cot)
    return prompt
    
if __name__ == "__main__":
    print("On Debug Mod Func")
    with open('/root/autodl-tmp/T2CoT/data/all_train_questions_text2cypher.txt', 'r', encoding='utf-8') as f:
        questions = [line.rstrip('\n') for line in f]
    with open('/root/autodl-tmp/T2CoT/data/all_train_answer_text2cypher.txt', 'r', encoding='utf-8') as f:
        answers = [line.rstrip('\n') for line in f]
    with open('/root/autodl-tmp/T2CoT/data/text2cypher/cot_prompt_gemma.jsonl', 'w', encoding='utf-8') as f:
        for test_question,test_answer in zip(questions, answers):
            result = create_lora_corpus(test_question,test_answer)
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
