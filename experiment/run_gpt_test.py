
import os
import json
from openai import OpenAI
from tqdm import tqdm
QUESTION_FILE = "/root/autodl-tmp/T2CoT/data/text2cypher/all_test_questions_text2cypher.txt"
SCHEMA_FILE = "/root/autodl-tmp/T2CoT/data/text2cypher/new_schema.json"
OUTPUT_FILE   = "/root/autodl-tmp/T2CoT/output/output_gpt.jsonl"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

client = OpenAI(
)


MODEL = "gpt-3.5-turbo-ca"
TEMP  = 0
MAX_TOKENS = 300

# ----------- 2. 极简 prompt -----------
SYSTEM_PROMPT = (
    "You are CypherAgent, an AI that only translates natural-language questions into Neo4j Cypher queries. "
    "Return **only** the query string, no explanations, no markdown, no greetings."
)

# ----------- 3. 批量调用 -----------
def main():
    with open(QUESTION_FILE, encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(questions)} questions.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for q in tqdm(questions, desc="GPT-3.5-turbo"):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": q}
            ]
            try:
                rsp = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=TEMP,
                    max_tokens=MAX_TOKENS
                )
                answer = rsp.choices[0].message.content.strip()
            except Exception as e:
                answer = f"ERROR: {e}"
            print(answer)
            fout.write(json.dumps({"question": q, "answer": answer}, ensure_ascii=False) + "\n")

    print(f"Done! Results -> {OUTPUT_FILE}")

if __name__ == "__main__":
    
    main()