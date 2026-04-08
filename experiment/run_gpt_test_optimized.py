
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import logging
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
QUESTION_FILE = "/root/autodl-tmp/T2CoT/data/text2cypher/all_test_questions_text2cypher.txt"
SCHEMA_FILE   = "/root/autodl-tmp/T2CoT/data/text2cypher/new_schema.json"
OUTPUT_DIR    = "/root/autodl-tmp/T2CoT/output"
USE_COT = False
os.makedirs(OUTPUT_DIR, exist_ok=True)
START_FROM = 0
import sys
sys.path.insert(0, str(Path(__file__).parent))
from gpt_schema_linking import run_schema_linking   # 返回 messages:list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true",
                        help="真正调用模型；否则只生成 prompt")
    parser.add_argument("--model", default="gpt-3.5-turbo",
                        help="模型名，默认 gpt-3.5-turbo")
    parser.add_argument("--max_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--question", required=True, help="问题文件 txt")
    parser.add_argument("--schema",   required=True, help="schema json")
    parser.add_argument("--out",      required=True, help="输出 jsonl（可含路径）")
    parser.add_argument("--cot", action="store_true", help="cot")
    parser.add_argument("--start_from", type=int, default=0, help="从第几条开始")
    return parser.parse_args()

def call_gpt(messages: list, client, model: str, max_tokens: int, temperature: float) -> str:
    import time, httpx          # 新增 httpx 用来读响应体
  
    try:
        rsp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        import traceback, httpx
        print("!!! exception type :", type(e).__name__)
        print("!!! exception msg  :", str(e))
        if hasattr(e, "response") and e.response:
            try:
                body = e.response.read().decode(errors="ignore")
            except Exception:
                body = "<no body>"
            print("!!! http body      :", body)
        traceback.print_exc()         
        return f"ERROR: {type(e).__name__} - {e}"
def main():
    args = parse_args()
    out_file = Path(OUTPUT_DIR) / args.out
    START_FROM = args.start_from -1
    print(f"本次任务从第{START_FROM}开始")
    # 读问题
    with open(QUESTION_FILE, encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(questions)} questions.")

    if args.run:

        from openai import OpenAI
        client = OpenAI(
        )
        
        
    else:
        client = None
    COUNT = 0
    with open(out_file, "w", encoding="utf-8") as fout:
        for q in tqdm(questions, desc="GPT-3.5" if args.run else "PromptOnly",mininterval=60.0):
            if COUNT < START_FROM:
                COUNT += 1
                continue
            messages = run_schema_linking(question=q, schema="", is_schema_solid=False,schema_file=SCHEMA_FILE,use_cot = True)
            if args.run:
                answer = call_gpt(messages, client, args.model,
                                  args.max_tokens, args.temperature)
            else:
                answer = {"messages": messages}   
            fout.write(json.dumps({"question": q, "answer": answer},
                                  ensure_ascii=False) + "\n")
            fout.flush()
    os.sync()
    print(f"Done! -> {out_file}")

if __name__ == "__main__":
    args = parse_args()
    QUESTION_FILE = Path(args.question)
    SCHEMA_FILE   = Path(args.schema)
    OUT_FILE      = Path(args.out)
    USE_COT = args.cot
    main()