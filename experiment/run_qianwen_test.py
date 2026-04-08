
import os
import sys
import json
import argparse
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

SL_DIR = Path("/root/autodl-tmp/T2CoT/data_process")
sys.path.insert(0, str(SL_DIR))
from schema_linking import run_schema_linking  # noqa

client = OpenAI(
)
MODEL = "qwen-turbo"
TEMP = 0
MAX_TOKENS = 300

def main(args):
    q_file = Path(args.question_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_suffix = os.getenv("PYTHON_OUT_SUFFIX", "output")   # 拿不到就用默认
    out_file = out_dir / f"{out_suffix}.jsonl"

    with q_file.open(encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    print(f"[INFO] Loaded {len(questions)} questions from {q_file}")
    with out_file.open("w", encoding="utf-8") as fout:
            for q in tqdm(questions, desc="Qwen-turbo"):
                prompt = run_schema_linking(q)
                messages = [{"role": "user", "content": prompt}]
                try:
                    rsp = client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        temperature=TEMP,
                        max_tokens=MAX_TOKENS,
                        extra_body={"enable_thinking": False}
                    )
                    answer = rsp.choices[0].message.content.strip()
                except Exception as e:
                    answer = f"ERROR: {e}"
                fout.write(json.dumps({"question": q, "answer": answer}, ensure_ascii=False) + "\n")
                fout.flush()   

    print(f"[INFO] Done! Results -> {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_file", required=True)
    parser.add_argument("--output_dir", required=True)

    main(parser.parse_args())