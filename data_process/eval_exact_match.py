
import json
import re
import pathlib
from typing import Union

PathLike = Union[str, pathlib.Path]

def normalize(sql: str) -> str:
    sql = sql.strip()
    sql = re.sub(r'\s+', ' ', sql)
    sql = sql.rstrip(';')
    return sql

def compute_exact_match(answer_path: PathLike, jsonl_path: PathLike, out_path: PathLike) -> None:
    answer_path, jsonl_path, out_path = map(pathlib.Path, [answer_path, jsonl_path, out_path])
    golds = [normalize(line) for line in answer_path.read_text(encoding='utf8').splitlines() if line.strip()]
    preds = []
    for ln in jsonl_path.read_text(encoding='utf8').splitlines():
        ln = ln.strip()
        if not ln:
            continue
        preds.append(normalize(json.loads(ln)['answer']))

    if len(preds) != len(golds):
        raise ValueError(f'样本数不一致：preds={len(preds)} vs golds={len(golds)}')

    correct = sum(p == g for p, g in zip(preds, golds))
    accuracy = correct / len(golds)
    print(f'Exact-match accuracy: {accuracy:.4f}  ({correct}/{len(golds)})')
    out_path.write_text(
        json.dumps({'total': len(golds), 'correct': correct, 'accuracy': accuracy}, ensure_ascii=False, indent=0),
        encoding='utf8'
    ) 
if __name__ == '__main__':
    compute_exact_match(
        '/root/autodl-tmp/T2CoT/data/synthetic-text2cypher-gpt4turbo/cypher.txt', 
        '/root/autodl-tmp/T2CoT/output/qwen_turbo/output3.jsonl', 
        '/root/autodl-tmp/T2CoT/output/eval_qianwen_em.txt'
        )