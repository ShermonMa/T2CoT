
import json
import pathlib
import sys
import re
from typing import List


def tokenize(cypher: str) -> List[str]:
    cypher = cypher.strip().lower()
    return re.findall(r'\w+', cypher)
def load_golds(p: pathlib.Path) -> List[str]:
    return [line.strip() for line in p.read_text(encoding='utf8').splitlines()
            if line.strip()]
def load_preds(p: pathlib.Path) -> List[str]:
    preds = []
    for ln in p.read_text(encoding='utf8').splitlines():
        ln = ln.strip()
        if not ln:
            continue
        preds.append(json.loads(ln)['answer'].strip())
    return preds


def corpus_bleu(golds: List[str], preds: List[str]) -> float:
    try:
        import nltk
        from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu
    except ImportError:
        raise SystemExit("nltk：pip install nltk")

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    list_of_references = [[tokenize(g)] for g in golds]
    hypotheses = [tokenize(p) for p in preds]

    return nltk_corpus_bleu(list_of_references, hypotheses)
def eval_bleu(answer_path: pathlib.Path,
                   jsonl_path: pathlib.Path,
                   out_path: pathlib.Path) -> None:
    answer_path, jsonl_path, out_path = map(pathlib.Path, [answer_path, jsonl_path, out_path])
    golds = load_golds(answer_path)
    preds = load_preds(jsonl_path)
    if len(golds) != len(preds):
        raise ValueError(f'样本数不一致：{len(golds)=} vs {len(preds)=}')

    bleu4 = corpus_bleu(golds, preds)
    print(f'BLEU-4: {bleu4:.4f}')
    out_path.write_text(json.dumps({'BLEU': bleu4}, ensure_ascii=False), encoding='utf8')

if __name__ == '__main__':
    answer_file = "/root/autodl-tmp/T2CoT/data/answer_text2cypher_500.txt"
    output_raw = "/root/autodl-tmp/T2CoT/output/qwen_turbo/output2.jsonl"
    bleu_file = "/root/autodl-tmp/T2CoT/output/qianwengpt.txt"
    eval_bleu(answer_file,output_raw,bleu_file)