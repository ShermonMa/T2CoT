#!/usr/bin/env python3
# extract_schema.py
import re
import json
import os
from collections import defaultdict

SRC_FILE = '/root/autodl-tmp/T2CoT/data/synthcypher_test/cypher.txt'
OUT_FILE = '/root/autodl-tmp/T2CoT/data/synthcypher_test/generate_schema.json'

# 预编译
NODE_DEC_RE = re.compile(r'\((\w+):(\w+)\)')                    # 变量:标签
PROP_ACC_RE = re.compile(r'(\w+)\.(\w+)\b')                     # 变量.属性
REL_RE      = re.compile(r'\)-\[(\w+):(\w+)\]->\(')            # 关系类型
BRACK_RE    = re.compile(r'\{([^{}]*)\}')                      # 最外层{...}

def extract_one(cypher: str, var2label: dict, node_props: dict, rel_triples: list, depth=0):
    # 1. 节点声明
    for m in NODE_DEC_RE.finditer(cypher):
        var, label = m.groups()
        if var in var2label and var2label[var] != label:
            raise ValueError(f"变量 {var} 标签冲突 {var2label[var]} vs {label}")
        var2label[var] = label
    for m in PROP_ACC_RE.finditer(cypher):
        var, prop = m.groups()
        if var in var2label:
            node_props[var2label[var]].add(prop)

    for m in BRACK_RE.finditer(cypher):
        inner = m.group(1)
        if '(' in inner or '[' in inner or '{' in inner:
            extract_one(inner, var2label, node_props, rel_triples, depth + 1)
        else:
            for kv in inner.split(','):
                if ':' in kv:
                    prop = kv.split(':')[0].strip()
                    if var2label:
                        last_var = next(reversed(var2label))
                        node_props[var2label[last_var]].add(prop)
    print("Relations extraction ")
    for m in REL_RE.finditer(cypher):
        rel_type = m.group(2)
        left_end  = m.start()+2
        right_start = m.end()-2
        left_match  = None
        right_match = None
        for n in NODE_DEC_RE.finditer(cypher):
            if n.end() <= left_end:
                left_match = n
        for n in NODE_DEC_RE.finditer(cypher, right_start):
            right_match = n
            break
        if not (left_match and right_match):
            raise ValueError(f"关系 [:{rel_type}] 左右节点未声明标签")
        from_l = var2label.get(left_match.group(1))
        to_l   = var2label.get(right_match.group(1))
        if from_l is None or to_l is None:
            raise ValueError(f"关系 [:{rel_type}] 左右变量未映射标签")
        rel_triples.append((from_l, rel_type, to_l))

def main():
    node_props = defaultdict(set)
    rel_map    = defaultdict(lambda: {"from": None, "to": None, "properties": set()})

    with open(SRC_FILE, encoding='utf-8') as f:
        count = 0
        for lineno, line in enumerate(f, 1):
            count += 1
            line = line.strip()
            if not line:
                continue
            var2label = {}
            rel_triples = []
            try:
                extract_one(line, var2label, node_props, rel_triples)
            except ValueError as e:
                print(f"[ERROR] Line {lineno}: {e}")
                print(f"  stmt: {line}")
                continue

            for from_l, rel_type, to_l in rel_triples:
                d = rel_map[rel_type]
                d["from"] = from_l
                d["to"]   = to_l

    schema = {
        "nodes": {lbl: sorted(ps) for lbl, ps in node_props.items()},
        "relationships": {
            rel: {"from": d["from"], "to": d["to"], "properties": sorted(list(d["properties"]))}
            for rel, d in rel_map.items()
        }
    }
    bad = [(k, type(k).__name__, k) for k in rel_map.keys()
       if not isinstance(k, (str, int, float, bool, type(None)))]
    if bad:
        for k, t, v in bad:
            for lineno, line in enumerate(open(SRC_FILE, encoding='utf-8'), 1):
                if str(v) in line:          # 粗略定位，够用
                    print(f"  可能语句行号:{lineno} 内容:{line.strip()}")
                    break
        raise TypeError("rel_map 中存在非 str 的 key，详见上方打印")
    with open(OUT_FILE, 'w', encoding='utf-8') as wf:
        json.dump(schema, wf, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()