
import json
import requests
from pathlib import Path
from typing import Iterable
import json

DATASET = "MohammadOthman/mo-customer-support-tweets-945k"


def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    return t


def local_rows(path: str, offset: int, length: int) -> list[dict]:
    rows: list[dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        # The file is a JSON array; stream-read by skipping to offset
        data = json.load(f)
        slice_ = data[offset: offset + length]
        for row in slice_:
            rows.append(row)
    return rows


def yield_docs(limit: int | None = None, page_size: int = 1000, local_path: str | None = 'data/preprocessed_data.json') -> Iterable[dict]:
    produced = 0
    offset = 0
    while True:
        to_get = page_size
        if local_path and Path(local_path).exists():
            rows = local_rows(local_path, offset, to_get)
        else:
            # fallback to datasets-server if needed (may 422)
            url = (
                "https://datasets-server.huggingface.co/rows?"
                f"dataset={DATASET}&config=default&split=train&offset={offset}&length={to_get}"
            )
            r = requests.get(url, timeout=60)
            if r.status_code != 200:
                break
            data = r.json()
            rows = [item.get('row', {}) for item in data.get('rows', [])]
        if not rows:
            break
        for row in rows:
            user = normalize(row.get("input", ""))
            agent = normalize(row.get("output", ""))
            if not user or not agent:
                continue
            qa_text = f"Customer: {user}\n\nAgent: {agent}"
            doc = {"input": user, "qa": qa_text}
            yield doc
            produced += 1
            if limit is not None and produced >= limit:
                return
        offset += to_get


def main(out_input: str, out_qa: str, limit: int | None = None) -> None:
    Path(out_input).parent.mkdir(parents=True, exist_ok=True)
    Path(out_qa).parent.mkdir(parents=True, exist_ok=True)
    with open(out_input, "w", encoding="utf-8") as fi, open(out_qa, "w", encoding="utf-8") as fq:
        for idx, doc in enumerate(yield_docs(limit)):
            fi.write(json.dumps({"id": idx, "text": doc["input"]}, ensure_ascii=False) + "\n")
            fq.write(json.dumps({"id": idx, "text": doc["qa"]}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--out_input', default='store/input_texts.jsonl')
    p.add_argument('--out_qa', default='store/qa_texts.jsonl')
    p.add_argument('--limit', type=int, default=100000)
    args = p.parse_args()
    main(args.out_input, args.out_qa, args.limit)
