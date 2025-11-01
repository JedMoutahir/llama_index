import argparse, os, json
from pathlib import Path
from typing import Optional, List, Dict, Any
from llama_index.core import Settings, load_index_from_storage, StorageContext
from llama_index.llms.openai import OpenAI

def load_llm_from_env():
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY", "dummy")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    # OpenAI-compatible (works with vLLM too)
    llm = OpenAI(model=model, api_key=api_key, base_url=base_url)
    return llm, {"model": model, "base_url": base_url}

def run_single(query_engine, question: str) -> Dict[str, Any]:
    resp = query_engine.query(question)
    # Extract sources if available
    sources = []
    try:
        for s in resp.source_nodes:
            sources.append({
                "doc_id": getattr(s.node, "doc_id", None),
                "score": getattr(s, "score", None),
                "meta": s.node.metadata if hasattr(s, "node") else {}
            })
    except Exception:
        pass
    return {"question": question, "answer": str(resp), "sources": sources}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persist-dir", required=True)
    ap.add_argument("--question", default=None, help="Single question to ask")
    ap.add_argument("--questions-file", default=None, help='JSONL with {"question": "..."} per line')
    ap.add_argument("--out", default="runs/rag_out")
    args = ap.parse_args()

    persist_dir = Path(args.persist_dir)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    answers_path = out_dir / "answers.jsonl"
    meta_path = out_dir / "run_meta.json"

    llm, llm_meta = load_llm_from_env()
    Settings.llm = llm

    storage = StorageContext.from_defaults(persist_dir=str(persist_dir))
    index = load_index_from_storage(storage)
    qe = index.as_query_engine(similarity_top_k=4)

    # collect questions
    questions: List[str] = []
    if args.question:
        questions.append(args.question)
    if args.questions_file:
        with open(args.questions_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                q = obj.get("question")
                if q: questions.append(q)

    if not questions:
        print("No question provided."); return

    with open(answers_path, "w", encoding="utf-8") as fo:
        for q in questions:
            rec = run_single(qe, q)
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(meta_path, "w", encoding="utf-8") as fm:
        json.dump({"llm": llm_meta, "n_questions": len(questions)}, fm, indent=2)

    print("Wrote:", answers_path)

if __name__ == "__main__":
    main()
