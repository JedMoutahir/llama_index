# Jed’s LlamaIndex PDF‑RAG Demo (OpenAI/vLLM compatible)

A compact, reproducible **PDF RAG** setup using **LlamaIndex**:
- Ingest PDFs → build a **VectorStoreIndex** (local, persisted on disk)
- Query with any **OpenAI‑compatible** endpoint (OpenAI, vLLM, etc.)
- Batch querying over a JSONL of questions
- Minimal deps; GPU optional (only helps for embedding speed)

> If you run vLLM from our other repo, you can point this demo to that endpoint.

---

## Folder
```
examples/jed_llamaindex_pdf_rag/
├── README.md
├── env.yml
├── build_index.py
├── query_rag.py
└── sample_questions.jsonl
```

## Quickstart

### 1) Create env
```bash
conda env create -f examples/jed_llamaindex_pdf_rag/env.yml
conda activate li-pdf-rag
```

### 2) Build index from PDFs
Put your PDFs under `./pdfs/` (or any folder), then:
```bash
python examples/jed_llamaindex_pdf_rag/build_index.py   --input-dir ./pdfs   --persist-dir ./storage_pdf_rag   --embed-model sentence-transformers/all-MiniLM-L6-v2
```

### 3) Query the index (single or batch)
Set your OpenAI-compatible endpoint via env vars (examples):
```bash
# OpenAI
export OPENAI_API_KEY=sk-... 
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=gpt-4o-mini

# or vLLM (from the vllm repo you set up)
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_MODEL=llama-3.1-8b-instruct
```

Single question:
```bash
python examples/jed_llamaindex_pdf_rag/query_rag.py   --persist-dir ./storage_pdf_rag   --question "What does section 3.2 say about evaluation metrics?"   --out runs/rag_single
```

Batch (JSONL with {"question": "..."} lines):
```bash
python examples/jed_llamaindex_pdf_rag/query_rag.py   --persist-dir ./storage_pdf_rag   --questions-file examples/jed_llamaindex_pdf_rag/sample_questions.jsonl   --out runs/rag_batch
```

Outputs land in `runs/.../answers.jsonl` with sources per answer.

---

## Notes
- Embeddings default: `sentence-transformers/all-MiniLM-L6-v2` (small & fast).
- You can switch to larger models (e.g., `bge-large`) by changing `--embed-model`.
- For CPU-only boxes, it still works—just slower on first embed pass (models are cached).

