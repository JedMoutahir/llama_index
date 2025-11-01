import argparse, os
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Directory containing PDFs")
    ap.add_argument("--persist-dir", required=True, help="Where to persist the index storage")
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--chunk-size", type=int, default=1024)
    ap.add_argument("--chunk-overlap", type=int, default=100)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    persist_dir = Path(args.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Set embedding
    Settings.embed_model = HuggingFaceEmbedding(model_name=args.embed_model)
    # Chunking parameters
    Settings.chunk_size = args.chunk_size
    Settings.chunk_overlap = args.chunk_overlap

    # Load PDFs
    reader = SimpleDirectoryReader(input_dir=str(input_dir), recursive=True, required_exts=[".pdf"])
    docs = reader.load_data()
    if not docs:
        print("No PDFs found. Place files under:", input_dir)
        return

    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=str(persist_dir))
    print("Index built & persisted to:", persist_dir)

if __name__ == "__main__":
    main()
