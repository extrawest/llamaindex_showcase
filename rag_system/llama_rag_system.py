import logging
import sys
import glob
from pathlib import Path

DATA_DIR = Path("../data")
INDEX_DIR = Path("../index")

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    node_parser,
    StorageContext,
    load_index_from_storage, service_context
)
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

llm = LlamaCPP(
    model_path="../llm/mistral-7b-instruct-v0.1.Q2_K.gguf",  # Local path to downloaded model
    context_window=4096,
    max_new_tokens=512,
    model_kwargs={
        'n_gpu_layers': 0,
        'n_threads': 8,
        'n_ctx': 4096,
        'n_batch': 512,
    },
    verbose=True
)

embedding_model = HuggingFaceEmbedding(model_name="WhereIsAI/UAE-Large-V1")

Settings.llm = llm
Settings.embed_model = embedding_model
Settings.node_parser = node_parser.SentenceSplitter()
Settings.chunk_size = 256
Settings.chunk_overlap = 20

Settings.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").encode

def save_or_load_index(index_dir: Path) -> VectorStoreIndex:
    """Load existing index or create new one if it doesn't exist."""
    index_exists = any(item for item in Path(index_dir).iterdir() if item.name != '.gitkeep')

    if index_exists:
        logging.info(f"Loading persisted index from: {index_dir}")
        storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
        index = load_index_from_storage(storage_context, service_context=service_context)

        return index
    else:
        logging.info("Persisted index not found, creating a new index...")

        pdf_files = glob.glob(str(DATA_DIR / '**/*.pdf'), recursive=True)

        documents = SimpleDirectoryReader(input_files=pdf_files).load_data()
        index = VectorStoreIndex.from_documents(
            documents,
            show_progress=True
        )
        index.storage_context.persist(persist_dir=str(index_dir))

        return index

def main():
    DATA_DIR.mkdir(exist_ok=True)
    INDEX_DIR.mkdir(exist_ok=True)

    index = save_or_load_index(INDEX_DIR)

    query_engine = index.as_query_engine()

    result = query_engine.query(
        "According to OCPP 1.6,  what is the best way to start charging session?"
    )
    print("\nQuery Result:")
    print(result)

if __name__ == "__main__":
    main() 