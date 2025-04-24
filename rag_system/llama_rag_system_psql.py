import logging
import sys
import glob
from pathlib import Path
from sqlalchemy import make_url
import psycopg2

DATA_DIR = Path("../data")

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    node_parser,
    StorageContext
)
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from transformers import AutoTokenizer

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

DB_NAME = "vector_db"
CONNECTION_STRING = "postgresql://postgres:123456@localhost:5432"

def setup_database():
    """Set up PostgreSQL database for vector storage."""
    conn = psycopg2.connect(CONNECTION_STRING)
    conn.autocommit = True

    with conn.cursor() as c:
        c.execute(f"DROP DATABASE IF EXISTS {DB_NAME}")
        c.execute(f"CREATE DATABASE {DB_NAME}")
    
    return conn

llm = LlamaCPP(
    model_path="../llm/mistral-7b-instruct-v0.1.Q2_K.gguf",
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

def save_or_load_index() -> VectorStoreIndex:
    """Create or load vector index using PostgreSQL."""
    setup_database()

    url = make_url(CONNECTION_STRING)
    vector_store = PGVectorStore.from_params(
        database=DB_NAME,
        host=url.host,
        password=url.password,
        port=url.port,
        user=url.username,
        table_name="documents",
        embed_dim=1024,
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    pdf_files = glob.glob(str(DATA_DIR / '**/*.pdf'), recursive=True)
    logging.info(f"Found {len(pdf_files)} PDF files to process")

    documents = SimpleDirectoryReader(input_files=pdf_files).load_data()
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    return index

def main():
    DATA_DIR.mkdir(exist_ok=True)

    index = save_or_load_index()

    query_engine = index.as_query_engine()

    result = query_engine.query(
        "According to OCPP 1.6, what is the best way to start charging session?"
    )
    print("\nQuery Result:")
    print(result)

if __name__ == "__main__":
    main() 