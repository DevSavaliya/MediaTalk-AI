from pathlib import Path
from functools import lru_cache
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = str(Path(__file__).resolve().parents[1] / "vector_db")
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs = {"device" : 'cpu'}
    )

def build_vector_store(transcript: str, collection_name=COLLECTION_NAME) -> Chroma:
    if not transcript.strip():
        raise ValueError("Cannot build a vector database from an empty transcript.")
    print("Building local vector database (loading the embedding model)...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata = {'chunk_index' : i})
        for i,chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents= docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_DIR
    )

    print(f"Vector database saved: {Path(CHROMA_DIR) / 'chroma.sqlite3'}")
    return vector_store



def load_vector_store(collection_name=COLLECTION_NAME) -> Chroma:
    if not (Path(CHROMA_DIR) / "chroma.sqlite3").is_file():
        raise FileNotFoundError("No saved video database found. Process a video first with python main.py.")
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function= embeddings,
        persist_directory=CHROMA_DIR,
        create_collection_if_not_exists=False,
    )

    if vector_store._collection.count() == 0:
        raise ValueError("The saved database contains no transcript. Process a video first.")
    return vector_store

def get_retriever(vector_store : Chroma, k :int = 4):
    return vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {"k":k}
    )

