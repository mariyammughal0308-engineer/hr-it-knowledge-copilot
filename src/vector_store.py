from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import BASE_DIR
from document_loader import load_documents
from chunker import split_documents


VECTORSTORE_DIR = BASE_DIR / "vectorstore"


def build_vector_store():
    print("Loading documents...")
    documents = load_documents()

    print("Creating chunks...")
    chunks = split_documents(documents)

    print(f"Total chunks: {len(chunks)}")

    print("Creating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Building vector database...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
        collection_name="hr_it_knowledge_base",
    )

    print("Vector database created successfully!")

    return vectorstore


if __name__ == "__main__":
    build_vector_store()