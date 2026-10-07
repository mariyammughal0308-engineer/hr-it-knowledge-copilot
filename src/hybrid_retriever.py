from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import BASE_DIR
from document_loader import load_documents
from chunker import split_documents
from bm25_retriever import BM25Retriever


VECTORSTORE_DIR = BASE_DIR / "vectorstore"


class HybridRetriever:
    def __init__(self):
        documents = load_documents()
        self.documents = split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectorstore = Chroma(
            collection_name="hr_it_knowledge_base",
            embedding_function=embeddings,
            persist_directory=str(VECTORSTORE_DIR),
        )

        self.bm25 = BM25Retriever(self.documents)

    def retrieve(self, query, top_k=20):
        dense_results = self.vectorstore.similarity_search(
            query,
            k=10
        )

        sparse_results = self.bm25.retrieve(
            query,
            top_k=10
        )

        # Combine both rankings using Reciprocal Rank Fusion.
        rrf_scores = {}

        for rank, document in enumerate(dense_results, start=1):
            key = self._document_key(document)
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (60 + rank)

        for rank, document in enumerate(sparse_results, start=1):
            key = self._document_key(document)
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (60 + rank)

        all_documents = dense_results + sparse_results

        unique_documents = {}
        for document in all_documents:
            unique_documents[self._document_key(document)] = document

        ranked_documents = sorted(
            unique_documents.values(),
            key=lambda doc: rrf_scores[self._document_key(doc)],
            reverse=True
        )

        return ranked_documents[:top_k]

    @staticmethod
    def _document_key(document):
        return (
            document.metadata.get("source_file", ""),
            document.metadata.get("page", ""),
            document.page_content,
        )