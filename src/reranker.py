from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self):
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(self, query, documents, top_k=5):
        pairs = [
            [query, document.page_content]
            for document in documents
        ]

        scores = self.model.predict(pairs)

        ranked_documents = sorted(
            zip(documents, scores),
            key=lambda item: item[1],
            reverse=True
        )

        return [
            {
                "document": document,
                "score": float(score)
            }
            for document, score in ranked_documents[:top_k]
        ]