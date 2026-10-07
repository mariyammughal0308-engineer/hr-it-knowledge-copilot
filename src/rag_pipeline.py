from hybrid_retriever import HybridRetriever
from reranker import CrossEncoderReranker
from answer_generator import generate_answer


class RAGPipeline:
    def __init__(self):
        print("Starting RAG pipeline...")

        self.retriever = HybridRetriever()
        print("Hybrid retriever ready.")

        self.reranker = CrossEncoderReranker()
        print("Reranker ready.")

    def retrieve(self, query: str, top_k: int = 5):
        candidates = self.retriever.retrieve(
            query=query,
            top_k=20
        )

        print(f"Retrieved candidates: {len(candidates)}")

        results = self.reranker.rerank(
            query=query,
            documents=candidates,
            top_k=top_k
        )

        return results

    def answer(self, question: str):
        results = self.retrieve(question)

        # Out-of-scope fallback
        if not results:
            return (
                "Information not found in the official company documentation.\n"
                "Please contact HR/IT directly."
            )

        # Check reranker confidence
        top_result = results[0]

        score = top_result.get("score", 0)

        if score < 0.25:
            return (
                "Information not found in the official company documentation.\n"
                "Please contact HR/IT directly."
            )

        final_answer = generate_answer(
            question,
            results
        )

        return final_answer


if __name__ == "__main__":
    pipeline = RAGPipeline()

    question = input("\nAsk your question: ")

    answer = pipeline.answer(question)

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(answer)