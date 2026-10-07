from langchain_ollama import ChatOllama


def build_context(results):
    context_parts = []

    for result in results:
        document = result["document"]

        source = document.metadata.get("source_file", "Unknown")
        page = document.metadata.get("page")

        if page:
            citation = f"[Source: {source}, Page {page}]"
        else:
            citation = f"[Source: {source}]"

        context_parts.append(
            f"{citation}\n{document.page_content}"
        )

    return "\n\n".join(context_parts)


def generate_answer(question, results):
    context = build_context(results)

    prompt = f"""
You are an internal HR/IT Knowledge Base Assistant.

Answer the user's question ONLY using the official company
documentation provided below.

Do not use outside knowledge.
Do not invent or assume information.

User question:
{question}

Official documentation:
{context}

Rules:
- Give a clear and concise answer.
- Cite each important factual statement.
- Use the exact source filename.
- Include the page number when available.
- If the information is not found, respond exactly:

Information not found in the official company documentation.
Please contact HR/IT directly.
"""

    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )

    response = llm.invoke(prompt)

    return response.content
