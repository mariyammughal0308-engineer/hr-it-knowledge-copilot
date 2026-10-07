from document_loader import load_documents
from chunker import split_documents


def main():
    print("Loading documents...")

    documents = load_documents()

    print(f"Loaded documents: {len(documents)}")

    print("Creating chunks...")

    chunks = split_documents(documents)

    print(f"Created chunks: {len(chunks)}")

    print("\nFirst chunk:")
    print(chunks[0].page_content)

    print("\nMetadata:")
    print(chunks[0].metadata)


if __name__ == "__main__":
    main()