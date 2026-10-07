from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)

from config import HR_DATA_DIR, IT_DATA_DIR


def load_documents():
    documents = []

    for folder in [HR_DATA_DIR, IT_DATA_DIR]:

        for file_path in Path(folder).iterdir():

            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))

            elif file_path.suffix.lower() == ".md":
                loader = TextLoader(
                    str(file_path),
                    encoding="utf-8"
                )

            elif file_path.suffix.lower() == ".docx":
                loader = Docx2txtLoader(str(file_path))

            else:
                continue

            docs = loader.load()

            for doc in docs:
                doc.metadata["source_file"] = file_path.name

                if folder == HR_DATA_DIR:
                    doc.metadata["department"] = "HR"
                else:
                    doc.metadata["department"] = "IT"

                if "policy" in file_path.name.lower():
                    doc.metadata["doc_type"] = "Policy"
                elif "handbook" in file_path.name.lower():
                    doc.metadata["doc_type"] = "Employee Handbook"
                elif "guide" in file_path.name.lower():
                    doc.metadata["doc_type"] = "Troubleshooting Guide"
                else:
                    doc.metadata["doc_type"] = "Document"

                # PDF page numbers start from 0 in many loaders.
                # Convert to human-readable page numbers.
                if "page" in doc.metadata:
                    doc.metadata["page"] = doc.metadata["page"] + 1

            documents.extend(docs)

    return documents