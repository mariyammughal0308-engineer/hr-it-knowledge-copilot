from pathlib import Path


def get_document_metadata(file_path: str) -> dict:
    path = Path(file_path)

    filename = path.name.lower()

    if path.parent.name == "hr":
        department = "HR"
    elif path.parent.name == "it":
        department = "IT"
    else:
        department = "Unknown"

    if "policy" in filename:
        doc_type = "Policy"
    elif "guide" in filename:
        doc_type = "Troubleshooting Guide"
    else:
        doc_type = "Unknown"

    return {
        "source_file": path.name,
        "department": department,
        "doc_type": doc_type,
    }