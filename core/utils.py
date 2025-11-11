from langchain_core.documents import Document

def format_docs(docs):
    """Join a list of Document objects into a single context string."""
    return "\n\n".join(d.page_content for d in docs)

def load_sample_docs():
    """Placeholder for document loading logic (replace later)."""
    return [
        Document(page_content="The company offers 20 annual leaves per year."),
        Document(page_content="Maternity leave allows 26 weeks of paid time off."),
        Document(page_content="There is currently no paternity leave policy mentioned.")
    ]
