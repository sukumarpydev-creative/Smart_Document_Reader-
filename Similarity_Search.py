"""
Similarity search on JSON document has the following flow
 1. JSON to Document Schema Conversion
    - Opening the JSON file and loading into the Object.
    - Setting up langchain Document schema and passing json object to schema
 2. Setting up the embedding
 3. Setting up the FAISS (Vector) Store from Document and embedding
 4. Setting up the query and passing it to the Similarity Search algorithm FAISS
 5. Printing the Results
"""
import json

from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from Trail_file import adaptive_retriever

load_dotenv()
def run_similarity_search(json_path, query):
    with open(json_path, 'r', encoding='utf-8') as f:
        file_data = json.load(f)

    document = [ Document (
        page_content= data['text'],
        metadata = data['metadata'],
    )
    for data in file_data]

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vec_store = FAISS.from_documents(document, embeddings)

    def adaptive_retrieve(query: str, vector_store, threshold: float = 0.6, max_k: int = 10):
        """
        Dynamically adjusts retrieval depth based on similarity scores.
        Lower score = more relevant.
        """
        results = vector_store.similarity_search_with_score(query, k=max_k)
        strong_matches = [doc for doc, score in results if score <= threshold]

        # fallback to top-1 if nothing passes threshold
        if not strong_matches:
            strong_matches = [results[0][0]]

        print(f"[AdaptiveRetriever] Retrieved {len(strong_matches)} chunks (threshold={threshold})")
        return strong_matches

    # Now we are going to do RAG (Retrival Augmented Generation) integration
    # Retriever is FAISS, LLM is Open AI / Ollama,

    # Defining the Retriever
    adaptive_retriever = RunnableLambda(lambda x: adaptive_retrieve(x, vec_store, threshold=0.55, max_k=8))

    # Defining the LLM
    llm = ChatOpenAI(model='gpt-4o-mini', temperature = 0)

    # Creating the Prompt Template
    prmt = """
    You are a helpful assistant that answers questions using the provided context.
    If the answer is not found in the context, respond with "Not enough information."

    Context:
    {context}

    Question:
    {question}

    """
    prompt_template = ChatPromptTemplate.from_template(prmt)

    ## Defining the RAG Pipeline - Manually
    rag_chain = (
        {
            "context" : adaptive_retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough()
        }
    | prompt_template
    | llm
    )

    response = rag_chain.invoke(query)
    return response.content

# Displaying the result
if __name__ == "__main__":
    query_a = "What is the relationship between fungi and trees"
    json_file = './JSON Data/Chunk2Json.json'
    search_result = run_similarity_search(json_file, query_a)
    print(search_result)

