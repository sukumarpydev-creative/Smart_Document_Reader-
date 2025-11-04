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

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
def run_similarity_search(json_path, query, k = 2):
    with open(json_path, 'r', encoding='utf-8') as f:
        file_data = json.load(f)

    document = [ Document (
        page_content= data['text'],
        metadata = data['metadata'],
    )
    for data in file_data]

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vec_store = FAISS.from_documents(document, embeddings)

    # Now we are going to do RAG (Retrival Augmented Generation) integration
    # Retriever is FAISS, LLM is Open AI / Ollama,

    # Defining the Retriever
    retriever = vec_store.as_retriever(search_kwargs = {'k':2})

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

    ## Defining the RAG Pipe Line
    rag_chain = (
        {
            "context" : retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough()
        }
    | prompt_template
    | llm
    )

    response = rag_chain.invoke(query)
    return response.content

# Displaying the result
if __name__ == "__main__":
    query = "What is the relationship between fungi and trees"
    json_file = './JSON Data/Chunk2Json.json'
    search_result = run_similarity_search(json_file, query)
    print(search_result)

