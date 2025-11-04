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

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Json to Document conversion
json_file = './JSON Data/Chunk2Json.json'

with open(json_file, 'r', encoding='utf-8') as f:
    file_data = json.load(f)

document = [ Document (
    page_content= data['text'],
    metadata = data['metadata'],
)
for data in file_data]

# Setting up the embedding
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Setting up the FAISS store
vec_store = FAISS.from_documents(document, embeddings)

# Similarity Search from the query
query = "What is the relationship between fungi and trees"
search_result = vec_store.similarity_search_with_score(query, k=2)

# Displaying the result
for i, (doc, sco) in enumerate(search_result, start=1):
    print(f"Result [{i}] has score of {sco} and the result is :")
    print(doc)

