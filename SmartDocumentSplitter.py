import os, re, json
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

file = 'Sample_PDF.pdf'
_, extension = os.path.splitext(file)

doc_loader = ''
if extension == '.pdf':
    doc_loader = PyPDFLoader(file)
elif extension == '.txt':
    doc_loader = TextLoader(file)
else:
    print(f"Unknown extension: {file}")

to_doc = doc_loader.load()

def cleanText(text):
    text = text.replace("\n", " ")
    text = re.sub("—", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

for doc in to_doc:
    doc.page_content = cleanText(doc.page_content)

chunk_setup = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=50
)

chnk = chunk_setup.split_documents(to_doc)
data = [{
    'text' : doc.page_content,
    'metadata' : doc.metadata
} for doc in chnk]

with open("chunk_to_json.json", "w", encoding="utf-8") as fo:
    json.dump(data, fo, ensure_ascii=False, indent=2)

with open("chunk_to_json.json", "r", encoding="utf-8") as fo:
    ch_json = json.load(fo)

texts = [item['text'] for item in ch_json]
metadatas = [item['metadata'] for item in ch_json]

embed_model = OllamaEmbeddings(model='nomic-embed-text')



vector_store = Chroma.from_texts(
    texts=texts,
    metadatas=metadatas,
    embedding=embed_model,
    collection_name='Write_up_Trees',
    persist_directory="./chroma_v4"
)

query = 'What is the relationship between the fungi and the trees'

embedings = embed_model.embed_documents(query)
print(embedings[0][:10])

results = vector_store.similarity_search_with_score(query, k = 3)

for i, (doc, sco) in enumerate(results, start=1):
    print(i)
    print(doc.page_content[:200])
    print(sco)