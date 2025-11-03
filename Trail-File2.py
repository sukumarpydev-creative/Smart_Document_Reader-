# PDF document loading, cleaning, chunking
import re
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import false

# PDF doc loading
load_pdf = PyPDFLoader('Sample_PDF.pdf')
text_doc = load_pdf.load()

print(f"Inintial Docuemnt :\n\n{text_doc[0].page_content[:300]}")
print(f"\n{10*'*'}\n")


#Text document cleaning
def text_cleaner(txt:str)->str:
    txt = txt.replace("\n"," ")
    txt = re.sub(r";"," ", txt)
    txt = re.sub(r"\s+"," ",txt)
    return txt.strip()

for doc in text_doc:
    doc.page_content = text_cleaner(doc.page_content)

print(f"Cleaned Document : \n\n{text_doc[0].page_content[:300]}")

text_spltng = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
)

chunks = text_spltng.split_documents(text_doc)
# print(f"\nChunks:\n\n{chunks[0]} \n\n{chunks[1]}")

## preparing the chunks in JSON format
data = [
    {
        'text':doc.page_content,
        'metadata':doc.metadata
     }
    for doc in chunks
]

with open("processed_junks.json", 'w+', encoding='utf-8' ) as f:
    json.dump(data, f, ensure_ascii = False ,indent=2)

print(f"Saved {len(data)} chunks")


