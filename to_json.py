import os, re, json
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter

## Checking the file extension and executing the appropriate loader
file_path = './DataFrom/Sample_PDF.pdf'
ext = os.path.splitext(file_path)[1]

file_loader = ''
try:
    if ext == '.pdf':
        print("File is of PDF Format")
        file_loader = PyPDFLoader(file_path)

    elif ext == '.txt':
        print("File is of TXT Format")
        file_loader = TextLoader(file_path)
    else:
        raise ValueError
except ValueError:
    print("File extension not supported")
    exit()

## Loading file to Object
file_reader = file_loader.load()

## Cleaning Function
def text_cleaner (text: str) -> str :
    text = re.sub('\n', ' ', text)
    text = re.sub('\t', ' ', text)
    text = re.sub(';', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

for doc in file_reader:
    doc.page_content = text_cleaner(doc.page_content)

## Splitting and Chunking

split_params = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = split_params.split_documents(file_reader)

# Conversion and storing chunks in JSON format
# Setting up the JSON data format
data = [
    {
        'text' : ch.page_content,
        'metadata' : ch.metadata
    }
    for ch in chunks
]

## Opening a file and Saving data to JSON
with open("./JSON Data/Chunk2Json.json","w", encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False,indent=2)

print(f"{ext} file converted to JSON and Stored at:   ./JSON Data/Chunk2Json.json")

