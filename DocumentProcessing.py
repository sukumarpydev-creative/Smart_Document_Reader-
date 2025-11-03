import re, json
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# loader = TextLoader('Sample.txt', encoding='utf-8')
# docuement = loader.load()
#
# print(docuement[0].page_content[:200])
#
# # Splitting the Text into Chunks
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=50
# )
#
# chunks = splitter.split_documents(docuement)
#
# print(chunks[0].page_content)
# # print(chunks[1].page_content)


## Loading from PDF
loader_pdf = PyPDFLoader('Sample_PDF.pdf')
doc_pdf = loader_pdf.load()

# print('\n\nPDF content is \n\n', doc_pdf[0].metadata)

def clean_text(text):
    text = text.replace('\n', ' ')  #Removal of Linebreaks
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'-\s+', ' ', text)
    return text.strip()

for doc in doc_pdf:
    doc.page_content = clean_text(doc.page_content)

pdf_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
)

chunks = pdf_splitter.split_documents(doc_pdf)
# print(chunks[0].page_content[:1000])

data = [{
    'text' : doc.page_content,
    'metadata' : doc.metadata
} for doc in chunks
]

with open("json_data.json", "w", encoding='utf-8') as fl:
    json.dump(data, fl, indent = 2, ensure_ascii = False)

with open("json_data.json", "r", encoding='utf-8') as fo:
    ch_one = json.load(fo)

# print(ch_one[0]['metadata']['producer'])

chnk_length = [len(ch['text']) for ch in ch_one]
print(chnk_length)
min_ch_len = min(chnk_length)
max_ch_len = max(chnk_length)
avg_ch_len = sum(chnk_length)/len(chnk_length)

print(f"Minimum length: {min_ch_len}")
print(f"Maximum length: {max_ch_len}")
print(f"Average length: {avg_ch_len}")