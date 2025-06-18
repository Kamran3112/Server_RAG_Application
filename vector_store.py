import os
import hashlib
from langchain_community.vectorstores import Chroma
import pdfplumber
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHROMA_DB_DIR, PDF_STORAGE_PATH, OLLAMA_BASE_URL
from langchain_community.embeddings import HuggingFaceEmbeddings

Embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")


# Load embedding model from Ollama
# embedding_model = OllamaEmbeddings(model="deepseek-r1:1.5b", base_url=OLLAMA_BASE_URL)
# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
# table content into text
def table_to_sentences(table, page_num, table_index, file_id):
    if not table or len(table) < 2:
        return []  # Skip if no headers or rows

    headers = table[0]
    data_rows = table[1:]
    sentences = []

    for row_num, row in enumerate(data_rows, start=1):
        if not any(row):  # Skip completely empty rows
            continue

        # Pair header and value
        cells = [f"{(header or '').strip()}: {(value or '').strip()}" for header, value in zip(headers, row)]

        sentence = f"Row {row_num}: " + ", ".join(cells)

        doc = Document(
            page_content=sentence,
            metadata={
                "file_id": file_id,
                "page": page_num + 1,
                "type": "table_row",
                "table_index": table_index,
                "row": row_num
            }
        )
        sentences.append(doc)

    return sentences

def process_and_index(path, file_id):
    documents = []
    table_count = 0

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                raw_doc = Document(page_content=text, metadata={
                    "file_id": file_id,
                    "page": page_num + 1,
                    "type": "text"
                })
                text_chunks = text_splitter.split_documents([raw_doc])
                documents.extend(text_chunks)

            tables = page.extract_tables()
            print(f"📊 Page {page_num + 1}: {len(tables)} tables found")

            for table in tables:
                row_documents = table_to_sentences(table, page_num, table_count, file_id)
                documents.extend(row_documents)
                table_count += 1


    # ✅ Step 1: Initialize embedding model
    Embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

    # ✅ Step 2: Create or load Chroma collection
    vectordb = Chroma(
        collection_name=file_id,
        embedding_function=Embedding_model,
        persist_directory="chroma_storage"
    )

    # ✅ Step 3: Add documents to Chroma
    if documents:
        vectordb.add_documents(documents)
        vectordb.persist()
        print(f"Indexed {len(documents)} documents into Chroma collection '{file_id}'")
    else:
        print("No documents to index.")

# save file

def save_file(file_bytes: bytes, filename: str, file_id: str, upload_dir="chroma_storage/uploads"):
    os.makedirs(upload_dir, exist_ok=True)  # create directories if not exist
    file_ext = filename.split('.')[-1]
    path = os.path.join(upload_dir, f"{file_id}.{file_ext}")
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path

# Retrieve similar docs from the corresponding collection
def search_similar_documents(query, file_id):
    try:
        vectordb = Chroma(
            collection_name=file_id,
            embedding_function=Embedding_model,
            persist_directory="chroma_storage"
        )
        return vectordb.similarity_search(query, k=3)
    except Exception as e:
        print(f"[!] Error retrieving documents for {file_id}: {e}")
        return []

# for delete chat history
def delete_index(file_id: str):
    vectordb = Chroma(
        collection_name=file_id,
        embedding_function=Embedding_model,
        persist_directory=CHROMA_DB_DIR
    )
    vectordb.delete_collection()

