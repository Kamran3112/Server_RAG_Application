import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

# ✅ Config
OLLAMA_BASE_URL = "http://localhost:11434"
CHROMA_DB_DIR = "chroma_storage"

# ✅ Setup
embedding_model = OllamaEmbeddings(model="deepseek-r1:1.5b", base_url=OLLAMA_BASE_URL)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def process_and_index_pdf(path, file_id):
    documents = []
    table_count = 0

    print(f"📥 Reading PDF: {path}")
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            print(f"📄 Page {page_num + 1}: text length = {len(text) if text else 0}")

            if text:
                raw_doc = Document(page_content=text, metadata={
                    "file_id": file_id,
                    "page": page_num + 1,
                    "type": "text"
                })
                text_chunks = text_splitter.split_documents([raw_doc])
                print(f"📝 Page {page_num + 1}: {len(text_chunks)} chunks")
                documents.extend(text_chunks)

            tables = page.extract_tables()
            print(f"📊 Page {page_num + 1}: {len(tables)} tables found")

            for table in tables:
                formatted_table = "\n".join(
                    [" | ".join(cell if cell else "" for cell in row) for row in table]
                )
                doc = Document(page_content=formatted_table, metadata={
                    "file_id": file_id,
                    "page": page_num + 1,
                    "type": "table",
                    "table_index": table_count
                })
                documents.append(doc)
                table_count += 1

    print(f"🧾 Total documents to index: {len(documents)}")

    if not documents:
        print("❌ No documents extracted. Skipping indexing.")
        return

    # ✅ Index into Chroma
    vectordb = Chroma(
        collection_name=file_id,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DB_DIR
    )

    vectordb.add_documents(documents)
    vectordb.persist()
    print(f"✅ Indexed {len(documents)} documents into Chroma collection '{file_id}'")

# ✅ MAIN: Replace these with your file path and ID
if __name__ == "__main__":
    pdf_path = "/home/onstak-dl2/Server_RAG_Application/chroma_storage/uploads/table_content.pdf"         # 🔁 Change this to your actual PDF path
    file_id = "testing"      # 🔁 Logical name for the collection

    if not os.path.isfile(pdf_path):
        print(f"❌ File not found: {pdf_path}")
    else:
        process_and_index_pdf(pdf_path, file_id)
