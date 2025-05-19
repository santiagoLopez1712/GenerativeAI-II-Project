import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber

def load_and_split_documents(data_dir="./data", chunk_size=500, chunk_overlap=100):
    """Loads and splits text and PDF documents."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The directory '{data_dir}' does not exist. Please create it and add .txt or .pdf files.")

    docs = []
    for filename in os.listdir(data_dir):
        path = os.path.join(data_dir, filename)
        try:
            if filename.endswith(".txt"):
                loader = TextLoader(path, encoding="utf-8")
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(path)
            else:
                print(f"⚠️ Skipping unsupported file format: {filename}")
                continue

            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["source"] = filename
            docs.extend(loaded_docs)
        except Exception as e:
            print(f"❌ Error loading the file {filename}: {e}")
            continue

    if len(docs) == 0:
        print("❌ No valid files were found in the directory.")
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"✅ {len(chunks)} chunks were loaded and split.")
    return chunks

def extract_pdf_with_pdfplumber(pdf_path):
    """Extrae texto y tablas de un archivo PDF."""
    text_content = []
    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # Extraer texto
            text = page.extract_text()
            if text:
                text_content.append({"page": page_num + 1, "text": text})

            # Extraer tablas
            page_tables = page.extract_tables()
            for table in page_tables:
                tables.append({"page": page_num + 1, "table": table})

    return text_content, tables

# Ejemplo de uso
pdf_path = "./data/example.pdf"
text_content, tables = extract_pdf_with_pdfplumber(pdf_path)

print("Texto extraído:")
for item in text_content:
    print(f"Página {item['page']}: {item['text'][:100]}...")  # Muestra los primeros 100 caracteres

print(f"\nSe encontraron {len(tables)} tablas.")

if __name__ == "__main__":
    chunks = load_and_split_documents()
    if chunks:
        print(f"First chunk: {chunks[0].page_content[:100]}...")
    else:
        print("No chunks were generated.")
