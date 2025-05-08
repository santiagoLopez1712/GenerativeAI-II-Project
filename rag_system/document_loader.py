import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_documents(data_dir="./data", chunk_size=500, chunk_overlap=100):
    """Carga y divide documentos de texto."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"El directorio '{data_dir}' no existe. Por favor, créalo y añade archivos .txt.")
    docs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            path = os.path.join(data_dir, filename)
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"✅ Se cargaron y dividieron {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    chunks = load_and_split_documents()
    print(f"Primer chunk: {chunks[0].page_content[:100]}...")