from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os

def load_and_index_documents():
    data_dir = "./data"
    persist_dir = "./chroma_index"
    docs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            path = os.path.join(data_dir, filename)
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
    db.persist()
    print(f"✅ Se cargaron {len(chunks)} chunks y se guardaron en {persist_dir}")

if __name__ == "__main__":
    load_and_index_documents()
