from langchain_community.vectorstores import Chroma
from langchain.embeddings import GoogleGenerativeAIEmbeddings
import os

def create_vectorstore(documents, persist_dir="./chroma_index"):
    """Crea y persiste la base de datos vectorial Chroma."""
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_dir)
    db.persist()
    print(f"✅ Base de datos vectorial guardada en {persist_dir}")
    return persist_dir

def load_vectorstore(persist_dir="./chroma_index"):
    """Carga la base de datos vectorial Chroma existente."""
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    print(f"✅ Base de datos vectorial cargada desde {persist_dir}")
    return db.as_retriever()

if __name__ == "__main__":
    # Ejemplo de uso (necesitarías tener documentos en la carpeta 'data')
    from document_loader import load_and_split_documents
    chunks = load_and_split_documents()
    persist_directory = create_vectorstore(chunks)
    retriever = load_vectorstore(persist_directory)
    print(f"Retriever creado: {retriever}")