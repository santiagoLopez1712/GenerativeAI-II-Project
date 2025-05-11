import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def create_vectorstore(documents, persist_dir="./chroma_index"):
    """Crea y guarda la base de datos vectorial Chroma."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    db = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_dir)
    print(f"✅ Vectorstore guardado en {persist_dir}")
    return persist_dir

def load_vectorstore(persist_dir="./chroma_index", search_kwargs={"k": 8}):
    """Carga la base de datos vectorial desde disco."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs=search_kwargs)
    print(f"✅ Vectorstore cargado desde {persist_dir} con parámetros de búsqueda: {search_kwargs}")
    return retriever

    retrieved_docs = retriever.get_relevant_documents(inputs["query"])
    print(f"🔍 Documentos recuperados para la pregunta: {inputs['query']}")
    for i, doc in enumerate(retrieved_docs):
        print(f"[{i+1}] {doc.page_content[:300]}...\n")

if __name__ == "__main__":
    from document_loader import load_and_split_documents
    chunks = load_and_split_documents()

    if not chunks:
        print("❌ No se cargaron documentos.")
    else:
        persist_directory = create_vectorstore(chunks)
        retriever = load_vectorstore(persist_directory)
        print(f"Retriever creado: {retriever}")

    
