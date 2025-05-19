import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def create_vectorstore(documents, persist_dir="./chroma_index"):
    """Creates and saves the Chroma vector database."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    db = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_dir)
    print(f"✅ Vectorstore saved in {persist_dir}")
    return persist_dir

def load_vectorstore(persist_dir="./chroma_index", search_kwargs={"k": 8}):
    """Loads the vector database from disk."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs=search_kwargs)
    print(f"✅ Vectorstore loaded from {persist_dir} with search parameters: {search_kwargs}")
    return retriever

    retrieved_docs = retriever.get_relevant_documents(inputs["query"])
    print(f"🔍 Documents retrieved for the query: {inputs['query']}")
    for i, doc in enumerate(retrieved_docs):
        print(f"[{i+1}] {doc.page_content[:300]}...\n")

if __name__ == "__main__":
    from document_loader import load_and_split_documents
    chunks = load_and_split_documents()

    if not chunks:
        print("❌ No documents were loaded.")
    else:
        persist_directory = create_vectorstore(chunks)
        retriever = load_vectorstore(persist_directory)
        print(f"Retriever created: {retriever}")


