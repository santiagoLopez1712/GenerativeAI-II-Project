from langchain_core.prompts import PromptTemplate
from langchain_core.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import Gemini
import os

def create_chat_chain(retriever):
    """Crea la cadena conversacional RAG."""
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    llm = Gemini(model_name="gemini-2.0-flash")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    prompt_template = """Utiliza la siguiente información para responder la pregunta del usuario.
    Si la respuesta no se encuentra en la información proporcionada, responde "No tengo la información para responder a esa pregunta."
    No hagas suposiciones.

    Contexto: {context}
    Pregunta: {question}

    Respuesta:"""
    QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chat_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )
    print("✅ Cadena de chat RAG creada.")
    return chat_chain

if __name__ == "__main__":
    # Ejemplo de uso (necesitas haber creado la base de datos vectorial primero)
    from retriever import load_vectorstore
    persist_directory = "./chroma_index" # Asegúrate de que esta ruta sea correcta
    retriever = load_vectorstore(persist_directory)
    if retriever:
        chat_chain = create_chat_chain(retriever)
        result = chat_chain({"question": "¿Cuál es la idea principal del documento?"})
        print(f"Respuesta: {result['answer']}")
        print(f"Documentos fuente: {[os.path.basename(doc.metadata['source']) for doc in result['source_documents']]}")