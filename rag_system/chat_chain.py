from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain

def create_chat_chain(retriever):
    """Crea una cadena de recuperación conversacional personalizada."""
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

    # Prompt para la respuesta basada en el contexto
    prompt_template = """Historial del chat: {chat_history}
    Utiliza la siguiente información para responder la pregunta del usuario.
    Si la respuesta no se encuentra en la información proporcionada, responde "No tengo la información para responder a esa pregunta."
    No hagas suposiciones.

    Contexto: {context}
    Pregunta: {question}

    Respuesta:"""
    RESPONSE_PROMPT = PromptTemplate(template=prompt_template, input_variables=["chat_history", "context", "question"])

    # Cadena para recuperar documentos
    retrieval_chain = retriever | RunnablePassthrough.assign(context=(lambda x: "\n\n".join(doc.page_content for doc in x)))

    # Cadena para generar la respuesta
    answer_chain = {
        "context": lambda x: x["context"],
        "question": lambda x: x["question"]
    } | RESPONSE_PROMPT | StrOutputParser()

    # Cadena conversacional completa
    @chain
    def conversational_rag_chain(query: str):
        # Cargar el historial de la memoria
        memory_variables = memory.load_memory_variables({})
        chat_history = memory_variables['chat_history']

        # Recuperar documentos
        retrieved_documents = retrieval_chain.invoke({"question": query})

        # Generar la respuesta usando el historial, el contexto y la pregunta
        response = answer_chain.invoke({"chat_history": chat_history, "question": query, "context": retrieved_documents["context"]})

        # Guardar la interacción en la memoria
        memory.save_context(inputs={"question": query}, outputs={"answer": response})

        return {"answer": response, "source_documents": retrieved_documents}

    return conversational_rag_chain

if __name__ == "__main__":
    # Ejemplo de uso (necesitas haber creado la base de datos vectorial primero)
    from retriever import load_vectorstore
    persist_directory = "./chroma_index" # Asegúrate de que esta ruta sea correcta
    retriever = load_vectorstore(persist_directory)
    if retriever:
        chat_chain = create_chat_chain(retriever)
        result = chat_chain.invoke({"question": "¿Cuál es la idea principal del documento?"})
        print(f"Respuesta: {result['answer']}")
        print(f"Documentos fuente: {[os.path.basename(doc.metadata['source']) for doc in result['source_documents']]}")

        # Ejemplo con memoria (esto necesitaría más trabajo para integrarse completamente)
        # result_seguimiento = chat_chain.invoke({"question": "¿Puedes dar más detalles sobre eso?", "chat_history": [HumanMessage(content="¿Cuál es la idea principal del documento?"), AIMessage(content=result['answer'])]})
        # print(f"\nRespuesta (seguimiento): {result_seguimiento['answer']}")