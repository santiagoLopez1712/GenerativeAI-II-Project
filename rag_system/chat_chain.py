import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap, chain

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Cargar las claves API desde el archivo .env
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


def create_chat_chain(retriever, prompt_version="original"):
    """Crea una cadena de recuperación conversacional personalizada con diferentes prompt versions."""
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

    if prompt_version == "original":
        prompt_template_to_use = """Historial del chat: {chat_history}
        Utiliza la siguiente información para responder la pregunta del usuario.
        Si la respuesta no se encuentra en la información proporcionada, responde "No tengo la información para responder a esa pregunta."
        No hagas suposiciones.

        Contexto: {context}
        Pregunta: {question}

        Respuesta:"""
    elif prompt_version == "v2":
        prompt_template_to_use = """Utiliza la información proporcionada a continuación para responder a la pregunta del usuario de forma concisa.
        Si la respuesta no se encuentra explícitamente en el contexto, responde: "La respuesta no se encuentra en los documentos proporcionados."
        Cita brevemente la fuente del documento al final de tu respuesta si está disponible.

        Contexto: {context}
        Pregunta: {question}

        Respuesta:"""
    elif prompt_version == "v3":
        prompt_template_to_use = """Historial del chat: {chat_history}
        El usuario ha hecho una pregunta. Por favor, utiliza la información de los siguientes documentos para responder de la manera más útil y conversacional posible.
        Confirma que tu respuesta se basa en la información proporcionada.
        Si la información no es suficiente para responder completamente, hazlo saber.

        Contexto: {context}
        Pregunta: {question}

        Respuesta (basada en los documentos):"""
    else:
        raise ValueError(f"Versión de prompt no válida: {prompt_version}")

    RESPONSE_PROMPT = PromptTemplate(
        template=prompt_template_to_use,
        input_variables=["chat_history", "context", "question"]
    )

    # ... (el resto de la función create_chat_chain) ...

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # ✅ Esta función se usará para recuperar y formatear los documentos
    def retrieve_and_format(inputs):
        retrieved_docs = retriever.get_relevant_documents(inputs["query"])
        return {
            "chat_history": str(inputs.get("chat_history", "")),
            "question": str(inputs["query"]),
            "context": format_docs(retrieved_docs)
        }

    full_chain = (
        RunnableLambda(retrieve_and_format) 
        | RESPONSE_PROMPT 
        | llm 
        | StrOutputParser()
    )

    @chain
    def conversational_chain(inputs: dict):
        response = full_chain.invoke(inputs)
        memory.save_context({"question": inputs["query"]}, {"answer": response})
        return {"answer": response}

    return conversational_chain
