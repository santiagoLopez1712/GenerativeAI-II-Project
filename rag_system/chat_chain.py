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


def create_chat_chain(retriever):
    from langchain_core.runnables import RunnableLambda

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    prompt_template = """Historial del chat: {chat_history}
    Usa el siguiente contexto para responder la pregunta. Si no está en el contexto, di: "No tengo la información."

    Contexto: {context}
    Pregunta: {question}

    Respuesta:"""

    RESPONSE_PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["chat_history", "context", "question"]
    )

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
