import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda
from langchain_core.runnables import chain

# Laden von Umgebungsvariablen
load_dotenv()
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


def create_chat_chain(retriever, prompt_version="original"):
    """Creates a custom conversational retrieval chain."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

    if prompt_version == "original":
        prompt_template_to_use = """Chat history: {chat_history}
        Verwenden Sie die folgenden Informationen, um die Frage des Benutzers zu beantworten.
        Wenn die Antwort nicht in den bereitgestellten Informationen gefunden wird, antworten Sie mit: "Ich habe keine Informationen, um diese Frage zu beantworten."
        Machen Sie keine Annahmen.

        Kontext: {context}
        Frage: {question}

        Antwort (auf Deutsch):"""
    elif prompt_version == "v2":
        prompt_template_to_use = """Verwenden Sie die unten bereitgestellten Informationen, um die Frage des Benutzers prägnant zu beantworten.
        Wenn die Antwort nicht explizit im Kontext gefunden wird, antworten Sie mit: "Die Antwort ist in den bereitgestellten Dokumenten nicht zu finden."
        Zitieren Sie am Ende Ihrer Antwort kurz die Quelle des Dokuments, falls verfügbar.

        Kontext: {context}
        Frage: {question}

        Antwort (auf Deutsch):"""
    elif prompt_version == "v3":
        prompt_template_to_use = """Chat history: {chat_history}
        Der Benutzer hat eine Frage gestellt. Bitte verwenden Sie die Informationen aus den folgenden Dokumenten, um so hilfreich und konversationsfreundlich wie möglich zu antworten.
        Bestätigen Sie, dass Ihre Antwort auf den bereitgestellten Informationen basiert.
        Wenn die Informationen nicht ausreichen, um vollständig zu antworten, lassen Sie den Benutzer dies wissen.

        Kontext: {context}
        Frage: {question}

        Antwort (auf Deutsch):"""
    else:
        raise ValueError(f"Invalid prompt version: {prompt_version}")

    RESPONSE_PROMPT = PromptTemplate(
        template=prompt_template_to_use,
        input_variables=["chat_history", "context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def retrieve_and_format(inputs):
        retrieved_docs = retriever.invoke(inputs["question"])
        return {
            "chat_history": str(inputs.get("chat_history", "")),
            "question": str(inputs["question"]),
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
        memory.save_context({"question": inputs["question"]}, {"answer": response})
        return {"answer": response}

    return conversational_chain
