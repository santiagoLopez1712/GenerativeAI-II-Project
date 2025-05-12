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


def create_chat_chain(retriever, prompt_version="v3"):
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
        # Verschiedene Abfragen generieren
        queries = [
            inputs["question"],  # Originalfrage
            f"Was bedeutet: {inputs['question']}?",  # Reformulierte Frage
            f"Erklären Sie detailliert: {inputs['question']}",  # Kontextanfrage
            f"Welche Beispiele gibt es für: {inputs['question']}?"  # Beispielanfrage
        ]
        
        # Dokumente für jede Abfrage abrufen
        all_retrieved_docs = []
        for query in queries:
            retrieved_docs = retriever.invoke(query)
            all_retrieved_docs.extend(retrieved_docs)
        
        # Doppelte Dokumente entfernen (falls erforderlich)
        unique_docs = {doc.page_content: doc for doc in all_retrieved_docs}.values()
        
        # Kontext aus den kombinierten Dokumenten erstellen
        return {
            "chat_history": str(inputs.get("chat_history", "")),  # Chatverlauf
            "question": str(inputs["question"]),  # Benutzerfrage
            "context": format_docs(unique_docs)  # Kombinierte Dokumente als Kontext
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

if __name__ == "__main__":
    # Beispiel-Retriever (Mock für Testzwecke)
    class MockRetriever:
        def invoke(self, query):
            # Simulierte Dokumente basierend auf der Abfrage
            return [
                {"page_content": f"Simulated content for query: {query}", "metadata": {"source": "MockSource"}}
            ]

    # Mock-Retriever erstellen
    retriever = MockRetriever()

    # Beispiel-Eingabe
    inputs = {
        "question": "Was ist Industrie 4.0?",
        "chat_history": "Vorherige Frage: Was ist Digitalisierung?"
    }

    # Funktion retrieve_and_format ausführen
    def format_docs(docs):
        return "\n\n".join(doc["page_content"] for doc in docs)

    def retrieve_and_format(inputs):
        # Verschiedene Abfragen generieren
        queries = [
            inputs["question"],  # Originalfrage
            f"Was bedeutet: {inputs['question']}?",  # Reformulierte Frage
            f"Erklären Sie detailliert: {inputs['question']}",  # Kontextanfrage
            f"Welche Beispiele gibt es für: {inputs['question']}?"  # Beispielanfrage
        ]
        
        # Dokumente für jede Abfrage abrufen
        all_retrieved_docs = []
        for query in queries:
            retrieved_docs = retriever.invoke(query)
            all_retrieved_docs.extend(retrieved_docs)
        
        # Doppelte Dokumente entfernen (falls erforderlich)
        unique_docs = {doc["page_content"]: doc for doc in all_retrieved_docs}.values()
        
        # Kontext aus den kombinierten Dokumenten erstellen
        return {
            "chat_history": str(inputs.get("chat_history", "")),  # Chatverlauf
            "question": str(inputs["question"]),  # Benutzerfrage
            "context": format_docs(unique_docs)  # Kombinierte Dokumente als Kontext
        }

    # Ergebnis drucken
    result = retrieve_and_format(inputs)
    print("\n--- Multi-Query Retrieval Ergebnis ---")
    print(f"Chatverlauf: {result['chat_history']}")
    print(f"Frage: {result['question']}")
    print(f"Kontext:\n{result['context']}")
