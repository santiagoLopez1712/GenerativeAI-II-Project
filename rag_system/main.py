import os
import json
from document_loader import load_and_split_documents
from retriever import create_vectorstore, load_vectorstore
from chat_chain import create_chat_chain


def main():
    data_dir = "./data"
    persist_dir = "./chroma_index"

    # 📄 Dokumente laden und aufteilen
    print("📄 Dokumente werden geladen und aufgeteilt...")
    documents = load_and_split_documents(data_dir=data_dir)

    # 📦 Vectorstore erstellen oder laden
    if not os.path.exists(persist_dir):
        print("📦 Neuer Vectorstore wird erstellt...")
        persist_dir = create_vectorstore(documents, persist_dir=persist_dir)
    retriever = load_vectorstore(persist_dir=persist_dir)

    # 🛠️ Chat-Kette erstellen
    chat_chain = create_chat_chain(retriever)

    # 🚀 Testfragen aus der JSON-Datei laden
    try:
        with open("test_questions.json", "r") as f:
            test_questions = json.load(f)
        print("\n--- 🚀 Testfragen ---")

        for item in test_questions:
            if isinstance(item, dict) and "question" in item:
                question = item["question"]
                try:
                    result = chat_chain.invoke({"question": question})  # Kann Zeit in Anspruch nehmen

                    if "answer" in result:
                        print(f"✅ Frage: {question}\n")  # Espacio entre pregunta y respuesta
                        print(f"💬 Antwort: {result['answer']}\n\n")  # Dos espacios entre respuesta y siguiente Frage
                    else:
                        print(f"⚠️ Keine Antwort gefunden für: {question}\n\n")
                except Exception as e:
                    print(f"❌ Fehler beim Abrufen der Antwort: {e}\n\n")
            else:
                print(f"⚠️ Ungültiges Format in der Frage: {item}\n\n")

    except FileNotFoundError:
        print("❌ test_questions.json nicht gefunden.\n\n")
    except json.JSONDecodeError:
        print("❌ Fehler beim Parsen von test_questions.json.\n\n")
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}\n\n")

if __name__ == "__main__":
    main()
