import os
import json
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from retriever import create_vectorstore, load_vectorstore
from chat_chain import create_chat_chain


def load_and_split_documents(data_dir="./data", chunk_size=500, chunk_overlap=100):
    """Loads and splits text and PDF documents, including files in subdirectories."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The directory '{data_dir}' does not exist. Please create it and add .txt or .pdf files.")

    docs = []
    # Recorre todas las carpetas y subcarpetas dentro de data_dir
    for root, _, files in os.walk(data_dir):
        for filename in files:
            path = os.path.join(root, filename)
            try:
                if filename.endswith(".txt"):
                    loader = TextLoader(path, encoding="utf-8")
                elif filename.endswith(".pdf"):
                    loader = PyPDFLoader(path)
                else:
                    print(f"⚠️ Skipping unsupported file format: {filename}")
                    continue

                # Carga los documentos y agrega la metadata de la fuente
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source"] = os.path.relpath(path, data_dir)
                docs.extend(loaded_docs)
            except Exception as e:
                print(f"❌ Error loading the file {filename}: {e}")
                continue

    if len(docs) == 0:
        print("❌ No valid files were found in the directory.")
        return []

    # Divide los documentos en chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"✅ {len(chunks)} chunks were loaded and split.")
    return chunks

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

        chat_history = ""  # Initialisiert den Chatverlauf als leer

        for item in test_questions:
            if isinstance(item, dict) and "question" in item:
                question = item["question"]
                try:
                    # Fügt den Chatverlauf hinzu
                    result = chat_chain.invoke({"question": question, "chat_history": chat_history})

                    if "answer" in result:
                        print(f"✅ Frage: {question}\n")
                        print(f"💬 Antwort: {result['answer']}\n\n")
                        
                        # Aktualisiert den Chatverlauf
                        chat_history += f"Frage: {question}\nAntwort: {result['answer']}\n\n"
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

    # Benutzerdefinierte Fragen
    ask_user_questions(chat_chain)

def ask_user_questions(chat_chain):
    """Permite al usuario hacer preguntas sobre los Dokumente."""
    print("\n--- Benutzerdefinierte Fragen ---")
    chat_history = ""  # Inicialisiert den Chatverlauf als leer

    while True:
        user_question = input("Geben Sie Ihre Frage ein (oder 'exit' zum Beenden): ")
        if user_question.lower() == "exit":
            print("Beenden der Benutzerfragen.")
            break

        try:
            # Fügt den Chatverlauf hinzu
            result = chat_chain.invoke({"question": user_question, "chat_history": chat_history})

            if "answer" in result:
                print(f"💬 Antwort: {result['answer']}\n")
                # Aktualisiert den Chatverlauf
                chat_history += f"Frage: {user_question}\nAntwort: {result['answer']}\n\n"
            else:
                print(f"⚠️ Keine Antwort gefunden für: {user_question}\n")
        except Exception as e:
            print(f"❌ Fehler beim Abrufen der Antwort: {e}\n")

if __name__ == "__main__":
    main()
    ask_user_questions(chat_chain)
