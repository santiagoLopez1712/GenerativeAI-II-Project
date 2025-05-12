import json
from chat_chain import create_chat_chain
from retriever import load_vectorstore

def run_and_document(prompt_version, test_questions_file="test_questions.json", persist_dir="./chroma_index"):
    """Führt Testfragen mit einer bestimmten Prompt-Version aus und dokumentiert die Ergebnisse."""
    print(f"\n--- Ausführung mit der Prompt-Version: {prompt_version} ---")

    retriever = load_vectorstore(persist_dir=persist_dir)
    if not retriever:
        print("Fehler beim Laden des Retrievers.")
        return

    chat_chain = create_chat_chain(retriever, prompt_version=prompt_version)

    try:
        with open(test_questions_file, "r") as f:
            test_questions = json.load(f)
        results = []
        chat_history = ""  # Initialisiert den Chatverlauf als leer

        for item in test_questions:
            if isinstance(item, dict) and "question" in item:
                question = item["question"]
                try:
                    # Fügt den Chatverlauf hinzu
                    result = chat_chain.invoke({"question": question, "chat_history": chat_history})

                    if "answer" in result:
                        print(f"\nFrage: {question}")
                        print(f"Antwort: {result['answer']}")
                        # Behandelt das Fehlen von 'source_documents'
                        print(f"Quellendokumente: {[doc['metadata'].get('source', 'N/A') for doc in result.get('source_documents', [])]}")

                        # Aktualisiert den Chatverlauf
                        chat_history += f"Frage: {question}\nAntwort: {result['answer']}\n\n"

                        results.append({
                            "frage": question,
                            "antwort": result['answer'],
                            "quellen": [doc['metadata'].get('source', 'N/A') for doc in result.get('source_documents', [])]
                        })
                    else:
                        print(f"⚠️ Keine Antwort gefunden für: {question}\n\n")
                except Exception as e:
                    print(f"❌ Fehler beim Abrufen der Antwort: {e}\n\n")
            else:
                print(f"⚠️ Ungültiges Frageformat: {item}\n\n")
        return results
    except FileNotFoundError:
        print(f"Die Datei {test_questions_file} wurde nicht gefunden.")
        return None
    except json.JSONDecodeError:
        print(f"Fehler beim Dekodieren der Datei {test_questions_file}.")
        return None

def main():
    original_prompt_results = run_and_document("original")
    variation_2_results = run_and_document("v2")
    variation_3_results = run_and_document("v3")

    print("\n--- Dokumentation der Ergebnisse ---")
    print("\n--- Ergebnisse mit dem Original-Prompt ---")
    if original_prompt_results:
        print(json.dumps(original_prompt_results, indent=4, ensure_ascii=False))

    print("\n--- Ergebnisse mit der Prompt-Variante 2 ---")
    if variation_2_results:
        print(json.dumps(variation_2_results, indent=4, ensure_ascii=False))

    print("\n--- Ergebnisse mit der Prompt-Variante 3 ---")
    if variation_3_results:
        print(json.dumps(variation_3_results, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()