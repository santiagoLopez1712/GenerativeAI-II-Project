import json
from rag_system.chat_chain import create_chat_chain  # Asegúrate de que la ruta sea correcta
from rag_system.retriever import load_vectorstore  # Asegúrate de que la ruta sea correcta

def run_and_document(prompt_version, test_questions_file="test_questions.json", persist_dir="./chroma_index"):
    """Ejecuta las preguntas de prueba con una versión del prompt y documenta los resultados."""
    print(f"\n--- Ejecutando con la versión del prompt: {prompt_version} ---")

    retriever = load_vectorstore(persist_dir=persist_dir)
    if not retriever:
        print("Error al cargar el retriever.")
        return

    chat_chain = create_chat_chain(retriever, prompt_version=prompt_version)  # Modificaremos create_chat_chain

    try:
        with open(test_questions_file, "r") as f:
            test_questions = json.load(f)
        results = []
        for item in test_questions:
            if isinstance(item, dict) and "question" in item:
                question = item["question"]
                result = chat_chain.invoke(question)
                print(f"\nPregunta: {question}")
                print(f"Respuesta: {result['answer']}")
                print(f"Documentos fuente: {[doc.metadata.get('source', 'N/A') for doc in result['source_documents']]}")
                results.append({
                    "pregunta": question,
                    "respuesta": result['answer'],
                    "fuentes": [doc.metadata.get('source', 'N/A') for doc in result['source_documents']]
                })
            else:
                print(f"Formato incorrecto de pregunta: {item}")
        return results
    except FileNotFoundError:
        print(f"El archivo {test_questions_file} no se encontró.")
        return None
    except json.JSONDecodeError:
        print(f"Error al decodificar el archivo {test_questions_file}.")
        return None

def main():
    original_prompt_results = run_and_document("original")
    variation_2_results = run_and_document("v2")
    variation_3_results = run_and_document("v3")

    print("\n--- Documentación de los Resultados ---")
    print("\n--- Resultados con el Prompt Original ---")
    if original_prompt_results:
        print(json.dumps(original_prompt_results, indent=4, ensure_ascii=False))

    print("\n--- Resultados con la Variación 2 del Prompt ---")
    if variation_2_results:
        print(json.dumps(variation_2_results, indent=4, ensure_ascii=False))

    print("\n--- Resultados con la Variación 3 del Prompt ---")
    if variation_3_results:
        print(json.dumps(variation_3_results, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()