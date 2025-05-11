import os
import json
from document_loader import load_and_split_documents
from retriever import create_vectorstore, load_vectorstore
from chat_chain import create_chat_chain

def main():
    data_dir = "./data"
    persist_dir = "./chroma_index"

    # Cargar y dividir los documentos
    print("Cargando y dividiendo documentos...")
    documents = load_and_split_documents(data_dir=data_dir)

    # Crear o cargar el vectorstore
    if not os.path.exists(persist_dir):
        print("Creando nuevo vectorstore...")
        persist_dir = create_vectorstore(documents, persist_dir=persist_dir)
    retriever = load_vectorstore(persist_dir=persist_dir)

    # Crear la cadena de chat
    chat_chain = create_chat_chain(retriever)

    # Cargar preguntas de prueba desde el archivo JSON
    try:
        with open("test_questions.json", "r") as f:
            test_questions = json.load(f)
        print("\n--- Preguntas de prueba ---")
        
        for item in test_questions:
            if isinstance(item, dict) and "question" in item:
                question = item["question"]
                try:
                    result = chat_chain.invoke({"query": question})
                    # Verificar que el resultado contenga la clave 'answer'
                    if "answer" in result:
                        print(f"\nPregunta: {question}")
                        print(f"Respuesta: {result['answer']}")
                    else:
                        print(f"❌ No se encontró una respuesta para la pregunta: {question}")
                except Exception as e:
                    print(f"❌ Error al obtener respuesta para la pregunta: {question}. Error: {e}")
            else:
                print(f"Formato inválido: {item}")
    
    except FileNotFoundError:
        print("❌ test_questions.json no encontrado.")
    except json.JSONDecodeError:
        print("❌ Error al parsear test_questions.json.")
    except Exception as e:
        print(f"Error inesperado: {e}")

if __name__ == "__main__":
    main()
