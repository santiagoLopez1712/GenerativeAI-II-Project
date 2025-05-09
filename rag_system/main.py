import os
import json
from dotenv import load_dotenv
from document_loader import load_and_split_documents
from retriever import create_vectorstore, load_vectorstore
from chat_chain import create_chat_chain

load_dotenv()
google_api_mainkey = os.getenv("GOOGLE_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY") # Aunque no se usa directamente aquí

# Configurar LangSmith (opcional)
if os.getenv("LANGCHAIN_TRACING_V2"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY_LANGSMITH")
    os.environ["LANGCHAIN_PROJECT"] = "rag-gemini-project"

# ... el resto del código de main.py ...

def main():
    data_dir = "./data"
    persist_dir = "./chroma_index"

    # 1. Cargar y dividir documentos
    documents = load_and_split_documents(data_dir=data_dir)

    # 2. Crear o cargar la base de datos vectorial
    if not os.path.exists(persist_dir):
        persist_dir = create_vectorstore(documents, persist_dir=persist_dir)
        retriever = load_vectorstore(persist_dir=persist_dir)
    else:
        retriever = load_vectorstore(persist_dir=persist_dir)

    # 3. Crear la cadena de chat RAG
    if retriever:
        chat_chain = create_chat_chain(retriever)

        # 4. Cargar y ejecutar preguntas de prueba
        try:
            with open("test_questions.json", "r") as f:
                test_questions = json.load(f)
            print("\n--- Ejecutando preguntas de prueba ---")
            for question in test_questions:
                result = chat_chain({"question": question})
                print(f"\nPregunta: {question}")
                print(f"Respuesta: {result['answer']}")
                print(f"Documentos fuente: {[os.path.basename(doc.metadata['source']) for doc in result['source_documents']]}")

            # Ejemplo de interacción multi-turno (puedes expandir esto)
            print("\n--- Ejemplo de conversación multi-turno ---")
            pregunta_inicial = "¿Cuál es el tema principal de uno de los documentos?"
            result_inicial = chat_chain({"question": pregunta_inicial})
            print(f"\nPregunta: {pregunta_inicial}")
            print(f"Respuesta: {result_inicial['answer']}")
            pregunta_seguimiento = "¿Puedes dar más detalles sobre eso?"
            result_seguimiento = chat_chain({"question": pregunta_seguimiento})
            print(f"\nPregunta: {pregunta_seguimiento}")
            print(f"Respuesta: {result_seguimiento['answer']}")

        except FileNotFoundError:
            print("El archivo test_questions.json no se encontró.")
        except json.JSONDecodeError:
            print("Error al decodificar el archivo test_questions.json.")

if __name__ == "__main__":
    main()