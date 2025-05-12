# RAG-basiertes Chatbot-System

Dies ist ein Python-basiertes Chatbot-System, das Retrieval Augmented Generation (RAG) verwendet, um Fragen zu beantworten. Das System lädt Dokumente, indiziert sie für die Suche und verwendet eine Chat-Kette, um relevante Informationen abzurufen und Antworten zu generieren.

## Übersicht

Das System besteht aus mehreren Hauptkomponenten:

* **Dokumentenverarbeitung:** Lädt und teilt Dokumente in Text- und PDF-Formate.
* **Vektorindexierung:** Erstellt einen Chroma-Vektorindex der Dokumente mit HuggingFace-Embeddings.
* **Retrieval:** Ruft relevante Dokumente basierend auf Benutzeranfragen ab.
* **Chat-Kette:** Generiert Antworten auf Benutzeranfragen unter Verwendung der abgerufenen Dokumente und eines Sprachmodells (Gemini).

## Verzeichnisstruktur

Die Verzeichnisstruktur des Projekts ist wie folgt:

```
GenerativeAI-II-Project/  # Hauptordner des Projekts
├── venv/                 # (Optional) Virtuelle Umgebung
├── requirements.txt      # Projektabhängigkeiten
├── .gitignore            # Dateien, die von Git ignoriert werden sollen
├── rag_system/           # Ordner mit dem RAG-System
│   ├── __init__.py       # Leere Datei, um rag_system als Python-Paket zu kennzeichnen
│   ├── chat_chain.py     # Definiert die Chat-Logik mit Retrieval und Prompting.
│   ├── document_loader.py # Lädt und teilt Dokumente aus verschiedenen Formaten.
│   ├── document_prompts.py # Enthält Code zum Ausführen von Testfragen und Dokumentieren der Ergebnisse für verschiedene Prompt-Versionen.
│   ├── main.py           # Hauptskript zum Ausführen des Chatbot-Systems.
│   ├── retriever.py      # Verwaltet das Erstellen und Laden des Vektorindex.
│   ├── test_questions.json # Enthält Testfragen im JSON-Format.
│   ├── langsmith.yaml    # Konfigurationsdatei für LangSmith
│   ├── readme.md         # Projekt Dokumentation
│   ├── data/             # Enthält die zu verarbeitenden Dokumente (.txt, .pdf).
│   ├── chroma_index/     # (Optional) Verzeichnis für den persistenten Chroma-Vektorindex.
│   └── __pycache__/      # Python-Cache-Dateien
```

## Voraussetzungen

* Python 3.x
* pip
* Ein Google Cloud-Konto und eine API-Key für den Zugriff auf das Gemini-Modell.
* Ein LangChain API-Key für Tracing (optional).

## Installation

1.  **Klonen Sie das Repository:**

    \`\`\`bash
    git clone <repository_url>
    cd rag_system
    \`\`\`

2.  **Erstellen Sie ein virtuelles Environment (empfohlen):**

    \`\`\`bash
    python3 -m venv venv
    source venv/bin/activate  # Unter Linux/MacOS
    venv\\Scripts\\activate.bat # Unter Windows
    \`\`\`

3.  **Installieren Sie die Abhängigkeiten:**

    \`\`\`bash
    pip install -r requirements.txt
    \`\`\`

4.  **Richten Sie Umgebungsvariablen ein:**

    * Erstellen Sie eine \`.env\`-Datei im Projektverzeichnis.
    * Fügen Sie Ihre API-Keys hinzu:

        \`\`\`
        LANGCHAIN_API_KEY="your_langchain_api_key" # Optional, für LangChain Tracing
        GOOGLE_API_KEY="your_google_api_key"
        \`\`\`
    * Ersetzen Sie `"your_langchain_api_key"` und `"your_google_api_key"` durch Ihre tatsächlichen Keys.

## Verwendung

1.  **Fügen Sie Dokumente hinzu:**

    * Legen Sie die Text- und PDF-Dokumente, die Sie für die Beantwortung von Fragen verwenden möchten, in den Ordner \`./data\`

2.  **Führen Sie das System aus:**

    \`\`\`bash
    python main.py
    \`\`\`

    Das Skript führt die folgenden Schritte aus:

    * Lädt und teilt die Dokumente aus dem Ordner \`./data\`.
    * Erstellt oder lädt einen Chroma-Vektorindex im Ordner \`./chroma_index\`.
    * Erstellt eine Chat-Kette, die den Retriever und das Gemini-Modell verwendet.
    * Lädt Testfragen aus der Datei \`test_questions.json\`.
    * Beantwortet die Testfragen und gibt die Antworten zusammen mit den zugehörigen Quelldokumenten aus.

## Struktur der Testfragen

Die Testfragen sollten in einer JSON-Datei namens \`test_questions.json\` im Hauptverzeichnis des Projekts gespeichert werden. Die Datei sollte eine Liste von Objekten enthalten, wobei jedes Objekt ein Feld "question" mit der Testfrage enthält. Zum Beispiel:

\`\`\`json
[
  { "question": "Was ist Industrie 4.0?" },
  { "question": "Wer war Angela Merkel?" }
]
\`\`\`

## Dokumentation der Ergebnisse

Das Skript \`document_prompts.py\` führt Testfragen mit verschiedenen Prompt-Versionen aus und dokumentiert die Ergebnisse in JSON-Ausgabe. Dies dient dazu, die Auswirkungen verschiedener Prompt-Strategien auf die Qualität der Antworten zu bewerten.

Um die Dokumentation der Ergebnisse zu generieren, führen Sie folgendes aus:

\`\`\`bash
python document_prompts.py
\`\`\`

## Fehlerbehebung

* **\`ModuleNotFoundError: No module named 'rag_system'\`:** Stellen Sie sicher, dass Sie den Befehl \`python main.py\` vom Hauptverzeichnis des Projekts ausführen (dem Verzeichnis, das \`main.py\` und den Ordner \`rag_system\` enthält).
* **\`FileNotFoundError: The directory './data' does not exist\`:** Erstellen Sie den Ordner \`./data\` und legen Sie Ihre Dokumente hinein.
* **\`HTTPError: 403 Client Error: Forbidden\`:** Überprüfen Sie, ob Ihre LangChain- und Google API-Keys korrekt in der \`.env\`-Datei konfiguriert sind.
* **\`Error beim Dekodieren der Datei test_questions.json\`:** Stellen Sie sicher, dass die Datei \`test_questions.json\` vorhanden und korrekt formatiert ist.
* **\`ValueError: Invalid prompt version\`:** Stellen Sie sicher, dass der Parameter \`prompt_version\` in \`create_chat_chain\` einen gültigen Wert hat ("original", "v2" oder "v3").

## Erweiterung

* **Unterstützung für weitere Dokumentformate:** Erweitern Sie die Funktion \`load_and_split_documents\` in \`document_loader.py\`, um andere Formate wie MS Word, HTML usw. zu verarbeiten.
* **Anpassbare Prompts:** Implementieren Sie eine flexiblere Prompt-Verwaltung, um verschiedene Prompt-Strategien zu unterstützen, ohne den Code direkt zu ändern.
* **Alternative Vektordatenbanken:** Integrieren Sie andere Vektordatenbanken wie Pinecone oder Weaviate.
* **Erweiterte Retrieval-Strategien:** Experimentieren Sie mit verschiedenen Retrieval-Methoden, z. B. dem Abrufen von Chunks mit Metadatenfiltern oder dem Abrufen von übergeordneten Dokumenten.
* **Zusätzliche Evaluierungsmetriken:** Implementieren Sie zusätzliche Metriken in \`document_prompts.py\` um die Leistung des Chatbots umfassender zu bewerten.

## Lizenz

[Fügen Sie hier die Lizenzinformationen ein]
