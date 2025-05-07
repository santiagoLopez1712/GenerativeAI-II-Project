# 🧠 Homework Project: Build a RAG (Retrieval-Augmented Generation) System

## 📌 Objective

The goal of this project is to build a **Retrieval-Augmented Generation (RAG)** system that fetches information from external documents and uses it to answer user questions. This project demonstrates how to ground a language model in real-world, up-to-date knowledge.

> 💡 **Note:** The selected model has a knowledge cutoff in **August 2024**. Your system must use **retrieved documents** to correctly answer questions about **events occurring after this date**.

---

## 🛠️ Core Requirements

### 1. Document Indexing
- Use **ChromaDB** with **persistence enabled**.
- Select a document describing an event that happened **after August 2024**.
- Split the document into **at least 50 chunks** using appropriate text splitting strategies.

### 2. System Architecture
- Use the model: `gemini-2.0-flash`
- Implement the pipeline using:
  - **LangChain** or **LlamaIndex**
  - **LangSmith** or **LangFuse** for observability and tracing
- Version control your code with **Git and GitHub**
- **Do not use pre-built agents**
- Implement:
  - 🗣️ **Dialog flow** (multi-turn interaction)
  - 🧠 **Memory** (to track context across messages)

### 3. Experimentation and Effectiveness Testing
- Create **at least 5 meaningful questions** that the system should answer using the retrieved document.
- The questions **must not be answerable** by the language model alone.
- Validate that the system answers correctly **only when using retrieval**.
- Compare and document the impact of different **system prompts** on model behavior.

### 4. Reproducibility & Clean Code Practices
- Use a clean GitHub repository:
  - ❌ No **large files** in git history
  - ❌ No **secret tokens** in commit history
- Your code should be:
  - Well-documented
  - Easy to run
  - Clearly structured

---

## 🚀 Submission Instructions

- **Deadline:** `11.05 at 23:59`
- Each student has a dedicated branch named after them.
- Open a **Pull Request (PR)** from your working branch **to your assigned branch** in this repository.
- Your PR must include:
  - ✅ Your full implementation code
  - ✅ A Jupyter notebook or script showing:
    - Document indexing
    - Retrieval steps
    - Question answering
    - Prompt variations and experiments
  - ✅ Link to your **LangSmith** or **LangFuse** project

---

## ⭐ Bonus (Mandatory for Extra Credit)

To earn bonus points, your system must implement **both** of the following features:

- 🔍 **Metadata filtering** to refine document retrieval
- 🔁 **Multi-Query retrieval** (e.g., query rephrasing or multiple simultaneous questions to improve answer quality)

---

Happy building! 🚀
