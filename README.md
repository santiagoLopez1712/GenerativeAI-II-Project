# Homework Project: Build a RAG (Retrieval-Augmented Generation) System

## Objective
Develop a basic Retrieval-Augmented Generation (RAG) system that retrieves information from an external document source and uses it to answer questions. This will demonstrate how language models can be grounded in up-to-date, domain-specific knowledge.

> The chosen model has a knowledge cutoff in **August 2024**, so answers to recent topics must rely on **retrieved documents**, not internal model knowledge.

---

## Core Requirements

### 1. Document Indexing
- Use **Chromadb** with **persistence enabled**.  
- Include document **splitting** (≥ 50 chunks) using appropriate text splitting strategies.

### 2. System Components
- Use `gemini-2.0-flash`
- Implement with **LangChain** or **LlamaIndex**.  
- Use **LangSmith** or **LangFuse**.  
- Use **Git and GitHub** for version control.

### 3. Experimentation  
- Compare **system prompts** and their effects on model behavior.  
- Use a variety of **questions** to evaluate system robustness (at leat 5 different questions with correct answers).

### 4. Reproducibility  
- Submit your code via **GitHub**.  
- Use a clean repository:  
  - ❌ **No large files** in git history  
  - ❌ **No secret tokens** in commit history

### 5. Submission  
- **Deadline:** `11.05 at 23:59`  
- **Deliverables:**
  - GitHub repo link
  - Link to your **LangSmith** or **LangFuse** project.
  - Jupyter notebook or script demonstrating:
    - Index creation
    - Retrieval
    - Answer generation
    - Prompt variations

---

## Bonus Features (Optional, for Extra Credit)

Implement one or more of the following to enhance your RAG system:

- ✅ **Dialog flow** (multi-turn interaction)  
- ✅ **Memory** (context tracking across interactions)  
- ✅ **Query rewriting** (to improve retrieval relevance)  
- ✅ **Metadata filtering** during document retrieval  
- ✅ **Multi-Query retrieval** (ask multiple questions or rephrase to get better context)

---
