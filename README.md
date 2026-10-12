# 🧪 Local RAG Playground & Multimodal Benchmark

A privacy-first, 100% offline evaluation testbed designed to benchmark local Generative AI models (e.g., **Gemma 2**, **Llama 3.2**, **Mistral**) on PDF document search, image/chart extraction, and Retrieval-Augmented Generation (RAG) performance.

Powered by **Ollama**, **LangChain**, **ChromaDB**, and **RAGAS** (Retrieval Augmented Generation Assessment System).

---

## 🛠️ Prerequisites & Environment Specifications

### 1. Runtime & Python Version
* **Python Runtime:** `Python 3.11.x` (**Strictly Recommended** on Windows to pull pre-compiled wheel binaries for packages like `scikit-network` and avoid C++ Build Tools compilation errors).
* **Virtual Environment:** Isolated `.venv` located in the root directory (`/rag-playground/.venv`).

### 2. Local Model Server Infrastructure
* **Ollama Server:** `v0.2.0` or higher running locally on default port `http://localhost:11434`.

### 3. Models Used & Roles

| Model Name | Model Type | Memory / VRAM | Purpose in Benchmark |
| :--- | :--- | :--- | :--- |
| **`gemma2:9b`** | Text LLM (Google) | ~6.5 GB | Primary Model Under Test & Fixed RAGAS "LLM-as-a-Judge" Evaluator |
| **`mxbai-embed-large`** | Embedding | ~670 MB | Text & chunk vectorization (1024-dimensional embeddings) |
| **`llama3.2-vision`** | Vision LLM (Meta) | ~7.9 GB | Transcribes images, charts, diagrams, and tables inside PDF streams |
| **`llama3.2`** *(Optional)* | Text LLM (Meta) | ~4.0 GB | Candidate model for multi-model benchmark loop |
| **`mistral`** *(Optional)* | Text LLM (Mistral) | ~4.5 GB | Candidate model for multi-model benchmark loop |

---

## ⚙️ Software Dependencies & Library Versions

The project relies on the following core Python libraries (pinned in `requirements.txt`):

* `python` = `3.11.x`
* `ragas` >= `0.1.15` (Evaluation Framework)
* `langchain-ollama` >= `0.1.0` (Ollama LLM & Embedding integration)
* `langchain-community` >= `0.2.0` (Document Loaders & Utilities)
* `langchain-core` >= `0.2.0` (Core abstractions)
* `chromadb` >= `0.5.0` (Local Vector Database)
* `pymupdf` >= `1.24.0` (Fast PDF rendering & image extraction)
* `pillow` >= `10.0.0` (Image processing)
* `pandas` >= `2.0.0` (Data manipulation & CSV export)
* `datasets` >= `2.14.0` (HuggingFace evaluation dataset formatting)

---

## 🚀 Quickstart & Setup

### Step 1: Pull Required Models in Ollama
Open your terminal or PowerShell and pull the baseline models:

```bash
# Baseline Generator & Judge LLM
ollama pull gemma2:9b

# Local Vector Embedding Model
ollama pull mxbai-embed-large

# Vision LLM for PDF Chart/Diagram Extraction
ollama pull llama3.2-vision