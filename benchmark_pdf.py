import sys
import types

# --------------------------------------------------------------------------
# 1. Monkey-patch missing VertexAI module in legacy RAGAS / LangChain builds
# --------------------------------------------------------------------------
try:
    from langchain_community.chat_models import vertexai
except ImportError:
    dummy_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
    dummy_vertex.ChatVertexAI = type("ChatVertexAI", (object,), {})
    sys.modules["langchain_community.chat_models.vertexai"] = dummy_vertex

# --------------------------------------------------------------------------
# 2. Standard Imports
# --------------------------------------------------------------------------
import os
import glob
import time
import re
import json
from datetime import datetime
import pandas as pd
from datasets import Dataset

import pymupdf
import ollama

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from ragas import evaluate, RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# --------------------------------------------------------------------------
# 3. Native JSON Interceptor for RAGAS Evaluator
# --------------------------------------------------------------------------
class CleanJSONChatOllama(ChatOllama):
    """Subclass of ChatOllama that sanitizes JSON strings for RAGAS."""

    def _clean_json_string(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        
        text = text.strip()
        
        # 1. Extract JSON from ```json ... ``` or ``` ... ``` blocks
        markdown_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
        if markdown_match:
            text = markdown_match.group(1).strip()

        # 2. Extract first valid JSON object {} or array [] from text
        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
        if json_match:
            return json_match.group(1)

        return text

    def invoke(self, input, config=None, **kwargs):
        response = super().invoke(input, config=config, **kwargs)
        if hasattr(response, 'content') and isinstance(response.content, str):
            response.content = self._clean_json_string(response.content)
        return response

# Helper function to guarantee all RAGAS metrics are instantiated instances
def init_ragas_metrics(evaluator_llm, evaluator_embeddings):
    raw_metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    initialized = []
    
    for m in raw_metrics:
        # Instantiate class if m is an uninstantiated type/class
        metric_obj = m() if isinstance(m, type) else m
        
        if hasattr(metric_obj, "llm"):
            metric_obj.llm = evaluator_llm
        if hasattr(metric_obj, "embeddings") and evaluator_embeddings is not None:
            metric_obj.embeddings = evaluator_embeddings
        if hasattr(metric_obj, "strictness"):
            metric_obj.strictness = 1
            
        initialized.append(metric_obj)
        
    return initialized

# --------------------------------------------------------------------------
# 4. Master Benchmark Configuration
# --------------------------------------------------------------------------
MODELS_TO_BENCHMARK = [
    "gemma2:9b",
    "llama3.2"
]

BENCHMARK_CONFIG = {
    "evaluator_model": "gemma2:9b",      # Fixed judge model for RAGAS
    "vision_model": "llama3.2-vision",   # Vision model for PDF images/charts
    "embedding_model": "mxbai-embed-large",
    "chunk_size": 600,
    "chunk_overlap": 100,
    "top_k": 3,
    "data_dir": "data",
    "results_dir": "results",
    "summary_csv": "results/all_benchmarks_summary.csv"
}

# --------------------------------------------------------------------------
# 5. Multimodal PDF Loader
# --------------------------------------------------------------------------
def ensure_sample_pdf_exists(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    if not pdf_files:
        print(f"[Info] No PDFs found in '{data_dir}/'. Creating sample PDF...")
        sample_path = os.path.join(data_dir, "sample_benchmark.pdf")
        pdf_bytes = (
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>/Contents 4 0 R>>endobj\n"
            b"4 0 obj<</Length 215>>stream\nBT /F1 12 Tf 50 700 Td "
            b"(Engine maintenance must be conducted every 10000 miles or 12 months.) Tj "
            b"0 -30 Td (Employees are eligible for remote work up to 2 days per week upon manager approval.) Tj ET\nendstream\nendobj\n"
            b"xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000056 00000 n \n0000000111 00000 n \n0000000212 00000 n \n"
            b"trailer<</Size 5/Root 1 0 R>>\nstartxref\n478\n%%EOF"
        )
        with open(sample_path, "wb") as f:
            f.write(pdf_bytes)

def extract_image_description(image_bytes, vision_model):
    try:
        response = ollama.chat(
            model=vision_model,
            messages=[{
                'role': 'user',
                'content': 'Extract all text, data from tables, numbers, chart labels, and summarize key insights clearly.',
                'images': [image_bytes]
            }]
        )
        return response['message']['content']
    except Exception:
        return ""

def load_pdf_multimodal(pdf_path, config):
    doc = pymupdf.open(pdf_path)
    page_documents = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text("text").strip()
        
        image_list = page.get_images(full=True)
        visual_descriptions = []

        if image_list and config["vision_model"]:
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                desc = extract_image_description(image_bytes, config["vision_model"])
                if desc:
                    visual_descriptions.append(
                        f"\n[VISUAL CONTENT / CHART DATA (Page {page_num+1}, Image {img_index+1})]:\n{desc}"
                    )

        full_content = page_text + "\n" + "\n".join(visual_descriptions)
        
        if full_content.strip():
            page_documents.append(
                Document(
                    page_content=full_content.strip(),
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "page": page_num + 1,
                        "has_visuals": len(image_list) > 0
                    }
                )
            )

    return page_documents

def build_vectorstore(config):
    data_dir = config["data_dir"]
    ensure_sample_pdf_exists(data_dir)

    print(f"[1/5] Extracting text and visual image data from PDFs in '{data_dir}/'...")
    pdf_paths = glob.glob(os.path.join(data_dir, "*.pdf"))
    
    all_documents = []
    for path in pdf_paths:
        print(f"      Processing PDF: {os.path.basename(path)}")
        docs = load_pdf_multimodal(path, config)
        all_documents.extend(docs)

    if not all_documents:
        raise ValueError(f"[ERROR] No extractable text or images found in PDFs inside '{data_dir}/'.")

    pdf_names = sorted(list(set(doc.metadata.get("source", "unknown") for doc in all_documents)))
    print(f"      Loaded {len(all_documents)} page(s) across {len(pdf_names)} PDF file(s).")

    print(f"      Chunking documents (size={config['chunk_size']}, overlap={config['chunk_overlap']})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"]
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"      Created {len(chunks)} text + vision chunk(s).")

    print(f"      Indexing chunks into Chroma vector database with '{config['embedding_model']}'...")
    embeddings = OllamaEmbeddings(model=config["embedding_model"])
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore, embeddings, pdf_names

# --------------------------------------------------------------------------
# 6. Response Generation & Output Logging
# --------------------------------------------------------------------------
def generate_rag_responses(model_name, vectorstore, questions, config):
    print(f"      Retrieving top-{config['top_k']} contexts & generating answers using model '{model_name}'...")
    llm = ChatOllama(model=model_name, temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": config["top_k"]})

    prompt_template = ChatPromptTemplate.from_template(
        "Answer the question based strictly on the provided context below. "
        "If the answer cannot be found in the context, state 'I do not know based on the provided document.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    retrieved_contexts = []
    generated_answers = []

    for q in questions:
        docs = retriever.invoke(q)
        chunk_texts = [doc.page_content for doc in docs]
        retrieved_contexts.append(chunk_texts)

        context_block = "\n\n".join(chunk_texts)
        formatted_prompt = prompt_template.format(context=context_block, question=q)
        
        try:
            response = llm.invoke(formatted_prompt)
            answer_str = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            answer_str = f"[Generation Error]: {e}"

        generated_answers.append(answer_str)

    return retrieved_contexts, generated_answers

def log_and_collect_results(model_name, results_df, config, pdf_names, execution_time_sec):
    os.makedirs(config["results_dir"], exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_safe_name = model_name.replace(':', '_')
    run_filename = f"run_{model_safe_name}_{timestamp_str}.csv"
    run_filepath = os.path.join(config["results_dir"], run_filename)
    results_df.to_csv(run_filepath, index=False)

    mean_faithfulness = results_df["faithfulness"].dropna().mean() if "faithfulness" in results_df else None
    mean_relevancy = results_df["answer_relevancy"].dropna().mean() if "answer_relevancy" in results_df else None
    mean_precision = results_df["context_precision"].dropna().mean() if "context_precision" in results_df else None
    mean_recall = results_df["context_recall"].dropna().mean() if "context_recall" in results_df else None

    summary_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "evaluator_judge": config["evaluator_model"],
        "embedding_model": config["embedding_model"],
        "chunk_size": config["chunk_size"],
        "chunk_overlap": config["chunk_overlap"],
        "top_k": config["top_k"],
        "pdf_sources": "; ".join(pdf_names),
        "query_count": len(results_df),
        "faithfulness_avg": round(mean_faithfulness, 4) if pd.notna(mean_faithfulness) else "N/A",
        "answer_relevancy_avg": round(mean_relevancy, 4) if pd.notna(mean_relevancy) else "N/A",
        "context_precision_avg": round(mean_precision, 4) if pd.notna(mean_precision) else "N/A",
        "context_recall_avg": round(mean_recall, 4) if pd.notna(mean_recall) else "N/A",
        "execution_time_sec": round(execution_time_sec, 2),
        "run_file": run_filepath
    }

    summary_df = pd.DataFrame([summary_data])
    summary_path = config["summary_csv"]

    try:
        if not os.path.exists(summary_path):
            summary_df.to_csv(summary_path, index=False)
        else:
            summary_df.to_csv(summary_path, mode="a", header=False, index=False)
    except PermissionError:
        fallback_path = os.path.join(config["results_dir"], f"summary_backup_{timestamp_str}.csv")
        print(f"\n[WARNING] '{summary_path}' is open in Excel. Saved summary to: '{fallback_path}'")
        summary_df.to_csv(fallback_path, index=False)
        summary_path = fallback_path

    return run_filepath, summary_path, summary_data

def print_model_report(model_name, summary_data, results_df):
    print("\n" + "="*65)
    print(f"         BENCHMARK REPORT: {model_name.upper()}         ")
    print("="*65)
    print(f" Evaluator Model     : {summary_data['evaluator_judge']}")
    print(f" Embedding Model     : {summary_data['embedding_model']}")
    print(f" PDF Sources         : {summary_data['pdf_sources']}")
    print(f" Execution Duration  : {summary_data['execution_time_sec']} seconds")
    print("-" * 65)
    print(" AVERAGE RAGAS SCORES:")
    print(f"  * Faithfulness        : {summary_data['faithfulness_avg']}")
    print(f"  * Answer Relevancy    : {summary_data['answer_relevancy_avg']}")
    print(f"  * Context Precision   : {summary_data['context_precision_avg']}")
    print(f"  * Context Recall      : {summary_data['context_recall_avg']}")
    print("-" * 65)
    print(" QUERY RESULTS:")
    
    for idx, row in results_df.iterrows():
        q_text = row.get('question') or row.get('user_input') or 'N/A'
        a_text = row.get('answer') or row.get('response') or 'N/A'
        f_score = row.get('faithfulness', 'N/A')
        r_score = row.get('answer_relevancy', 'N/A')

        print(f"\n [Q{idx+1}]: {q_text}")
        print(f"  Ans : {a_text}")
        print(f"  Scores -> Faithfulness: {f_score} | Relevancy: {r_score}")
    print("="*65 + "\n")

# --------------------------------------------------------------------------
# 7. Main Orchestrator
# --------------------------------------------------------------------------
def run_benchmark():
    print("=== Starting PDF RAG Benchmark (hard_stress_test.pdf) ===")

    test_questions = [
        "What was the Q2 revenue and YoY Growth percentage for AI Services?",
        "What is the maximum allowable latency for microservices and what downtime breach triggers a Severity 1 escalation?",
        "How much of the Cloud Infra growth came from strategic acquisitions in Q2, and what is its audit compliance status?"
    ]

    ground_truths = [
        "The Q2 revenue for AI Services was $12,450,000 and the YoY Growth was +53.7%.",
        "The maximum allowable latency for microservices is strictly 45ms under peak load, and any downtime breach exceeding 120 seconds triggers a Severity 1 escalation.",
        "Cloud Infra growth includes $3.2M from strategic acquisitions in Q2, and its audit compliance status is Audited Passed."
    ]

    # 1. Index Vector Store
    vectorstore, embedding_model, pdf_names = build_vectorstore(BENCHMARK_CONFIG)

    # 2. Configure Evaluator Judge
    print(f"[2/5] Setting up RAGAS Evaluator Judge ({BENCHMARK_CONFIG['evaluator_model']})...")
    evaluator_chat = CleanJSONChatOllama(
        model=BENCHMARK_CONFIG["evaluator_model"], 
        temperature=0,
        format="json"
    )
    evaluator_llm = LangchainLLMWrapper(evaluator_chat)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embedding_model)

    # Instantiate and configure metric objects cleanly
    active_metrics = init_ragas_metrics(evaluator_llm, evaluator_embeddings)

    # Single-worker sequential execution prevents Ollama concurrency timeouts
    run_config = RunConfig(max_workers=1, timeout=120)

    # 3. Loop through benchmark models
    print(f"\n[3/5] Starting Evaluation Loop for models: {MODELS_TO_BENCHMARK}")
    
    for model_index, model_name in enumerate(MODELS_TO_BENCHMARK, start=1):
        print(f"\n>>> [{model_index}/{len(MODELS_TO_BENCHMARK)}] Benchmarking Model: {model_name} <<<")
        start_time = time.time()

        retrieved_contexts, generated_answers = generate_rag_responses(
            model_name, vectorstore, test_questions, BENCHMARK_CONFIG
        )

        eval_dict = {
            "question": test_questions,
            "user_input": test_questions,
            "contexts": retrieved_contexts,
            "answer": generated_answers,
            "response": generated_answers,
            "ground_truth": ground_truths,
            "reference": ground_truths
        }
        dataset = Dataset.from_dict(eval_dict)

        print(f"      Evaluating output metrics sequentially for '{model_name}'...")
        results = evaluate(
            dataset=dataset,
            metrics=active_metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=run_config
        )
        results_df = results.to_pandas()
        execution_time = time.time() - start_time

        run_file, summary_path, summary_data = log_and_collect_results(
            model_name, results_df, BENCHMARK_CONFIG, pdf_names, execution_time
        )

        print_model_report(model_name, summary_data, results_df)

    print("\n=== All Model Benchmarks Completed ===")
    print(f"Master comparative log updated: '{BENCHMARK_CONFIG['summary_csv']}'")

if __name__ == "__main__":
    run_benchmark()