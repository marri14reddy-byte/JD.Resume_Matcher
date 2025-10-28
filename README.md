<<<<<<< HEAD
# JD.Resume_Matcher
=======
# JD ↔ Resume Matcher (Streamlit)

Two-tab app for matching resumes against a job description at scale, with deterministic scoring and ANN-based fast retrieval.

## Features

- Two tabs: RAG Matcher and Resumes
- Deterministic score with sliders (Skills/Experience/Industry) and boosts (company/location)
- ANN HNSW chunk index (FAISS) for fast retrieval on 10k→100k+ resumes
- Bulk ingest from ZIP or Gmail; incremental embedding cache; float16 on disk
- GPU auto-detect, batched encodes, and adjustable batch size
- Duplicate detection and cache compaction
- Configurable chunking and ANN parameters; metrics panel

## Quickstart

1. Install dependencies (Windows PowerShell):

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the app:

```powershell
streamlit run resume_matcher_rag.py
```

3. Optional: enable ANN
- In the Resumes tab → "Vector index (ANN)" → click "Index new resumes (chunks)".

## Notes

- LLM summaries use an Ollama endpoint and llama3.1 8B instruct; adjust host/model inside `resume_matcher_rag.py` if needed.
- ANN compaction/rebuild can be heavy; start it when the machine is idle.
- For very large datasets or multi-instance, consider a remote vector store (Qdrant/Milvus). The app keeps the retrieval contract simple for swapping.
>>>>>>> c1818ad (Initial commit: Streamlit + Chroma resume matcher)
