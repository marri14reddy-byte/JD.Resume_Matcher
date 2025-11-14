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

## Gmail Integration (Optional)

To enable automatic resume ingestion from Gmail:

1. **Review Permissions**: See [PERMISSIONS.md](PERMISSIONS.md) for details on what access is needed
2. **Setup Gmail API**: Follow the step-by-step guide in [GMAIL_SETUP.md](GMAIL_SETUP.md)
3. **Place credentials**: Download `credentials.json` from Google Cloud Console and place it in the app directory

**Note:** Gmail integration is completely optional. You can use the app without it by uploading resumes manually (ZIP files or individual PDFs).

## Permissions & Security

This application requires certain permissions to function:
- **Gmail API** (optional): For automatic resume fetching - [Setup Guide](GMAIL_SETUP.md)
- **File System**: For storing resumes, embeddings, and cache locally
- **Network**: For downloading ML models and accessing Gmail API (if enabled)

All data stays local on your machine. See [PERMISSIONS.md](PERMISSIONS.md) for complete details.

## Notes

- LLM summaries use an Ollama endpoint and llama3.1 8B instruct; adjust host/model inside `resume_matcher_rag.py` if needed.
- ANN compaction/rebuild can be heavy; start it when the machine is idle.
- For very large datasets or multi-instance, consider a remote vector store (Qdrant/Milvus). The app keeps the retrieval contract simple for swapping.
