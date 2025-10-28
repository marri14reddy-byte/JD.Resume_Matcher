---
title: JD ↔ Resume Matcher
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.39.0
app_file: resume_matcher_rag.py
pinned: false
---

# Deploy to Hugging Face Spaces (no credit card)

This app ranks resumes against a Job Description using Streamlit + ChromaDB. It now auto-detects Spaces and defaults Chroma storage to `/data/chroma_store` so your index persists across restarts.

## Requirements
Ensure your `requirements.txt` includes at least:
- streamlit
- chromadb
- sentence-transformers
- numpy
- pandas
- scikit-learn
- requests
- pdfplumber
- python-docx
- mysql-connector-python
- rank-bm25
- google-api-python-client
- google-auth
- google-auth-oauthlib

## Steps
1. Create a new Space at https://huggingface.co/spaces → SDK: Streamlit, Hardware: CPU basic
2. Upload this repository’s files (or connect to Git and push)
3. Settings → App:
   - SDK: Streamlit
   - App file: `resume_matcher_rag.py`
4. Settings → Variables and secrets:
   - `LLM_BASE_URL` (optional): e.g., `http://67.11.191.239:11434`
   - `LLM_MODEL` (optional): e.g., `llama3.1:8b`
   - Optional MySQL: `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DB`
5. Settings → Persistence: Enable Persistent storage
   - The app will store Chroma at `/data/chroma_store` automatically
6. Open the app:
   - Settings → Search index → Build/Update index to index resumes
   - RAG Matcher → Upload a JD to view top matches

## Notes
- If the remote LLM isn’t reachable (e.g., port 11434 blocked), the app falls back to a deterministic summary. Ranking still works.
- MySQL must be reachable from the Space if you use import or BLOB fallback.
- Free Spaces may sleep when idle, but data in `/data` persists.
