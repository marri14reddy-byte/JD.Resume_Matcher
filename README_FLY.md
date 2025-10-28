# Fly.io deployment for Resume Matcher (Streamlit + Chroma)

This guide gets your app running 24/7 on Fly.io with a persistent volume for ChromaDB.

## Prerequisites
- A Fly.io account and the Fly CLI installed on Windows.
- Your project has `resume_matcher_rag.py`, `requirements.txt`, this `Dockerfile`, and `fly.toml`.

## One-time setup
1. Login
```
flyctl auth login
```

2. Initialize app (pick a unique app name when prompted; this will update `fly.toml`)
```
flyctl launch --no-deploy
```

3. Create a persistent volume for Chroma (change region as desired)
```
flyctl volumes create chroma_data --region iad --size 3
```

4. Set secrets (replace values accordingly)
```
flyctl secrets set \
  LLM_BASE_URL=http://67.11.191.239:11434 \
  LLM_MODEL=llama3.1:8b \
  MYSQL_HOST=your-mysql-host \
  MYSQL_USER=your-user \
  MYSQL_PASSWORD=your-password \
  MYSQL_DB=your-db
```

## Deploy
```
flyctl deploy
```

Once deployed, visit the URL shown (e.g., https://<app-name>.fly.dev).

## Notes
- Streamlit runs on port 8080 internally; Fly maps 80/443 for public access.
- The Chroma index is stored in `/app/chroma_store` on the `chroma_data` volume.
- If your network blocks outbound traffic to port 11434, consider exposing your LLM endpoint via HTTPS 443 or placing a reverse proxy in front of it.
- Update `requirements.txt` to include all deps your app needs (Streamlit, Chroma, sentence-transformers, mysql-connector-python, pdfplumber, python-docx, requests, etc.).
