from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Allow CORS for local development (optional, but useful for Flowise)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESUME_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloaded_resumes')

@app.get("/resumes")
def list_resumes():
    if not os.path.exists(RESUME_FOLDER):
        return {"resumes": []}
    resumes = [f for f in os.listdir(RESUME_FOLDER) if f.lower().endswith((".pdf", ".txt", ".docx"))]
    return {"resumes": resumes}

@app.get("/resume/{filename}")
def get_resume_text(filename: str):
    import urllib.parse
    safe_filename = urllib.parse.unquote(filename)
    file_path = os.path.join(RESUME_FOLDER, safe_filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Resume not found")
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.pdf':
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
        elif ext == '.docx':
            import docx
            doc = docx.Document(file_path)
            text = '\n'.join([para.text for para in doc.paragraphs])
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        else:
            text = ''
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")
    return {"filename": filename, "text": text}
