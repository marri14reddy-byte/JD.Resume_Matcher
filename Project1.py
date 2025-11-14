import streamlit as st
import os

# Check for credentials.json at startup
if not os.path.exists('credentials.json'):
    st.warning("⚠️ WARNING: credentials.json not found in project directory! Gmail sync will be disabled.")
    st.info("📖 To enable Gmail integration, see [GMAIL_SETUP.md](https://github.com/marri14reddy-byte/JD.Resume_Matcher/blob/main/GMAIL_SETUP.md) for setup instructions.")

st.write("App started! (after credentials check)")

import re
import sys
import subprocess
import os
import base64
import streamlit as st
# Gmail sync imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Gmail API OAuth Configuration
# SCOPES: Defines what Gmail permissions the app requests
# - gmail.modify: Allows reading emails, accessing attachments, and marking emails as read
# - Does NOT allow sending, deleting, or accessing sensitive settings
# For more details, see PERMISSIONS.md and GMAIL_SETUP.md
# If modifying these scopes, delete the file token.json to force re-authorization.
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

# Gmail sync function
def sync_gmail():
    # Check if credentials.json exists
    if not os.path.exists('credentials.json'):
        st.sidebar.error("❌ credentials.json not found!")
        st.sidebar.info("📖 See [GMAIL_SETUP.md](https://github.com/marri14reddy-byte/JD.Resume_Matcher/blob/main/GMAIL_SETUP.md) for setup instructions.")
        return
    
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    try:
        service = build('gmail', 'v1', credentials=creds)
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], q="is:unread has:attachment").execute()
        messages = results.get('messages', [])
        if not messages:
            st.sidebar.info('No new resumes found in your Gmail.')
            return
        st.sidebar.info(f'Found {len(messages)} new emails with attachments. Processing...')
        for message in messages:
            try:
                msg = service.users().messages().get(userId='me', id=message['id']).execute()
                payload = msg.get('payload', {})
                parts = payload.get('parts', [])
                for part in parts:
                    if part.get('filename'):
                        body = part.get('body', {})
                        if 'data' in body:
                            data = body['data']
                        elif 'attachmentId' in body:
                            att_id = body['attachmentId']
                            att = service.users().messages().attachments().get(userId='me', messageId=message['id'], id=att_id).execute()
                            data = att['data']
                        else:
                            continue
                        file_data = base64.urlsafe_b64decode(data.encode('UTF-8'))
                        path = os.path.join('resumes', part['filename'])
                        with open(path, 'wb') as f:
                            f.write(file_data)
                service.users().messages().modify(userId='me', id=message['id'], body={'removeLabelIds': ['UNREAD']}).execute()
            except Exception as e:
                st.sidebar.warning(f"Error processing message {message['id']}: {e}")
        st.sidebar.success('Resumes synced from Gmail.')
    except HttpError as error:
        st.sidebar.error(f'An error occurred: {error}')
        if 'insufficient' in str(error).lower() or 'permission' in str(error).lower():
            st.sidebar.info("💡 This might be a permission issue. See [PERMISSIONS.md](https://github.com/marri14reddy-byte/JD.Resume_Matcher/blob/main/PERMISSIONS.md) for details.")

# Ensure Hugging Face transformers and dependencies are installed
def ensure_hf_dependencies():
    try:
        import faiss
        from transformers import pipeline
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu", "transformers", "torch", "sentencepiece"])
        import faiss
        from transformers import pipeline

ensure_hf_dependencies()

# Load RAG model and tokenizer
def load_rag_model():
    from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
    try:
        tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
        retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True)
        model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)
        return tokenizer, model
    except Exception as e:
        return None, None

# Use RAG for Q&A extraction
def rag_extract(text, question):
    tokenizer, model = load_rag_model()
    if tokenizer is None or model is None:
        return None
    input_dict = tokenizer.prepare_seq2seq_batch(
        question, [text], return_tensors="pt"
    )
    generated = model.generate(
        input_ids=input_dict["input_ids"],
        attention_mask=input_dict["attention_mask"]
    )
    answer = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return answer.strip()

# Robust contact extraction using spaCy for name and regex for email/phone

# Enhanced extraction using RAG if available, fallback to spaCy/regex
def is_valid_email(email):
    """Simple regex check for email validity."""
    if not email or not isinstance(email, str):
        return False
    email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(email_regex, email) is not None

def is_plausible_name(name):
    """Check if the name is plausible (not a long sentence or generic answer)."""
    if not name or not isinstance(name, str):
        return False
    # Check for sentence-like structures or common failure modes
    if len(name.split()) > 5 or any(kw in name.lower() for kw in ['candidate', 'email', 'phone', 'resume', 'n/a', 'unknown']):
        return False
    return True

def extract_contact_details_hybrid(text):
    """
    A hybrid approach to extract contact details, using both RAG and a robust spaCy/regex fallback,
    then validating and selecting the best result.
    """
    # --- Step 1: Get results from both methods ---

    # Method A: RAG Extraction
    rag_name = rag_extract(text, "What is the candidate's full name?")
    rag_email = rag_extract(text, "What is the candidate's email address?")
    rag_phone = rag_extract(text, "What is the candidate's phone number?")

    # Method B: Rule-based Extraction (spaCy and Regex)
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    
    doc = nlp(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Name extraction
    rule_name = "N/A"
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    if persons:
        rule_name = persons[0]
    else: # Fallback if no PERSON entity is found
        for line in lines[:5]:
            if re.match(r"^[A-Z][a-z']+(?:\s[A-Z][a-z']+)+$", line.strip()):
                rule_name = line.strip()
                break
    if rule_name == "N/A" and lines:
        rule_name = lines[0] # Final fallback

    # Email extraction
    rule_email = "N/A"
    email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    if email_match:
        rule_email = email_match.group(0)

    # Phone extraction
    rule_phone = "N/A"
    phone_match = re.search(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text)
    if phone_match:
        rule_phone = phone_match.group(0)

    # --- Step 2: Validate and Select the Best Result ---
    
    final_name = rag_name if is_plausible_name(rag_name) else rule_name
    final_email = rag_email if is_valid_email(rag_email) else rule_email
    final_phone = rag_phone if rag_phone and len(rag_phone) > 6 else rule_phone # Simple validation for phone

    return final_name, final_email, final_phone
import importlib
import subprocess
import sys

# Ensure spaCy model is available
def ensure_spacy_model():
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except (OSError, ImportError):
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        import spacy
        spacy.load("en_core_web_sm")
ensure_spacy_model()
import streamlit as st
from PyPDF2 import PdfReader
import faiss
import shutil  # For clearing stored resumes if needed
import os  # Add missing import for os


# Directories to store resumes and job descriptions
RESUME_DIR = 'resumes'
JD_DIR = 'jds'
if not os.path.exists(RESUME_DIR):
    os.makedirs(RESUME_DIR)
if not os.path.exists(JD_DIR):
    os.makedirs(JD_DIR)



# Sidebar design improvements

st.sidebar.markdown("""
<h2 style='text-align:center; color:#4F8BF9;'>📄 Resume & JD Manager</h2>
<hr style='border:1px solid #4F8BF9;'>
""", unsafe_allow_html=True)

st.write("Sidebar rendered!")

# Mode selector
match_mode = st.sidebar.radio(
    "Select Matching Mode",
    ('Match Resumes to JD', 'Match JDs to Resume'),
    help="Choose whether to find the best resumes for a job, or the best jobs for a resume."
)
 script.


with st.sidebar.expander("🧑‍💼 Manage Resumes", expanded=True):
    uploaded_resumes = st.file_uploader("Upload resumes (PDF or TXT)", type=['pdf', 'txt'], accept_multiple_files=True, key='resume_uploader')
    if uploaded_resumes:
        for resume in uploaded_resumes:
            file_path = os.path.join(RESUME_DIR, resume.name)
            with open(file_path, 'wb') as f:
                f.write(resume.getbuffer())
        st.success(f"Uploaded {len(uploaded_resumes)} resume(s).")
    
    if st.button("🔄 Sync Resumes from Gmail"):
        sync_gmail()
    
    if st.button("🗑️ Clear All Stored Resumes"):
        shutil.rmtree(RESUME_DIR)
        os.makedirs(RESUME_DIR)
        st.success("All resumes cleared.")

with st.sidebar.expander("💼 Manage Job Descriptions", expanded=True):
    uploaded_jds = st.file_uploader("Upload job descriptions (PDF or TXT)", type=['pdf', 'txt'], accept_multiple_files=True, key='jd_uploader')
    if uploaded_jds:
        for jd in uploaded_jds:
            file_path = os.path.join(JD_DIR, jd.name)
            with open(file_path, 'wb') as f:
                f.write(jd.getbuffer())
        st.success(f"Uploaded {len(uploaded_jds)} JD(s).")

    if st.button("🗑️ Clear All Stored JDs"):
        shutil.rmtree(JD_DIR)
        os.makedirs(JD_DIR)
        st.success("All JDs cleared.")

# Custom CSS for cards
st.markdown("""
<style>
    .card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: box-shadow 0.3s;
    }
    .card:hover {
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .contact-box {
        margin-bottom:0.5em; 
        padding:0.5em 1em; 
        background:#f0f2f6; 
        border-radius:6px; 
        border:1px solid #d1d9e6; 
        color:#333;
    }
</style>
""", unsafe_allow_html=True)




st.write("Before main section processing!")
# Main section processing
if match_mode == 'Match Resumes to JD':
    # List all stored JDs for selection
    jd_files = [f for f in os.listdir(JD_DIR) if f.endswith('.pdf') or f.endswith('.txt')]
    job_text = None  # Ensure job_text is always defined before use
    if not jd_files:
        st.title("Welcome to the Resume & JD Matching App!")
        st.info("To get started, please upload one or more job descriptions using the sidebar on the left.")
    else:
        selected_jd = st.selectbox("Select a job description to match resumes:", jd_files)
        if selected_jd:
            st.header("Job Description and Resume Matching")
            jd_path = os.path.join(JD_DIR, selected_jd)
            try:
                if selected_jd.endswith('.pdf'):
                    reader = PdfReader(jd_path)
                    job_text = ''
                    for page in reader.pages:
                        job_text += page.extract_text() + '\n'
                elif selected_jd.endswith('.txt'):
                    with open(jd_path, 'r', encoding='utf-8') as f:
                        job_text = f.read()
            except Exception as e:
                st.warning(f"Error processing selected JD: {str(e)}")

    if job_text:
        st.write("Job description loaded successfully.")

        threshold = st.slider(
            "Set similarity threshold for eligible resumes",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.01,
            help="Only resumes with a similarity score above this threshold will be shown."
        )

        show_resume_content = st.checkbox("Show matched resume content", value=True)

        # Load and extract text from all stored resumes
        resumes = []
        resume_texts = []
        for filename in os.listdir(RESUME_DIR):
            path = os.path.join(RESUME_DIR, filename)
            try:
                if filename.endswith('.pdf'):
                    reader = PdfReader(path)
                    text = ''
                    for page in reader.pages:
                        text += page.extract_text() + '\n'
                elif filename.endswith('.txt'):
                    with open(path, 'r', encoding='utf-8') as f:
                        text = f.read()
                else:
                    continue
                resumes.append((filename, path))
                resume_texts.append(text)
            except Exception as e:
                st.warning(f"Error processing {filename}: {str(e)}")

        if resumes and resume_texts:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                import subprocess
                import sys
                # Upgrade pip to latest version to avoid installation issues
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--upgrade", "pip"
                ])
                # Try to install a specific version of torch to avoid missing file errors
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "--force-reinstall", "--upgrade", "--no-cache-dir", "torch==1.13.1", "torchvision", "torchaudio", "--extra-index-url", "https://download.pytorch.org/whl/cpu"
                    ])
                except Exception as e:
                    print("Warning: Failed to install torch packages:", e)
                # Finally, install sentence-transformers with force reinstall
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--force-reinstall", "--upgrade", "--no-cache-dir", "sentence-transformers"
                ])
                from sentence_transformers import SentenceTransformer
            st.info("Generating embeddings for job description and resumes. This may take a few seconds...")
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
            jd_emb = embedder.encode([job_text], normalize_embeddings=True)
            resume_embs = embedder.encode(resume_texts, normalize_embeddings=True)

            # Build FAISS index for cosine similarity
            dim = jd_emb.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(resume_embs)

            # Search for top matches
            D, I = index.search(jd_emb, len(resume_embs))
            similarities = D[0]
            sorted_indices = similarities.argsort()[::-1]

            st.header("Eligible Resumes")
            st.write(f"Resumes sorted by embedding similarity score (threshold: {threshold:.2f}). Higher scores indicate better matches based on skills, experience, and other content.")

            eligible_count = 0
            for idx in sorted_indices:
                score = similarities[idx]
                if score > threshold:
                    filename, path = resumes[idx]
                    if filename == selected_jd:
                        continue
                    st.subheader(f"{filename} (Similarity: {score:.2f})")
                    with open(path, 'rb') as f:
                        st.download_button(
                            label="Download Resume",
                            data=f,
                            file_name=filename,
                            mime='application/pdf' if filename.endswith('.pdf') else 'text/plain'
                        )
                    # Extract and display contact details
                    text = resume_texts[idx]
                    name, email, phone = extract_contact_details_hybrid(text)
                    st.markdown(f"""
    <div class='contact-box'>
        <b>Contact Details:</b><br>
        <b>Name:</b> {name}<br>
        <b>Email:</b> {email}<br>
        <b>Contact Number:</b> {phone}
    </div>
    """, unsafe_allow_html=True)
                    if show_resume_content:
                        with st.expander("🔎 View Resume Content"):
                            st.text(text)
                    st.markdown("</div>", unsafe_allow_html=True) # Close card
                    eligible_count += 1
                else:
                    break  # Stop after ineligible ones

            if eligible_count == 0:
                st.info("No resumes meet the eligibility threshold. Try adjusting the threshold or uploading more resumes.")
        else:
            st.info("No resumes stored yet. Upload some in the sidebar.")
    else:
        st.info("Upload and select a job description to start matching.")

elif match_mode == 'Match JDs to Resume': # Match JDs to Resume
    resume_files = [f for f in os.listdir(RESUME_DIR) if f.endswith(('.pdf', '.txt'))]
    if not resume_files:
        st.title("Welcome to the Resume & JD Matching App!")
        st.info("To get started, please upload one or more resumes using the sidebar on the left.")
    else:
        selected_resume = st.selectbox("Select a resume to match job descriptions against:", resume_files)
        if selected_resume:
            # This is the new logic
            st.header(f"Matching Job Descriptions for: {selected_resume}")
            
            # Load selected resume text
            resume_path = os.path.join(RESUME_DIR, selected_resume)
            resume_text = ""
            try:
                if selected_resume.endswith('.pdf'):
                    reader = PdfReader(resume_path)
                    for page in reader.pages:
                        resume_text += page.extract_text() + '\n'
                else:
                    with open(resume_path, 'r', encoding='utf-8') as f:
                        resume_text = f.read()
            except Exception as e:
                st.error(f"Error reading resume: {e}")

            if resume_text:
                # Load all JDs
                jds = []
                jd_texts = []
                for filename in os.listdir(JD_DIR):
                    path = os.path.join(JD_DIR, filename)
                    try:
                        if filename.endswith('.pdf'):
                            reader = PdfReader(path)
                            text = ''
                            for page in reader.pages:
                                text += page.extract_text() + '\n'
                        elif filename.endswith('.txt'):
                            with open(path, 'r', encoding='utf-8') as f:
                                text = f.read()
                        else:
                            continue
                        jds.append((filename, path))
                        jd_texts.append(text)
                    except Exception as e:
                        st.warning(f"Error processing {filename}: {str(e)}")

                if jds and jd_texts:
                    st.info("Generating embeddings... This may take a moment.")
                    from sentence_transformers import SentenceTransformer
                    embedder = SentenceTransformer('all-MiniLM-L6-v2')
                    
                    resume_emb = embedder.encode([resume_text], normalize_embeddings=True)
                    jd_embs = embedder.encode(jd_texts, normalize_embeddings=True)  

                    dim = resume_emb.shape[1]
                    index = faiss.IndexFlatIP(dim)
                    index.add(jd_embs)

                    D, I = index.search(resume_emb, len(jds))
                    similarities = D[0]
                    sorted_indices = similarities.argsort()[::-1]

                    st.header("Matching Job Descriptions")
                    threshold = st.slider(
                        "Set similarity threshold for eligible JDs",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.2,
                        step=0.01,
                        help="Only JDs with a similarity score above this threshold will be shown."
                    )
                    for idx in sorted_indices:
                        score = similarities[idx]
                        if score >= threshold:
                            filename, path = jds[idx]
                            st.subheader(f"{filename} (Similarity: {score:.2f})")
                            # Optionally show JD content
                            with st.expander("View Job Description"):
                                st.text(jd_texts[idx])
                else:
                    st.info("No job descriptions found to match against. Please upload some.")

