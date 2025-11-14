import pdfplumber
# Fallback: Extract text from PDF or TXT if pyresparser fails
def extract_text_fallback(file_path):
	try:
		if file_path.lower().endswith('.pdf'):
			text = ""
			with pdfplumber.open(file_path) as pdf:
				for page in pdf.pages:
					text += page.extract_text() or ""
			return text
		elif file_path.lower().endswith('.txt'):
			with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
				return f.read()
	except Exception:
		return None
	return None
import streamlit as st
import os
import pickle
import tempfile
import traceback
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from pyresparser import ResumeParser
from concurrent.futures import ThreadPoolExecutor

# Gmail API OAuth Configuration
# SCOPES: Defines what Gmail permissions the app requests
# - gmail.readonly: Allows reading emails and accessing attachments only
# For more details, see PERMISSIONS.md and GMAIL_SETUP.md
# If modifying these scopes, delete the file token.pickle to force re-authorization.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_gmail():
	creds = None
	try:
		if os.path.exists('token.pickle'):
			with open('token.pickle', 'rb') as token:
				creds = pickle.load(token)
		if not creds or not creds.valid:
			if creds and creds.expired and creds.refresh_token:
				creds.refresh(Request())
			else:
				cred_path = r'C:\\Users\\Srinidh\\Desktop\\prj\\credentials.json'
				if not os.path.exists(cred_path):
					st.error(f"credentials.json not found at {cred_path}. Please place your Gmail API credentials there.")
					st.info("📖 Need help? See the [Gmail Setup Guide](https://github.com/marri14reddy-byte/JD.Resume_Matcher/blob/main/GMAIL_SETUP.md) for step-by-step instructions.")
					return None
				flow = InstalledAppFlow.from_client_secrets_file(cred_path, SCOPES)
				creds = flow.run_local_server(port=0)
			with open('token.pickle', 'wb') as token:
				pickle.dump(creds, token)
		return creds
	except Exception as e:
		st.error(f"Failed to authenticate with Gmail: {e}")
		st.info("If you see a browser window, complete the Google login and allow access. If not, check your credentials.json file and network connection.")
		import traceback as tb
		st.error(tb.format_exc())
		return None

def download_attachments_from_gmail():
	creds = authenticate_gmail()
	if creds is None:
		st.error("Gmail authentication failed. Cannot download resumes.")
		return 0, None
	try:
		service = build('gmail', 'v1', credentials=creds)
		results = service.users().messages().list(userId='me', q='has:attachment', maxResults=5).execute()
		messages = results.get('messages', [])
		download_dir = 'downloaded_resumes'
		os.makedirs(download_dir, exist_ok=True)
		count = 0
		for msg in messages:
			msg_id = msg['id']
			msg_data = service.users().messages().get(userId='me', id=msg_id).execute()
			for part in msg_data['payload'].get('parts', []):
				filename = part.get('filename')
				if filename and (filename.endswith('.pdf') or filename.endswith('.txt')):
					att_id = part['body'].get('attachmentId')
					if att_id:
						att = service.users().messages().attachments().get(userId='me', messageId=msg_id, id=att_id).execute()
						file_data = att['data']
						import base64
						file_bytes = base64.urlsafe_b64decode(file_data.encode('UTF-8'))
						with open(os.path.join(download_dir, filename), 'wb') as f:
							f.write(file_bytes)
						count += 1
		return count, download_dir
	except Exception as e:
		st.error(f"Failed to connect to Gmail or download attachments: {e}")
		import traceback as tb
		st.error(tb.format_exc())
		return 0, None

# Parse JD text for skills, certifications, experience
def parse_jd_text(jd_text):
	skills, certifications = set(), set()
	experience = 0
	lines = jd_text.lower().split('\n')
	for line in lines:
		if 'skill' in line:
			skills.update([w.strip() for w in line.split(':')[-1].split(',')])
		if 'certification' in line:
			certifications.update([w.strip() for w in line.split(':')[-1].split(',')])
		if 'experience' in line:
			import re
			match = re.search(r'(\d+)', line)
			if match:
				experience = max(experience, int(match.group(1)))
	return {
		'skills': {s for s in skills if s},
		'certifications': {c for c in certifications if c},
		'experience': experience
	}


# Parse resume using pyresparser and pdfplumber, combine results
import hashlib
parse_cache = {}
def parse_resume_combined(file_path):
	# Use file hash for cache key
	try:
		with open(file_path, 'rb') as f:
			file_hash = hashlib.md5(f.read()).hexdigest()
	except Exception:
		file_hash = file_path  # fallback
	if file_hash in parse_cache:
		return parse_cache[file_hash]
	result = None
	try:
		data = ResumeParser(file_path).get_extracted_data()
		if not data:
			data = {}
	except Exception:
		data = {}
	# Always extract raw text for accuracy
	text = extract_text_fallback(file_path)
	if text:
		data['text'] = text
	# Clean up fields
	data['name'] = data.get('name', os.path.basename(file_path))
	data['skills'] = list(set([s.strip() for s in data.get('skills', []) if s]))
	data['certifications'] = list(set([c.strip() for c in data.get('certifications', []) if c]))
	data['total_experience'] = data.get('total_experience', 0) or 0
	data['email'] = data.get('email', '')
	data['mobile_number'] = data.get('mobile_number', '')
	parse_cache[file_hash] = data
	return data

# Match resumes to JD fields
def match_resumes_structured(jd_fields, parsed_resumes):
	results = []
	for r in parsed_resumes:
		if not r or not r.get('skills'): continue
		skill_overlap = len(set(map(str.lower, r.get('skills', []))) & set(map(str.lower, jd_fields['skills'])))
		cert_overlap = len(set(map(str.lower, r.get('certifications', []))) & set(map(str.lower, jd_fields['certifications'])))
		exp = r.get('total_experience', 0) or 0
		exp_score = 1 if exp >= jd_fields['experience'] else 0
		score = skill_overlap * 2 + cert_overlap + exp_score
		results.append({'filename': r.get('name', 'Unknown'), 'score': score, 'data': r})
	results.sort(key=lambda x: x['score'], reverse=True)
	return results

st.title("Resume Matcher")

st.write("Upload a job description (PDF or text file):")
job_desc_file = st.file_uploader("Job Description", type=["pdf", "txt"])

# Option to upload resumes manually

st.write("Or upload resumes manually (PDF or text files, multiple allowed):")
manual_resume_files = st.file_uploader("Upload Resumes", type=["pdf", "txt"], accept_multiple_files=True, key="manual_resumes")

# Add Gmail sync button
if st.button("Sync Resumes from Gmail"):
	with st.spinner("Syncing resumes from Gmail and storing locally..."):
		count, folder = download_attachments_from_gmail()
		if count > 0:
			st.success(f"Downloaded and stored {count} resumes from Gmail in '{folder}' folder.")
		else:
			st.info("No new resumes were downloaded from Gmail.")


job_desc_text = None
jd_fields = None
similarity_threshold = None
if job_desc_file:
	if job_desc_file.name.lower().endswith('.pdf'):
		with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
			tmp.write(job_desc_file.read())
			jd_path = tmp.name
		jd_data = ResumeParser(jd_path).get_extracted_data()
		job_desc_text = jd_data.get('skills', [])
		os.unlink(jd_path)
	else:
		job_desc_text = job_desc_file.read().decode('utf-8', errors='ignore')
	st.success("Job description uploaded.")
	jd_fields = parse_jd_text(job_desc_text if isinstance(job_desc_text, str) else '\n'.join(job_desc_text))
	similarity_threshold = st.slider("Minimum Match Score", min_value=0, max_value=10, value=0, step=1, help="Only resumes with score >= this value will be highlighted after download.")

if job_desc_file and similarity_threshold is not None:
	# Collect resumes from manual upload
	resume_files = []
	temp_manual_files = []
	if manual_resume_files:
		import tempfile
		for uploaded in manual_resume_files:
			suffix = '.pdf' if uploaded.name.lower().endswith('.pdf') else '.txt'
			with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
				tmp.write(uploaded.read())
				temp_manual_files.append(tmp.name)
		resume_files.extend(temp_manual_files)

	# Always fetch Gmail resumes from local storage (downloaded_resumes)
	gmail_folder = 'downloaded_resumes'
	gmail_files = []
	if os.path.exists(gmail_folder):
		gmail_files = [os.path.join(gmail_folder, f) for f in os.listdir(gmail_folder) if f.lower().endswith(('.pdf', '.txt'))]

	manual_files = list(resume_files)
	files = manual_files + gmail_files

	if st.button("Parse and Match Resumes"):
		with st.spinner("Parsing and matching resumes..."):
			with ThreadPoolExecutor(max_workers=16) as executor:
				parsed_resumes = list(executor.map(parse_resume_combined, files))

			# Separate manual and Gmail resumes
			manual_count = len(manual_files)
			gmail_count = len(gmail_files)
			# Compute scores for all resumes
			scored_manual = []
			scored_gmail = []
			for i, r in enumerate(parsed_resumes):
				if r:
					score = 0
					if r.get('skills') or r.get('certifications') or r.get('total_experience'):
						ranked = match_resumes_structured(jd_fields, [r])
						score = ranked[0]['score'] if ranked else 0
					entry = {'index': i, 'score': score, 'resume': r}
					if i < manual_count:
						scored_manual.append(entry)
					else:
						scored_gmail.append(entry)
			# Sort and take top 10 for each
			top_manual = sorted(scored_manual, key=lambda x: x['score'], reverse=True)[:10]
			top_gmail = sorted(scored_gmail, key=lambda x: x['score'], reverse=True)[:10]
			if top_manual:
				st.subheader(f"Top {len(top_manual)} Manually Uploaded Resumes:")
				for item in top_manual:
					i = item['index']
					r = item['resume']
					score = item['score']
					highlight = score >= similarity_threshold
					with st.expander(f"[Manual] {r.get('name', os.path.basename(files[i]))} - Score: {score}"):
						st.markdown(f"**Name:** {r.get('name', 'Not found')}")
						st.markdown(f"**Email:** {r.get('email', 'Not found')}")
						st.markdown(f"**Contact:** {r.get('mobile_number', 'Not found')}")
						st.markdown(f"**Skills:** {', '.join(r.get('skills', []))}")
						st.markdown(f"**Certifications:** {', '.join(r.get('certifications', []))}")
						st.markdown(f"**Experience:** {r.get('total_experience', 'Not found')} years")
						st.markdown("---")
						st.markdown("**Full Resume Text:**")
						st.text(r.get('text', '')[:5000] + ("..." if len(r.get('text', '')) > 5000 else ""))
						with open(files[i], 'rb') as f:
							st.download_button(label="Download Resume File", data=f, file_name=os.path.basename(files[i]))
						if highlight:
							st.success("This resume meets or exceeds the minimum match score.")
			if top_gmail:
				st.subheader(f"Top {len(top_gmail)} Gmail Downloaded Resumes:")
				for item in top_gmail:
					i = item['index']
					r = item['resume']
					score = item['score']
					highlight = score >= similarity_threshold
					with st.expander(f"[Gmail] {r.get('name', os.path.basename(files[i]))} - Score: {score}"):
						st.markdown(f"**Name:** {r.get('name', 'Not found')}")
						st.markdown(f"**Email:** {r.get('email', 'Not found')}")
						st.markdown(f"**Contact:** {r.get('mobile_number', 'Not found')}")
						st.markdown(f"**Skills:** {', '.join(r.get('skills', []))}")
						st.markdown(f"**Certifications:** {', '.join(r.get('certifications', []))}")
						st.markdown(f"**Experience:** {r.get('total_experience', 'Not found')} years")
						st.markdown("---")
						st.markdown("**Full Resume Text:**")
						st.text(r.get('text', '')[:5000] + ("..." if len(r.get('text', '')) > 5000 else ""))
						with open(files[i], 'rb') as f:
							st.download_button(label="Download Resume File", data=f, file_name=os.path.basename(files[i]))
						if highlight:
							st.success("This resume meets or exceeds the minimum match score.")
			# Optionally, show a message if there are resumes that could not be parsed
			unparsed = [i for i, r in enumerate(parsed_resumes) if not r]
			if unparsed:
				st.subheader(f"{len(unparsed)} resumes could not be parsed:")
				for i in unparsed:
					label = '[Manual]' if i < manual_count else '[Gmail]'
					with st.expander(f"{label} {os.path.basename(files[i])} - Could not be parsed"):
						st.warning("This resume could not be parsed or read.")
else:
	st.info("Please upload a job description to enable downloading resumes from Gmail.")
