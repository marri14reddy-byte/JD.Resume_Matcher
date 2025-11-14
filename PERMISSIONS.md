# Permissions Documentation

This document outlines all permissions required by the JD.Resume_Matcher application.

## Overview

The JD.Resume_Matcher application requires specific permissions to function properly, particularly when using the Gmail integration feature for automatic resume ingestion.

## Required Permissions

### 1. Gmail API Permissions

**Scope:** `https://www.googleapis.com/auth/gmail.modify`

**Purpose:** This scope allows the application to:
- Read emails from your Gmail inbox
- Access email attachments (resume files)
- Mark emails as read after processing
- Search for specific emails with attachments

**Why Gmail.Modify?**
- The application needs to mark emails as read after downloading resume attachments to avoid reprocessing
- This is the minimum scope required for both reading emails AND modifying their labels/status
- The app does NOT send emails, delete emails, or access sensitive settings

### 2. File System Permissions

**Purpose:** The application needs local file system access to:
- Store downloaded resumes in `downloaded_resumes/` directory
- Cache embeddings and text extractions for performance
- Store OAuth tokens securely (`token.json`)
- Maintain a local SQLite database (`resumes.db`)
- Store FAISS indexes for fast resume retrieval

**Directories Created:**
- `downloaded_resumes/` - Resume files storage
- `downloaded_resumes/.text_cache/` - Cached text extractions
- `downloaded_resumes/.chunk_cache/` - Cached document chunks
- `chroma_store/` - Vector database storage (optional)
- `.llm_cache/` - LLM response cache (optional)

### 3. Network Permissions

**Purpose:** The application makes network requests to:
- Google Gmail API servers (when Gmail sync is enabled)
- Ollama endpoint for LLM summaries (optional, local or remote)
- Hugging Face model hub (for downloading embedding models)

## Security Considerations

### Credentials Storage

- **credentials.json**: Contains OAuth 2.0 client credentials (NOT your password)
  - Should NEVER be committed to version control
  - Already included in `.gitignore`
  - Place in application root directory

- **token.json**: Contains OAuth access/refresh tokens
  - Automatically generated after first OAuth flow
  - Should NEVER be committed to version control
  - Already included in `.gitignore` (as `token.pickle`)
  - Will be regenerated if deleted

### Data Privacy

- **All resume data stays local**: Resumes are stored only on your machine
- **No cloud processing**: Embedding and matching happen locally
- **Gmail access**: Only reads emails you explicitly grant access to
- **No data sharing**: The app does not send your resumes or data to any third party

### Permission Revocation

You can revoke Gmail API access at any time:
1. Visit [Google Account Permissions](https://myaccount.google.com/permissions)
2. Find "JD Resume Matcher" (or your OAuth app name)
3. Click "Remove Access"
4. Delete `token.json` from the application directory

## Optional Features

### Features NOT Requiring Gmail Permissions

You can use the following features without Gmail setup:
- Manual resume upload (ZIP files, individual PDFs)
- Resume scoring and ranking
- Skills/Experience/Industry matching
- Vector similarity search
- Resume database management

### Features Requiring Gmail Permissions

- Automatic resume sync from Gmail
- Mark processed emails as read
- Bulk ingest from Gmail attachments

## Troubleshooting

### "credentials.json not found"
- You need to set up Google Cloud OAuth credentials
- See [GMAIL_SETUP.md](GMAIL_SETUP.md) for step-by-step instructions

### "Invalid scope" or "Access denied"
- Ensure your OAuth consent screen includes the `gmail.modify` scope
- You may need to re-authorize after changing scopes
- Delete `token.json` and try authenticating again

### "Insufficient permissions"
- Verify your Google Cloud project has the Gmail API enabled
- Check that your OAuth credentials are for a "Desktop app" type
- Ensure you're signing in with the correct Google account

## Questions?

If you have security concerns or questions about permissions:
1. Review the source code - it's open source!
2. Check file: `resume_matcher_rag.py` function `gmail_authorize()` 
3. Check file: `Project1.py` function `sync_gmail()`
4. Open an issue on GitHub for clarification
