# Permissions Documentation

This document outlines all permissions required by the JD.Resume_Matcher application.

## Overview

The JD.Resume_Matcher application requires specific permissions to function properly, particularly when using the Gmail integration feature for automatic resume ingestion.

## Required Permissions

### 1. Gmail API Permissions

The application uses different Gmail API scopes depending on which file you're running:

#### Main Application (resume_matcher_rag.py)

**Scope:** `https://www.googleapis.com/auth/gmail.readonly`

**Purpose:** This scope allows the application to:
- Read emails from your Gmail inbox (read-only)
- Access email attachments (resume files)
- Search for specific emails with attachments
- Download attachments without modifying email status

**Why Gmail.Readonly?**
- The main application only needs to download resumes from email attachments
- It does NOT mark emails as read or modify any email properties
- This is the minimum permission needed for read-only access
- Provides better security with least-privilege principle

#### Alternative Application (Project1.py)

**Scope:** `https://www.googleapis.com/auth/gmail.modify`

**Purpose:** This scope allows the application to:
- Read emails from your Gmail inbox
- Access email attachments (resume files)
- Mark emails as read after processing
- Search for specific emails with attachments

**Why Gmail.Modify?**
- This version marks emails as read after downloading to avoid reprocessing
- This is the minimum scope required for both reading emails AND modifying their labels/status
- The app still does NOT send emails, delete emails, or access sensitive settings

**Which Scope to Use?**
- Use `gmail.readonly` (resume_matcher_rag.py) for maximum security if you don't need emails marked as read
- Use `gmail.modify` (Project1.py) if you want processed emails automatically marked as read

### 2. File System Permissions

**Purpose:** The application needs local file system access to:
- Store downloaded resumes in `downloaded_resumes/` directory
- Cache embeddings and text extractions for performance
- Store OAuth tokens securely (`token.json` or `token.pickle`)
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

- **token.json / token.pickle**: Contains OAuth access/refresh tokens
  - Automatically generated after first OAuth flow
  - Different files may use different names (token.json or token.pickle)
  - Should NEVER be committed to version control
  - Both filenames are included in `.gitignore`
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
4. Delete `token.json` or `token.pickle` from the application directory

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
