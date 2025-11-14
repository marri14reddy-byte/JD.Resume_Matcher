# Gmail API Setup Guide

This guide walks you through setting up Gmail API credentials for the JD.Resume_Matcher application to enable automatic resume ingestion from your Gmail account.

## Prerequisites

- A Google account
- Access to [Google Cloud Console](https://console.cloud.google.com/)
- Basic familiarity with following step-by-step instructions

## Why Do I Need This?

The Gmail integration allows the application to:
- Automatically fetch resumes from email attachments
- Process resumes sent to your inbox
- Mark processed emails as read to avoid duplicates

**Note:** If you only want to upload resumes manually (ZIP files or PDFs), you can skip this setup entirely.

## Step-by-Step Setup

### Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click on the project dropdown at the top (next to "Google Cloud")
3. Click "New Project"
4. Enter a project name: e.g., "Resume Matcher"
5. Click "Create"
6. Wait for the project to be created (you'll see a notification)

### Step 2: Enable Gmail API

1. Make sure your new project is selected (check the dropdown at the top)
2. Go to "APIs & Services" > "Library" (use the left sidebar or search)
3. Search for "Gmail API"
4. Click on "Gmail API"
5. Click the "Enable" button
6. Wait for it to be enabled (takes a few seconds)

### Step 3: Configure OAuth Consent Screen

1. Go to "APIs & Services" > "OAuth consent screen"
2. Select "External" user type (unless you're in a Google Workspace org)
3. Click "Create"

**App Information:**
- App name: `JD Resume Matcher` (or any name you prefer)
- User support email: Select your email from the dropdown
- Developer contact email: Enter your email address

4. Click "Save and Continue"

**Scopes:**
5. Click "Add or Remove Scopes"
6. Filter or search for "Gmail API"
7. Select the scope: `https://www.googleapis.com/auth/gmail.modify`
   - This allows reading emails and marking them as read
8. Click "Update"
9. Click "Save and Continue"

**Test Users (Important!):**
10. Click "Add Users"
11. Enter your Gmail address (the one you'll use to fetch resumes)
12. Click "Add"
13. Click "Save and Continue"

**Summary:**
14. Review your settings
15. Click "Back to Dashboard"

### Step 4: Create OAuth Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth client ID"
3. Application type: Select "Desktop app"
4. Name: `Resume Matcher Desktop` (or any name)
5. Click "Create"

**Download Credentials:**
6. A dialog will appear showing your Client ID and Secret
7. Click "Download JSON"
8. The file will be downloaded (usually named like `client_secret_XXXX.json`)

### Step 5: Install Credentials

1. Rename the downloaded file to `credentials.json`
2. Move/copy `credentials.json` to your JD.Resume_Matcher application directory
   - This is the same folder where `resume_matcher_rag.py` is located
   - On Windows: Likely `C:\Users\YourName\path\to\JD.Resume_Matcher\`
   - On Mac/Linux: Likely `/home/username/path/to/JD.Resume_Matcher/`

**Security Check:**
- ✅ File is named exactly `credentials.json`
- ✅ File is in the application root directory (same folder as the .py files)
- ✅ File should NOT be committed to git (already in .gitignore)

### Step 6: First-Time Authorization

1. Start the application:
   ```bash
   streamlit run resume_matcher_rag.py
   ```

2. Navigate to the "Resumes" tab in the web interface

3. Find the "Gmail Sync" section

4. Click "Authorize Gmail" or "Connect Gmail" button

5. **OAuth Flow:**
   - A browser window will open
   - Sign in to your Google account (the one you added as a test user)
   - You'll see a warning: "Google hasn't verified this app"
     - Click "Advanced"
     - Click "Go to [Your App Name] (unsafe)"
     - This is safe because it's YOUR app, not a public app
   
6. **Grant Permissions:**
   - Review the permissions requested
   - Should show: "Read, compose, send, and permanently delete all your email from Gmail"
     - This is the description of `gmail.modify` scope
     - The app only READS and MARKS as read (doesn't delete!)
   - Click "Allow"

7. **Success:**
   - The browser should show "The authentication flow has completed"
   - You can close the browser window
   - Return to the Streamlit app
   - You should now see "Connected" status

8. **Token Storage:**
   - A file named `token.json` is created in your app directory
   - This contains your access token (keep it secret!)
   - The app will use this token for future Gmail access
   - If you delete this file, you'll need to re-authorize

## Using Gmail Sync

Once authorized:

1. In the Streamlit app, go to the "Resumes" tab
2. Find the "Gmail Sync" section
3. Click "Fetch from Gmail" button
4. The app will:
   - Search for unread emails with attachments
   - Download resume attachments (PDF, DOCX, TXT)
   - Mark processed emails as read
   - Store resumes in the `downloaded_resumes/` folder

## Troubleshooting

### "credentials.json not found"
- Ensure the file is named exactly `credentials.json` (not `credentials (1).json`)
- Ensure it's in the correct directory (application root)
- Check file permissions (should be readable)

### "Access blocked: This app's request is invalid"
- Make sure you added your email as a Test User in Step 3
- Verify the Gmail API is enabled in Step 2
- Check that the OAuth consent screen is configured with the correct scope

### "Invalid grant" or "Token expired"
- Delete `token.json` from the application directory
- Re-run the authorization flow (Step 6)

### "The application has been blocked from creating sessions"
- Your OAuth consent screen might be in "Testing" mode with expired test users
- Go back to OAuth consent screen and re-add your email as a test user
- Or publish your app (not recommended for personal use)

### "No module named 'google'" or import errors
- Install required dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### "Permission denied" when fetching emails
- Verify your OAuth scope includes `gmail.modify`
- Delete `token.json` and re-authorize to refresh permissions
- Check that your Google account has the correct permissions

## Security Best Practices

1. **Never share `credentials.json`** - It contains your OAuth client secret
2. **Never share `token.json`** - It contains your personal access token
3. **Never commit these files to git** - Already protected by `.gitignore`
4. **Use a dedicated email** - Consider using a separate Gmail account for resume collection
5. **Regular audits** - Periodically review authorized apps at https://myaccount.google.com/permissions
6. **Revoke when done** - If you stop using the app, revoke its access

## Alternative: Manual Upload

If Gmail setup seems complicated, you can always:
- Upload resumes manually via the "Upload Resumes" section
- Drag and drop PDF/DOCX files
- Upload ZIP files with multiple resumes
- No Gmail setup required!

## Publishing Your App (Optional)

If you want to share this app with others without test user restrictions:

1. Complete OAuth consent screen verification (Google review process)
2. Provide privacy policy URL
3. Provide terms of service URL
4. Submit for verification
5. Wait for Google approval (can take several weeks)

**Note:** For personal use, staying in "Testing" mode is fine and recommended.

## Need Help?

- Check [PERMISSIONS.md](PERMISSIONS.md) for permission details
- Review the application logs in Streamlit
- Open an issue on GitHub with your (non-sensitive) error messages
- Remember to NEVER share your `credentials.json` or `token.json` files

## Video Tutorial (Coming Soon)

We're working on a video walkthrough of this setup process. Check the README for updates!
