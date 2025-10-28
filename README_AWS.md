# AWS quick start: S3 + Qdrant + ECS/Fargate worker + Streamlit app

This guide sets up the recommended combo:

- S3 for durable resume storage and presigned links
- Qdrant on EC2 for vector search
- ECS/Fargate worker to ingest from MySQL, upload to S3, and upsert vectors to Qdrant
- Streamlit app points to Qdrant (and shows S3 links)

---

## 1) S3 bucket + IAM

1. Create a bucket (unique name), e.g., `my-resume-bucket` in `us-east-1`.
2. Enable bucket versioning (recommended) and default encryption (AES-256 or KMS).
3. Create an IAM role for ECS Task with this inline policy (scope to your bucket/prefix):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": ["arn:aws:s3:::my-resume-bucket"] },
    { "Effect": "Allow", "Action": ["s3:GetObject","s3:PutObject"], "Resource": ["arn:aws:s3:::my-resume-bucket/resumes/*"] }
  ]
}
```

App settings:
- Enable S3, set Bucket, Region, Prefix (e.g., `resumes/`).

Worker env:
- `S3_ENABLED=true`, `S3_BUCKET=my-resume-bucket`, `S3_REGION=us-east-1`, `S3_PREFIX=resumes/`

---

## 2) Qdrant on EC2

Launch an EC2 (Ubuntu 22.04, t3.medium or better) with security group allowing TCP 6333 from your office IP.

SSH in and run:

```bash
sudo mkdir -p /qdrant_storage
sudo docker run -d --name qdrant -p 6333:6333 -v /qdrant_storage:/qdrant/storage qdrant/qdrant:latest
```

Note the instance public DNS as `QDRANT_HOST`.

---

## 3) Build and push the worker image

From the project root:

```bash
# Authenticate to ECR and create a repo (one-time)
aws ecr create-repository --repository-name resume-ingest --region us-east-1
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build and push
docker build -f Dockerfile.worker -t resume-ingest:latest .
docker tag resume-ingest:latest $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/resume-ingest:latest
docker push $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/resume-ingest:latest
```

---

## 4) ECS/Fargate task (ingest)

Create an ECS cluster (Fargate). Create a Task Definition with:
- Image: `$ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/resume-ingest:latest`
- CPU/Memory: 0.5 vCPU / 1GB+ (or higher for batching)
- Task Role: the S3 policy role
- Env vars (examples):
```
MYSQL_HOST=your.mysql.host
MYSQL_PORT=3306
MYSQL_USER=readonly
MYSQL_PASSWORD=******
MYSQL_DATABASE=resumes_db
MYSQL_SQL=SELECT filename, directory_name, stored_filename FROM resumes WHERE updated_at >= NOW() - INTERVAL 1 DAY;

FILENAME_COL=filename
DIR_COL=directory_name
STORED_NAME_COL=stored_filename
DERIVE_URL=false
LOCAL_COPY=true
BASE_DIR=\\\\fileserver\\share\\uploads

# Or use DERIVE_URL=true with BASE_URL=https://files.example.com/

S3_ENABLED=true
S3_BUCKET=my-resume-bucket
S3_REGION=us-east-1
S3_PREFIX=resumes/

QDRANT_HOST=<ec2-public-dns>
QDRANT_PORT=6333
QDRANT_COLLECTION=resume_chunks
```
- Run task on a schedule using EventBridge (hourly/nightly) or trigger manually.

The worker:
- Pulls rows from MySQL, copies/downloads resumes
- Uploads to S3 (optional)
- Extracts text, chunks, embeds
- Upserts vectors to Qdrant

---

## 5) Run the Streamlit app on EC2

On another EC2 (or the same), deploy the app:

```bash
sudo apt update && sudo apt -y install python3-venv python3-pip git
python3 -m venv appenv && source appenv/bin/activate
pip install --upgrade pip
pip install streamlit numpy sentence-transformers pdfplumber python-docx requests boto3 qdrant-client mysql-connector-python

# copy your app code to ~/app, then
cd ~/app
streamlit run resume_matcher_rag.py --server.headless true --server.address 0.0.0.0 --server.port 8501
```

In the app Settings → Search index → Advanced ANN settings:
- Vector store: `Qdrant (beta)`
- Host: `<ec2-public-dns>`
- Port: `6333`
- Collection: `resume_chunks`

Cloud storage (optional): enable S3 and set bucket/region/prefix. The UI will show presigned links.

---

## 6) Security tips
- Restrict Qdrant SG to trusted IPs or place behind VPN.
- Use IAM roles for ECS and the app EC2 instead of static keys.
- Enable S3 server-side encryption and (optionally) bucket policy to enforce TLS and KMS.
- If exposing the app publicly, run Nginx with HTTPS and optional basic auth.

---

## 7) Troubleshooting
- Worker logs show: fetched rows, new files, S3 uploads, chunks upserted.
- If no files are saved: verify BASE_DIR or BASE_URL and your SQL returns directory_name/stored_filename.
- If Qdrant upserts are 0: ensure port 6333 reachable, collection auto-created.
- App shows no results: click Build/Update index only if using local FAISS; for Qdrant, ensure worker has run and app points to Qdrant.
