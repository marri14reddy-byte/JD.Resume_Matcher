import os
import sys
import time
import json
import re
import hashlib
import tempfile
from typing import List, Tuple

# Optional deps imported lazily where possible

RESUME_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloaded_resumes')
os.makedirs(RESUME_DIR, exist_ok=True)

DEFAULT_CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', '1200'))
DEFAULT_CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', '300'))
DEFAULT_EMBED_BATCH = int(os.environ.get('EMBED_BATCH', '64'))


def extract_text_fallback(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.pdf':
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                return '\n'.join(page.extract_text() or '' for page in pdf.pages)
        elif ext == '.docx':
            import docx
            doc = docx.Document(file_path)
            return '\n'.join([para.text for para in doc.paragraphs])
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        else:
            return ''
    except Exception:
        return ''


def _chunk_text(txt: str, size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    txt = txt or ''
    if size <= 0:
        size = 800
    if overlap < 0:
        overlap = 0
    chunks = []
    i = 0
    n = len(txt)
    step = max(1, size - overlap)
    while i < n:
        chunks.append(txt[i:i+size])
        i += step
    return chunks


def _embed_texts(texts: List[str], batch: int = DEFAULT_EMBED_BATCH):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # try GPU if available
    try:
        import torch
        if torch.cuda.is_available():
            model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    except Exception:
        pass
    try:
        vecs = model.encode(texts, convert_to_numpy=True, batch_size=batch, normalize_embeddings=True, show_progress_bar=False)
    except TypeError:
        vecs = np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False))
    return vecs.astype('float32')


def _s3_upload_if_enabled(filename: str) -> str | None:
    if not os.environ.get('S3_ENABLED', '').lower() in ('1', 'true', 'yes'):
        return None
    bucket = os.environ.get('S3_BUCKET')
    region = os.environ.get('S3_REGION', 'us-east-1')
    prefix = os.environ.get('S3_PREFIX', 'resumes/')
    if not bucket:
        return None
    try:
        import boto3
        import hashlib
        path = os.path.join(RESUME_DIR, filename)
        h = hashlib.sha1()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        sha1 = h.hexdigest()
        key = f"{prefix}{sha1}_{filename}"
        cli = boto3.client('s3', region_name=region)
        cli.upload_file(path, bucket, key)
        return key
    except Exception as e:
        print(f"[S3] upload failed for {filename}: {e}")
        return None


def _qdrant_upsert_chunks(files: List[str]) -> int:
    host = os.environ.get('QDRANT_HOST', 'localhost')
    port = int(os.environ.get('QDRANT_PORT', '6333'))
    collection = os.environ.get('QDRANT_COLLECTION', 'resume_chunks')
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams, PointStruct
    except Exception as e:
        print(f"[Qdrant] missing qdrant-client: {e}")
        return 0
    cli = QdrantClient(host=host, port=port)
    # Ensure collection
    try:
        cli.get_collection(collection)
    except Exception:
        try:
            cli.recreate_collection(collection_name=collection, vectors_config=VectorParams(size=384, distance=Distance.COSINE))
        except Exception as e:
            print(f"[Qdrant] create collection failed: {e}")
            return 0
    added = 0
    import numpy as np
    for fname in files:
        fpath = os.path.join(RESUME_DIR, fname)
        text = extract_text_fallback(fpath)
        chunks = _chunk_text(text)
        if not chunks:
            continue
        vecs = _embed_texts(chunks)
        points = []
        base_id = int(abs(hash(fname)) % (10**12))
        for i, v in enumerate(vecs):
            pid = base_id + i
            payload = {"resume_file": fname, "chunk_id": i}
            points.append(PointStruct(id=pid, vector=v.tolist(), payload=payload))
        try:
            cli.upsert(collection_name=collection, points=points)
            added += len(points)
        except Exception as e:
            print(f"[Qdrant] upsert failed for {fname}: {e}")
    return added


def _download_from_row(row: dict, cfg: dict) -> str | None:
    """Return saved basename or None."""
    import requests
    # filename to use locally
    fname = str(row.get(cfg['filename_col']) or '').strip()
    if not fname:
        return None
    base = os.path.basename(fname)
    target = os.path.join(RESUME_DIR, base)
    # avoid overwrite
    if os.path.exists(target):
        r, e = os.path.splitext(base)
        k = 1
        while os.path.exists(os.path.join(RESUME_DIR, f"{r}_{k}{e}")):
            k += 1
        base = f"{r}_{k}{e}"
        target = os.path.join(RESUME_DIR, base)

    wrote = False
    # 1) BLOB
    bcol = cfg.get('blob_col')
    if bcol and (bcol in row) and (row[bcol] is not None) and not wrote:
        try:
            data = row[bcol]
            with open(target, 'wb') as f:
                f.write(data if isinstance(data, (bytes, bytearray)) else bytes(data))
            wrote = True
        except Exception:
            wrote = False
    # 2) Direct URL
    ucol = cfg.get('url_col')
    if ucol and (ucol in row) and row[ucol] and not wrote:
        try:
            r = requests.get(str(row[ucol]), timeout=30)
            r.raise_for_status()
            with open(target, 'wb') as f:
                f.write(r.content)
            wrote = True
        except Exception:
            wrote = False
    # 3) Derived URL
    if (not wrote) and cfg.get('derive_url') and cfg.get('base_url') and cfg.get('dir_col') and cfg.get('stored_name_col'):
        try:
            dval = str(row.get(cfg['dir_col']) or '').strip('/')
            sval = str(row.get(cfg['stored_name_col']) or '').strip()
            if dval and sval:
                from urllib.parse import urljoin, quote
                base_clean = cfg['base_url'] if cfg['base_url'].endswith('/') else (cfg['base_url'] + '/')
                dir_url = urljoin(base_clean, dval + '/')
                full_url = urljoin(dir_url, quote(sval))
                r = requests.get(full_url, timeout=30)
                r.raise_for_status()
                with open(target, 'wb') as f:
                    f.write(r.content)
                wrote = True
        except Exception:
            wrote = False
    # 4) Local/UNC copy
    if (not wrote) and cfg.get('local_copy') and cfg.get('base_dir') and cfg.get('dir_col') and cfg.get('stored_name_col'):
        try:
            import shutil
            dval = str(row.get(cfg['dir_col']) or '').strip('/\\')
            sval = str(row.get(cfg['stored_name_col']) or '').strip()
            if dval and sval:
                src_path = os.path.normpath(os.path.join(cfg['base_dir'], dval, sval))
                base_norm = os.path.normpath(cfg['base_dir'])
                if src_path.startswith(base_norm) and os.path.exists(src_path):
                    shutil.copyfile(src_path, target)
                    wrote = True
        except Exception:
            wrote = False

    return base if wrote else None


def run_once():
    # Read config from env
    mysql_host = os.environ.get('MYSQL_HOST', '')
    mysql_port = int(os.environ.get('MYSQL_PORT', '3306'))
    mysql_user = os.environ.get('MYSQL_USER', '')
    mysql_pass = os.environ.get('MYSQL_PASSWORD', '')
    mysql_db = os.environ.get('MYSQL_DATABASE', '')
    mysql_sql = os.environ.get('MYSQL_SQL', '')
    connect_timeout = int(os.environ.get('MYSQL_CONNECT_TIMEOUT', '10'))

    cfg = {
        'filename_col': os.environ.get('FILENAME_COL', 'filename'),
        'blob_col': os.environ.get('BLOB_COL') or None,
        'url_col': os.environ.get('URL_COL') or None,
        'derive_url': os.environ.get('DERIVE_URL', 'false').lower() in ('1','true','yes'),
        'base_url': os.environ.get('BASE_URL') or None,
        'dir_col': os.environ.get('DIR_COL') or None,
        'stored_name_col': os.environ.get('STORED_NAME_COL') or None,
        'local_copy': os.environ.get('LOCAL_COPY', 'false').lower() in ('1','true','yes'),
        'base_dir': os.environ.get('BASE_DIR') or None,
    }

    if not (mysql_host and mysql_user and mysql_db and mysql_sql):
        print('[Worker] Missing MySQL config. Set MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE, MYSQL_SQL.')
        return 1

    try:
        import mysql.connector
    except Exception:
        print('[Worker] Install mysql-connector-python')
        return 1

    conn_args = {
        'host': mysql_host,
        'port': mysql_port,
        'user': mysql_user,
        'password': mysql_pass,
        'database': mysql_db,
    }
    if connect_timeout > 0:
        conn_args['connection_timeout'] = connect_timeout

    downloaded: List[str] = []
    try:
        conn = mysql.connector.connect(**conn_args)
        cur = conn.cursor(dictionary=True)
        cur.execute(mysql_sql)
        rows = cur.fetchall()
        total = len(rows)
        print(f"[Worker] Fetched {total} row(s) from MySQL.")
        for i, row in enumerate(rows, 1):
            nm = _download_from_row(row, cfg)
            if nm:
                downloaded.append(nm)
            if i % 25 == 0 or i == total:
                print(f"[Worker] Processed {i}/{total}. New files: {len(downloaded)}")
        try:
            cur.close(); conn.close()
        except Exception:
            pass
    except Exception as e:
        print(f"[Worker] MySQL error: {e}")
        return 1

    # Upload to S3 if enabled
    s3_keys = []
    for nm in downloaded:
        k = _s3_upload_if_enabled(nm)
        if k:
            s3_keys.append((nm, k))
    if s3_keys:
        print(f"[Worker] Uploaded {len(s3_keys)} file(s) to S3.")

    # Upsert to Qdrant
    if downloaded:
        added = _qdrant_upsert_chunks(downloaded)
        print(f"[Worker] Upserted {added} chunk vectors to Qdrant.")
    else:
        print('[Worker] No new files to index.')
    return 0


if __name__ == '__main__':
    sys.exit(run_once())
