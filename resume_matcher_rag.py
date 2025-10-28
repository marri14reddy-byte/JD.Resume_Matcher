# --- RAG tab logic (diagnostics and embedding loading) ---
import streamlit as st
import os
import pickle
import numpy as np
import concurrent.futures
import traceback
import re
import json
from urllib.parse import urljoin, quote
import tempfile
import zipfile
import base64
import hashlib
import sqlite3
import time
import math
from typing import Tuple

# Set paths
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_RESUME_DIR = os.path.join(_ROOT_DIR, 'downloaded_resumes')
embedding_npy_path = os.path.join(_RESUME_DIR, 'resume_embeddings.npy')
embedding_idx_path = os.path.join(_RESUME_DIR, 'resume_embeddings_index.pkl')
text_cache_dir = os.path.join(_RESUME_DIR, '.text_cache')
chunk_cache_dir = os.path.join(_RESUME_DIR, '.chunk_cache')
# ANN chunk index (HNSW) paths
_CHUNK_INDEX_PATH = os.path.join(_RESUME_DIR, 'chunks_hnsw.faiss')
_CHUNK_MAP_PATH = os.path.join(_RESUME_DIR, 'chunks_map.pkl')  # list of (resume_file, chunk_id)
_SHARD_MANIFEST_PATH = os.path.join(_RESUME_DIR, 'chunks_manifest.json')  # tracks shard files

# ANN defaults and thresholds
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 300
DEFAULT_TOP_CHUNKS = 500
SHARD_ROTATE_THRESHOLD = 400_000  # rotate shard after this many chunks
MAX_SHARDS_CACHED_DEFAULT = 2

# Remote Vector Store (optional)
REMOTE_VS_MODES = ["Local FAISS", "Qdrant (beta)"]

# Performance/config toggles
USE_FLOAT16_EMBEDS = True  # store embeddings as float16 to reduce disk size; cast to float32 when using
DEFAULT_EMBED_BATCH = 64   # batch size for SentenceTransformer.encode

# Ensure required folders exist
os.makedirs(_RESUME_DIR, exist_ok=True)
os.makedirs(text_cache_dir, exist_ok=True)
os.makedirs(chunk_cache_dir, exist_ok=True)

# Default to Chroma as the vector store for a simpler out-of-the-box setup
if 'vector_store_mode' not in st.session_state:
    st.session_state['vector_store_mode'] = 'Chroma (beta)'
if 'chroma_path' not in st.session_state:
    # If running on Hugging Face Spaces, default to the persistent storage mount at /data
    try:
        _is_hf = bool(os.environ.get('SPACE_ID') or os.environ.get('HF_SPACE') or os.environ.get('SPACE_RUNTIME'))
    except Exception:
        _is_hf = False
    _default_chroma = '/data/chroma_store' if _is_hf else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chroma_store')
    st.session_state['chroma_path'] = _default_chroma
if 'chroma_collection' not in st.session_state:
    st.session_state['chroma_collection'] = 'resume_chunks'

# Small helpers for status metrics
def _list_resume_files(folder: str = _RESUME_DIR) -> list:
    try:
        return [f for f in os.listdir(folder) if f.lower().endswith(('.pdf', '.txt', '.docx'))]
    except Exception:
        return []

def _load_embedding_index(idx_path: str = embedding_idx_path) -> list:
    if os.path.exists(idx_path):
        try:
            with open(idx_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return []
    return []

def get_resume_cache_status():
    """Return (total_files:int, embedded_count:int, remaining_to_embed:int)."""
    files = _list_resume_files()
    embedded = _load_embedding_index()
    total = len(files)
    emb = len([fn for fn in embedded if fn in set(files)])
    return total, emb, max(0, total - emb)

def get_remaining_files(folder: str = _RESUME_DIR) -> list:
    """Return list of filenames that exist on disk but are not in the embedding index."""
    files = set(_list_resume_files(folder))
    embedded = set(_load_embedding_index())
    remaining = [fn for fn in files if fn not in embedded]
    remaining.sort()
    return remaining

# Diagnostics
print(f"[RAG] Embedding .npy path: {embedding_npy_path}")
print(f"[RAG] Embedding index path: {embedding_idx_path}")
print(f"[RAG] .npy exists: {os.path.exists(embedding_npy_path)}")
print(f"[RAG] index exists: {os.path.exists(embedding_idx_path)}")
# ---- Metadata DB (SQLite) helpers ----
_DB_PATH = os.path.join(_ROOT_DIR, 'resumes.db')

def _get_db():
    try:
        conn = sqlite3.connect(_DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        return None

def _init_db():
    conn = _get_db()
    if not conn:
        return
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS resumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            sha1 TEXT,
            mtime REAL,
            size INTEGER,
            text_hash TEXT,
            s3_key TEXT,
            created_at REAL,
            updated_at REAL
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS resume_attrs (
            resume_id INTEGER,
            years INTEGER,
            skills_json TEXT,
            locations_json TEXT,
            industries_json TEXT,
            FOREIGN KEY(resume_id) REFERENCES resumes(id)
        );
        """
    )
    conn.commit()
    conn.close()

def _normalize_text_for_hash(text: str) -> str:
    s = (text or '').lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _sha1_file(path: str) -> str | None:
    try:
        h = hashlib.sha1()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def _db_upsert_resume(filename: str, text: str, s3_key: str | None = None):
    conn = _get_db()
    if not conn:
        return
    path = os.path.join(_RESUME_DIR, filename)
    try:
        st_mtime = os.path.getmtime(path)
        st_size = os.path.getsize(path)
    except Exception:
        st_mtime = 0.0
        st_size = 0
    sha1 = _sha1_file(path)
    text_hash = hashlib.sha1((_normalize_text_for_hash(text)).encode('utf-8', errors='ignore')).hexdigest() if text else None
    now = time.time()
    cur = conn.cursor()
    cur.execute("SELECT id FROM resumes WHERE filename = ?", (filename,))
    row = cur.fetchone()
    if row:
        cur.execute(
            "UPDATE resumes SET sha1=?, mtime=?, size=?, text_hash=?, s3_key=COALESCE(?, s3_key), updated_at=? WHERE id=?",
            (sha1, st_mtime, st_size, text_hash, s3_key, now, row['id'])
        )
        resume_id = row['id']
    else:
        cur.execute(
            "INSERT INTO resumes(filename, sha1, mtime, size, text_hash, s3_key, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (filename, sha1, st_mtime, st_size, text_hash, s3_key, now, now)
        )
        resume_id = cur.lastrowid
    # Attributes
    years = _extract_years_exp(text)
    skills = sorted(set([sk for sk in _COMMON_SKILLS if sk in (text or '').lower()]))
    locs = extract_locations(text)
    inds = sorted(list(_extract_industries(text)))
    cur.execute("DELETE FROM resume_attrs WHERE resume_id=?", (resume_id,))
    cur.execute(
        "INSERT INTO resume_attrs(resume_id, years, skills_json, locations_json, industries_json) VALUES (?,?,?,?,?)",
        (resume_id, years, json.dumps(skills), json.dumps(locs), json.dumps(inds))
    )
    conn.commit()
    conn.close()

def _db_get_s3_key(filename: str) -> str | None:
    conn = _get_db()
    if not conn:
        return None
    cur = conn.cursor()
    cur.execute("SELECT s3_key FROM resumes WHERE filename=?", (filename,))
    row = cur.fetchone()
    conn.close()
    return row['s3_key'] if row and row['s3_key'] else None

def _db_list_filenames() -> list[str]:
    """Return all known filenames from SQLite resumes table."""
    try:
        conn = _get_db()
        if not conn:
            return []
        cur = conn.cursor()
        cur.execute("SELECT filename FROM resumes")
        rows = cur.fetchall()
        conn.close()
        out: list[str] = []
        for r in rows:
            try:
                if isinstance(r, tuple):
                    out.append(r[0])
                elif isinstance(r, sqlite3.Row):
                    out.append(r['filename'])
            except Exception:
                pass
        return [x for x in out if x]
    except Exception:
        return []

_init_db()

# Remote vector store stubs (overridden later if implemented)
def remote_vs_enabled() -> bool:
    return False
def remote_vs_query(*args, **kwargs):
    return None
def remote_vs_upsert(*args, **kwargs):
    return 0
def _get_qdrant_client():
    return None

# Load embedding_index before using it
embedding_index = []
if os.path.exists(embedding_idx_path):
    with open(embedding_idx_path, 'rb') as f:
        embedding_index = pickle.load(f)

# Cached SentenceTransformer model with GPU auto-detect
@st.cache_resource(show_spinner=False)
def get_st_model():
    try:
        from sentence_transformers import SentenceTransformer
        # Try to choose the best available device
        device = 'cpu'
        try:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
        except Exception:
            device = 'cpu'
        try:
            return SentenceTransformer('all-MiniLM-L6-v2', device=device)
        except TypeError:
            # Older versions may not support device kwarg
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model
    except Exception:
        # Lazy import fallback inside callers if needed
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')

# --- Single extract_text_fallback definition at top ---
def extract_text_fallback(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    # Simple on-disk text cache by filename + mtime
    try:
        base = os.path.basename(file_path)
        # If original file is missing, try latest cached text by basename
        if not os.path.exists(file_path):
            try:
                candidates = [fn for fn in os.listdir(text_cache_dir) if fn.startswith(base + ".") and fn.endswith(".txt")]
                if candidates:
                    candidates.sort(key=lambda fn: os.path.getmtime(os.path.join(text_cache_dir, fn)), reverse=True)
                    with open(os.path.join(text_cache_dir, candidates[0]), 'r', encoding='utf-8', errors='ignore') as cf:
                        return cf.read()
            except Exception:
                pass
        mtime = int(os.path.getmtime(file_path)) if os.path.exists(file_path) else 0
        cache_name = f"{base}.{mtime}.txt"
        cache_path = os.path.join(text_cache_dir, cache_name)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8', errors='ignore') as cf:
                    return cf.read()
            except Exception:
                pass
    except Exception:
        cache_path = None

    def _extract():
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

    text = _extract()
    # Save to cache; also clean old cache variants for the same basename
    try:
        if cache_path and text:
            # Remove stale caches for same base prefix
            prefix = os.path.basename(file_path) + '.'
            for fn in os.listdir(text_cache_dir):
                if fn.startswith(prefix) and fn != os.path.basename(cache_path):
                    try:
                        os.remove(os.path.join(text_cache_dir, fn))
                    except Exception:
                        pass
            with open(cache_path, 'w', encoding='utf-8', errors='ignore') as cf:
                cf.write(text)
    except Exception:
        pass
    return text

def update_embedding_cache_for_new_files(folder: str = None, files_to_add: list | None = None):
    """Append embeddings for resumes not yet present in the cache.

    Keeps resume_embeddings.npy and resume_embeddings_index.pkl in sync and only processes new files.
    If files_to_add is provided, only those basenames are considered (when not already embedded).
    Returns the number of resumes added to the cache.
    """
    folder = folder or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloaded_resumes')
    os.makedirs(folder, exist_ok=True)
    embedding_npy_path = os.path.join(folder, 'resume_embeddings.npy')
    embedding_idx_path = os.path.join(folder, 'resume_embeddings_index.pkl')

    # Load or initialize index and embeddings
    if os.path.exists(embedding_idx_path):
        with open(embedding_idx_path, 'rb') as f:
            embedding_index = pickle.load(f)
    else:
        embedding_index = []
    emb_dim = 384
    if os.path.exists(embedding_npy_path) and len(embedding_index) > 0:
        embeddings = np.load(embedding_npy_path)
        # Cast to float32 for math; stored dtype may be float16
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
    else:
        embeddings = np.zeros((0, emb_dim), dtype=np.float32)

    # Collect new files (optionally restricted to files_to_add)
    if files_to_add is not None:
        cand = [f for f in files_to_add if f.lower().endswith(('.pdf', '.txt', '.docx'))]
        new_files = [f for f in cand if (f not in embedding_index and os.path.exists(os.path.join(folder, f)))]
    else:
        all_files = [f for f in os.listdir(folder) if f.lower().endswith(('.pdf', '.txt', '.docx'))]
        new_files = [f for f in all_files if f not in embedding_index]
    if not new_files:
        return 0

    model = get_st_model()

    added = 0
    if new_files:
        # Batch extract text
        # Parallel text extraction
        def _read_text(fname: str) -> str:
            return extract_text_fallback(os.path.join(folder, fname))
        max_workers = max(2, min(8, (os.cpu_count() or 4)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            texts = list(ex.map(_read_text, new_files))
        # Upsert metadata to SQLite
        for fname, txt in zip(new_files, texts):
            try:
                _db_upsert_resume(fname, txt, None)
            except Exception:
                pass
        # Batch encode (normalized)
        try:
            new_embeds = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=DEFAULT_EMBED_BATCH,
            )
        except TypeError:
            # Some versions may not support convert_to_numpy or batch_size args
            new_embeds = np.asarray(
                model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            )
        # Normalize and cast
        new_embeds = new_embeds.astype(np.float32)
        # Persist
        added = len(new_files)
        embedding_index.extend(new_files)
        if embeddings.shape[0] == 0:
            embeddings = new_embeds
        else:
            embeddings = np.vstack([embeddings, new_embeds])
        # Store as float16 if enabled to save disk; load path will upcast to float32 for math
        to_store = embeddings.astype(np.float16) if USE_FLOAT16_EMBEDS else embeddings.astype(np.float32)
        np.save(embedding_npy_path, to_store)
        with open(embedding_idx_path, 'wb') as f:
            pickle.dump(embedding_index, f)
    return added

# ---------------- ANN (HNSW) chunk index helpers ---------------- #
@st.cache_resource(show_spinner=False)
def get_hnsw_index():
    """Load HNSW index and mapping if available. Returns (index, mapping) or (None, None)."""
    try:
        import faiss  # type: ignore
    except Exception:
        return None, None
    try:
        if os.path.exists(_CHUNK_INDEX_PATH) and os.path.exists(_CHUNK_MAP_PATH):
            index = faiss.read_index(_CHUNK_INDEX_PATH)
            with open(_CHUNK_MAP_PATH, 'rb') as f:
                mapping = pickle.load(f)
            # Some defaults for search perf
            try:
                ef = int(st.session_state.get('ann_efSearch', 64))
                index.hnsw.efSearch = ef
            except Exception:
                pass
            return index, mapping
    except Exception:
        pass
    return None, None

def _save_hnsw(index, mapping):
    try:
        import faiss  # type: ignore
        faiss.write_index(index, _CHUNK_INDEX_PATH)
        with open(_CHUNK_MAP_PATH, 'wb') as f:
            pickle.dump(mapping, f)
    except Exception:
        pass

def _create_empty_hnsw(dim=384):
    try:
        import faiss  # type: ignore
        index = faiss.IndexHNSWFlat(dim, 32)
        try:
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 64
        except Exception:
            pass
        return index
    except Exception:
        return None

def _get_mapped_files_set() -> set[str]:
    s = set()
    try:
        if os.path.exists(_CHUNK_MAP_PATH):
            with open(_CHUNK_MAP_PATH, 'rb') as f:
                mp = pickle.load(f)
                s.update([p[0] for p in mp])
    except Exception:
        pass
    try:
        for _, mp in _list_shard_files():
            try:
                with open(mp, 'rb') as f:
                    mm = pickle.load(f)
                    s.update([p[0] for p in mm])
            except Exception:
                pass
    except Exception:
        pass
    return s

def index_new_shard_from_remaining() -> tuple[int, str | None]:
    """Create a new shard index from files not yet present in any mapping. Returns (chunks_added, shard_name)."""
    try:
        import faiss  # type: ignore
    except Exception:
        return 0, None
    mapped = _get_mapped_files_set()
    present = [f for f in _list_resume_files(_RESUME_DIR) if f not in mapped]
    if not present:
        return 0, None
    index = _create_empty_hnsw(384)
    if index is None:
        return 0, None
    mapping: list[tuple[str, int]] = []
    total_added = 0
    for fname in present:
        fpath = os.path.join(_RESUME_DIR, fname)
        text = extract_text_fallback(fpath)
        chunks = _chunk_text(text)
        if not chunks:
            continue
        vecs = _embed_texts(chunks)
        try:
            index.add(vecs)
            mapping.extend([(fname, i) for i in range(vecs.shape[0])])
            total_added += int(vecs.shape[0])
        except Exception:
            continue
    # Determine next shard id
    shard_files = _list_shard_files()
    next_id = 0
    if shard_files:
        try:
            nums = []
            for ip, _ in shard_files:
                base = os.path.basename(ip)
                n = int(base.replace('chunks_hnsw_', '').replace('.faiss', ''))
                nums.append(n)
            next_id = (max(nums) + 1) if nums else 0
        except Exception:
            next_id = len(shard_files)
    idx_path = os.path.join(_RESUME_DIR, f'chunks_hnsw_{next_id:03d}.faiss')
    map_path = os.path.join(_RESUME_DIR, f'chunks_map_{next_id:03d}.pkl')
    try:
        import faiss  # type: ignore
        faiss.write_index(index, idx_path)
        with open(map_path, 'wb') as f:
            pickle.dump(mapping, f)
    except Exception:
        return 0, None
    return total_added, os.path.basename(idx_path)

def _chunk_text(txt: str, size: int = 1200, overlap: int = 300) -> list[str]:
    # Allow dynamic settings via session_state
    try:
        size = int(st.session_state.get('ann_chunk_size', size))
        overlap = int(st.session_state.get('ann_chunk_overlap', overlap))
    except Exception:
        pass
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

def _embed_texts(texts: list[str], batch: int = DEFAULT_EMBED_BATCH) -> np.ndarray:
    model = get_st_model()
    # Heuristic batch tuning + retry on OOM/TypeError
    b = int(st.session_state.get('embed_batch_size', batch)) if 'embed_batch_size' in st.session_state else batch
    while True:
        try:
            vecs = model.encode(texts, convert_to_numpy=True, batch_size=b, normalize_embeddings=True, show_progress_bar=False)
            break
        except TypeError:
            vecs = np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False))
            break
        except Exception:
            if b <= 8:
                # give up
                vecs = np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False))
                break
            b = max(8, b // 2)
            st.session_state['embed_batch_size'] = b
    return vecs.astype(np.float32)

# --------- Sharded index helpers ---------
def _list_shard_files() -> list[tuple[str, str]]:
    """Return list of (index_path, map_path) for shards found with naming chunks_hnsw_XXX.faiss and chunks_map_XXX.pkl."""
    shards = []
    try:
        for fn in os.listdir(_RESUME_DIR):
            if fn.startswith('chunks_hnsw_') and fn.endswith('.faiss'):
                base = fn[len('chunks_hnsw_'):-len('.faiss')]
                idx = os.path.join(_RESUME_DIR, fn)
                mp = os.path.join(_RESUME_DIR, f'chunks_map_{base}.pkl')
                if os.path.exists(mp):
                    shards.append((idx, mp))
    except Exception:
        pass
    shards.sort()
    return shards

def _load_index_and_map(idx_path: str, map_path: str):
    try:
        import faiss  # type: ignore
        index = faiss.read_index(idx_path)
        with open(map_path, 'rb') as f:
            mapping = pickle.load(f)
        try:
            ef = int(st.session_state.get('ann_efSearch', 64))
            index.hnsw.efSearch = ef
        except Exception:
            pass
        return index, mapping
    except Exception:
        return None, None

def load_shard_cached(idx_path: str, map_path: str):
    """LRU-cached shard loader to limit memory footprint when many shards exist."""
    key = (idx_path, map_path)
    cache = st.session_state.get('shard_cache')
    order = st.session_state.get('shard_cache_order')
    cap = int(st.session_state.get('max_shards_cached', MAX_SHARDS_CACHED_DEFAULT))
    if cache is None or order is None:
        cache, order = {}, []
        st.session_state['shard_cache'] = cache
        st.session_state['shard_cache_order'] = order
    if key in cache:
        # refresh LRU order
        try:
            order.remove(key)
        except ValueError:
            pass
        order.append(key)
        return cache[key]
    # load
    idx, mp = _load_index_and_map(idx_path, map_path)
    if idx is None or not mp:
        return None, None
    cache[key] = (idx, mp)
    order.append(key)
    # enforce capacity
    while len(order) > max(1, cap):
        evict = order.pop(0)
        try:
            del cache[evict]
        except Exception:
            pass
    return idx, mp

def ingest_new_files_to_hnsw(files: list[str] | None = None) -> int:
    """Add chunk embeddings for new files into HNSW. Returns number of chunks added.

    If files is None, considers any resume files not yet present in mapping.
    """
    try:
        import faiss  # type: ignore
    except Exception:
        # FAISS not installed; nothing to do
        return 0

    # Load or create
    index, mapping = get_hnsw_index()
    if index is None:
        index = _create_empty_hnsw(384)
        mapping = []
    if index is None:
        return 0

    # Determine candidate files
    present_files = set([f for f in os.listdir(_RESUME_DIR) if f.lower().endswith(('.pdf','.docx','.txt'))])
    if files is not None:
        cand_files = [f for f in files if f in present_files]
    else:
        mapped_files = set([p[0] for p in (mapping or [])])
        cand_files = [f for f in present_files if f not in mapped_files]
    if not cand_files:
        return 0

    # Extract, chunk, embed in batches per file
    total_added = 0
    for fname in cand_files:
        fpath = os.path.join(_RESUME_DIR, fname)
        text = extract_text_fallback(fpath)
        chunks = _chunk_text(text)
        if not chunks:
            continue
        vecs = _embed_texts(chunks)
        try:
            index.add(vecs)
            mapping.extend([(fname, i) for i in range(vecs.shape[0])])
            total_added += int(vecs.shape[0])
        except Exception:
            continue

    # Save back to disk and clear cached loader so subsequent calls see updates
    _save_hnsw(index, mapping)
    try:
        get_hnsw_index.clear()
    except Exception:
        pass
    return total_added

# ---- Background ingestion worker (simple) ----
def _ann_worker(files: list[str]):
    st.session_state['ann_worker_total'] = len(files)
    st.session_state['ann_worker_done'] = 0
    for fn in files:
        try:
            ingest_new_files_to_hnsw(files=[fn])
        except Exception:
            pass
        st.session_state['ann_worker_done'] += 1
    st.session_state['ann_worker_running'] = False

def start_ann_background_worker(files: list[str] | None = None):
    try:
        remaining = files if files is not None else [
            f for f in _list_resume_files(_RESUME_DIR)
            if True
        ]
        import threading
        st.session_state['ann_worker_running'] = True
        if st.session_state.get('vector_store_mode') == 'Qdrant (beta)':
            # background upsert to remote
            def _remote_worker(fs):
                st.session_state['ann_worker_total'] = len(fs)
                st.session_state['ann_worker_done'] = 0
                for fn in fs:
                    try:
                        remote_vs_upsert([fn])
                    except Exception:
                        pass
                    st.session_state['ann_worker_done'] += 1
                st.session_state['ann_worker_running'] = False
            t = threading.Thread(target=_remote_worker, args=(remaining,), daemon=True)
        else:
            t = threading.Thread(target=_ann_worker, args=(remaining,), daemon=True)
        t.start()
    except Exception:
        st.session_state['ann_worker_running'] = False

    # ---- Remote Vector Store (Qdrant) helpers ----
    def _get_qdrant_client():
        try:
            from qdrant_client import QdrantClient
            host = st.session_state.get('qdrant_host', 'localhost')
            port = int(st.session_state.get('qdrant_port', 6333))
            api_key = st.session_state.get('qdrant_api_key') or None
            return QdrantClient(host=host, port=port, api_key=api_key)
        except Exception:
            return None

    def remote_vs_enabled() -> bool:
        return st.session_state.get('vector_store_mode') == 'Qdrant (beta)'

    def remote_vs_upsert(files: list[str]) -> int:
        if not remote_vs_enabled():
            return 0
        client = _get_qdrant_client()
        if client is None:
            st.warning("Qdrant client not available. Install qdrant-client: pip install qdrant-client")
            return 0
        try:
            from qdrant_client.http.models import Distance, VectorParams, PointStruct
        except Exception:
            st.warning("qdrant-client HTTP models unavailable.")
            return 0
        collection = st.session_state.get('qdrant_collection', 'resume_chunks')
        # Ensure collection exists
        try:
            client.get_collection(collection)
        except Exception:
            try:
                client.recreate_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
            except Exception as e:
                st.error(f"Failed to create Qdrant collection: {e}")
                return 0
        # Upsert points
        added = 0
        for fname in files:
            fpath = os.path.join(_RESUME_DIR, fname)
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
                client.upsert(collection_name=collection, points=points)
                added += len(points)
            except Exception:
                pass
        return added

    def remote_vs_query(jd_text: str, top_chunks: int = 500, max_resumes: int = 100, return_highlights: bool = False):
        if not remote_vs_enabled():
            return None
        client = _get_qdrant_client()
        if client is None:
            return None
        try:
            from qdrant_client.http.models import Filter
        except Exception:
            return None
        collection = st.session_state.get('qdrant_collection', 'resume_chunks')
        q = _embed_texts([jd_text], batch=1)[0].tolist()
        limit = int(st.session_state.get('ann_top_chunks', top_chunks))
        try:
            res = client.search(collection_name=collection, query_vector=q, limit=limit)
        except Exception:
            return None
        resume_best = {}
        highlights = {}
        for pt in res:
            pl = getattr(pt, 'payload', {}) or {}
            fname = pl.get('resume_file')
            cid = pl.get('chunk_id', 0)
            score = float(getattr(pt, 'score', 0.0) or 0.0)
            if not fname:
                continue
            prev = resume_best.get(fname)
            if prev is None or score > prev:
                resume_best[fname] = score
            if return_highlights:
                fpath = os.path.join(_RESUME_DIR, fname)
                full_txt = extract_text_fallback(fpath)
                chunks = _chunk_text(full_txt)
                if 0 <= int(cid) < len(chunks):
                    snippet = chunks[int(cid)][:300].replace('\n', ' ')
                    lst = highlights.setdefault(fname, [])
                    if len(lst) < 3:
                        lst.append((snippet, score))
        if not resume_best:
            return None
        ordered = sorted(resume_best.items(), key=lambda kv: kv[1], reverse=True)
        top_files = [fn for fn, _ in ordered[:max_resumes]]
        if return_highlights:
            return (top_files, {fn: highlights.get(fn, [])[:3] for fn in top_files})
        return top_files

def ann_shortlist_resumes(jd_text: str, top_chunks: int = 400, max_resumes: int = 100, return_highlights: bool = False):
    """Use ANN HNSW chunk index to shortlist resume filenames for a given JD.

    Returns either:
      - list[str] of resume filenames ordered by relevance, or
      - (list[str], dict[str, list[tuple[str,float]]]) when return_highlights=True where highlights map to list of (chunk_text, sim)
    Returns None if ANN is unavailable.
    """
    try:
        import faiss  # type: ignore
    except Exception:
        return None

    # Prepare query vector
    try:
        q = _embed_texts([jd_text], batch=1)
    except Exception:
        return None

    # Gather candidate (index, mapping) pairs from shards if present, else monolithic
    shard_files = _list_shard_files()
    pairs = []
    if shard_files:
        for ip, mp in shard_files:
            idx, mping = load_shard_cached(ip, mp)
            if idx is not None and mping:
                pairs.append((idx, mping))
    else:
        index, mapping = get_hnsw_index()
        if index is not None and mapping:
            pairs.append((index, mapping))
    if not pairs:
        return None

    # Query each index and aggregate
    resume_best: dict[str, float] = {}
    highlights: dict[str, list[tuple[str, float]]] = {}
    topN = int(st.session_state.get('ann_top_chunks', top_chunks))
    # Optionally run shard searches in parallel to reduce latency
    parallel = bool(st.session_state.get('ann_parallel_shard_queries', True))
    max_workers = int(st.session_state.get('ann_shard_query_workers', min(4, max(1, len(pairs))))) if parallel else 1
    def _search_single(idx_obj, mapping_obj):
        try:
            return idx_obj.search(q, min(topN, len(mapping_obj)))
        except Exception:
            return None, None

    if parallel and len(pairs) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for index, mapping in pairs:
                futures.append((ex.submit(_search_single, index, mapping), mapping))
            for fut, mapping in futures:
                try:
                    D, I = fut.result()
                except Exception:
                    D, I = None, None
                if D is None or I is None or len(I) == 0:
                    continue
                for dist, idx in zip(D[0], I[0]):
                    try:
                        fname, chunk_id = mapping[int(idx)]
                    except Exception:
                        continue
                    try:
                        sim = 1.0 - float(dist) / 2.0
                    except Exception:
                        sim = 0.0
                    prev = resume_best.get(fname)
                    if prev is None or sim > prev:
                        resume_best[fname] = sim
                    if return_highlights:
                        fpath = os.path.join(_RESUME_DIR, fname)
                        full_txt = extract_text_fallback(fpath)
                        chunks = _chunk_text(full_txt)
                        if 0 <= int(chunk_id) < len(chunks):
                            snippet = chunks[int(chunk_id)][:300].replace('\n', ' ')
                            lst = highlights.setdefault(fname, [])
                            if len(lst) < 3:
                                lst.append((snippet, sim))
    else:
        for index, mapping in pairs:
            try:
                D, I = index.search(q, min(topN, len(mapping)))
            except Exception:
                continue
            if D is None or I is None or len(I) == 0:
                continue
            for dist, idx in zip(D[0], I[0]):
                try:
                    fname, chunk_id = mapping[int(idx)]
                except Exception:
                    continue
                try:
                    sim = 1.0 - float(dist) / 2.0
                except Exception:
                    sim = 0.0
                prev = resume_best.get(fname)
                if prev is None or sim > prev:
                    resume_best[fname] = sim
                if return_highlights:
                    fpath = os.path.join(_RESUME_DIR, fname)
                    full_txt = extract_text_fallback(fpath)
                    chunks = _chunk_text(full_txt)
                    if 0 <= int(chunk_id) < len(chunks):
                        snippet = chunks[int(chunk_id)][:300].replace('\n', ' ')
                        lst = highlights.setdefault(fname, [])
                        if len(lst) < 3:
                            lst.append((snippet, sim))
    if not resume_best:
        return None
    ordered = sorted(resume_best.items(), key=lambda kv: kv[1], reverse=True)
    top_files = [fn for fn, _ in ordered[:max_resumes]]
    if return_highlights:
        # Restrict highlights to top_files and keep at most 3
        h2 = {fn: highlights.get(fn, [])[:3] for fn in top_files}
        return top_files if not h2 else (top_files, h2)
    return top_files if top_files else None

def get_resume_texts_and_files():
    """Load resume texts and filenames from the downloaded_resumes folder.

    Returns:
        tuple[list[str], list[str], dict]: (resume_texts, resume_files, parsed_resume_data)
    """
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloaded_resumes')
    os.makedirs(folder, exist_ok=True)

    # Collect resume files present locally
    resume_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(('.pdf', '.txt', '.docx'))
    ]

    # If embeddings-only mode with Chroma, augment filenames from DB even if files are gone
    if _chroma_enabled() and bool(st.session_state.get('store_embeddings_only', False)):
        try:
            known = _db_list_filenames()
            for fn in known:
                if fn and fn.lower().endswith(('.pdf', '.txt', '.docx')) and fn not in resume_files:
                    resume_files.append(fn)
        except Exception:
            pass

    resume_texts = []
    parsed_resume_data = {}
    for fname in resume_files:
        fpath = os.path.join(folder, fname)
        if os.path.exists(fpath):
            text = extract_text_fallback(fpath)
        else:
            # Reconstruct from Chroma documents when file is missing
            text = ""
            if _chroma_enabled():
                try:
                    col = _get_chroma_collection()
                    if col:
                        got = col.get(where={"resume_file": fname}, include=["documents"], limit=100000)
                        docs = got.get("documents") or []
                        if docs and isinstance(docs[0], list):
                            docs = docs[0]
                        if docs:
                            text = "\n".join([d for d in docs if d])
                except Exception:
                    text = ""
        resume_texts.append(text or "")
        parsed_resume_data[fname] = {"text": text or ""}

    return resume_texts, resume_files, parsed_resume_data

# --- BM25 lexical prefilter (no extra dependency) ---
def _bm25_tokenize(text: str) -> list[str]:
    if not text:
        return []
    # alnum + tech chars; lowercase
    toks = re.findall(r"[A-Za-z][A-Za-z0-9\+\#\.\-]{1,}", text.lower())
    # drop very short tokens
    return [t for t in toks if len(t) >= 2]


# --- Cross-encoder re-ranker helpers ---
@st.cache_resource(show_spinner=False)
def get_cross_encoder_model(model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(model_name)
    except Exception:
        return None

def cross_encoder_rerank(jd_text: str, candidate_texts: list[str], candidate_files: list[str], model_name: str, top_k: int = 30) -> Tuple[list[str], list[str]]:
    """Return (ordered_files, ordered_texts) after scoring with cross-encoder."""
    if not candidate_texts or not candidate_files:
        return candidate_files, candidate_texts
    model = get_cross_encoder_model(model_name)
    if model is None:
        return candidate_files, candidate_texts
    # Prepare pairs for cross-encoder
    pairs = [(jd_text, t) for t in candidate_texts]
    try:
        scores = model.predict(pairs, show_progress_bar=False)
    except Exception:
        try:
            scores = model.predict(pairs)
        except Exception:
            return candidate_files, candidate_texts
    # Sort by score desc
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ordered_files = [candidate_files[i] for i in idxs[:top_k]]
    ordered_texts = [candidate_texts[i] for i in idxs[:top_k]]
    return ordered_files, ordered_texts

def _ensure_bm25_index(resume_files: list[str], resume_texts: list[str]):
    # Build and cache a simple BM25 index in session_state
    key = 'bm25_index'
    cache = st.session_state.get(key)
    # Use a simple fingerprint: tuple(sorted(filenames)) and count
    fp = tuple(sorted(resume_files))
    if cache and cache.get('fingerprint') == fp:
        return
    N = len(resume_files)
    doc_tfs: dict[str, dict[str, int]] = {}
    doc_len: dict[str, int] = {}
    df: dict[str, int] = {}
    for fname, txt in zip(resume_files, resume_texts):
        toks = _bm25_tokenize(txt)
        doc_len[fname] = len(toks)
        tf_local: dict[str, int] = {}
        for tk in toks:
            tf_local[tk] = tf_local.get(tk, 0) + 1
        doc_tfs[fname] = tf_local
        for tk in tf_local.keys():
            df[tk] = df.get(tk, 0) + 1
    avgdl = (sum(doc_len.values()) / max(1, N)) if N else 0.0
    st.session_state[key] = {
        'fingerprint': fp,
        'N': N,
        'df': df,
        'doc_tfs': doc_tfs,
        'doc_len': doc_len,
        'avgdl': avgdl,
    }

def bm25_top_docs(query: str, resume_files: list[str], resume_texts: list[str], top_n: int = 2000) -> list[str]:
    if not query or not resume_files:
        return []
    _ensure_bm25_index(resume_files, resume_texts)
    idx = st.session_state.get('bm25_index') or {}
    N = int(idx.get('N') or 0)
    if N <= 0:
        return []
    df: dict = idx.get('df') or {}
    doc_tfs: dict = idx.get('doc_tfs') or {}
    doc_len: dict = idx.get('doc_len') or {}
    avgdl: float = float(idx.get('avgdl') or 0.0)
    # Params
    k1 = 1.2
    b = 0.75
    q_terms = _bm25_tokenize(query)
    if not q_terms:
        return []
    # Precompute idf for query terms
    idf: dict[str, float] = {}
    for t in set(q_terms):
        dft = int(df.get(t, 0))
        idf[t] = math.log((N - dft + 0.5) / (dft + 0.5) + 1.0)
    scores: list[tuple[float, str]] = []
    for fname in resume_files:
        dl = max(1, int(doc_len.get(fname, 0)))
        tfmap: dict[str, int] = doc_tfs.get(fname, {})
        s = 0.0
        for t in q_terms:
            if t not in idf:
                continue
            tf = int(tfmap.get(t, 0))
            if tf <= 0:
                continue
            denom = tf + k1 * (1 - b + b * (dl / max(1e-9, avgdl)))
            s += idf[t] * (tf * (k1 + 1)) / max(1e-9, denom)
        if s > 0:
            scores.append((s, fname))
    scores.sort(reverse=True)
    return [fn for _, fn in scores[:max(1, top_n)]]

# --- Lightweight deterministic scoring and facts extraction helpers ---
def _extract_years_exp(text: str) -> int:
    if not text:
        return 0
    yrs = 0
    # Look for patterns like: 8 years, 5yrs, 10+ years
    for m in re.finditer(r"(\d{1,2})\s*\+?\s*(years|yrs|y)\b", text, flags=re.IGNORECASE):
        try:
            yrs = max(yrs, int(m.group(1)))
        except Exception:
            pass
    return yrs

def _token_set(text: str) -> set:
    if not text:
        return set()
    toks = re.findall(r"[A-Za-z][A-Za-z\+\#\.\-]{1,}", text.lower())
    return set(toks)

def compute_match_score(jd_text: str, resume_text: str, weights: tuple | None = None):
    """Return (match_pct:int 0..100, stars:str like '★★★★★', breakdown:dict).

    Prioritizes required skills and industry match, with extra boosts for company affinity and location match.
    Additionally, strongly favors resumes whose years of experience are CLOSEST to the JD requirement
    (symmetrically: e.g., 3 or 5 is preferred over 2 or 6 when JD requires 4), largely independent of skills.
    Use breakdown['skills_overlap_count'] to filter.
    weights: optional (skills_w, exp_w, industry_w) in fractions. If None, defaults to 0.60/0.30/0.10.
    """
    # Extract skills and industries
    def _extract_skills_set_local(text: str) -> set:
        tl = (text or '').lower()
        return {s for s in _COMMON_SKILLS if s in tl}
    def _extract_industries_set_local(text: str) -> set:
        COMMON_INDS = [
            'it','software','finance','banking','insurance','fintech','healthcare','pharma','biotech','retail',
            'e-commerce','ecommerce','manufacturing','telecom','telecommunications','education','edtech','energy',
            'government','public sector','automotive','aerospace','logistics','supply chain','media','entertainment',
            'real estate','gaming','travel','hospitality'
        ]
        tl = (text or '').lower()
        inds = set()
        for term in COMMON_INDS:
            if term in tl:
                inds.add(term)
        if 'telecommunications' in inds:
            inds.add('telecom')
        if 'ecommerce' in inds:
            inds.add('e-commerce')
        return inds

    jd_sk = _extract_skills_set_local(jd_text)
    rs_sk = _extract_skills_set_local(resume_text)
    jd_inds = _extract_industries_set_local(jd_text)
    rs_inds = _extract_industries_set_local(resume_text)

    # Skills overlap
    common_skills = jd_sk.intersection(rs_sk)
    skills_ratio = (len(common_skills) / max(1, len(jd_sk))) if jd_sk else 0.0

    # Industry match
    common_inds = jd_inds.intersection(rs_inds)
    industry_score = 1.0 if common_inds else (0.5 if (rs_inds and not jd_inds) else 0.0)

    # Experience proximity heuristic (symmetric closeness to JD requirement)
    # - Centered at jd_yrs; linearly decays to 0 by a tolerance window (default 2 years)
    # - Ensures candidates closest to the target (e.g., 3 or 5 for JD=4) are ranked above far ones (2 or 6+),
    #   even if skills are similar.
    jd_yrs = _extract_years_exp(jd_text)
    rs_yrs = _extract_years_exp(resume_text)
    if jd_yrs > 0 and rs_yrs > 0:
        delta = abs(rs_yrs - jd_yrs)
        tol_years = 2.0  # decay to zero by +/- 2 years from the target
        exp_prox = max(0.0, 1.0 - (delta / tol_years))
    else:
        exp_prox = 0.0
    # Keep a basic experience presence signal (very small) for resumes with any experience mentioned
    exp_presence = 1.0 if rs_yrs > 0 else 0.0

    # Extras: company affinity + location match
    jd_info_local = extract_jd_details(jd_text)
    jd_company = (jd_info_local.get('company') or '').strip().lower()
    jd_locations = set([s.lower() for s in (jd_info_local.get('locations') or [])])
    resume_locations = set([s.lower() for s in extract_locations(resume_text)])
    company_score = 0.0
    if jd_company and len(jd_company) >= 3:
        if jd_company in (resume_text or '').lower():
            company_score = 1.0
    location_score = 1.0 if (jd_locations and (jd_locations & resume_locations)) else 0.0

    # Weighted total: Skills 60%, Experience 30%, Industry 10% (configurable) + proximity bonus + small boosts
    if weights is None:
        sw, ew, iw = 0.60, 0.30, 0.10
    else:
        try:
            sw, ew, iw = weights
        except Exception:
            sw, ew, iw = 0.60, 0.30, 0.10
    total_w = max(1e-9, (sw + ew + iw))
    sw, ew, iw = sw/total_w, ew/total_w, iw/total_w
    # Replace prior experience score with proximity; retain a tiny presence component
    exp_component = 0.9 * exp_prox + 0.1 * exp_presence
    base = sw * skills_ratio + ew * exp_component + iw * industry_score
    # Add a strong proximity bonus so closeness can outweigh skills when needed (meets "despite skills" ask)
    proximity_bonus = 0.35 * exp_prox  # up to +0.35
    # Add gentle boosts (8% company, 7% location) then clamp
    score = base + proximity_bonus + 0.08 * company_score + 0.07 * location_score
    match_pct = int(round(max(0.0, min(1.0, score)) * 100))
    # Stars mapping
    if match_pct >= 80:
        stars = '★★★★★'
    elif match_pct >= 65:
        stars = '★★★★☆'
    elif match_pct >= 50:
        stars = '★★★☆☆'
    elif match_pct >= 35:
        stars = '★★☆☆☆'
    else:
        stars = '★☆☆☆☆'

    breakdown = {
        'skills_overlap_ratio': round(skills_ratio, 3),
        'skills_overlap_count': len(common_skills),
        'jd_skills_count': len(jd_sk),
        'common_skills': sorted(common_skills),
        'industry_match': 1 if bool(common_inds) else 0,
        'common_industries': sorted(common_inds),
        'experience': rs_yrs,
        'jd_experience': jd_yrs,
        'experience_proximity': round(exp_prox, 3),
        'company_match': 1 if company_score >= 1.0 else 0,
        'jd_company': jd_company,
        'location_match': 1 if location_score >= 1.0 else 0,
        'jd_locations': sorted(jd_locations),
        'resume_locations': sorted(resume_locations),
    }
    return match_pct, stars, breakdown

def _extract_industries(text: str) -> set:
    COMMON_INDS = [
        'it','software','finance','banking','insurance','fintech','healthcare','pharma','biotech','retail',
        'e-commerce','ecommerce','manufacturing','telecom','telecommunications','education','edtech','energy',
        'government','public sector','automotive','aerospace','logistics','supply chain','media','entertainment',
        'real estate','gaming','travel','hospitality'
    ]
    tl = (text or '').lower()
    inds = set()
    for term in COMMON_INDS:
        if term in tl:
            inds.add(term)
    if 'telecommunications' in inds:
        inds.add('telecom')
    if 'ecommerce' in inds:
        inds.add('e-commerce')
    return inds


# --- Skill synonyms and must-have gating helpers ---
def _parse_skill_synonyms(lines: str) -> dict:
    """Parse lines like 'python: py, python3' into a dict canonical -> set(variants).
    Returns mapping with all lowercased tokens and includes the canonical itself.
    """
    out = {}
    if not lines:
        return out
    for raw in (lines or '').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        # split on ':' or '=' or '->'
        m = re.split(r"\s*[:=\-\>]+\s*", line, maxsplit=1)
        if not m:
            continue
        key = m[0].strip().lower()
        vals = []
        if len(m) > 1 and m[1]:
            vals = [v.strip().lower() for v in re.split(r"[,;]", m[1]) if v.strip()]
        variants = set([key])
        for v in vals:
            variants.add(v)
        out[key] = variants
    return out


def _expand_skill_variants(skill: str, synonyms_map: dict) -> set:
    k = (skill or '').strip().lower()
    if not k:
        return set()
    res = set([k])
    if k in synonyms_map:
        res.update(synonyms_map.get(k, set()))
    # Also check if any synonym canonical maps to this skill (reverse lookup)
    for can, variants in synonyms_map.items():
        if k == can or k in variants:
            res.update(variants)
            res.add(can)
    return res


def resume_has_required_skills(resume_text: str, required_skills_csv: str, synonyms_lines: str, require_all: bool = True) -> bool:
    """Return True if resume_text satisfies the required skills constraint.

    If require_all is True then every required skill must be present (or one of its variants).
    If False then at least one required skill must be present.
    Matching checks token presence and substring fallback in lowercase text.
    """
    if not required_skills_csv or not resume_text:
        return True
    synonyms_map = _parse_skill_synonyms(synonyms_lines or '')
    txt = (resume_text or '').lower()
    tokens = _token_set(txt)
    # Parse CSV
    parts = [p.strip() for p in re.split(r"[,;]", required_skills_csv) if p.strip()]
    if not parts:
        return True
    results = []
    for sk in parts:
        variants = _expand_skill_variants(sk, synonyms_map)
        found = False
        for v in variants:
            if v in tokens:
                found = True
                break
            # substring fallback for multi-word tokens
            if v in txt:
                found = True
                break
        results.append(found)
    return all(results) if require_all else any(results)

_COMMON_SKILLS = [
    'python','java','javascript','typescript','react','node','nodejs','aws','azure','gcp','sql','nosql',
    'docker','kubernetes','terraform','linux','git','pandas','numpy','pytorch','tensorflow','scikit',
    'spark','hadoop','kafka','go','golang','c++','c#','.net','php','ruby','swift','kotlin','django','flask',
    'spring','redux','graphql','mysql','postgresql','mongodb','oracle','tableau','powerbi','excel'
]

def _extract_contact_info(text: str) -> dict:
    if not text:
        return {}
    info = {}
    m = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, flags=re.IGNORECASE)
    if m:
        info['email'] = m.group(0)
    m = re.search(r"(\+?\d[\d\s\-\(\)]{6,}\d)", text)
    if m:
        info['phone'] = m.group(1)
    m = re.search(r"https?://(www\.)?linkedin\.com/[\w\-/]+", text, flags=re.IGNORECASE)
    if m:
        info['linkedin'] = m.group(0)
    m = re.search(r"https?://(www\.)?github\.com/[\w\-]+", text, flags=re.IGNORECASE)
    if m:
        info['github'] = m.group(0)
    return info

def retrieve_resume_facts(resume_text: str) -> dict:
    text_low = (resume_text or '').lower()
    found_skills = []
    for sk in _COMMON_SKILLS:
        if sk in text_low:
            found_skills.append(sk)
    years = _extract_years_exp(resume_text)
    contact = _extract_contact_info(resume_text)
    return {
        'years': years,
        'skills': sorted(set(found_skills)),
        'contact': contact,
    }

# --- Location extraction helper ---
def extract_locations(text: str) -> list:
    if not text:
        return []
    t = text.lower()
    locs = set()
    # Common explicit label
    m = re.search(r"(?im)^(?:location|based in|work location)\s*[:\-]\s*(.+)$", text)
    if m:
        val = m.group(1).strip()
        # keep first 60 chars, split at delimiters
        parts = re.split(r"[,/|]", val[:60])
        for p in parts:
            p2 = p.strip().lower()
            if 2 <= len(p2) <= 40:
                locs.add(p2)
    # Keyword list (add or tune as needed)
    COMMON_LOCS = [
        'remote','hybrid','onsite','work from home',
        'bangalore','bengaluru','hyderabad','mumbai','pune','chennai','delhi','new delhi','noida','gurgaon','gurugram','kolkata','kochi','ahmedabad',
        'london','manchester','edinburgh','glasgow','paris','berlin','munich','amsterdam','dublin','rome','madrid','lisbon',
        'new york','nyc','san francisco','sfo','seattle','austin','dallas','chicago','boston','los angeles','la','washington dc','dc',
        'toronto','vancouver','ottawa','montreal',
        'sydney','melbourne',
        'singapore','dubai','abu dhabi','riyadh','jeddah','doha',
        'india','usa','united states','canada','uk','united kingdom','europe','middle east'
    ]
    for kw in COMMON_LOCS:
        if kw in t:
            locs.add(kw)
    # Normalize aliases
    if 'bengaluru' in locs:
        locs.add('bangalore')
    if 'nyc' in locs:
        locs.add('new york')
    if 'la' in locs:
        locs.add('los angeles')
    if 'united kingdom' in locs:
        locs.add('uk')
    if 'united states' in locs:
        locs.add('usa')
    return sorted(locs)

# --- JD parsing helper: extract company, required experience, skills, industry ---
def extract_jd_details(jd_text: str) -> dict:
    txt = (jd_text or '').strip()
    low = txt.lower()
    # Experience (years) using existing heuristic
    years_req = _extract_years_exp(txt)
    # Skills present in JD from our common list
    jd_skills = sorted({s for s in _COMMON_SKILLS if s in low})
    # Simple industry detection similar to compute_match_score
    COMMON_INDS = [
        'it','software','finance','banking','insurance','fintech','healthcare','pharma','biotech','retail',
        'e-commerce','ecommerce','manufacturing','telecom','telecommunications','education','edtech','energy',
        'government','public sector','automotive','aerospace','logistics','supply chain','media','entertainment',
        'real estate','gaming','travel','hospitality'
    ]
    inds = []
    for term in COMMON_INDS:
        if term in low:
            inds.append(term)
    if 'telecommunications' in inds and 'telecom' not in inds:
        inds.append('telecom')
    if 'ecommerce' in inds and 'e-commerce' not in inds:
        inds.append('e-commerce')
    inds = sorted(set(inds))
    # Company name heuristics
    company = ''
    # 1) Look for explicit labels
    m = re.search(r"(?im)^(?:company|company name)\s*[:\-]\s*(.+)$", txt)
    if m:
        company = m.group(1).strip().strip(' .')
    # 2) Look for lines like "About <Company>" or "Join <Company>"
    if not company:
        m = re.search(r"(?im)^(?:about|join|at)\s+([A-Z][A-Za-z0-9&., \-]{2,})$", txt)
        if m:
            company = m.group(1).strip().strip(' .')
    # 3) Look for patterns in first lines: "We are hiring at <Company>" or "Role at <Company>"
    if not company:
        m = re.search(r"(?i)\bat\s+([A-Z][A-Za-z0-9&., \-]{2,})", txt[:400])
        if m:
            company = m.group(1).strip().strip(' .')
    # Title/Position (optional)
    title = ''
    m = re.search(r"(?im)^(?:title|position|role)\s*[:\-]\s*(.+)$", txt)
    if m:
        title = m.group(1).strip().strip(' .')
    # Fallback: first line if short
    if not title:
        first_line = txt.splitlines()[0].strip() if txt.splitlines() else ''
        if 4 <= len(first_line) <= 120:
            title = first_line
    # Locations
    jd_locations = extract_locations(txt)
    return {
        'company': company,
        'title': title,
        'experience_required_years': years_req,
        'skills': jd_skills,
        'industries': inds,
        'locations': jd_locations,
    }

# Legacy FAISS-based similarity function removed; replaced by cached embeddings + cosine prefilter elsewhere.

# -------- Fast-mode, caching, and LLM summary helpers (placed before usage) -------- #
def _norm(s: str) -> str:
    try:
        return re.sub(r"\s+", " ", (s or "").strip())
    except Exception:
        return s or ""

def _jd_hash(jd_text: str) -> str:
    try:
        return hashlib.sha1(_norm(jd_text).encode("utf-8", errors="ignore")).hexdigest()
    except Exception:
        return str(abs(hash(_norm(jd_text))))

def _get_llm_cache():
    cache = st.session_state.get("llm_summary_cache")
    if cache is None:
        cache = {}
        st.session_state["llm_summary_cache"] = cache
    return cache

LLM_CACHE_DIR = os.path.join(_RESUME_DIR, '.llm_cache')
os.makedirs(LLM_CACHE_DIR, exist_ok=True)

def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name or "")

def get_llm_summary_cached(jd_text: str, resume_text: str, fname: str, timeout: int = 60) -> tuple[int, str] | None:
    """Return cached (score, summary) for (jd_hash, fname) or None if not present."""
    key = (_jd_hash(jd_text), fname)
    cache = _get_llm_cache()
    if key in cache:
        return cache[key]
    # Check disk cache
    try:
        fn = os.path.join(LLM_CACHE_DIR, f"{key[0]}_{_safe_name(fname)}.json")
        if os.path.exists(fn):
            with open(fn, 'r', encoding='utf-8') as f:
                data = json.load(f)
            sc = int(data.get('score', 0) or 0)
            sm = str(data.get('summary', '') or '')
            cache[key] = (sc, sm)
            st.session_state["llm_summary_cache"] = cache
            return sc, sm
    except Exception:
        pass
    return None

def set_llm_summary_cache(jd_text: str, fname: str, score: int, summary: str):
    key = (_jd_hash(jd_text), fname)
    cache = _get_llm_cache()
    cache[key] = (int(score), str(summary or ""))
    st.session_state["llm_summary_cache"] = cache
    # Persist to disk
    try:
        fn = os.path.join(LLM_CACHE_DIR, f"{key[0]}_{_safe_name(fname)}.json")
        with open(fn, 'w', encoding='utf-8') as f:
            json.dump({"score": int(score), "summary": str(summary or "")}, f)
    except Exception:
        pass

def deterministic_summary(jd_text: str, resume_text: str) -> str:
    """Fast, no-network summary using detected skills, experience, and overlap."""
    mp, _, br = compute_match_score(jd_text, resume_text, st.session_state.get('computed_user_weights', (0.60,0.30,0.10)))
    yrs = _extract_years_exp(resume_text)
    common = br.get('common_skills', [])
    sk = ", ".join(common[:6]) if common else "relevant skills"
    bits = []
    if yrs:
        bits.append(f"~{yrs} yrs exp")
    bits.append(f"match {mp}%")
    return f"Strong in {sk}; {', '.join(bits)}."

@st.cache_data(show_spinner=False, ttl=300)
def presign_cached(key: str, expires: int = 3600) -> str | None:
    # AWS removed: no presign available
    return None

def download_attachments_from_gmail():
    import os
    import pickle
    import base64
    from googleapiclient.discovery import build
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    import streamlit as st
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            project_dir = os.path.dirname(os.path.abspath(__file__))
            cred_path = os.path.join(project_dir, 'credentials.json')
            if not os.path.exists(cred_path):
                st.error(f"Missing credentials.json for Gmail OAuth. Please place it in {project_dir}.")
                return 0, None
            flow = InstalledAppFlow.from_client_secrets_file(cred_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    service = build('gmail', 'v1', credentials=creds)
    results = service.users().messages().list(userId='me', q='has:attachment', maxResults=20).execute()
    messages = results.get('messages', [])
    resume_folder = 'downloaded_resumes'
    os.makedirs(resume_folder, exist_ok=True)
    downloaded = 0
    for msg in messages:
        msg_id = msg['id']
        msg_data = service.users().messages().get(userId='me', id=msg_id).execute()
        parts = msg_data.get('payload', {}).get('parts', [])
        for part in parts:
            filename = part.get('filename')
            body = part.get('body', {})
            if filename and filename.lower().endswith(('.pdf', '.docx', '.txt')):
                att_id = body.get('attachmentId')
                if att_id:
                    att = service.users().messages().attachments().get(userId='me', messageId=msg_id, id=att_id).execute()
                    data = att.get('data')
                    if data:
                        file_data = base64.urlsafe_b64decode(data.encode('UTF-8'))
                        save_path = os.path.join(resume_folder, filename)
                        if not os.path.exists(save_path):
                            with open(save_path, 'wb') as f:
                                f.write(file_data)
                            downloaded += 1
    return downloaded, resume_folder if downloaded > 0 else None
    # AWS/S3 helpers removed

# ---- ChromaDB helpers (optional local vector store with per-row metadata) ----
def _chroma_enabled() -> bool:
    try:
        return st.session_state.get('vector_store_mode') in ('Chroma (beta)', 'Chroma (local)')
    except Exception:
        return False


def _get_chroma_collection():
    """Open/create a persistent Chroma collection for resume chunks.
    Uses cosine space for SBERT-style embeddings. Path and collection are configurable in Settings.
    """
    try:
        import chromadb
        base_path = st.session_state.get('chroma_path') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chroma_store')
        os.makedirs(base_path, exist_ok=True)
        client = chromadb.PersistentClient(path=base_path)
        coll_name = st.session_state.get('chroma_collection', 'resume_chunks')
        try:
            return client.get_collection(coll_name)
        except Exception:
            return client.create_collection(name=coll_name, metadata={"hnsw:space": "cosine"})
    except Exception as e:
        try:
            st.warning(f"Chroma not available: {e}")
        except Exception:
            pass
        return None


def _chroma_get_any_metadata_for_file(fname: str) -> dict | None:
    """Fetch one metadata row from Chroma for the given resume filename."""
    try:
        col = _get_chroma_collection()
        if not col:
            return None
        got = col.get(where={"resume_file": fname}, include=["metadatas"], limit=1)
        m = (got.get("metadatas") or [])
        if m and isinstance(m[0], list):
            m = m[0]
        return m[0] if m else None
    except Exception:
        return None


def purge_originals_if_indexed() -> int:
    """Delete local resume files that are already indexed in Chroma. Returns count removed."""
    if not _chroma_enabled():
        return 0
    col = _get_chroma_collection()
    if col is None:
        return 0
    removed = 0
    try:
        local = set(_list_resume_files(_RESUME_DIR))
        for fname in list(local):
            try:
                got = col.get(where={"resume_file": fname}, limit=1)
                has = bool(got and (got.get("ids") or []))
            except Exception:
                has = False
            if has:
                p = os.path.join(_RESUME_DIR, fname)
                try:
                    os.remove(p)
                    removed += 1
                except Exception:
                    pass
        return removed
    except Exception:
        return removed


def _chroma_chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Simple character-based chunker to avoid coupling to app-specific helpers.
    Ensures overlap < size and returns non-empty, stripped chunks.
    """
    try:
        s = (text or "")
        size = max(100, int(size))
        overlap = max(0, int(overlap))
        if overlap >= size:
            overlap = size - 1
        chunks: list[str] = []
        n = len(s)
        i = 0
        while i < n:
            end = min(n, i + size)
            part = s[i:end].strip()
            if part:
                chunks.append(part)
            if end >= n:
                break
            i = end - overlap
            if i <= 0:
                i = end
        return chunks
    except Exception:
        return [text] if text else []


def _chroma_embed_texts(texts: list[str]) -> np.ndarray:
    """Embed texts using the existing sentence-transformers model in this app."""
    model = get_st_model()
    bs = int(st.session_state.get('embed_batch_size', DEFAULT_EMBED_BATCH))
    try:
        vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False, batch_size=bs)
    except TypeError:
        vecs = np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False))
    return vecs.astype(np.float32)


def chroma_upsert_files(file_paths: list[str]) -> int:
    """
    Upsert the given resume files as chunks into Chroma with per-row metadata.
    Returns number of chunks added.
    """
    col = _get_chroma_collection()
    if col is None or not file_paths:
        return 0

    added = 0
    batch_ids: list[str] = []
    batch_docs: list[str] = []
    batch_embs: list[list[float]] = []
    batch_metas: list[dict] = []

    for path in file_paths:
        try:
            fname = os.path.basename(path)
            text = extract_text_fallback(path) or ""
            if not text.strip():
                continue

            try:
                stat = os.stat(path)
                size = stat.st_size
                mtime = int(stat.st_mtime)
            except Exception:
                size = None
                mtime = None

            chunks = _chroma_chunk_text(
                text,
                size=int(st.session_state.get('ann_chunk_size', DEFAULT_CHUNK_SIZE)),
                overlap=int(st.session_state.get('ann_chunk_overlap', DEFAULT_CHUNK_OVERLAP))
            )
            if not chunks:
                continue

            embeds = _chroma_embed_texts(chunks)

            try:
                import hashlib
                with open(path, 'rb') as f:
                    sha1 = hashlib.sha1(f.read()).hexdigest()
            except Exception:
                sha1 = None

            for idx, (chunk_text, vec) in enumerate(zip(chunks, embeds)):
                cid = f"{fname}:{idx}:{(sha1 or 'nohash')[:8]}"
                meta = {
                    "resume_file": fname,
                    "chunk_id": idx,
                    "sha1": sha1,
                    "size": size,
                    "mtime": mtime,
                }
                batch_ids.append(cid)
                batch_docs.append(chunk_text)
                batch_embs.append(vec.tolist())
                batch_metas.append(meta)
                if len(batch_ids) >= 512:
                    args = {"ids": batch_ids, "embeddings": batch_embs, "metadatas": batch_metas}
                    if bool(st.session_state.get('chroma_store_documents', True)):
                        args["documents"] = batch_docs
                    col.add(**args)
                    added += len(batch_ids)
                    batch_ids, batch_docs, batch_embs, batch_metas = [], [], [], []

        except Exception as e:
            try:
                st.warning(f"Chroma upsert skipped {path}: {e}")
            except Exception:
                pass
            continue

    if batch_ids:
        args = {"ids": batch_ids, "embeddings": batch_embs, "metadatas": batch_metas}
        if bool(st.session_state.get('chroma_store_documents', True)):
            args["documents"] = batch_docs
        col.add(**args)
        added += len(batch_ids)

    return added


def _extract_text_from_bytes(data: bytes, filename: str) -> str:
    """Extract text from in-memory file bytes (PDF/DOCX/TXT)."""
    if not data:
        return ""
    name = (filename or "").lower()
    try:
        if name.endswith(".pdf"):
            import pdfplumber, io
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        elif name.endswith(".docx"):
            import docx, io
            doc = docx.Document(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs)
        elif name.endswith(".txt"):
            try:
                return data.decode("utf-8", errors="ignore")
            except Exception:
                return data.decode("latin-1", errors="ignore")
        else:
            # Fallback: try text decode
            return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def chroma_upsert_from_mysql(sql: str, id_col: str = "id", filename_col: str = "filename",
                             blob_col: str | None = "content", url_col: str | None = None,
                             limit: int | None = 500) -> int:
    """Read rows from MySQL and upsert into Chroma (no local storage)."""
    col = _get_chroma_collection()
    if col is None:
        try:
            st.warning("Chroma client not available.")
        except Exception:
            pass
        return 0
    try:
        import mysql.connector, requests  # type: ignore
    except Exception:
        try:
            st.warning("Missing mysql-connector-python or requests. Install them to use MySQL indexing.")
        except Exception:
            pass
        return 0

    host = st.session_state.get('mysql_host', '127.0.0.1')
    port = int(st.session_state.get('mysql_port', 3306))
    user = st.session_state.get('mysql_user', '')
    password = st.session_state.get('mysql_pass', '')
    database = st.session_state.get('mysql_db', '')
    connect_timeout = int(st.session_state.get('mysql_connect_timeout', 10) or 10)

    conn = mysql.connector.connect(host=host, port=port, user=user, password=password,
                                   database=database, connection_timeout=connect_timeout)
    cur = conn.cursor(dictionary=True)
    cur.execute(sql)
    rows = cur.fetchmany(size=limit) if limit else cur.fetchall()

    added = 0
    batch_ids: list[str] = []
    batch_docs: list[str] = []
    batch_embs: list[list[float]] = []
    batch_metas: list[dict] = []

    for row in rows or []:
        rid = row.get(id_col)
        fname = os.path.basename(str(row.get(filename_col) or f"{rid or 'row'}.txt"))
        data = b""
        if blob_col and (blob_col in row) and (row[blob_col] is not None):
            v = row[blob_col]
            data = v if isinstance(v, (bytes, bytearray)) else bytes(v)
        elif url_col and row.get(url_col):
            try:
                r = requests.get(str(row[url_col]), timeout=30)
                r.raise_for_status()
                data = r.content
            except Exception:
                data = b""

        text = _extract_text_from_bytes(data, fname) if data else ""
        if not text.strip():
            continue

        chunks = _chroma_chunk_text(
            text,
            size=int(st.session_state.get('ann_chunk_size', DEFAULT_CHUNK_SIZE)),
            overlap=int(st.session_state.get('ann_chunk_overlap', DEFAULT_CHUNK_OVERLAP))
        )
        if not chunks:
            continue
        vecs = _chroma_embed_texts(chunks)

        sha_part = hashlib.sha1((str(rid) + "|" + fname).encode("utf-8", errors="ignore")).hexdigest()[:8]
        for idx, (chunk_text, v) in enumerate(zip(chunks, vecs)):
            cid = f"mysql:{rid}:{fname}:{idx}:{sha_part}"
            meta = {
                "resume_file": fname,
                "chunk_id": idx,
                "mysql_id": rid,
                "source": "mysql",
                "size": len(data) if data else None,
            }
            batch_ids.append(cid)
            batch_docs.append(chunk_text)
            batch_embs.append(v.tolist())
            batch_metas.append(meta)
            if len(batch_ids) >= 512:
                args = {"ids": batch_ids, "embeddings": batch_embs, "metadatas": batch_metas}
                if bool(st.session_state.get('chroma_store_documents', True)):
                    args["documents"] = batch_docs
                col.add(**args)
                added += len(batch_ids)
                batch_ids, batch_docs, batch_embs, batch_metas = [], [], [], []

    if batch_ids:
        args = {"ids": batch_ids, "embeddings": batch_embs, "metadatas": batch_metas}
        if bool(st.session_state.get('chroma_store_documents', True)):
            args["documents"] = batch_docs
        col.add(**args)
        added += len(batch_ids)

    try:
        cur.close(); conn.close()
    except Exception:
        pass
    return added


def mysql_fetch_blob_by_id(mysql_id) -> bytes | None:
    """Fetch original file bytes by id using configured MySQL settings and download SQL template."""
    try:
        import mysql.connector  # type: ignore
    except Exception:
        try:
            st.warning("Missing mysql-connector-python. Install it to enable MySQL download fallback.")
        except Exception:
            pass
        return None
    host = st.session_state.get('mysql_host', '127.0.0.1')
    port = int(st.session_state.get('mysql_port', 3306))
    user = st.session_state.get('mysql_user', '')
    password = st.session_state.get('mysql_pass', '')
    database = st.session_state.get('mysql_db', '')
    connect_timeout = int(st.session_state.get('mysql_connect_timeout', 10) or 10)
    sql = st.session_state.get('mysql_download_sql', 'SELECT content FROM resumes WHERE id=%s')
    try:
        conn = mysql.connector.connect(host=host, port=port, user=user, password=password,
                                       database=database, connection_timeout=connect_timeout)
        cur = conn.cursor()
        cur.execute(sql, (mysql_id,))
        row = cur.fetchone()
        cur.close(); conn.close()
        if not row:
            return None
        # Support tuple or dict cursor
        if isinstance(row, (list, tuple)):
            val = row[0]
        elif isinstance(row, dict):
            # take the first column value
            val = next(iter(row.values()))
        else:
            val = row
        if isinstance(val, (bytes, bytearray)):
            return bytes(val)
        try:
            return bytes(val)
        except Exception:
            return None
    except Exception:
        return None


def chroma_query(jd_text: str, top_chunks: int = 500, max_resumes: int = 100, return_highlights: bool = True):
    """
    Query Chroma by embedding the JD text, aggregate by resume_file, and return shortlist.
    Returns (files, highlights) if return_highlights else files.
    """
    col = _get_chroma_collection()
    if col is None:
        return None
    try:
        qv = _embed_texts([jd_text], batch=1)[0].tolist()
        res = col.query(query_embeddings=[qv], n_results=int(top_chunks), include=["metadatas", "distances", "documents"])
    except Exception as e:
        try:
            st.warning(f"Chroma query failed: {e}")
        except Exception:
            pass
        return None

    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    # Documents may be absent when we don't store them in Chroma; pad to length of metas so zip doesn't truncate
    _docs_raw = res.get("documents")
    if _docs_raw and len(_docs_raw) > 0 and _docs_raw[0]:
        docs = _docs_raw[0]
    else:
        docs = [None] * len(metas)

    resume_best: dict[str, float] = {}
    highlights: dict[str, list[tuple[str, float]]] = {}

    for md, d, doc in zip(metas, dists, docs):
        try:
            fname = (md or {}).get("resume_file")
            if not fname:
                continue
            sim = 1.0 - float(d)
            prev = resume_best.get(fname)
            if prev is None or sim > prev:
                resume_best[fname] = sim
            if return_highlights and doc:
                lst = highlights.setdefault(fname, [])
                if len(lst) < 3:
                    lst.append((doc[:300].replace("\n", " "), sim))
        except Exception:
            continue

    if not resume_best:
        return None

    ordered = sorted(resume_best.items(), key=lambda kv: kv[1], reverse=True)
    top_files = [fn for fn, _ in ordered[:max_resumes]]
    return (top_files, {fn: highlights.get(fn, [])[:3] for fn in top_files}) if return_highlights else top_files

# ---- MySQL remote import helper ----
def _sync_from_mysql(host: str, port: int, user: str, password: str, database: str,
                     sql: str, filename_col: str = "filename",
                     blob_col: str | None = "content", url_col: str | None = None,
                     limit: int | None = 200,
                     ssl_ca: str | None = None, ssl_cert: str | None = None, ssl_key: str | None = None,
                     ssl_disabled: bool | None = None,
                     progress_cb=None,
                     derive_url: bool | None = None, base_url: str | None = None,
                     dir_col: str | None = None, stored_name_col: str | None = None,
                     local_copy: bool | None = None, base_dir: str | None = None,
                     connect_timeout: int | None = 10) -> tuple[list[str], dict]:
    """
    Fetch resumes from MySQL into downloaded_resumes.
    - If blob_col provided (BLOB), writes to disk.
    - If url_col provided (HTTP/HTTPS), downloads via requests.
    Returns list of downloaded basenames.
    """
    downloaded: list[str] = []
    stats = {
        'fetched': 0,
        'missing_filename': 0,
        'blob_saved': 0,
        'url_saved': 0,
        'local_saved': 0,
        'renamed': 0,
        'errors': 0,
    }
    if not (host and user and database and sql):
        return downloaded, stats
    try:
        try:
            import mysql.connector  # type: ignore
        except Exception:
            st.warning("mysql-connector-python not installed. Run: pip install mysql-connector-python")
            return downloaded, stats
        try:
            import requests  # type: ignore
        except Exception:
            st.warning("requests not installed. Run: pip install requests")
            return downloaded, stats

        os.makedirs(_RESUME_DIR, exist_ok=True)
        conn_args = {
            "host": host,
            "port": int(port),
            "user": user,
            "password": password,
            "database": database,
        }
        try:
            if connect_timeout and int(connect_timeout) > 0:
                conn_args["connection_timeout"] = int(connect_timeout)
        except Exception:
            pass
        # SSL options (optional)
        if ssl_disabled is True:
            conn_args["ssl_disabled"] = True
        else:
            if ssl_ca:
                conn_args["ssl_ca"] = ssl_ca
            if ssl_cert:
                conn_args["ssl_cert"] = ssl_cert
            if ssl_key:
                conn_args["ssl_key"] = ssl_key
        conn = mysql.connector.connect(**conn_args)
        cur = conn.cursor(dictionary=True)
        cur.execute(sql)
        rows = cur.fetchmany(size=limit) if limit else cur.fetchall()
        total = len(rows) if rows else 0
        stats['fetched'] = total
        if callable(progress_cb):
            try:
                progress_cb(0, total, f"Fetched {total} row(s) – starting downloads…")
            except Exception:
                pass
        processed = 0
        for row in rows:
            processed += 1
            fname = str(row.get(filename_col) or "").strip()
            if not fname:
                stats['missing_filename'] += 1
                continue
            base = os.path.basename(fname)
            target = os.path.join(_RESUME_DIR, base)
            # Avoid overwrite by renaming if file exists
            if os.path.exists(target):
                root, ext = os.path.splitext(base)
                k = 1
                while os.path.exists(os.path.join(_RESUME_DIR, f"{root}_{k}{ext}")):
                    k += 1
                base = f"{root}_{k}{ext}"
                target = os.path.join(_RESUME_DIR, base)
                stats['renamed'] += 1

            wrote = False
            if blob_col and (blob_col in row) and (row[blob_col] is not None):
                try:
                    data = row[blob_col]
                    with open(target, "wb") as f:
                        f.write(data if isinstance(data, (bytes, bytearray)) else bytes(data))
                    wrote = True
                    stats['blob_saved'] += 1
                except Exception:
                    wrote = False
                    stats['errors'] += 1
            if (not wrote) and url_col and (url_col in row) and row[url_col]:
                try:
                    url = str(row[url_col])
                    r = requests.get(url, timeout=30)
                    r.raise_for_status()
                    with open(target, "wb") as f:
                        f.write(r.content)
                    wrote = True
                    stats['url_saved'] += 1
                except Exception:
                    wrote = False
                    stats['errors'] += 1
            # Derived URL from base + directory_name + stored_filename
            if (not wrote) and derive_url and base_url and dir_col and stored_name_col:
                try:
                    dval = str(row.get(dir_col) or '').strip('/')
                    sval = str(row.get(stored_name_col) or '').strip()
                    if dval and sval:
                        # Build URL safely
                        # Ensure base ends with /, join directory, then append quoted filename
                        base_clean = base_url if base_url.endswith('/') else (base_url + '/')
                        dir_url = urljoin(base_clean, dval + '/')
                        full_url = urljoin(dir_url, quote(sval))
                        r = requests.get(full_url, timeout=30)
                        r.raise_for_status()
                        with open(target, 'wb') as f:
                            f.write(r.content)
                        wrote = True
                        stats['url_saved'] += 1
                except Exception:
                    wrote = False
                    stats['errors'] += 1
            # Local/UNC copy from base_dir + directory_name + stored_filename
            if (not wrote) and local_copy and base_dir and dir_col and stored_name_col:
                try:
                    dval = str(row.get(dir_col) or '').strip('/\\')
                    sval = str(row.get(stored_name_col) or '').strip()
                    if dval and sval:
                        src_path = os.path.normpath(os.path.join(base_dir, dval, sval))
                        base_norm = os.path.normpath(base_dir)
                        # Prevent path traversal outside base_dir
                        if src_path.startswith(base_norm) and os.path.exists(src_path):
                            shutil.copyfile(src_path, target)
                            wrote = True
                            stats['local_saved'] += 1
                except Exception:
                    wrote = False
                    stats['errors'] += 1
            if wrote:
                downloaded.append(base)
            if callable(progress_cb):
                try:
                    progress_cb(processed, total, f"Processed {processed}/{total} – downloaded {len(downloaded)} file(s)")
                except Exception:
                    pass

        try:
            cur.close()
            conn.close()
        except Exception:
            pass
    except Exception as e:
        st.error(f"MySQL sync failed: {e}")
    return downloaded, stats

def _test_mysql_connection(host: str, port: int, user: str, password: str, database: str,
                           sql: str,
                           ssl_ca: str | None = None, ssl_cert: str | None = None, ssl_key: str | None = None,
                           ssl_disabled: bool | None = None,
                           connect_timeout: int | None = 10) -> bool:
    """Attempt a lightweight connection and single-row fetch to validate creds, host, and SQL.
    Raises an exception if any step fails; returns True on success.
    """
    try:
        import mysql.connector  # type: ignore
    except Exception:
        raise RuntimeError("mysql-connector-python not installed. Run: pip install mysql-connector-python")
    conn_args = {
        "host": host,
        "port": int(port),
        "user": user,
        "password": password,
        "database": database,
    }
    try:
        if connect_timeout and int(connect_timeout) > 0:
            conn_args["connection_timeout"] = int(connect_timeout)
    except Exception:
        pass
    if ssl_disabled is True:
        conn_args["ssl_disabled"] = True
    else:
        if ssl_ca:
            conn_args["ssl_ca"] = ssl_ca
        if ssl_cert:
            conn_args["ssl_cert"] = ssl_cert
        if ssl_key:
            conn_args["ssl_key"] = ssl_key
    conn = None
    cur = None
    try:
        conn = mysql.connector.connect(**conn_args)
        cur = conn.cursor()
        cur.execute(sql)
        _ = cur.fetchone()
        return True
    finally:
        try:
            cur.close() if cur else None
        except Exception:
            pass
        try:
            conn.close() if conn else None
        except Exception:
            pass

def _peek_mysql_rows(host: str, port: int, user: str, password: str, database: str,
                     sql: str,
                     ssl_ca: str | None = None, ssl_cert: str | None = None, ssl_key: str | None = None,
                     ssl_disabled: bool | None = None,
                     connect_timeout: int | None = 10) -> tuple[list[str], list[dict]]:
    """Fetch up to 5 rows and return (columns, sample_rows) with values sanitized for display."""
    import mysql.connector  # type: ignore
    conn_args = {
        "host": host,
        "port": int(port),
        "user": user,
        "password": password,
        "database": database,
    }
    try:
        if connect_timeout and int(connect_timeout) > 0:
            conn_args["connection_timeout"] = int(connect_timeout)
    except Exception:
        pass
    if ssl_disabled is True:
        conn_args["ssl_disabled"] = True
    else:
        if ssl_ca: conn_args["ssl_ca"] = ssl_ca
        if ssl_cert: conn_args["ssl_cert"] = ssl_cert
        if ssl_key: conn_args["ssl_key"] = ssl_key
    conn = mysql.connector.connect(**conn_args)
    cur = conn.cursor(dictionary=True)
    cur.execute(sql)
    rows = cur.fetchmany(size=5)
    cols = list(rows[0].keys()) if rows else []
    out = []
    for r in rows:
        m = {}
        for k, v in r.items():
            if isinstance(v, (bytes, bytearray)):
                m[k] = f"<{len(v)} bytes>"
            else:
                s = str(v)
                m[k] = s if len(s) <= 160 else (s[:157] + '...')
        out.append(m)
    try:
        cur.close(); conn.close()
    except Exception:
        pass
    return cols, out

# ---- Persisted MySQL settings (local JSON; plaintext password) ----
_MYSQL_SETTINGS_PATH = os.path.join(_ROOT_DIR, 'mysql_settings.json')

def _load_mysql_settings() -> dict:
    try:
        if os.path.exists(_MYSQL_SETTINGS_PATH):
            with open(_MYSQL_SETTINGS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {}

def _save_mysql_settings(cfg: dict) -> bool:
    try:
        with open(_MYSQL_SETTINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2)
        return True
    except Exception:
        return False

import streamlit as st
import os
import pdfplumber
import docx
from sentence_transformers import SentenceTransformer
import tempfile
import base64
import pickle
from googleapiclient.discovery import build
import zipfile
import io
import shutil
from pathlib import Path

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
body, .stApp {
    background: linear-gradient(120deg, #f8fafc 0%, #e0eafc 100%) !important;
    color: #23272F !important;
    font-family: 'Inter', Arial, sans-serif !important;
    font-size: 1.08rem;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #e0eafc 60%, #cfdef3 100%) !important;
    color: #23272F !important;
    border-radius: 0 16px 16px 0;
    box-shadow: 2px 0 12px #e0eafc;
}
section[data-testid="stSidebar"] * {
    color: #23272F !important;
    font-family: 'Inter', Arial, sans-serif !important;
}
header[data-testid="stHeader"] {
    background: none !important;
    color: #23272F !important;
}
header[data-testid="stHeader"] * {
    color: #23272F !important;
}
.app-title {
    font-size: 2rem; font-weight: 800; color: #1b2a41; letter-spacing: -0.5px;
    margin: 0.25em 0 0.75em 0; padding-bottom: 0.25em;
    border-bottom: 2px solid #e0eafc;
    font-family: 'Inter', Arial, sans-serif !important;
}
.big-title {
    font-size:2.5rem; font-weight:700; color:#23272F; letter-spacing: -1px; margin-bottom: 0.5em;
    font-family: 'Inter', Arial, sans-serif !important;
}
.section-title {
    font-size:1.25rem; font-weight:600; color:#2193b0; margin-top:2em; margin-bottom:0.5em;
    letter-spacing: -0.5px;
}
.resume-card {
    background:#fff; border-radius:14px; padding:1.2em 1.5em; margin-bottom:1.2em; box-shadow:0 4px 16px #e0eafc;
    color:#23272F;
    text-align: left;
    border: 1.5px solid #e0eafc;
    transition: box-shadow 0.2s, border 0.2s;
}
.resume-card:hover {
    box-shadow:0 8px 24px #b6d0f7;
    border: 1.5px solid #2193b0;
}
.sim-score {
    color:#2193b0; font-weight:600;
    text-align: left;
}
.stButton>button, .stDownloadButton>button {
    background: linear-gradient(90deg, #2193b0 60%, #6DD5ED 100%);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.55em 1.3em;
    margin: 0.2em 0.2em 0.2em 0;
    transition: background 0.2s, color 0.2s, box-shadow 0.2s;
    box-shadow: 0 2px 8px #e0eafc;
    font-family: 'Inter', Arial, sans-serif !important;
    font-size: 1.05rem;
    display: inline-block;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    background: linear-gradient(90deg, #6DD5ED 60%, #2193b0 100%);
    color: #fff;
    box-shadow: 0 4px 16px #b6d0f7;
}
.stTextInput>div>input, .stSelectbox>div>div>div {
    background: #fff;
    color: #23272F;
    border-radius: 8px;
    border: 1.5px solid #b6d0f7;
    text-align: left;
    font-size: 1.05rem;
    padding: 0.5em 1em;
    font-family: 'Inter', Arial, sans-serif !important;
}
.stExpanderHeader {
    color: #2193b0 !important;
    text-align: left;
    font-weight: 600;
    font-size: 1.1rem;
}
.stInfo, .stWarning, .stSuccess {
    background: #f0f8ff !important;
    color: #2193b0 !important;
    border-radius: 8px;
    text-align: left;
    font-size: 1.05rem;
    font-family: 'Inter', Arial, sans-serif !important;
}
pre, .resume-card pre {
    color: #23272F !important;
    background: #f8f9fa !important;
    border-radius: 8px;
    padding: 0.7em 1em;
    text-align: left;
    font-size: 1.05rem;
    font-family: 'Inter', Arial, sans-serif !important;
}
::-webkit-scrollbar {
    height: 10px;
    background: #e0eafc;
    border-radius: 8px;
}
::-webkit-scrollbar-thumb {
    background: #b6d0f7;
    border-radius: 8px;
}
::-webkit-scrollbar-thumb:hover {
    background: #2193b0;
}
</style>
""", unsafe_allow_html=True)

# Change tab names
st.markdown("""
<style>
.stTabs [role="tablist"] {
    gap: 1rem !important;
    justify-content: flex-start !important;
    margin-bottom: 0.5em !important;
    overflow-x: auto !important;
    white-space: nowrap !important;
    scrollbar-width: thin;
    scrollbar-color: #2193b0 #e0eafc;
}
.stTabs [role="tab"] {
    font-size: 1.08rem !important;
    font-weight: 600 !important;
    padding: 0.7em 1.2em !important;
    border-radius: 8px 8px 0 0 !important;
    margin-right: 0.3em !important;
    background: #f7fafd !important;
    color: #2193b0 !important;
    border: 1.5px solid #e0eafc !important;
    transition: background 0.2s, color 0.2s;
    display: inline-block !important;
}
.stTabs [role="tab"][aria-selected="true"] {
    background: #2193b0 !important;
    color: #fff !important;
    border-bottom: 2.5px solid #6DD5ED !important;
}
.stTabs [role="tablist"]::-webkit-scrollbar {
    height: 8px;
    background: #e0eafc;
}
.stTabs [role="tablist"]::-webkit-scrollbar-thumb {
    background: #2193b0;
    border-radius: 8px;
}
.stTabs [role="tablist"]::-webkit-scrollbar-thumb:hover {
    background: #6DD5ED;
}
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<div class="app-title">JD ↔ Resume Matcher</div>', unsafe_allow_html=True)

# Single tabs bar
main_tabs = st.tabs(["RAG Matcher", "Resumes", "Settings"])

# --- Custom CSS for radio button highlighting ---
st.markdown("""
<style>
div[data-testid="stRadio"] > label {
    font-weight: 700;
    color: #2193b0;
}
div[data-testid="stRadio"] div[role="radiogroup"] > div {
    border: 2px solid #2193b0;
    border-radius: 8px;
    margin-bottom: 0.3em;
    background: #e0eafc;
    transition: background 0.2s, border 0.2s;
}
div[data-testid="stRadio"] div[role="radiogroup"] > div[aria-checked="true"] {
    background: linear-gradient(90deg, #2193b0 60%, #6DD5ED 100%);
    color: #fff !important;
    border: 2.5px solid #6DD5ED;
    font-weight: 900;
}
div[data-testid="stRadio"] div[role="radiogroup"] > div[aria-checked="true"] label {
    color: #fff !important;
}
</style>
""", unsafe_allow_html=True)

    # (Removed: Browse by Skills & Experience tab content)
with main_tabs[0]:
    st.markdown('<div class="big-title">RAG Matcher: Resume Match Summaries</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload Job Description</div>', unsafe_allow_html=True)
    # Show global resume/embedding status for awareness
    t_cnt, e_cnt, r_cnt = get_resume_cache_status()
    st.caption(f"Resumes on disk: {t_cnt} • Embedded cached: {e_cnt} • Remaining to embed: {r_cnt}")
    jd_file = st.file_uploader("Upload Job Description (any file type)", type=None, key="jd_file_raggen")
    jd_text = ""
    if jd_file:
        try:
            ext = os.path.splitext(jd_file.name)[1].lower()
            jd_text = ""
            try:
                if ext == '.pdf':
                    import pdfplumber
                    with pdfplumber.open(jd_file) as pdf:
                        jd_text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
                elif ext == '.docx':
                    import docx
                    doc = docx.Document(jd_file)
                    jd_text = '\n'.join([para.text for para in doc.paragraphs])
                elif ext == '.txt':
                    jd_text = jd_file.read().decode('utf-8', errors='ignore').strip()
                else:
                    # Try to read as text
                    try:
                        jd_text = jd_file.read().decode('utf-8', errors='ignore').strip()
                    except Exception:
                        jd_text = ""
            except Exception:
                jd_text = ""

            # Check for empty or invalid JD text after reading
            if not jd_text or len(jd_text.strip()) < 10:
                st.warning("Job description text is empty or too short. Please upload a valid JD file.")
                st.stop()

            # Show extracted JD details
            jd_info = extract_jd_details(jd_text)
            comp = jd_info.get('company') or '—'
            title = jd_info.get('title') or '—'
            yrs = jd_info.get('experience_required_years') or 0
            yrs_str = f"{yrs} yrs" if yrs else '—'
            jd_skills = jd_info.get('skills') or []
            jd_skills_str = ", ".join(jd_skills[:15]) if jd_skills else '—'
            jd_inds = jd_info.get('industries') or []
            jd_inds_str = ", ".join(jd_inds) if jd_inds else '—'
            jd_locs = jd_info.get('locations') or []
            jd_locs_str = ", ".join(jd_locs[:5]) if jd_locs else '—'
            st.markdown('<div class="section-title">Job details (extracted)</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="resume-card" style="border-color:#cfe8ff;">
                  <div style="margin:2px 0;"><b>Company:</b> {comp}</div>
                  <div style="margin:2px 0;"><b>Title/Role:</b> {title}</div>
                  <div style="margin:2px 0;"><b>Experience required:</b> {yrs_str}</div>
                  <div style="margin:2px 0;"><b>Skills in JD:</b> {jd_skills_str}</div>
                  <div style="margin:2px 0;"><b>Industry:</b> {jd_inds_str}</div>
                  <div style="margin:2px 0;"><b>Location(s) in JD:</b> {jd_locs_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # --- Must-have skills and synonyms (compact UI) ---
            with st.expander("Must-have skills (optional)", expanded=False):
                c_mh1, c_mh2 = st.columns([2, 1])
                with c_mh1:
                    _ = st.text_input(
                        "Must-have skills (comma-separated)",
                        value=st.session_state.get('must_have_skills', ''),
                        placeholder='e.g. python, aws, terraform',
                        key='must_have_skills',
                        help="Leave empty to disable gating."
                    )
                with c_mh2:
                    _ = st.checkbox(
                        "Require all",
                        value=bool(st.session_state.get('must_have_require_all', True)),
                        key='must_have_require_all',
                        help="If checked, candidate must contain every listed skill (AND). If unchecked, any one suffices (OR).",
                    )
                with st.expander("Synonyms (optional)"):
                    _ = st.text_area(
                        "Skill synonyms (one per line: canonical: alias1, alias2)",
                        value=st.session_state.get('skill_synonyms_lines', ''),
                        key='skill_synonyms_lines',
                        height=70,
                        help="Example: python: py, python3"
                    )

            # Emphasis & weights (compact with optional fine-tune)
            with st.expander("Emphasis & weights", expanded=False):
                presets = {
                    "Default (60/30/10)": (60, 30, 10),
                    "Skills-heavy (80/15/5)": (80, 15, 5),
                    "Experience-heavy (40/55/5)": (40, 55, 5),
                    "Industry-heavy (50/20/30)": (50, 20, 30),
                    "Custom": None,
                }
                if 'weights_preset' not in st.session_state:
                    st.session_state['weights_preset'] = "Default (60/30/10)"
                # Initialize slider values if missing
                if 'skills_pct' not in st.session_state or 'exp_pct' not in st.session_state or 'ind_pct' not in st.session_state:
                    s0, e0, i0 = presets[st.session_state['weights_preset']] or (60, 30, 10)
                    st.session_state['skills_pct'] = s0
                    st.session_state['exp_pct'] = e0
                    st.session_state['ind_pct'] = i0

                col_p1, col_p2 = st.columns([2, 1])
                with col_p1:
                    chosen = st.selectbox(
                        "Preset",
                        list(presets.keys()),
                        key="weights_preset",
                        help="Choose a preset. Enable Fine-tune to adjust sliders."
                    )
                with col_p2:
                    fine_tune = st.checkbox("Fine-tune sliders", value=bool(st.session_state.get('weights_fine_tune', False)), key='weights_fine_tune')

                # Apply preset to sliders when not fine-tuning or when a non-Custom preset is chosen
                if not fine_tune and chosen != "Custom":
                    s, e, i = presets[chosen]
                    st.session_state['skills_pct'] = s
                    st.session_state['exp_pct'] = e
                    st.session_state['ind_pct'] = i

                # Show sliders only if fine-tuning
                if fine_tune:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        _ = st.slider("Skills %", 0, 100, value=int(st.session_state.get('skills_pct', 60)), step=1, key='skills_pct')
                    with c2:
                        _ = st.slider("Experience %", 0, 100, value=int(st.session_state.get('exp_pct', 30)), step=1, key='exp_pct')
                    with c3:
                        _ = st.slider("Industry %", 0, 100, value=int(st.session_state.get('ind_pct', 10)), step=1, key='ind_pct')
                    # Auto-balance to 100
                    if 'prev_skills_pct' not in st.session_state:
                        st.session_state['prev_skills_pct'] = st.session_state['skills_pct']
                        st.session_state['prev_exp_pct'] = st.session_state['exp_pct']
                        st.session_state['prev_ind_pct'] = st.session_state['ind_pct']
                    cur_s = int(st.session_state['skills_pct']); cur_e = int(st.session_state['exp_pct']); cur_i = int(st.session_state['ind_pct'])
                    diffs = {
                        'skills': abs(cur_s - int(st.session_state['prev_skills_pct'])),
                        'exp': abs(cur_e - int(st.session_state['prev_exp_pct'])),
                        'ind': abs(cur_i - int(st.session_state['prev_ind_pct'])),
                    }
                    changed = max(diffs, key=diffs.get)
                    total_now = cur_s + cur_e + cur_i
                    if total_now != 100:
                        delta = 100 - total_now
                        if changed == 'skills':
                            denom = max(1, cur_e + cur_i)
                            new_e = cur_e + round(delta * (cur_e / denom))
                            new_i = cur_i + (delta - (new_e - cur_e))
                            new_e = max(0, min(100, new_e)); new_i = max(0, min(100, new_i))
                            st.session_state['exp_pct'] = new_e; st.session_state['ind_pct'] = new_i
                            cur_e, cur_i = new_e, new_i
                        elif changed == 'exp':
                            denom = max(1, cur_s + cur_i)
                            new_s = cur_s + round(delta * (cur_s / denom))
                            new_i = cur_i + (delta - (new_s - cur_s))
                            new_s = max(0, min(100, new_s)); new_i = max(0, min(100, new_i))
                            st.session_state['skills_pct'] = new_s; st.session_state['ind_pct'] = new_i
                            cur_s, cur_i = new_s, new_i
                        else:
                            denom = max(1, cur_s + cur_e)
                            new_s = cur_s + round(delta * (cur_s / denom))
                            new_e = cur_e + (delta - (new_s - cur_s))
                            new_s = max(0, min(100, new_s)); new_e = max(0, min(100, new_e))
                            st.session_state['skills_pct'] = new_s; st.session_state['exp_pct'] = new_e
                            cur_s, cur_e = new_s, new_e
                        # Final correction
                        sum2 = cur_s + cur_e + cur_i
                        if sum2 != 100:
                            remainder = 100 - sum2
                            if max(cur_s, cur_e, cur_i) == cur_s:
                                cur_s = max(0, min(100, cur_s + remainder)); st.session_state['skills_pct'] = cur_s
                            elif max(cur_s, cur_e, cur_i) == cur_e:
                                cur_e = max(0, min(100, cur_e + remainder)); st.session_state['exp_pct'] = cur_e
                            else:
                                cur_i = max(0, min(100, cur_i + remainder)); st.session_state['ind_pct'] = cur_i
                    # Update prev trackers
                    st.session_state['prev_skills_pct'] = st.session_state['skills_pct']
                    st.session_state['prev_exp_pct'] = st.session_state['exp_pct']
                    st.session_state['prev_ind_pct'] = st.session_state['ind_pct']

                # Compute weights from either sliders or current preset
                if fine_tune or st.session_state.get('weights_preset') == 'Custom':
                    skills_pct = int(st.session_state['skills_pct']); exp_pct = int(st.session_state['exp_pct']); ind_pct = int(st.session_state['ind_pct'])
                else:
                    s, e, i = presets[st.session_state['weights_preset']] or (60, 30, 10)
                    skills_pct, exp_pct, ind_pct = s, e, i
                total_pct = max(1, skills_pct + exp_pct + ind_pct)
                user_weights = (skills_pct/total_pct, exp_pct/total_pct, ind_pct/total_pct)
                st.session_state['computed_user_weights'] = user_weights

            resume_texts, resume_files, parsed_resume_data = get_resume_texts_and_files()
            if not resume_files:
                st.warning("No resumes found. Please upload some resumes first.")
                st.stop()

            # (FAISS embedding ranking removed as requested)

        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
            st.write("Stack trace:", traceback.format_exc())
        # Refresh resumes; then shortlist using ANN (preferred) or embedding prefilter (fallback) before LLM
        # Ensure we have safe default weights even if earlier UI block failed
        user_weights = st.session_state.get('computed_user_weights', (0.60, 0.30, 0.10))
        resume_texts, resume_files, parsed_resume_data = get_resume_texts_and_files()
        filtered_resume_texts = []
        filtered_resume_files = []
        # Build a pipeline signature and reuse cached shortlist when only pagination changes
        try:
            _jh = _jd_hash(jd_text)
            bm25_en_cfg = bool(st.session_state.get('enable_bm25', False))
            bm25_n_cfg = int(st.session_state.get('bm25_top_docs', 400))
            ann_tc_cfg = int(st.session_state.get('ann_top_chunks', 250))
            mh_csv_cfg = st.session_state.get('must_have_skills', '')
            req_all_cfg = bool(st.session_state.get('must_have_require_all', True))
            syn_lines_cfg = st.session_state.get('skill_synonyms_lines', '')
            ce_en_cfg = bool(st.session_state.get('enable_cross_encoder'))
            ce_model_cfg = st.session_state.get('cross_encoder_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
            ce_topk_cfg = int(st.session_state.get('cross_encoder_topk', 12))
            pipeline_sig = (
                _jh,
                tuple(sorted(resume_files)),
                tuple(round(x, 3) for x in user_weights),
                bm25_en_cfg, bm25_n_cfg, ann_tc_cfg,
                mh_csv_cfg, req_all_cfg, syn_lines_cfg,
                ce_en_cfg, ce_model_cfg, ce_topk_cfg,
            )
            if st.session_state.get('last_filtered_sig') == pipeline_sig:
                # Reuse cached shortlist and highlights to make pagination instant
                filtered_resume_files = list(st.session_state.get('last_filtered_files', []))
                filtered_resume_texts = list(st.session_state.get('last_filtered_texts', []))
                st.session_state['ann_highlights'] = st.session_state.get('last_ann_highlights', {})
                st.session_state['need_recompute'] = False
            else:
                st.session_state['need_recompute'] = True
        except Exception:
            st.session_state['need_recompute'] = True
            pipeline_sig = None
        try:
            # Skip heavy pipeline if we already have a cached shortlist for this JD/settings
            if not st.session_state.get('need_recompute', True):
                raise RuntimeError("__SKIP_PIPELINE__")
            # Optional BM25 prefilter (timed)
            # Default to disabled for better performance; if enabled, use a smaller default top-N
            bm25_enabled = bool(st.session_state.get('enable_bm25', False))
            bm25_topN = int(st.session_state.get('bm25_top_docs', 400))
            _t_bm0 = time.perf_counter()
            bm25_files = bm25_top_docs(jd_text, resume_files, resume_texts, top_n=bm25_topN) if bm25_enabled else []
            st.session_state['bm25_ms'] = int((time.perf_counter() - _t_bm0) * 1000)

            # Preferred path: If BM25 returned candidates, do a fast cosine prefilter restricted to those; else use ANN
            shortlist_indices: list[int] = []
            ann_highlights = {}
            if bm25_enabled and bm25_files:
                _ = update_embedding_cache_for_new_files()
                folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloaded_resumes')
                emb_npy = os.path.join(folder, 'resume_embeddings.npy')
                emb_idx = os.path.join(folder, 'resume_embeddings_index.pkl')
                if os.path.exists(emb_npy) and os.path.exists(emb_idx):
                    with open(emb_idx, 'rb') as f:
                        emb_files = pickle.load(f)
                    embeddings = np.load(emb_npy)
                    if embeddings.dtype != np.float32:
                        embeddings = embeddings.astype(np.float32)
                    bm25_set = set(bm25_files)
                    present = [(i, fn) for i, fn in enumerate(emb_files) if fn in bm25_set]
                    if present:
                        idxs, fns = zip(*present)
                        E = embeddings[list(idxs)]
                        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
                        model = get_st_model()
                        # Cache JD embedding by hash to avoid recomputing across reruns for the same JD
                        try:
                            _emb_cache = st.session_state.setdefault('jd_emb_cache', {})
                            _jh = _jd_hash(jd_text)
                            if _jh in _emb_cache:
                                q = np.asarray(_emb_cache[_jh], dtype=np.float32)
                            else:
                                q = model.encode([jd_text], convert_to_numpy=True, batch_size=DEFAULT_EMBED_BATCH)[0]
                                # store as list to keep session serializable
                                _emb_cache[_jh] = q.astype(np.float32).tolist()
                        except TypeError:
                            q = np.asarray(model.encode([jd_text]))[0]
                            try:
                                _emb_cache = st.session_state.setdefault('jd_emb_cache', {})
                                _jh = _jd_hash(jd_text)
                                _emb_cache[_jh] = q.astype(np.float32).tolist()
                            except Exception:
                                pass
                        qn = q / (np.linalg.norm(q) + 1e-12)
                        sims = En.dot(qn)
                        K = min(100, len(fns))
                        topK_local = np.argsort(-sims)[:K]
                        shortlist_files_fb = [fns[i] for i in topK_local]
                        name_to_idx = {fn: i for i, fn in enumerate(resume_files)}
                        shortlist_indices = [name_to_idx[fn] for fn in shortlist_files_fb if fn in name_to_idx]
            if not shortlist_indices:
                # ANN over chunks (Chroma/Qdrant/Local FAISS)
                _t0 = time.perf_counter()
                # Reduce default ANN top chunks for faster queries
                top_chunks_cfg = int(st.session_state.get('ann_top_chunks', 250))
                if _chroma_enabled():
                    # Lazy one-time build per session (ids deduplicate; safe to call repeatedly)
                    try:
                        if not st.session_state.get('chroma_index_ready'):
                            files_abs = [os.path.join(_RESUME_DIR, f) for f in resume_files]
                            added = chroma_upsert_files(files_abs)
                            st.session_state['chroma_index_ready'] = True
                            if added:
                                st.session_state['chroma_last_added'] = int(added)
                                if bool(st.session_state.get('store_embeddings_only', False)):
                                    try:
                                        removed = purge_originals_if_indexed()
                                        if removed:
                                            st.session_state['last_purge_removed'] = removed
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                    shortlist_out = chroma_query(jd_text, top_chunks=top_chunks_cfg, max_resumes=100, return_highlights=True)
                elif remote_vs_enabled():
                    shortlist_out = remote_vs_query(jd_text, top_chunks=top_chunks_cfg, max_resumes=100, return_highlights=True)
                else:
                    shortlist_out = ann_shortlist_resumes(jd_text, top_chunks=top_chunks_cfg, max_resumes=100, return_highlights=True)
                st.session_state['last_query_ms'] = (time.perf_counter() - _t0) * 1000.0
                shortlist_files = None
                if isinstance(shortlist_out, tuple):
                    shortlist_files, ann_highlights = shortlist_out
                else:
                    shortlist_files = shortlist_out
                if shortlist_files:
                    # If BM25 provided candidates, intersect to sharpen precision (fallback to ANN list if intersection empty)
                    if bm25_enabled and bm25_files:
                        bm25_set2 = set(bm25_files)
                        filtered = [fn for fn in shortlist_files if fn in bm25_set2]
                        shortlist_files = filtered or shortlist_files
                    name_to_idx = {fn: i for i, fn in enumerate(resume_files)}
                    shortlist_indices = [name_to_idx[fn] for fn in shortlist_files if fn in name_to_idx]

            # Rank shortlist by deterministic match score (skills/exp/industry + boosts),
            # but include only resumes with relevant skills overlap and must-have gating
            if shortlist_indices:
                _t_rank0 = time.perf_counter()
                scored = []
                must_csv = st.session_state.get('must_have_skills', '')
                require_all = bool(st.session_state.get('must_have_require_all', True))
                syn_lines = st.session_state.get('skill_synonyms_lines', '')
                for idx in shortlist_indices:
                    rs_text = resume_texts[idx]
                    # Enforce must-have gating if configured
                    if must_csv:
                        try:
                            if not resume_has_required_skills(rs_text, must_csv, syn_lines, require_all):
                                continue
                        except Exception:
                            # If gating helper errors for any reason, skip gating instead of failing the flow
                            pass
                    match_pct, stars, breakdown = compute_match_score(jd_text, rs_text, user_weights)
                    if breakdown.get('skills_overlap_count', 0) > 0:
                        scored.append((match_pct, idx))
                if scored:
                    scored.sort(reverse=True, key=lambda t: t[0])
                    maxM = int(st.session_state.get('rag_max_results', 50))
                    M = min(maxM, len(scored))
                    final_idxs = [idx for _, idx in scored[:M]]
                    filtered_resume_texts = [resume_texts[i] for i in final_idxs]
                    filtered_resume_files = [resume_files[i] for i in final_idxs]
                    # Store highlights for UI
                    st.session_state['ann_highlights'] = {fn: ann_highlights.get(fn, []) for fn in filtered_resume_files}
                st.session_state['rank_ms'] = int((time.perf_counter() - _t_rank0) * 1000)
        except Exception:
            # Fall back to length-based shortlist if embeddings are unavailable
            pass
        # If none selected yet (e.g., cache missing or no shortlist hits),
        # do a skills-based scan over all resumes and still enforce relevant-skills filter and must-have gating
        if not filtered_resume_texts and st.session_state.get('need_recompute', True):
            scored_all = []
            must_csv = st.session_state.get('must_have_skills', '')
            require_all = bool(st.session_state.get('must_have_require_all', True))
            syn_lines = st.session_state.get('skill_synonyms_lines', '')
            for i in range(len(resume_texts)):
                # Enforce must-have gating if configured
                if must_csv:
                    try:
                        if not resume_has_required_skills(resume_texts[i], must_csv, syn_lines, require_all):
                            continue
                    except Exception:
                        pass
                mp, stars_local, br = compute_match_score(jd_text, resume_texts[i], user_weights)
                if br.get('skills_overlap_count', 0) > 0:
                    scored_all.append((mp, i))
            if scored_all:
                scored_all.sort(reverse=True, key=lambda t: t[0])
                maxM = int(st.session_state.get('rag_max_results', 50))
                M = min(maxM, len(scored_all))
                final_idxs = [i for _, i in scored_all[:M]]
                filtered_resume_texts = [resume_texts[i] for i in final_idxs]
                filtered_resume_files = [resume_files[i] for i in final_idxs]
            else:
                st.info("No resumes with relevant skills found for this job description. Nothing to display.")
                st.stop()
        # Save shortlist cache for fast pagination if we computed it this run
        try:
            if st.session_state.get('need_recompute', True) is True and filtered_resume_files:
                st.session_state['last_filtered_files'] = list(filtered_resume_files)
                st.session_state['last_filtered_texts'] = list(filtered_resume_texts)
                st.session_state['last_ann_highlights'] = st.session_state.get('ann_highlights', {})
                st.session_state['last_filtered_sig'] = pipeline_sig
        except Exception:
            pass
    def ollama_mistral_match(jd_text, resume_text):
            import requests, re, json, os
            jd_short = jd_text[:600]
            resume_short = resume_text[:600]
            prompt = (
                f"Job description:\n{jd_short}\n\n"
                f"Candidate resume:\n{resume_short}\n\n"
                "Rate the match between this candidate and the job on a scale of 0-100. "
                "Give a short summary (40-60 words) focused on skills, experience, and gaps. "
                "Return ONLY a single JSON object with double quotes and no extra text, strictly this shape: "
                '{"score": 75, "summary": "..."}'
            )

            def parse_llm_json_response(text: str):
                s = (text or "").strip()
                if not s:
                    return {"score": 0, "summary": ""}
                # Try direct JSON
                try:
                    return json.loads(s)
                except Exception:
                    pass
                # Try extract JSON block and normalize quotes
                m = re.search(r"\{.*\}", s, flags=re.DOTALL)
                if m:
                    cand = m.group(0)
                    cand2 = re.sub(r"'", '"', cand)
                    cand2 = re.sub(r",\s*([}\]])", r"\1", cand2)  # remove trailing commas
                    try:
                        return json.loads(cand2)
                    except Exception:
                        pass
                # Fallback: extract score number and return raw text as summary
                m = re.search(r"score\D+(\d{1,3})", s, flags=re.IGNORECASE)
                score = int(m.group(1)) if m else 0
                return {"score": score, "summary": s}

            try:
                base_url = st.session_state.get('llm_base_url') or os.environ.get('LLM_BASE_URL') or "http://67.11.191.239:11434"
                model_id = st.session_state.get('llm_model') or os.environ.get('LLM_MODEL') or "llama3.1:8b-instruct-q4_K_M"
                timeout_s = int(st.session_state.get('llm_timeout', 60))
                r = requests.post(
                    f"{base_url.rstrip('/')}/api/generate",
                    json={
                        "model": model_id,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": 0.2, "max_tokens": 300},
                    },
                    timeout=timeout_s,
                )
                r.raise_for_status()
                data = r.json()
                response = data.get("response", "")
                result = parse_llm_json_response(response)
                score = int(result.get("score", 0) or 0)
                # Normalize score to 0..100
                score = max(0, min(100, score))
                summary = str(result.get("summary", "") or "").strip()
                # Strip accidental wrapping quotes and collapse whitespace
                summary = re.sub(r'^\s*["\']|["\']\s*$', '', summary)
                summary = re.sub(r"\s+", " ", summary)
                return score, summary
            except Exception as e:
                # Graceful fallback: deterministic summary when LLM is unreachable
                try:
                    return 0, deterministic_summary(jd_text, resume_text)
                except Exception:
                    return 0, f"[LLM error: {e}]"
        # Title
        st.markdown('<div class="section-title">Top Matches</div>', unsafe_allow_html=True)
        # Optional cross-encoder re-ranking before display
        if st.session_state.get('enable_cross_encoder') and filtered_resume_texts:
            try:
                model_name = st.session_state.get('cross_encoder_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
                # limit to a small multiple of page size to reduce latency
                page_sz = int(st.session_state.get('rag_view_limit', 5) or 5)
                default_topk = int(st.session_state.get('cross_encoder_topk', 12))
                topk = max(5, min(default_topk, page_sz * 3))
                ordered_files, ordered_texts = cross_encoder_rerank(jd_text, filtered_resume_texts, filtered_resume_files, model_name, top_k=topk)
                if ordered_texts:
                    filtered_resume_texts = ordered_texts
                    filtered_resume_files = ordered_files
            except Exception:
                pass

        # Build display for current page only; summaries generated lazily per item
        if 'rag_view_limit' not in st.session_state:
            st.session_state['rag_view_limit'] = 5
        view_limit = int(st.session_state.get('rag_view_limit', 5))
        total_items = len(filtered_resume_files)
        _t_render0 = time.perf_counter()
        # Generate LLM summaries for visible items in parallel (with cache)
        vis_n = min(view_limit, total_items)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results: list[tuple[int, str]] = [(-1, "") for _ in range(vis_n)]
        # First, fill from cache
        pending = []
        for i in range(vis_n):
            fname = filtered_resume_files[i]
            resume_text = filtered_resume_texts[i]
            got = get_llm_summary_cached(jd_text, resume_text, fname)
            if got is not None:
                results[i] = got
            else:
                pending.append((i, fname, resume_text))
        # Compute pending in parallel
        if pending:
            def _gen_one(idx, fname, rtxt):
                sc, sm = ollama_mistral_match(jd_text, rtxt)
                set_llm_summary_cache(jd_text, fname, sc, sm)
                return idx, (sc, sm)
            maxw = max(2, min(8, (os.cpu_count() or 4)))
            with ThreadPoolExecutor(max_workers=maxw) as ex:
                futs = [ex.submit(_gen_one, idx, fn, rt) for idx, fn, rt in pending]
                for fut in as_completed(futs):
                    try:
                        idx, tup = fut.result()
                        results[idx] = tup
                    except Exception:
                        pass
        st.session_state['llm_ms'] = int((time.perf_counter() - _t_render0) * 1000)

        for i in range(vis_n):
            fname = filtered_resume_files[i]
            resume_text = filtered_resume_texts[i]
            match_pct, stars, breakdown = compute_match_score(jd_text, resume_text, user_weights)
            facts = retrieve_resume_facts(resume_text)
            exp_str = f"{facts['years']} yrs" if facts.get('years', 0) else ""
            skills_list = facts.get('skills', [])
            skills_str = ", ".join(skills_list[:8]) if skills_list else ""
            matched_skills = breakdown.get('common_skills', [])
            matched_str = ", ".join(matched_skills[:8]) if matched_skills else ""
            inds = breakdown.get('common_industries', [])
            inds_str = ", ".join(inds) if inds else ""
            jd_locs_disp = breakdown.get('jd_locations', [])
            jd_locs_str = ", ".join(jd_locs_disp[:3]) if jd_locs_disp else "—"
            rs_locs_disp = breakdown.get('resume_locations', [])
            rs_locs_str = ", ".join(rs_locs_disp[:3]) if rs_locs_disp else "—"
            contact = facts.get('contact', {}) or {}
            contact_parts = []
            if contact.get('email'): contact_parts.append(contact['email'])
            if contact.get('phone'): contact_parts.append(contact['phone'])
            if contact.get('linkedin'): contact_parts.append('LinkedIn')
            if contact.get('github'): contact_parts.append('GitHub')
            contact_str = " | ".join(contact_parts)
            line1_parts = []
            if contact_str:
                line1_parts.append(f"<b>Contact:</b> {contact_str}")
            if exp_str:
                line1_parts.append(f"<b>Experience:</b> {exp_str}")
            line1_html = " &nbsp;•&nbsp; ".join(line1_parts)
            line2_html = f"<b>Matched skills:</b> {matched_str}" if matched_str else ""
            line3_html = f"<b>Core skills:</b> {skills_str}" if skills_str else ""
            line4_html = f"<b>Industry:</b> {inds_str}" if inds_str else ""
            line5_html = f"<b>Location:</b> JD — {jd_locs_str} &nbsp;•&nbsp; Candidate — {rs_locs_str}"
            facts_html = ""
            if line1_html:
                facts_html += f"<div class='facts-line' style='font-size:1.05em;margin:2px 0;'>{line1_html}</div>"
            if line2_html:
                facts_html += f"<div class='facts-line' style='font-size:1.05em;margin:2px 0;'>{line2_html}</div>"
            if line3_html:
                facts_html += f"<div class='facts-line' style='font-size:1.05em;margin:2px 0;'>{line3_html}</div>"
            if line4_html:
                facts_html += f"<div class='facts-line' style='font-size:1.05em;margin:2px 0;'>{line4_html}</div>"
            if line5_html:
                facts_html += f"<div class='facts-line' style='font-size:1.05em;margin:2px 0;'>{line5_html}</div>"
            summary_text = results[i][1] if results[i][1] else "(no summary)"
            preview = resume_text[:800] + ("..." if len(resume_text) > 800 else "")
            st.markdown(
                f'<div class="resume-card"><b>{fname}</b> '
                f'<span class="sim-score">{stars} ({match_pct}%)</span><br>'
                f'{facts_html}'
                f'<b>Summary:</b> {summary_text}',
                unsafe_allow_html=True)
            ac1, ac2, ac3 = st.columns([1,1,1])
            state_key = f"view_match_state_{i}_{fname}"
            with ac1:
                if st.button("👁️ View", key=f"view_match_btn_{i}_{fname}"):
                    st.session_state[state_key] = not st.session_state.get(state_key, False)
            with ac2:
                resume_path = os.path.join('downloaded_resumes', fname)
                if os.path.exists(resume_path):
                    with open(resume_path, "rb") as file:
                        st.download_button(
                            label="Download",
                            data=file,
                            file_name=fname,
                            mime="application/octet-stream",
                            key=f"download_match_{i}_{fname}"
                        )
                else:
                    # MySQL fallback download or TXT fallback
                    def _mysql_download_blob_for_file(name: str) -> bytes | None:
                        try:
                            meta = _chroma_get_any_metadata_for_file(name) or {}
                            mysql_id = meta.get('mysql_id')
                            if mysql_id is None:
                                return None
                            data = mysql_fetch_blob_by_id(mysql_id)
                            return data
                        except Exception:
                            return None
                    blob = _mysql_download_blob_for_file(fname)
                    if blob:
                        st.download_button(
                            label="Download",
                            data=blob,
                            file_name=fname,
                            mime="application/octet-stream",
                            key=f"download_match_mysql_{i}_{fname}"
                        )
                    else:
                        # Fallback to text content
                        st.download_button(
                            label="Download as .txt",
                            data=resume_text.encode('utf-8', errors='ignore'),
                            file_name=os.path.splitext(fname)[0] + ".txt",
                            mime="text/plain",
                            key=f"download_match_txt_{i}_{fname}"
                        )
            with ac3:
                # No external storage links
                st.empty()
            hi = (st.session_state.get('ann_highlights') or {}).get(fname) or []
            if hi:
                with st.expander("Top matching snippets", expanded=False):
                    for snip, sim in hi:
                        st.markdown(f"- {snip} <span style='color:#888'>(sim {sim:.2f})</span>", unsafe_allow_html=True)
            if st.session_state.get(state_key, False):
                with st.expander("Preview", expanded=True):
                    resume_path_exp = os.path.join('downloaded_resumes', fname)
                    if fname.lower().endswith('.pdf') and os.path.exists(resume_path_exp):
                        with open(resume_path_exp, 'rb') as pdf_file:
                            base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
                            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="900" type="application/pdf"></iframe>'
                            st.markdown(pdf_display, unsafe_allow_html=True)
                    else:
                        st.markdown(f"<pre style='font-size:0.9em;'>{preview}</pre>", unsafe_allow_html=True)
            else:
                with st.expander("Preview", expanded=False):
                    st.markdown(f"<pre style='font-size:0.9em;'>{preview}</pre>", unsafe_allow_html=True)

        st.session_state['render_ms'] = int((time.perf_counter() - _t_render0) * 1000)
        try:
            ann_ms = float(st.session_state.get('last_query_ms', 0.0))
            bm_ms = int(st.session_state.get('bm25_ms', 0))
            rk_ms = int(st.session_state.get('rank_ms', 0))
            rd_ms = int(st.session_state.get('render_ms', 0))
            llm_ms = int(st.session_state.get('llm_ms', 0))
            st.caption(f"Perf: BM25 {bm_ms} ms • ANN {ann_ms:.1f} ms • Rank {rk_ms} ms • LLM {llm_ms} ms • Render {rd_ms} ms")
        except Exception:
            pass

        # See more / Show less controls
        if total_items > view_limit:
            if st.button("See more matches", key="see_more_rag"):
                st.session_state['rag_view_limit'] = min(total_items, view_limit + 5)
                st.rerun()
        elif total_items > 5 and view_limit > 5:
            if st.button("Show less", key="show_less_rag"):
                st.session_state['rag_view_limit'] = 5
                st.rerun()

    else:
        st.info("Upload a job description to start RAG-based resume matching.")

def ollama_summarize(prompt, model="llama3.1:8b-instruct-q4_K_M"):
    import requests, os
    base_url = st.session_state.get('llm_base_url') or os.environ.get('LLM_BASE_URL') or "http://67.11.191.239:11434"
    model_id = st.session_state.get('llm_model') or os.environ.get('LLM_MODEL') or model
    timeout_s = int(st.session_state.get('llm_timeout', 180))
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model_id,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload, timeout=timeout_s)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        return f"Error from remote LLM: {e}"

with main_tabs[1]:
    st.markdown('<div class="big-title">Manage resumes</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Quick start: upload or sync</div>', unsafe_allow_html=True)
    # Status metrics
    total_cnt, embedded_cnt, remaining_cnt = get_resume_cache_status()
    m1, m2, m3 = st.columns(3)
    m1.metric("Total resumes", total_cnt)
    m2.metric("Embedded (cached)", embedded_cnt)
    m3.metric("Remaining to embed", remaining_cnt)
    try:
        ratio = (embedded_cnt / total_cnt) if total_cnt > 0 else 0.0
        st.progress(ratio, text=f"Embedding cache: {embedded_cnt}/{total_cnt}")
    except Exception:
        pass

    
    # --- Bulk Import: ZIP of resumes ---
    st.markdown('<div class="section-title">Import ZIP of resumes</div>', unsafe_allow_html=True)
    zip_up = st.file_uploader("Import a ZIP containing PDF/DOCX/TXT resumes", type=["zip"], key="zip_import_resumes")
    if zip_up is not None:
        try:
            dest_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'downloaded_resumes')
            os.makedirs(dest_folder, exist_ok=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                tmp.write(zip_up.read())
                tmp_path = tmp.name
            count_extracted = 0
            extracted_names = []
            with zipfile.ZipFile(tmp_path, 'r') as zf:
                for member in zf.infolist():
                    name = member.filename
                    if member.is_dir():
                        continue
                    lower = name.lower()
                    if not lower.endswith((".pdf", ".docx", ".txt")):
                        continue
                    # Avoid path traversal
                    base = os.path.basename(name)
                    if not base:
                        continue
                    target = os.path.join(dest_folder, base)
                    # If same filename exists, append a numeric suffix
                    if os.path.exists(target):
                        root, ext = os.path.splitext(base)
                        k = 1
                        while os.path.exists(os.path.join(dest_folder, f"{root}_{k}{ext}")):
                            k += 1
                        base = f"{root}_{k}{ext}"
                        target = os.path.join(dest_folder, base)
                    with zf.open(member) as src, open(target, 'wb') as out:
                        out.write(src.read())
                    count_extracted += 1
                    extracted_names.append(base)
            # Update embedding cache only for these extracted files
            added = update_embedding_cache_for_new_files(dest_folder, extracted_names)
            t, e, r = get_resume_cache_status()
            # S3 upload removed
            st.success(f"Imported {count_extracted} file(s) from ZIP. Cached embeddings for {added} new resume(s). Total: {t}, Embedded: {e}, Remaining: {r}.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to import ZIP: {e}")

    # Manual update cache button (incremental)
    if st.button("Update Embedding Cache (Incremental)", key="update_embed_cache", help="Embed any new resumes added since the last cache update."):
        try:
            added = update_embedding_cache_for_new_files()
            t, e, r = get_resume_cache_status()
            st.success(f"Embedding cache updated. {added} new resume(s) embedded. Total: {t}, Embedded: {e}, Remaining: {r}.")
        except Exception as e:
            st.error(f"Failed to update embedding cache: {e}")

    # --- Direct Resume Upload ---
    st.markdown('<div class="section-title">Upload resumes (PDF/DOCX/TXT)</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload one or more resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="resume_upload_tab3")
    if uploaded_files:
        os.makedirs('downloaded_resumes', exist_ok=True)
        saved_names = []
        for up_file in uploaded_files:
            # Deduplicate by renaming if file exists
            base = up_file.name
            target = os.path.join('downloaded_resumes', base)
            if os.path.exists(target):
                root, ext = os.path.splitext(base)
                k = 1
                while os.path.exists(os.path.join('downloaded_resumes', f"{root}_{k}{ext}")):
                    k += 1
                base = f"{root}_{k}{ext}"
                target = os.path.join('downloaded_resumes', base)
            with open(target, "wb") as f:
                f.write(up_file.read())
            saved_names.append(base)
        # Update embedding cache only for these files
        added = update_embedding_cache_for_new_files('downloaded_resumes', saved_names)
        t, e, r = get_resume_cache_status()
        # S3 upload removed
        st.success(f"Uploaded {len(uploaded_files)} file(s). Cached embeddings for {added} new resume(s). Total: {t}, Embedded: {e}, Remaining: {r}.")
        st.rerun()

    # Sync from Gmail (improved UI)
    with st.expander("📧 Sync from Gmail", expanded=False):
        # Determine current Gmail sign-in status
        gmail_user = None
        import shutil
        if os.path.exists('token.pickle'):
            try:
                with open('token.pickle', 'rb') as token:
                    creds = pickle.load(token)
                try:
                    from googleapiclient.discovery import build
                except Exception:
                    build = None
                service = build('gmail', 'v1', credentials=creds) if build else None
                profile = service.users().getProfile(userId='me').execute()
                gmail_user = profile.get('emailAddress')
            except Exception:
                gmail_user = None

        # Header row with status and actions
        c_status, c_actions = st.columns([2, 2])
        with c_status:
            if gmail_user:
                st.markdown(f"🟢 <b>Connected:</b> {gmail_user}", unsafe_allow_html=True)
            else:
                st.markdown("🔴 <b>Not connected</b>", unsafe_allow_html=True)
                st.caption("Place credentials.json in this app's folder to enable Gmail OAuth.")

        with c_actions:
            ac1, ac2 = st.columns([1,1])
            with ac1:
                if gmail_user:
                    if st.button("Sign out", key="signout_gmail_tab1", help="Forget your Gmail token; next sync will prompt login."):
                        try:
                            os.remove('token.pickle')
                        except Exception:
                            pass
                        st.success("Signed out. Next sync will prompt for Gmail login.")
                        st.rerun()
                else:
                    if st.button("Sign in", key="login_gmail_tab1", help="Open a browser window to grant Gmail read access."):
                        with st.spinner("Opening Gmail sign-in window..."):
                            try:
                                download_attachments_from_gmail()
                                st.success("Gmail sign-in complete. You can now sync resumes.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Gmail sign-in failed: {e}")
            with ac2:
                if st.button("Sync attachments", key="sync_gmail_tab1", help="Fetch recent Gmail attachments and save as resumes."):
                    with st.spinner("Syncing resumes from Gmail and storing locally..."):
                        count, folder = download_attachments_from_gmail()
                        if count > 0:
                            st.success(f"Downloaded and stored {count} new resumes from Gmail in '{folder}' folder.")
                            # S3 upload removed
                        else:
                            st.info("No new resumes were downloaded from Gmail.")

        st.caption("Scope: Gmail read-only. We only fetch attachments with extensions: PDF, DOCX, TXT.")

    # Delete all resumes button (kept outside Gmail expander)
    action_cols = st.columns([1,1,1])
    with action_cols[2]:
        if os.path.exists('downloaded_resumes'):
            if st.button("Delete all", key="delete_all_resumes_action_row", help="Remove all local resumes from disk."):
                import shutil
                try:
                    shutil.rmtree('downloaded_resumes')
                    os.makedirs('downloaded_resumes', exist_ok=True)
                    st.success("All resumes deleted.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete all resumes: {e}")
    if os.path.exists('downloaded_resumes'):
        def extract_resume_info(text):
            import re
            lines = text.splitlines()
            name = lines[0].strip() if lines else "N/A"
            phone_match = re.search(r'(\+?\d[\d\s\-]{7,}\d)', text)
            phone = phone_match.group(0) if phone_match else "N/A"
            email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
            email = email_match.group(0) if email_match else "N/A"
            skills_match = re.search(r'Skills[:\s]*([\w\s,\-]+)', text, re.IGNORECASE)
            skills = skills_match.group(1).strip() if skills_match else "N/A"
            exp_match = re.search(r'Experience[:\s]*([\w\s,\-]+)', text, re.IGNORECASE)
            experience = exp_match.group(1).strip() if exp_match else "N/A"
            return name, phone, email, skills, experience

        # Resume listing disabled by request
        # If needed later, this area can display summary metrics or actions only.
        pass

    


# --- Keyword Matcher Tab removed ---




with main_tabs[2]:
    st.markdown('<div class="big-title">Settings</div>', unsafe_allow_html=True)
    # Build info and quick tip to locate MySQL expander (helps confirm correct file is running)
    # Hidden per request
    try:
        _ = os.path.abspath(__file__)
        _ = os.path.getmtime(__file__)
    except Exception:
        pass
    # Inline Status (moved to top, not collapsible)
    try:
        tot_chunks = 0
        if os.path.exists(_CHUNK_MAP_PATH):
            with open(_CHUNK_MAP_PATH, 'rb') as f:
                mp = pickle.load(f)
                tot_chunks += len(mp)
        for _, mp in _list_shard_files():
            try:
                with open(mp, 'rb') as f:
                    tot_chunks += len(pickle.load(f))
            except Exception:
                pass
        dev = 'cuda' if 'cuda' in str(getattr(get_st_model(), 'device', 'cpu')).lower() else 'cpu'
        c1, c2, c3 = st.columns(3)
        c1.metric("Chunks indexed", tot_chunks)
        c2.metric("Device", dev)
        c3.metric("Embed batch size", int(st.session_state.get('embed_batch_size', DEFAULT_EMBED_BATCH)))
        if 'last_query_ms' in st.session_state:
            st.caption(f"Last ANN query latency: {st.session_state['last_query_ms']:.1f} ms")
        st.caption(f"Vector store: {st.session_state.get('vector_store_mode', 'Chroma (beta)')}")
        # Storage status simplified (no cloud)
    except Exception:
        st.caption("Metrics unavailable.")
    # Embedding cache actions
    with st.expander("Embedding cache"):
        st.write("Maintain the embedding cache for faster scoring.")
        if st.button("Compact embedding cache (drop orphans)", key="compact_embed_cache_settings", help="Remove embeddings for files that no longer exist on disk and shrink the cache file."):
            try:
                idx_path = embedding_idx_path
                npy_path = embedding_npy_path
                if not (os.path.exists(idx_path) and os.path.exists(npy_path)):
                    st.info("No existing embedding cache to compact.")
                else:
                    with open(idx_path, 'rb') as f:
                        emb_files = pickle.load(f)
                    embeddings = np.load(npy_path)
                    if embeddings.dtype != np.float32:
                        embeddings = embeddings.astype(np.float32)
                    present_pairs = [(i, fn) for i, fn in enumerate(emb_files) if os.path.exists(os.path.join(_RESUME_DIR, fn))]
                    if not present_pairs:
                        new_E = np.zeros((0, 384), dtype=np.float32)
                        new_idx = []
                    else:
                        idxs, fns = zip(*present_pairs)
                        new_E = embeddings[list(idxs)]
                        new_idx = list(fns)
                    to_store = new_E.astype(np.float16) if USE_FLOAT16_EMBEDS else new_E.astype(np.float32)
                    np.save(npy_path, to_store)
                    with open(idx_path, 'wb') as f:
                        pickle.dump(new_idx, f)
                    t, e, r = get_resume_cache_status()
                    st.success(f"Compacted cache. Total: {t}, Embedded: {e}, Remaining: {r}.")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to compact cache: {e}")

    # Search index actions
    with st.expander("Search index"):
        st.write("Build/update the vector index for fast shortlist (ANN).")
        # Compact vector store selector (Chroma default)
        row_vs = st.columns([2,2,2])
        with row_vs[0]:
            vs_options = ['Chroma (beta)', 'Local FAISS', 'Qdrant (beta)']
            current_vs = st.session_state.get('vector_store_mode', 'Chroma (beta)')
            try:
                default_index = vs_options.index(current_vs) if current_vs in vs_options else 0
            except Exception:
                default_index = 0
            st.session_state['vector_store_mode'] = st.selectbox(
                "Vector store",
                options=vs_options,
                index=default_index,
                help="Chroma (recommended), or choose FAISS/Qdrant."
            )
        if st.session_state.get('vector_store_mode') == 'Qdrant (beta)':
            with st.expander("Qdrant options (optional)"):
                c6, c7, c8 = st.columns(3)
                with c6:
                    st.session_state['qdrant_host'] = st.text_input("Host", value=st.session_state.get('qdrant_host', 'localhost'))
                    st.session_state['qdrant_collection'] = st.text_input("Collection", value=st.session_state.get('qdrant_collection', 'resume_chunks'))
                with c7:
                    st.session_state['qdrant_port'] = st.number_input("Port", min_value=1, max_value=65535, value=int(st.session_state.get('qdrant_port', 6333)))
                    st.session_state['qdrant_api_key'] = st.text_input("API key (optional)", value=st.session_state.get('qdrant_api_key', ''), type='password')
                with c8:
                    st.caption("Ensure your Qdrant instance is reachable. Vectors use COSINE distance, dim=384.")
        if st.session_state.get('vector_store_mode') in ('Chroma (beta)', 'Chroma (local)'):
            with st.expander("Chroma options (optional)", expanded=False):
                c6a, c7a = st.columns(2)
                with c6a:
                    st.session_state['chroma_path'] = st.text_input(
                        "Store path",
                        value=st.session_state.get('chroma_path', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chroma_store')),
                        help="Directory for Chroma persistent store."
                    )
                with c7a:
                    st.session_state['chroma_collection'] = st.text_input(
                        "Collection",
                        value=st.session_state.get('chroma_collection', 'resume_chunks'),
                        help="Chroma collection for resume chunks."
                    )
        colA, colB, colC = st.columns([1,1,1])
        with colA:
            if st.button("Build/Update index", key="build_ann_index_settings", help="Add new resume chunks to the vector index so ANN search is fast."):
                try:
                    if st.session_state.get('vector_store_mode') == 'Qdrant (beta)':
                        files = _list_resume_files(_RESUME_DIR)
                        added = remote_vs_upsert(files)
                    elif st.session_state.get('vector_store_mode') in ('Chroma (beta)', 'Chroma (local)'):
                        files = _list_resume_files(_RESUME_DIR)
                        files_abs = [os.path.join(_RESUME_DIR, f) for f in files]
                        added = chroma_upsert_files(files_abs)
                        st.session_state['chroma_index_ready'] = True
                    else:
                        added = ingest_new_files_to_hnsw()
                    if added > 0:
                        st.success(f"Indexed {added} chunk(s).")
                        # Optional auto-purge when embeddings-only mode is enabled
                        if bool(st.session_state.get('store_embeddings_only', False)) and st.session_state.get('vector_store_mode') in ('Chroma (beta)', 'Chroma (local)'):
                            try:
                                removed = purge_originals_if_indexed()
                                if removed:
                                    st.caption(f"Embeddings-only: removed {removed} original file(s) after indexing.")
                            except Exception:
                                pass
                    else:
                        st.info("No new resumes to index or vector store not available.")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
        with colB:
            if st.button("Rebuild index", key="rebuild_ann_index_settings", help="Recreate the index from current resumes (drops stale entries)."):
                try:
                    if st.session_state.get('vector_store_mode') == 'Qdrant (beta)':
                        client = _get_qdrant_client()
                        if client is None:
                            st.warning("Qdrant client not available.")
                        else:
                            collection = st.session_state.get('qdrant_collection', 'resume_chunks')
                            try:
                                client.delete_collection(collection)
                            except Exception:
                                pass
                            cnt = remote_vs_upsert(_list_resume_files(_RESUME_DIR))
                            st.success(f"Rebuilt remote collection '{collection}'. Added {cnt} chunks.")
                    elif st.session_state.get('vector_store_mode') in ('Chroma (beta)', 'Chroma (local)'):
                        try:
                            import chromadb
                            base_path = st.session_state.get('chroma_path') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chroma_store')
                            client = chromadb.PersistentClient(path=base_path)
                            coll_name = st.session_state.get('chroma_collection', 'resume_chunks')
                            try:
                                client.delete_collection(coll_name)
                            except Exception:
                                pass
                        except Exception as e:
                            st.error(f"Chroma client not available: {e}")
                        files = _list_resume_files(_RESUME_DIR)
                        files_abs = [os.path.join(_RESUME_DIR, f) for f in files]
                        cnt = chroma_upsert_files(files_abs)
                        st.session_state['chroma_index_ready'] = True
                        st.success(f"Rebuilt Chroma collection. Added {cnt} chunks.")
                    else:
                        removed = 0
                        for p in [_CHUNK_INDEX_PATH, _CHUNK_MAP_PATH, _SHARD_MANIFEST_PATH]:
                            if os.path.exists(p):
                                try:
                                    os.remove(p); removed += 1
                                except Exception:
                                    pass
                        for ip, mp in _list_shard_files():
                            for p in [ip, mp]:
                                if os.path.exists(p):
                                    try:
                                        os.remove(p); removed += 1
                                    except Exception:
                                        pass
                        cnt = ingest_new_files_to_hnsw(files=_list_resume_files(_RESUME_DIR))
                        st.success(f"Rebuilt index. Added {cnt} chunk(s). Removed {removed} old file(s).")
                except Exception as e:
                    st.error(f"Failed to rebuild index: {e}")
        with colC:
            # Compact status by selected store
            vs = st.session_state.get('vector_store_mode', 'Chroma (beta)')
            if vs == 'Local FAISS':
                if os.path.exists(_CHUNK_INDEX_PATH) and os.path.exists(_CHUNK_MAP_PATH):
                    st.caption("FAISS index present: chunks_hnsw.faiss + chunks_map.pkl")
                else:
                    shard_list = _list_shard_files()
                    if shard_list:
                        tot = 0
                        for _, mp in shard_list:
                            try:
                                with open(mp, 'rb') as f:
                                    tot += len(pickle.load(f))
                            except Exception:
                                pass
                        st.caption(f"FAISS shards: {len(shard_list)} shard(s), {tot} chunks.")
                    else:
                        st.caption("FAISS index not found.")
            elif vs in ('Chroma (beta)', 'Chroma (local)'):
                st.caption(f"Chroma: {st.session_state.get('chroma_collection', 'resume_chunks')} @ {os.path.basename(st.session_state.get('chroma_path', 'chroma_store'))}")
            else:
                st.caption("Qdrant: remote collection")

        # Advanced search options grouped
        with st.expander("Advanced search settings"):
            # Build new shard (advanced)
            if st.session_state.get('vector_store_mode') == 'Local FAISS' and st.button("Build new shard (FAISS)", key="build_new_shard_settings", help="Create a new FAISS shard for files not yet in any shard (advanced)."):
                try:
                    added, shard_name = index_new_shard_from_remaining()
                    if added > 0 and shard_name:
                        st.success(f"Created shard {shard_name} with {added} chunk(s).")
                    else:
                        st.info("No remaining files to shard or FAISS not installed.")
                except Exception as e:
                    st.error(f"Shard creation failed: {e}")

            # Cross-encoder options
            st.markdown("<div class='section-title'>Cross-encoder re-ranker</div>", unsafe_allow_html=True)
            c1x, c2x = st.columns([1,1])
            with c1x:
                st.session_state['enable_cross_encoder'] = st.checkbox(
                    "Enable cross-encoder re-ranker",
                    value=bool(st.session_state.get('enable_cross_encoder', False)),
                    help="Run a Cross-Encoder over the top candidates to improve final ranking (higher accuracy, added latency).")
            with c2x:
                st.session_state['cross_encoder_model'] = st.selectbox(
                    "Cross-encoder model",
                    options=[
                        'cross-encoder/ms-marco-MiniLM-L-6-v2',
                        'cross-encoder/ms-marco-MiniLM-L-12-v2'
                    ],
                    index=0,
                    help="Model used for cross-encoder re-ranking."
                )
            st.session_state['cross_encoder_topk'] = st.number_input(
                "Cross-encoder top-k",
                min_value=5,
                max_value=200,
                value=int(st.session_state.get('cross_encoder_topk', 30)),
                step=5,
                help="How many top candidates to re-rank with the cross-encoder (applies after deterministic filtering)."
            )

            # Hybrid retrieval options
            st.markdown("<div class='section-title'>Hybrid retrieval</div>", unsafe_allow_html=True)
            c_h1, c_h2 = st.columns([1,1])
            with c_h1:
                st.session_state['enable_bm25'] = st.checkbox(
                    "Enable BM25 lexical prefilter",
                    value=bool(st.session_state.get('enable_bm25', True)),
                    help="Use a lexical BM25 pass to find top documents before vector shortlist. Speeds up and often improves precision.")
            with c_h2:
                st.session_state['bm25_top_docs'] = st.number_input(
                    "BM25 top docs",
                    min_value=200,
                    max_value=20000,
                    value=int(st.session_state.get('bm25_top_docs', 2000)),
                    step=100,
                    help="How many documents to keep from the fast BM25 pass before vector filtering.")

    # Storage mode
    with st.expander("Storage mode"):
        st.session_state['store_embeddings_only'] = st.checkbox(
            "Store only embeddings (delete originals after indexing)",
            value=bool(st.session_state.get('store_embeddings_only', False)),
            help="After successful indexing into the vector store, delete local resume files to save disk space. Previews still work from Chroma documents."
        )
        st.session_state['chroma_store_documents'] = st.checkbox(
            "Store chunk text in Chroma (recommended)",
            value=bool(st.session_state.get('chroma_store_documents', True)),
            help="Keep chunk text in Chroma so you can preview and quote snippets even when originals are deleted."
        )
        if st.button("Purge indexed originals now", key="purge_indexed_originals_btn"):
            try:
                removed = purge_originals_if_indexed()
                if removed > 0:
                    st.success(f"Removed {removed} local file(s) that were already indexed.")
                else:
                    st.info("No indexed originals found to purge.")
            except Exception as e:
                st.error(f"Purge failed: {e}")

    # Remote import (MySQL)
    with st.expander("Remote import (MySQL)"):
        st.write("Sync resumes from a MySQL database into the app, then embed. You can also index directly into Chroma without storing local files.")
        # One-time load of saved defaults into session state
        if not st.session_state.get('mysql_settings_loaded'):
            saved = _load_mysql_settings()
            if saved:
                # Apply saved defaults if not already set in this session
                st.session_state['mysql_host'] = st.session_state.get('mysql_host', saved.get('host', ''))
                st.session_state['mysql_port'] = st.session_state.get('mysql_port', saved.get('port', 3306))
                st.session_state['mysql_user'] = st.session_state.get('mysql_user', saved.get('user', ''))
                st.session_state['mysql_pass'] = st.session_state.get('mysql_pass', saved.get('password', ''))
                st.session_state['mysql_db'] = st.session_state.get('mysql_db', saved.get('database', ''))
                st.session_state['mysql_sql'] = st.session_state.get('mysql_sql', saved.get('sql', ''))
                st.session_state['mysql_use_ssl'] = st.session_state.get('mysql_use_ssl', bool(saved.get('use_ssl', False)))
                st.session_state['mysql_ssl_ca'] = st.session_state.get('mysql_ssl_ca', saved.get('ssl_ca', ''))
                st.session_state['mysql_ssl_cert'] = st.session_state.get('mysql_ssl_cert', saved.get('ssl_cert', ''))
                st.session_state['mysql_ssl_key'] = st.session_state.get('mysql_ssl_key', saved.get('ssl_key', ''))
                st.session_state['mysql_filename_col'] = st.session_state.get('mysql_filename_col', saved.get('filename_col', 'filename'))
                st.session_state['mysql_blob_col'] = st.session_state.get('mysql_blob_col', saved.get('blob_col', 'content'))
                st.session_state['mysql_url_col'] = st.session_state.get('mysql_url_col', saved.get('url_col', ''))
                st.session_state['mysql_connect_timeout'] = st.session_state.get('mysql_connect_timeout', saved.get('connect_timeout', 10))
                st.session_state['mysql_derive_url'] = st.session_state.get('mysql_derive_url', bool(saved.get('derive_url', False)))
                st.session_state['mysql_dir_col'] = st.session_state.get('mysql_dir_col', saved.get('dir_col', 'directory_name'))
                st.session_state['mysql_stored_name_col'] = st.session_state.get('mysql_stored_name_col', saved.get('stored_name_col', 'stored_filename'))
                st.session_state['mysql_base_url'] = st.session_state.get('mysql_base_url', saved.get('base_url', ''))
                st.session_state['mysql_local_copy'] = st.session_state.get('mysql_local_copy', bool(saved.get('local_copy', False)))
                st.session_state['mysql_base_dir'] = st.session_state.get('mysql_base_dir', saved.get('base_dir', ''))
            st.session_state['mysql_settings_loaded'] = True
        c1, c2, c3 = st.columns(3)
        with c1:
            mysql_host = st.text_input("MySQL host", value=st.session_state.get('mysql_host', '127.0.0.1'))
            st.session_state['mysql_host'] = mysql_host
            mysql_db = st.text_input("Database", value=st.session_state.get('mysql_db', 'resumes_db'))
            st.session_state['mysql_db'] = mysql_db
        with c2:
            mysql_port = st.number_input("Port", min_value=1, max_value=65535, value=int(st.session_state.get('mysql_port', 3306)))
            st.session_state['mysql_port'] = mysql_port
            mysql_user = st.text_input("Username", value=st.session_state.get('mysql_user', 'readonly'))
            st.session_state['mysql_user'] = mysql_user
        with c3:
            mysql_pass = st.text_input("Password", value=st.session_state.get('mysql_pass', ''), type="password")
            st.session_state['mysql_pass'] = mysql_pass

        # Optional SSL settings
        st.session_state['mysql_use_ssl'] = st.checkbox("Use SSL/TLS", value=bool(st.session_state.get('mysql_use_ssl', False)), help="Enable if your MySQL server requires SSL/TLS. Provide CA/cert/key if needed.")
        ssl_ca = ssl_cert = ssl_key = None
        if st.session_state['mysql_use_ssl']:
            c_ssl1, c_ssl2, c_ssl3 = st.columns(3)
            with c_ssl1:
                ssl_ca = st.text_input("SSL CA (path)", value=st.session_state.get('mysql_ssl_ca', ''))
                st.session_state['mysql_ssl_ca'] = ssl_ca
            with c_ssl2:
                ssl_cert = st.text_input("SSL cert (path)", value=st.session_state.get('mysql_ssl_cert', ''))
                st.session_state['mysql_ssl_cert'] = ssl_cert
            with c_ssl3:
                ssl_key = st.text_input("SSL key (path)", value=st.session_state.get('mysql_ssl_key', ''))
                st.session_state['mysql_ssl_key'] = ssl_key

        st.caption("SQL must return columns: filename, plus either content (BLOB) or url (HTTP/HTTPS). If your column names differ, set the mappings below.")
        default_mysql_sql = "SELECT id, filename, content FROM resumes WHERE updated_at >= NOW() - INTERVAL 30 DAY LIMIT 500;"
        mysql_sql = st.text_area("SQL to fetch resumes", value=st.session_state.get('mysql_sql', default_mysql_sql), height=90)
        st.session_state['mysql_sql'] = mysql_sql

        # Column mappings and timeout
        mc1, mc2, mc3, mc4 = st.columns([1,1,1,1])
        with mc1:
            filename_col = st.text_input("Filename column", value=st.session_state.get('mysql_filename_col', 'filename'))
            st.session_state['mysql_filename_col'] = filename_col
        with mc2:
            blob_col = st.text_input("BLOB column (optional)", value=st.session_state.get('mysql_blob_col', 'content'))
            st.session_state['mysql_blob_col'] = blob_col
        with mc3:
            url_col = st.text_input("URL column (optional)", value=st.session_state.get('mysql_url_col', ''))
            st.session_state['mysql_url_col'] = url_col
        with mc4:
            connect_timeout = st.number_input("Connect timeout (s)", min_value=3, max_value=60, value=int(st.session_state.get('mysql_connect_timeout', 10)), step=1, help="Seconds to wait when connecting before giving up.")
            st.session_state['mysql_connect_timeout'] = connect_timeout

        md1, md2, md3 = st.columns([1,1.2,1.8])
        with md1:
            st.session_state['mysql_derive_url'] = st.checkbox(
                "Derive URL",
                value=bool(st.session_state.get('mysql_derive_url', False)),
                help="If your table stores directory and stored filename, build the download URL as base_url + directory + stored_filename.")
        with md2:
            dir_col = st.text_input("Directory column", value=st.session_state.get('mysql_dir_col', 'directory_name'))
            st.session_state['mysql_dir_col'] = dir_col
        with md3:
            stored_name_col = st.text_input("Stored filename column", value=st.session_state.get('mysql_stored_name_col', 'stored_filename'))
            st.session_state['mysql_stored_name_col'] = stored_name_col
        base_url = st.text_input("Base URL prefix (for derived URL)", value=st.session_state.get('mysql_base_url', ''), help="Example: https://files.example.com/ or https://server/uploads/. We will append directory and stored filename.")
        st.session_state['mysql_base_url'] = base_url

        # Optional: Copy from local/UNC share instead of HTTP
        lc1, lc2 = st.columns([1,2])
        with lc1:
            st.session_state['mysql_local_copy'] = st.checkbox(
                "Copy from local/UNC path",
                value=bool(st.session_state.get('mysql_local_copy', False)),
                help="If your files are on a local disk or network share, copy from base_dir + directory + stored_filename (e.g., \\\ileserver\\share)."
            )
        with lc2:
            st.session_state['mysql_base_dir'] = st.text_input(
                "Base directory (UNC or local)",
                value=st.session_state.get('mysql_base_dir', ''),
                help="Example: \\\ileserver\\share\\uploads or D:\\uploads. We'll append directory and stored filename."
            )

        # Save/Clear defaults row
        colSave, colClear, colNote = st.columns([1,1,3])
        with colSave:
            if st.button("Save as default", key="mysql_save_defaults", help="Save these MySQL settings so they load automatically next time."):
                config = {
                    'host': st.session_state.get('mysql_host', ''),
                    'port': int(st.session_state.get('mysql_port', 3306) or 3306),
                    'user': st.session_state.get('mysql_user', ''),
                    'password': st.session_state.get('mysql_pass', ''),  # stored in plaintext locally
                    'database': st.session_state.get('mysql_db', ''),
                    'sql': st.session_state.get('mysql_sql', ''),
                    'use_ssl': bool(st.session_state.get('mysql_use_ssl', False)),
                    'ssl_ca': st.session_state.get('mysql_ssl_ca', ''),
                    'ssl_cert': st.session_state.get('mysql_ssl_cert', ''),
                    'ssl_key': st.session_state.get('mysql_ssl_key', ''),
                    'filename_col': st.session_state.get('mysql_filename_col', 'filename'),
                    'blob_col': st.session_state.get('mysql_blob_col', 'content'),
                    'url_col': st.session_state.get('mysql_url_col', ''),
                    'connect_timeout': int(st.session_state.get('mysql_connect_timeout', 10) or 10),
                    'derive_url': bool(st.session_state.get('mysql_derive_url', False)),
                    'dir_col': st.session_state.get('mysql_dir_col', 'directory_name'),
                    'stored_name_col': st.session_state.get('mysql_stored_name_col', 'stored_filename'),
                    'base_url': st.session_state.get('mysql_base_url', ''),
                    'local_copy': bool(st.session_state.get('mysql_local_copy', False)),
                    'base_dir': st.session_state.get('mysql_base_dir', ''),
                }
                if _save_mysql_settings(config):
                    st.success("Saved MySQL defaults. They will auto-load on next app start.")
                else:
                    st.error("Failed to save MySQL defaults. Check write permissions in the app folder.")
        with colClear:
            if st.button("Clear saved defaults", key="mysql_clear_defaults", help="Remove saved MySQL settings from disk."):
                try:
                    if os.path.exists(_MYSQL_SETTINGS_PATH):
                        os.remove(_MYSQL_SETTINGS_PATH)
                        st.success("Cleared saved MySQL defaults.")
                    else:
                        st.info("No saved MySQL defaults found.")
                except Exception as e:
                    st.error(f"Failed to clear defaults: {e}")
        with colNote:
            st.caption("Note: Defaults are saved locally in mysql_settings.json (password stored in plaintext). For production, prefer Streamlit secrets or environment variables.")

        # Preview rows button to help map columns
        if st.button("Preview first 5 rows", key="mysql_preview_btn", help="Run the SQL with LIMIT 5 and display column names and a sample of values."):
            try:
                pv_sql = f"SELECT * FROM ({mysql_sql.rstrip(';')}) AS q LIMIT 5;"
                cols, rows = _peek_mysql_rows(
                    mysql_host, int(mysql_port), mysql_user, mysql_pass, mysql_db, pv_sql,
                    ssl_ca=st.session_state.get('mysql_ssl_ca') or None if st.session_state.get('mysql_use_ssl') else None,
                    ssl_cert=st.session_state.get('mysql_ssl_cert') or None if st.session_state.get('mysql_use_ssl') else None,
                    ssl_key=st.session_state.get('mysql_ssl_key') or None if st.session_state.get('mysql_use_ssl') else None,
                    ssl_disabled=False if st.session_state.get('mysql_use_ssl') else True,
                    connect_timeout=int(st.session_state.get('mysql_connect_timeout', 10) or 10),
                )
                st.write("Columns:", cols)
                if rows:
                    st.json(rows)
                else:
                    st.info("No rows returned by preview.")
            except Exception as e:
                st.error(f"Preview failed: {e}")

        # Download SQL template for on-demand downloads (embeddings-only)
        st.session_state['mysql_download_sql'] = st.text_input(
            "Download SQL (by id)",
            value=st.session_state.get('mysql_download_sql', 'SELECT content FROM resumes WHERE id=%s'),
            help="SQL used to fetch the original BLOB for a single resume by id. Use %s placeholder for the id."
        )

        # Direct Chroma indexing without local copy
        cdi1, cdi2 = st.columns([1,1])
        with cdi1:
            if st.button("Index from MySQL (no local files)", key="mysql_index_chroma_btn", help="Fetch rows and upsert into Chroma directly without saving local files."):
                try:
                    cnt = chroma_upsert_from_mysql(
                        st.session_state.get('mysql_sql', default_mysql_sql),
                        id_col='id', filename_col=st.session_state.get('mysql_filename_col','filename'),
                        blob_col=(st.session_state.get('mysql_blob_col') or '').strip() or 'content',
                        url_col=(st.session_state.get('mysql_url_col') or '').strip() or None,
                        limit=None
                    )
                    st.session_state['chroma_index_ready'] = True
                    if cnt:
                        st.success(f"Indexed {cnt} chunk(s) to Chroma.")
                        if bool(st.session_state.get('store_embeddings_only', False)):
                            try:
                                removed = purge_originals_if_indexed()
                                if removed:
                                    st.caption(f"Embeddings-only: removed {removed} original file(s) after indexing.")
                            except Exception:
                                pass
                    else:
                        st.info("No chunks were added. Check your SQL and column mappings.")
                except Exception as e:
                    st.error(f"Chroma upsert from MySQL failed: {e}")
        with cdi2:
            st.empty()

        colA, colB = st.columns([1,1])
        with colA:
            if st.button("Test DB", key="test_mysql_btn", help="Connect and fetch 1 row to verify connection and SQL."):
                try:
                    test_sql = f"SELECT * FROM ({mysql_sql.rstrip(';')}) AS q LIMIT 1;"
                    _test_mysql_connection(
                        mysql_host, int(mysql_port), mysql_user, mysql_pass, mysql_db, test_sql,
                        ssl_ca=st.session_state.get('mysql_ssl_ca') or None if st.session_state.get('mysql_use_ssl') else None,
                        ssl_cert=st.session_state.get('mysql_ssl_cert') or None if st.session_state.get('mysql_use_ssl') else None,
                        ssl_key=st.session_state.get('mysql_ssl_key') or None if st.session_state.get('mysql_use_ssl') else None,
                        ssl_disabled=False if st.session_state.get('mysql_use_ssl') else True,
                        connect_timeout=int(st.session_state.get('mysql_connect_timeout', 10) or 10),
                    )
                    st.success("DB connection OK.")
                except Exception as e:
                    st.error(f"DB test failed: {e}")
        with colB:
            if st.button("Sync from MySQL", key="sync_mysql_btn", help="Fetch resumes and store locally; then embed."):
                with st.spinner("Syncing from MySQL…"):
                    pb = st.progress(0.0, text="Connecting to database…")
                    def _mysql_progress(cur:int, total:int, text:str):
                        try:
                            ratio = (cur/total) if total else 0.0
                            pb.progress(ratio, text=text)
                        except Exception:
                            pass
                    new_files, stats = _sync_from_mysql(
                        mysql_host, int(mysql_port), mysql_user, mysql_pass, mysql_db, mysql_sql, limit=None,
                        ssl_ca=st.session_state.get('mysql_ssl_ca') or None if st.session_state.get('mysql_use_ssl') else None,
                        ssl_cert=st.session_state.get('mysql_ssl_cert') or None if st.session_state.get('mysql_use_ssl') else None,
                        ssl_key=st.session_state.get('mysql_ssl_key') or None if st.session_state.get('mysql_use_ssl') else None,
                        ssl_disabled=False if st.session_state.get('mysql_use_ssl') else True,
                        filename_col=st.session_state.get('mysql_filename_col') or 'filename',
                        blob_col=(st.session_state.get('mysql_blob_col') or '').strip() or None,
                        url_col=(st.session_state.get('mysql_url_col') or '').strip() or None,
                        derive_url=bool(st.session_state.get('mysql_derive_url', False)),
                        base_url=(st.session_state.get('mysql_base_url') or '').strip() or None,
                        dir_col=(st.session_state.get('mysql_dir_col') or '').strip() or None,
                        stored_name_col=(st.session_state.get('mysql_stored_name_col') or '').strip() or None,
                        local_copy=bool(st.session_state.get('mysql_local_copy', False)),
                        base_dir=(st.session_state.get('mysql_base_dir') or '').strip() or None,
                        connect_timeout=int(st.session_state.get('mysql_connect_timeout', 10) or 10),
                        progress_cb=_mysql_progress,
                    )
                    if new_files:
                        added = update_embedding_cache_for_new_files(files_to_add=new_files)
                        pb.progress(1.0, text="Sync complete – embedding and uploads done.")
                        st.success(f"Synced {len(new_files)} file(s). Embedded {added} new.")
                        st.caption(f"Fetched: {stats.get('fetched',0)} • Missing filename: {stats.get('missing_filename',0)} • Saved (blob): {stats.get('blob_saved',0)} • Saved (url): {stats.get('url_saved',0)} • Saved (local): {stats.get('local_saved',0)} • Renamed: {stats.get('renamed',0)} • Errors: {stats.get('errors',0)}")
                        st.rerun()
                    else:
                        pb.progress(1.0, text="No rows to download or downloads failed.")
                        st.info("No new rows or downloads failed.")
                        try:
                            st.caption(f"Fetched: {stats.get('fetched',0)} • Missing filename: {stats.get('missing_filename',0)} • Saved (blob): {stats.get('blob_saved',0)} • Saved (url): {stats.get('url_saved',0)} • Saved (local): {stats.get('local_saved',0)} • Renamed: {stats.get('renamed',0)} • Errors: {stats.get('errors',0)}")
                            st.caption("Tip: Use 'Preview first 5 rows' and adjust 'Filename/BLOB/URL column' mappings if column names differ from your schema.")
                        except Exception:
                            pass

    # LLM settings
    with st.expander("LLM settings"):
        st.caption("Configure the remote LLM service (defaults to your remote Ollama server).")
        c_llm1, c_llm2, c_llm3 = st.columns([2,2,1])
        with c_llm1:
            st.session_state['llm_base_url'] = st.text_input(
                "Remote LLM base URL",
                value=st.session_state.get('llm_base_url', os.environ.get('LLM_BASE_URL', 'http://67.11.191.239:11434')),
                help="Example: http://host:11434 (Ollama). We'll call /api/generate on this base URL."
            )
        with c_llm2:
            st.session_state['llm_model'] = st.text_input(
                "Model id",
                value=st.session_state.get('llm_model', os.environ.get('LLM_MODEL', 'llama3.1:8b-instruct-q4_K_M')),
                help="Ollama model tag, e.g., llama3.1:8b-instruct-q4_K_M"
            )
        with c_llm3:
            st.session_state['llm_timeout'] = st.number_input(
                "Timeout (s)", min_value=10, max_value=300, value=int(st.session_state.get('llm_timeout', 60)), step=5,
                help="Maximum seconds to wait for LLM response."
            )

    # (Status expander removed; status is shown inline at top)

    # Advanced maintenance grouped
    with st.expander("Advanced maintenance"):
        # Build embeddings for remaining
        st.write("Build embedding cache (remaining): Embed only resumes not yet cached. Shows live progress.")
        chunk_size = st.number_input("Batch size", min_value=8, max_value=256, value=DEFAULT_EMBED_BATCH, step=8, help="Number of resumes to embed per batch.")
        if st.button("Start build", key="cache_remaining_live_settings", help="Compute and store embeddings for resumes not yet in the cache."):
            try:
                remaining = get_remaining_files(_RESUME_DIR)
                if not remaining:
                    st.info("All resumes are already embedded.")
                else:
                    total_left = len(remaining)
                    pb = st.progress(0.0, text=f"Embedding 0/{total_left} remaining…")
                    log_ph = st.empty()
                    idx_path = embedding_idx_path
                    npy_path = embedding_npy_path
                    if os.path.exists(idx_path):
                        with open(idx_path, 'rb') as f:
                            embedding_index_local = pickle.load(f)
                    else:
                        embedding_index_local = []
                    emb_dim = 384
                    if os.path.exists(npy_path) and len(embedding_index_local) > 0:
                        embeddings_local = np.load(npy_path)
                        if embeddings_local.dtype != np.float32:
                            embeddings_local = embeddings_local.astype(np.float32)
                    else:
                        embeddings_local = np.zeros((0, emb_dim), dtype=np.float32)
                    model = get_st_model()
                    for i in range(0, total_left, int(chunk_size)):
                        batch_files = remaining[i:i+int(chunk_size)]
                        def _read_text(fname: str) -> str:
                            return extract_text_fallback(os.path.join(_RESUME_DIR, fname))
                        max_workers = max(2, min(8, (os.cpu_count() or 4)))
                        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                            texts = list(ex.map(_read_text, batch_files))
                        try:
                            batch_embeds = model.encode(
                                texts,
                                convert_to_numpy=True,
                                normalize_embeddings=True,
                                show_progress_bar=False,
                                batch_size=int(chunk_size),
                            )
                        except TypeError:
                            batch_embeds = np.asarray(
                                model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                            )
                        batch_embeds = batch_embeds.astype(np.float32)
                        embedding_index_local.extend(batch_files)
                        if embeddings_local.shape[0] == 0:
                            embeddings_local = batch_embeds
                        else:
                            embeddings_local = np.vstack([embeddings_local, batch_embeds])
                        to_store = embeddings_local.astype(np.float16) if USE_FLOAT16_EMBEDS else embeddings_local.astype(np.float32)
                        np.save(npy_path, to_store)
                        with open(idx_path, 'wb') as f:
                            pickle.dump(embedding_index_local, f)
                        done = min(i + len(batch_files), total_left)
                        t, e, r = get_resume_cache_status()
                        pb.progress(done/total_left, text=f"Embedding {done}/{total_left} remaining…  |  Total: {t}, Embedded: {e}, Remaining: {r}")
                        log_ph.write(f"Processed batch {i//int(chunk_size)+1}: {len(batch_files)} files. Embedded {done}/{total_left} of remaining.")
                    t, e, r = get_resume_cache_status()
                    st.success(f"Caching complete. Total: {t}, Embedded: {e}, Remaining: {r}.")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to cache remaining resumes: {e}")

        st.markdown("---")
        st.write("Duplicates: Detect and remove duplicates (keep newest per group).")
        st.write("Detect and remove duplicates (keep newest per group).")
        auto_remove = st.checkbox("Auto remove duplicates (keep newest per group)", value=True, help="When enabled, duplicates are deleted right after scan and the embedding cache is compacted.")
        def _norm_text_for_hash(text: str) -> str:
            s = (text or '').lower()
            s = re.sub(r"\s+", " ", s).strip()
            return s
        def _file_mtime(path: str) -> float:
            try:
                return os.path.getmtime(path)
            except Exception:
                return 0.0
        if st.button("Scan duplicates", key="scan_dupes_settings", help="Identify duplicate resumes by normalized content; you can auto-remove or review first."):
            try:
                files = _list_resume_files(_RESUME_DIR)
                text_hash_map = {}
                for fn in files:
                    p = os.path.join(_RESUME_DIR, fn)
                    txt = extract_text_fallback(p)
                    key = hashlib.sha256(_norm_text_for_hash(txt).encode('utf-8', errors='ignore')).hexdigest() if txt else None
                    if not key:
                        try:
                            with open(p, 'rb') as f:
                                key = hashlib.sha256(f.read()).hexdigest()
                        except Exception:
                            key = None
                    if key:
                        text_hash_map.setdefault(key, []).append(fn)
                dup_groups = {k: v for k, v in text_hash_map.items() if len(v) > 1}
                if not dup_groups:
                    st.success("No duplicate groups found.")
                else:
                    st.warning(f"Found {len(dup_groups)} duplicate group(s).")
                    shown = 0
                    for i, (k, group) in enumerate(dup_groups.items(), 1):
                        if shown >= 5:
                            st.write("… (more groups not shown)")
                            break
                        st.write(f"Group {i}: {len(group)} files → {', '.join(sorted(group))}")
                        shown += 1
                    def _remove_dupes_and_fix_cache(groups: dict):
                        removed = 0
                        kept_files = []
                        for _, group in groups.items():
                            paths = [os.path.join(_RESUME_DIR, g) for g in group]
                            newest = max(paths, key=_file_mtime)
                            kept_files.append(os.path.basename(newest))
                            for p in paths:
                                if p != newest and os.path.exists(p):
                                    try:
                                        # External storage deletion removed
                                        os.remove(p)
                                        removed += 1
                                    except Exception:
                                        pass
                        idx_path = embedding_idx_path
                        npy_path = embedding_npy_path
                        if os.path.exists(idx_path) and os.path.exists(npy_path):
                            try:
                                with open(idx_path, 'rb') as f:
                                    emb_files = pickle.load(f)
                                embeddings = np.load(npy_path)
                                if embeddings.dtype != np.float32:
                                    embeddings = embeddings.astype(np.float32)
                                present_pairs = [(i, fn) for i, fn in enumerate(emb_files) if os.path.exists(os.path.join(_RESUME_DIR, fn))]
                                if present_pairs:
                                    idxs, fns = zip(*present_pairs)
                                    new_E = embeddings[list(idxs)]
                                    new_idx = list(fns)
                                else:
                                    new_E = np.zeros((0, 384), dtype=np.float32)
                                    new_idx = []
                                to_store = new_E.astype(np.float16) if USE_FLOAT16_EMBEDS else new_E.astype(np.float32)
                                np.save(npy_path, to_store)
                                with open(idx_path, 'wb') as f:
                                    pickle.dump(new_idx, f)
                                missing_kept = [k for k in kept_files if k not in set(new_idx)]
                                if missing_kept:
                                    try:
                                        update_embedding_cache_for_new_files(_RESUME_DIR, missing_kept)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        return removed
                    if auto_remove:
                        removed = _remove_dupes_and_fix_cache(dup_groups)
                        t, e, r = get_resume_cache_status()
                        st.success(f"Auto-removed {removed} duplicate file(s). Total: {t}, Embedded: {e}, Remaining: {r}.")
                        st.rerun()
                    else:
                        if st.button("Remove duplicates (keep newest per group)", key="remove_dupes_keep_newest_settings", help="Deletes older duplicates and compacts the embedding cache to match."):
                            removed = _remove_dupes_and_fix_cache(dup_groups)
                            if removed:
                                st.success(f"Removed {removed} duplicate file(s).")
                                st.rerun()
                            else:
                                st.info("No files were removed.")
            except Exception as e:
                st.error(f"Failed to scan/remove duplicates: {e}")





