from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .headings import parse_heading, Heading

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

from .aggregate import compute_summary, dedupe_findings
from .fewshot import cuad_file_meta, load_or_build_fewshot, resolve_cuad_json_path
from .llm import LLMConfig, LLMRunner, load_categories_payload, normalize_llm_text
from .validate_legal_refs import filter_valid_legal_references as _filter_valid_legal_references
from .models import (
    Finding,
    JobStatus,
    JobStatusResponse,
    Paragraph,
    ResultDocument,
    ResultMeta,
    ResultPayload,
    UploadResponse,
)
from .parser import parse_pdf_to_paragraphs, _split_into_paragraphs
from .annotated_pdf import build_annotated_text_pdf
from .storage import (
    Paths,
    cleanup_files_and_db,
    create_job,
    ensure_dirs,
    get_job,
    init_db,
    job_result_path,
    job_upload_path,
    update_job,
)

LOGGER = logging.getLogger("demo")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

security = HTTPBasic(auto_error=False)

def _auth_enabled() -> bool:
    return False


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def get_limits() -> dict:
    return {
        "max_paragraphs": _env_int("MAX_PARAGRAPHS", 200),
        "max_file_mb": _env_int("MAX_FILE_MB", 0),
        "ttl_hours": _env_int("TTL_HOURS", 24),
        "max_llm_concurrency": _env_int("MAX_LLM_CONCURRENCY", 12),
    }


def require_basic_auth(creds: HTTPBasicCredentials = Depends(security)) -> None:
    if not _auth_enabled():
        return
    if creds is None:
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Basic"})
    user = os.environ.get("BASIC_AUTH_USER", "")
    pwd = os.environ.get("BASIC_AUTH_PASS", "")
    ok = secrets.compare_digest(creds.username or "", user) and secrets.compare_digest(creds.password or "", pwd)
    if not ok:
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Basic"})


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TEMPLATES_DIR = BASE_DIR / "templates"
CONFIGS_DIR = BASE_DIR / "configs"
STATIC_DIR = BASE_DIR / "static"

paths = Paths(base=DATA_DIR)

app = FastAPI(title="Contract Red-Flag Detector (Demo)")

@app.get("/static/example_contract.pdf")
async def get_example_contract() -> FileResponse:
    """Serve example contract without authentication."""
    example_path = _resolve_example_contract_path()
    if not example_path.exists():
        raise HTTPException(status_code=404, detail="Example contract not found")
    return FileResponse(
        path=str(example_path),
        media_type="application/pdf",
        filename="example_contract.pdf",
    )

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory, ephemeral “live trace” storage (demo-friendly).
# Keys are NOT persisted; this is for real-time UI visibility during a running job.
_LIVE_TRACE: dict[str, dict] = {}
_CANCELLED: set[str] = set()
_JOB_TASKS: dict[str, set[asyncio.Task]] = {}


def _register_job_task(job_id: str, task: asyncio.Task) -> None:
    tasks = _JOB_TASKS.setdefault(job_id, set())
    tasks.add(task)

    def _cleanup(_t: asyncio.Task) -> None:
        tasks.discard(_t)

    task.add_done_callback(_cleanup)


def _cancel_job_tasks(job_id: str) -> None:
    for t in list(_JOB_TASKS.get(job_id, set())):
        if not t.done():
            t.cancel()


def _set_live_trace(job_id: str, payload: dict) -> None:
    _LIVE_TRACE[job_id] = payload


def _clear_live_trace(job_id: str) -> None:
    _LIVE_TRACE.pop(job_id, None)

def _is_cancelled(job_id: str) -> bool:
    return job_id in _CANCELLED


def _is_heading_paragraph(text: str) -> bool:
    """
    Only treat TOP-LEVEL headings as headings for section splitting/removal.
    Subsection headers like "1.3" should stay in the content.
    """
    h = parse_heading(text, max_len=160)
    if h is None:
        return False
    if h.level == 0:
        return True
    if h.level == 1:
        label = (h.label or "").strip()
        m = re.search(r"(?:Section|Clause|§)\s*(\d+(?:\.\d+)*)", label, re.IGNORECASE)
        if m:
            return "." not in m.group(1)
        return True
    if h.level == 2:
        label = (h.label or "").strip()
        return "." not in label
    return False


def _is_signature_paragraph(text: str) -> bool:
    """
    Check if a paragraph is likely a signature line, date, or name-only line.
    Such paragraphs should be filtered out before LLM processing as they're not contract content.
    """
    if not text:
        return False
    # Never treat headings as signatures/garbage.
    if parse_heading(text, max_len=160) is not None:
        return False
    t = text.strip()
    # Protect sub-clause markers like "2.4. Prices." or "5.7 Nonsolicitation." —
    # parse_heading rejects them (subsection numbers) but they ARE contract content.
    if re.match(r"^\s*\d+(?:\.\d+)+", t):
        return False
    if len(t) < 3:
        return False
    
    words = t.split()
    word_count = len(words)
    
    # Pattern: Address lines - "City, State ZIP" or "City, State ZIP Attn: Name"
    # Examples: "Northfield, Illinois 60093", "Northfield, Illinois 60093 Attn: Philip E. Ruben, Esq."
    address_pattern = re.compile(
        r"^\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s+[A-Z][a-z]+\s+\d{5}(?:\s+Attn:.*)?$",
        re.IGNORECASE
    )
    if address_pattern.match(t):
        return True
    
    # Pattern: Lines starting with "Attn:" or "Attention:"
    if re.match(r"^\s*Attn:\s+", t, re.IGNORECASE) or re.match(r"^\s*Attention:\s+", t, re.IGNORECASE):
        return True
    
    # Pattern: Contact labels like "Email:", "Phone:", "Fax:" etc.
    if re.match(r"^\s*(Email|E-mail|Phone|Tel|Fax)\s*:\s*", t, re.IGNORECASE):
        return True
    
    # Pattern: Very short line (1-2 words) - likely garbage (name, address fragment, etc.)
    if word_count <= 2:
        return True
    
    # Pattern: "Name Surname" or "Name Surname, Title" (2-4 capitalized words, no punctuation except comma)
    name_only = re.compile(r"^\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)?\s*$")
    if name_only.match(t):
        return True
    
    # Pattern: "Name Surname /s/" or "/s/ Name Surname" (signature marker)
    if re.search(r"/s/", t, re.IGNORECASE):
        return True
    
    # Pattern: Just a date (standalone date line)
    date_only_patterns = [
        r"^\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\s*$",  # 01/01/2024
        r"^\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\s*$",  # January 1, 2024
        r"^\s*\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s*$",  # 1 January 2024
    ]
    if any(re.match(dp, t, re.IGNORECASE) for dp in date_only_patterns):
        return True
    
    # Pattern: "Name Surname Date" where Date is like "January 1, 2024" or "01/01/2024"
    # Short line (2-8 words) with both name and date
    date_patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",  # 01/01/2024 or 01-01-2024
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",  # January 1, 2024
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",  # 1 January 2024
    ]
    if 2 <= word_count <= 8:  # Short line with 2-8 words
        has_date = any(re.search(dp, t, re.IGNORECASE) for dp in date_patterns)
        # Check if it's mostly capitalized words (names) and date, with minimal other content
        capitalized_words = sum(1 for w in words if re.match(r"^[A-Z][a-z]+$", w))
        if has_date and capitalized_words >= 2 and capitalized_words >= word_count * 0.6:
            return True
    
    # Pattern: "Full name: ____", "Name: ____", "Print name: ____", "Date: ____", "Title: ____"
    if re.match(r"^\s*(Full\s+name|Print\s+name|Name|Tenant\s+Name|Landlord\s+Name)\s*:\s*[_A-Za-z\s]*$", t, re.IGNORECASE):
        return True
    if re.match(r"^\s*Date\s*:\s*[_\d/\-\s]*$", t, re.IGNORECASE):
        return True
    if re.match(r"^\s*Title\s*:\s*[_A-Za-z\s]*$", t, re.IGNORECASE):
        return True
    
    # Pattern: Very short line (1-3 words) that's just capitalized words (likely name)
    if 1 <= word_count <= 3:
        if all(re.match(r"^[A-Z][a-z]+$", w) for w in words):
            return True
    
    return False


def _filter_garbage_paragraphs(paragraphs: list[Paragraph]) -> list[Paragraph]:
    """
    Filter out garbage paragraphs (signatures, dates, very short lines, etc.)
    that shouldn't go to LLM processing.
    Also filters sequences of very short lines (1-2 words) that are likely garbage.
    """
    if not paragraphs:
        return paragraphs
    
    filtered = []
    i = 0
    while i < len(paragraphs):
        p = paragraphs[i]
        text = (p.text or "").strip()
        
        # Skip empty paragraphs
        if not text:
            i += 1
            continue
        
        # Keep headings even if they look short or name-like.
        if parse_heading(text, max_len=160) is not None:
            filtered.append(p)
            i += 1
            continue

        # Check if it's a signature/date/name paragraph
        if _is_signature_paragraph(text):
            i += 1
            continue
        
        # Check for sequences of very short lines (1-2 words) - likely garbage
        words = text.split()
        if len(words) <= 2:
            # Check if next few paragraphs are also very short (sequence of garbage)
            short_count = 1
            j = i + 1
            while j < len(paragraphs) and short_count < 5:  # Check up to 4 more paragraphs
                next_text = (paragraphs[j].text or "").strip()
                if not next_text:
                    break
                if parse_heading(next_text, max_len=160) is not None:
                    # Heading within short lines means this isn't garbage sequence.
                    short_count = 0
                    break
                next_words = next_text.split()
                if len(next_words) <= 2:
                    short_count += 1
                    j += 1
                else:
                    break
            
            # If we have 3+ consecutive very short lines, it's likely garbage (address, name list, etc.)
            if short_count >= 3:
                # Skip all these short paragraphs
                i = j
                continue
        
        # Keep this paragraph
        filtered.append(p)
        i += 1
    
    return filtered


def _normalize_paragraphs(paragraphs: list[Paragraph]) -> list[Paragraph]:
    """
    Normalize each paragraph's text (whitespace, PDF artefacts, control chars).
    Use this same normalized text everywhere: LLM chunks, quote matching, PDF, frontend.
    Note: Signature paragraphs are filtered later (before LLM processing), not here,
    so they remain in PDF and frontend for display.
    """
    return [
        Paragraph(
            text=normalize_llm_text(p.text or ""),
            page=p.page,
            paragraph_index=p.paragraph_index,
            bbox=p.bbox,
        )
        for p in paragraphs
    ]


def _approx_chars_limit() -> int:
    return _env_int("STAGE1_MAX_INPUT_CHARS", 60000)


_EXCEPTION_TRIGGER_RE = __import__("re").compile(
    r"\b(except|provided that|subject to|notwithstanding|unless)\b",
    __import__("re").IGNORECASE,
)


def _has_exception_trigger(text: str) -> bool:
    return bool(_EXCEPTION_TRIGGER_RE.search(text or ""))


def _approx_tokens(text: str) -> int:
    """
    Provider-agnostic token estimate.
    For English-ish text, ~4 chars/token is a common heuristic.
    """
    s = (text or "").strip()
    if not s:
        return 0
    return max(1, int((len(s) + 3) / 4))


def _normalize_for_match(text: str) -> str:
    """
    Aggressive normalization for quote matching: first apply normalize_llm_text to ensure
    consistency with the text that LLM sees, then collapse all whitespace to single space
    and lowercase for robust matching.
    
    This ensures we always compare against the same normalized base text, preventing
    mismatches from different normalization approaches.
    """
    # First normalize using the same function that processes text for LLM
    # This ensures consistency: we're matching against the same normalized text
    # that was sent to LLM and stored in paragraphs
    normalized = normalize_llm_text(text or "")
    # Then apply aggressive normalization for matching: lowercase + collapse whitespace
    t = normalized.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _find_trigger_edges(text: str, edge_frac: float) -> tuple[bool, bool]:
    s = text or ""
    if not s:
        return (False, False)
    edge = int(max(1, round(len(s) * edge_frac)))
    hits = list(_EXCEPTION_TRIGGER_RE.finditer(s))
    if not hits:
        return (False, False)
    first = hits[0].start()
    last = hits[-1].start()
    return (first <= edge, last >= max(0, len(s) - edge))


def _extend_window_by_stride(
    sec: list[Paragraph],
    start: int,
    end: int,
    *,
    stride_tokens: int,
    extend_left: bool,
    extend_right: bool,
) -> tuple[int, int]:
    # Extend left by ~stride_tokens worth of paragraphs
    if extend_left:
        need = stride_tokens
        i = start - 1
        while i >= 0 and need > 0:
            need -= _approx_tokens(sec[i].text)
            i -= 1
        start = max(0, i + 1)
    if extend_right:
        need = stride_tokens
        i = end
        while i < len(sec) and need > 0:
            need -= _approx_tokens(sec[i].text)
            i += 1
        end = min(len(sec), i)
    return start, end


def _heading_stack_to_path(stack: list[Heading]) -> str:
    parts: list[str] = []
    for h in stack:
        if h.label:
            parts.append(h.label)
        if h.title:
            parts.append(h.title)
    cleaned: list[str] = []
    for p in parts:
        p = " ".join(str(p).split())
        if not p:
            continue
        if cleaned and cleaned[-1].lower() == p.lower():
            continue
        cleaned.append(p)
    return " > ".join(cleaned)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail}, headers=exc.headers)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


@app.on_event("startup")
async def on_startup() -> None:
    ensure_dirs(paths)
    await init_db(paths.db)
    ttl = get_limits()["ttl_hours"]
    cleanup_files_and_db(paths, ttl_hours=ttl)
    LOGGER.info("Startup complete. TTL=%sh", ttl)


@app.get("/", response_class=HTMLResponse)
async def index(_: None = Depends(require_basic_auth)) -> HTMLResponse:
    return FileResponse(str(TEMPLATES_DIR / "index.html"))


@app.post("/auth/login")
async def login(username: str = Form(""), password: str = Form("")) -> JSONResponse:
    """Optional login endpoint. Returns user info if credentials match."""
    auth_user = os.environ.get("BASIC_AUTH_USER", "demo")
    auth_pass = os.environ.get("BASIC_AUTH_PASS", "demo")
    
    is_valid = secrets.compare_digest(username or "", auth_user) and secrets.compare_digest(password or "", auth_pass)
    
    if is_valid:
        is_demo = username.lower() == "demo" or auth_user.lower() == "demo"
        return JSONResponse(
            content={
                "success": True,
                "username": username,
                "is_demo": is_demo,
            }
        )
    else:
        return JSONResponse(
            status_code=401,
            content={"success": False, "error": "Invalid credentials"}
        )


@app.get("/health")
async def health(_: None = Depends(require_basic_auth)) -> JSONResponse:
    """Debug endpoint (no secrets) for demo setup."""
    provider = "nebius" if os.environ.get("NEBIUS_API_KEY") else "openai"
    base_url = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/") if provider == "nebius" else None
    if provider == "nebius":
        stage1 = os.environ.get("NEBIUS_MODEL_STAGE1") or os.environ.get("OPENAI_MODEL_STAGE1", "Qwen/Qwen3-30B-A3B-Thinking-2507")
        stage2 = (
            os.environ.get("NEBIUS_MODEL_STAGE2")
            or os.environ.get("NEBIUS_MODEL")
            or os.environ.get("OPENAI_MODEL_STAGE2", "Qwen/Qwen3-30B-A3B-Thinking-2507")
        )
    else:
        stage1 = os.environ.get("OPENAI_MODEL_STAGE1", "Qwen/Qwen3-30B-A3B-Thinking-2507")
        stage2 = os.environ.get("OPENAI_MODEL_STAGE2", "Qwen/Qwen3-30B-A3B-Thinking-2507")
    return JSONResponse(
        content={
            "provider": provider,
            "base_url": base_url,
            "models": {
                "stage_1": stage1,
                "stage_2": stage2,
            },
            "limits": get_limits(),
        }
    )


def _resolve_example_contract_path() -> Path:
    """Resolve path to the example contract PDF. Prefer EXAMPLE_CONTRACT_PATH env if set and exists."""
    env_path = os.environ.get("EXAMPLE_CONTRACT_PATH", "").strip()
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
    default = STATIC_DIR / "example_contract.pdf"
    return default


@app.post("/upload-example", response_model=UploadResponse)
async def upload_example(
    background: BackgroundTasks,
    provider: str = Form("nebius"),
    api_key: str = Form(""),
    base_url: str = Form(""),
    model: str = Form(""),
    mode: str = Form(""),
    prompt_version: str = Form(""),
    use_demo_key: str = Form("0"),
    _: None = Depends(require_basic_auth),
) -> UploadResponse:
    """Create a job from the built-in example contract."""
    example_path = _resolve_example_contract_path()
    if not example_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Example contract not found. Add static/example_contract.pdf or set EXAMPLE_CONTRACT_PATH to your PDF path.",
        )
    limits = get_limits()
    job_id = secrets.token_hex(12)
    upload_path = job_upload_path(paths, job_id, example_path.name)
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(example_path.read_bytes())
    await create_job(paths.db, job_id, example_path.name, upload_path, debug_paragraph_enabled=False, debug_window_enabled=False)
    use_demo = (use_demo_key or "").strip() in ("1", "true", "yes", "on")
    final_api_key = (os.environ.get("NEBIUS_API_KEY") or os.environ.get("OPENAI_API_KEY") or "") if use_demo else (api_key or "").strip()
    if not final_api_key:
        raise HTTPException(status_code=400, detail="API key is required (or sign in as demo).")
    llm_overrides = {
        "provider": (provider or "").strip().lower(),
        "api_key": final_api_key,
        "base_url": (base_url or "").strip(),
        "model": (model or "").strip(),
        "mode": (mode or "").strip().lower(),
        "prompt_version": (prompt_version or "").strip().lower(),
    }
    background.add_task(process_job, job_id, llm_overrides)
    return UploadResponse(job_id=job_id, status=JobStatus.queued, limits=limits)


@app.post("/upload", response_model=UploadResponse)
async def upload(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    provider: str = Form("nebius"),
    api_key: str = Form(""),
    base_url: str = Form(""),
    model: str = Form(""),
    enable_debug_paragraph: str = Form("0"),
    enable_debug_window: str = Form("0"),
    mode: str = Form(""),
    prompt_version: str = Form(""),
    use_demo_key: str = Form("0"),
    _: None = Depends(require_basic_auth),
) -> UploadResponse:
    limits = get_limits()
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    raw = await file.read()
    max_mb = int(limits.get("max_file_mb") or 0)
    if max_mb > 0:
        max_bytes = max_mb * 1024 * 1024
        if len(raw) > max_bytes:
            raise HTTPException(status_code=400, detail=f"File too large. Max {max_mb} MB.")

    job_id = secrets.token_hex(12)
    upload_path = job_upload_path(paths, job_id, file.filename or "upload.pdf")
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(raw)

    dbg_para = (enable_debug_paragraph or "").strip() in ("1", "true", "yes", "on")
    dbg_win = (enable_debug_window or "").strip() in ("1", "true", "yes", "on")
    await create_job(
        paths.db,
        job_id,
        file.filename or "upload.pdf",
        upload_path,
        debug_paragraph_enabled=dbg_para,
        debug_window_enabled=dbg_win,
    )
    use_demo = (use_demo_key or "").strip() in ("1", "true", "yes", "on")
    final_api_key = ""
    if use_demo:
        final_api_key = os.environ.get("NEBIUS_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    else:
        final_api_key = (api_key or "").strip()
    if not final_api_key:
        raise HTTPException(status_code=400, detail="API key is required (or sign in as demo).")
    if not (model or "").strip():
        raise HTTPException(status_code=400, detail="Model is required.")
    
    llm_overrides = {
        "provider": (provider or "").strip().lower(),
        "api_key": final_api_key,
        "base_url": (base_url or "").strip(),
        "model": (model or "").strip(),
        "mode": (mode or "").strip().lower(),
        "prompt_version": (prompt_version or "").strip().lower(),
    }
    background.add_task(process_job, job_id, llm_overrides)

    return UploadResponse(job_id=job_id, status=JobStatus.queued, limits=limits)


@app.post("/upload-text", response_model=UploadResponse)
async def upload_text(
    background: BackgroundTasks,
    text: str = Form(...),
    provider: str = Form("nebius"),
    api_key: str = Form(""),
    base_url: str = Form(""),
    model: str = Form(""),
    enable_debug_paragraph: str = Form("0"),
    enable_debug_window: str = Form("0"),
    mode: str = Form(""),
    prompt_version: str = Form(""),
    use_demo_key: str = Form("0"),
    _: None = Depends(require_basic_auth),
) -> UploadResponse:
    """Create a job from plain text input."""
    limits = get_limits()
    
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    # Create a temporary text file (we'll parse it as text, not PDF)
    job_id = secrets.token_hex(12)
    upload_path = job_upload_path(paths, job_id, "text_input.txt")
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_text(text, encoding="utf-8")
    
    # For text input, we'll parse it directly in process_job
    # Mark it as text input by using .txt extension
    dbg_para = (enable_debug_paragraph or "").strip() in ("1", "true", "yes", "on")
    dbg_win = (enable_debug_window or "").strip() in ("1", "true", "yes", "on")
    await create_job(
        paths.db,
        job_id,
        "text_input.txt",
        upload_path,
        debug_paragraph_enabled=dbg_para,
        debug_window_enabled=dbg_win,
    )
    use_demo = (use_demo_key or "").strip() in ("1", "true", "yes", "on")
    final_api_key = ""
    if use_demo:
        final_api_key = os.environ.get("NEBIUS_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    else:
        final_api_key = (api_key or "").strip()
    if not final_api_key:
        raise HTTPException(status_code=400, detail="API key is required (or sign in as demo).")
    if not (model or "").strip():
        raise HTTPException(status_code=400, detail="Model is required.")
    
    llm_overrides = {
        "provider": (provider or "").strip().lower(),
        "api_key": final_api_key,
        "base_url": (base_url or "").strip(),
        "model": (model or "").strip(),
        "mode": (mode or "").strip().lower(),
        "prompt_version": (prompt_version or "").strip().lower(),
    }
    background.add_task(process_job, job_id, llm_overrides)

    return UploadResponse(job_id=job_id, status=JobStatus.queued, limits=limits)


@app.get("/job/{job_id}", response_model=JobStatusResponse)
async def job_status(job_id: str, _: None = Depends(require_basic_auth)) -> JobStatusResponse:
    row = await get_job(paths.db, job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=row["job_id"],
        status=JobStatus(row["status"]),
        stage=row.get("stage"),
        progress=int(row.get("progress") or 0),
        error=row.get("error"),
        started_at=datetime.fromisoformat(row["started_at"]) if row.get("started_at") else None,
        updated_at=datetime.fromisoformat(row["updated_at"]),
        filename=row.get("filename"),
        estimated_minutes=int(row["estimated_minutes"]) if row.get("estimated_minutes") is not None else None,
    )

@app.post("/job/{job_id}/cancel")
async def cancel_job(job_id: str, _: None = Depends(require_basic_auth)) -> JSONResponse:
    """
    Request cooperative cancellation of a running job.
    This does not kill in-flight network calls, but it stops scheduling further work.
    """
    row = await get_job(paths.db, job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    status = row.get("status")
    if status in (JobStatus.done.value, JobStatus.failed.value, JobStatus.cancelled.value):
        return JSONResponse(content={"ok": True, "status": status})

    _CANCELLED.add(job_id)
    _cancel_job_tasks(job_id)
    # Show immediate feedback in UI.
    await update_job(paths.db, job_id, stage="cancelling", error="CANCEL_REQUESTED")
    return JSONResponse(content={"ok": True, "status": "cancelling"})


@app.get("/job/{job_id}/live-trace")
async def job_live_trace(job_id: str, _: None = Depends(require_basic_auth)) -> JSONResponse:
    """
    Returns the current in-progress trace for a running job (single object),
    intended to be polled by the UI and replaced each time.
    """
    row = await get_job(paths.db, job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    enabled = bool(int(row.get("debug_window_enabled") or 0))
    if not enabled and bool(int(row.get("debug_enabled") or 0)):
        enabled = True
    if not enabled:
        return JSONResponse(content={"enabled": False, "trace": None})
    return JSONResponse(content={"enabled": True, "trace": _LIVE_TRACE.get(job_id)})


@app.get("/job/{job_id}/result")
async def job_result(job_id: str, _: None = Depends(require_basic_auth)) -> JSONResponse:
    row = await get_job(paths.db, job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    if row["status"] != JobStatus.done.value:
        raise HTTPException(status_code=400, detail="Job not finished")
    result_path = row.get("result_path")
    if not result_path or not Path(result_path).exists():
        raise HTTPException(status_code=404, detail="Result not found")
    return JSONResponse(content=json.loads(Path(result_path).read_text(encoding="utf-8")))


@app.get("/job/{job_id}/pdf")
async def job_pdf(job_id: str, _: None = Depends(require_basic_auth)) -> FileResponse:
    row = await get_job(paths.db, job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    upload_path_s = row.get("upload_path")
    if not upload_path_s:
        raise HTTPException(status_code=404, detail="Document not found")
    upload_path = Path(upload_path_s)
    if not upload_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    if upload_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="PDF is not available for this job (text input was used)")
    filename = row.get("filename") or "document.pdf"
    return FileResponse(path=str(upload_path), media_type="application/pdf", filename=filename)

@app.get("/job/{job_id}/annotated", response_class=HTMLResponse)
async def job_annotated(job_id: str, _: None = Depends(require_basic_auth)) -> HTMLResponse:
    return FileResponse(str(TEMPLATES_DIR / "annotated.html"))


@app.get("/job/{job_id}/annotated-data")
async def job_annotated_data(job_id: str, _: None = Depends(require_basic_auth)) -> JSONResponse:
    row = await get_job(paths.db, job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    upload_path_s = row.get("upload_path")
    result_path_s = row.get("result_path")
    if not upload_path_s or not Path(upload_path_s).exists():
        raise HTTPException(status_code=404, detail="Document not found")
    if not result_path_s or not Path(result_path_s).exists():
        raise HTTPException(status_code=404, detail="Result not found")

    upload_path = Path(upload_path_s)
    result = json.loads(Path(result_path_s).read_text(encoding="utf-8"))
    findings = result.get("findings", [])

    # Filter out items explicitly marked as fair (is_unfair=False) and dedupe.
    total_findings = len(findings)
    unfair_findings = [f for f in findings if f.get("is_unfair") is not False]
    filtered_count = total_findings - len(unfair_findings)
    if filtered_count > 0:
        LOGGER.info(
            "Filtered %s findings with is_unfair=False (out of %s total)",
            filtered_count,
            total_findings,
        )
    unfair_findings = dedupe_findings([Finding(**f) for f in unfair_findings])
    unfair_findings_dict = [f.model_dump() for f in unfair_findings]

    limits = get_limits()
    if upload_path.suffix.lower() == ".txt":
        text_content = upload_path.read_text(encoding="utf-8")
        para_infos = _split_into_paragraphs(text_content)
        paragraphs = [
            Paragraph(
                text=pi.text,
                page=1,
                paragraph_index=i,
                bbox=None,
            )
            for i, pi in enumerate(para_infos)
        ]
        page_count = 1
        if limits["max_paragraphs"] > 0 and len(paragraphs) > limits["max_paragraphs"]:
            raise HTTPException(
                status_code=400,
                detail=f"Text produced > {limits['max_paragraphs']} paragraphs, exceeds MAX_PARAGRAPHS={limits['max_paragraphs']}.",
            )
    else:
        paragraphs, page_count = parse_pdf_to_paragraphs(
            upload_path,
            max_paragraphs=limits["max_paragraphs"],
        )

    paragraphs = _normalize_paragraphs(paragraphs)
    payload = {
        "page_count": page_count,
        "paragraphs": [
            {"paragraph_index": p.paragraph_index, "page": p.page, "text": p.text}
            for p in paragraphs
        ],
        "findings": unfair_findings_dict,
    }
    return JSONResponse(content=payload)

@app.get("/job/{job_id}/annotated.pdf")
async def job_annotated_pdf(job_id: str, _: None = Depends(require_basic_auth)) -> FileResponse:
    row = await get_job(paths.db, job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    upload_path_s = row.get("upload_path")
    result_path_s = row.get("result_path")
    if not upload_path_s or not Path(upload_path_s).exists():
        raise HTTPException(status_code=404, detail="Document not found")
    if not result_path_s or not Path(result_path_s).exists():
        raise HTTPException(status_code=404, detail="Result not found")

    upload_path = Path(upload_path_s)
    out_path = paths.results / f"{job_id}_annotated.pdf"

    try:
        result = json.loads(Path(result_path_s).read_text(encoding="utf-8"))
        findings = result.get("findings", [])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read result: {exc}")

    limits = get_limits()
    if upload_path.suffix.lower() == ".txt":
        text_content = upload_path.read_text(encoding="utf-8")
        para_infos = _split_into_paragraphs(text_content)
        paragraphs = [
            Paragraph(
                text=pi.text,
                page=1,
                paragraph_index=i,
                bbox=None,
            )
            for i, pi in enumerate(para_infos)
        ]
    else:
        paragraphs, _page_count = parse_pdf_to_paragraphs(
            upload_path,
            max_paragraphs=limits["max_paragraphs"],
        )

    paragraphs = _normalize_paragraphs(paragraphs)
    build_annotated_text_pdf(paragraphs=paragraphs, findings=findings, out_path=out_path)
    return FileResponse(path=str(out_path), media_type="application/pdf", filename=f"annotated_{job_id}.pdf")

async def process_job(job_id: str, llm_overrides: Optional[dict] = None) -> None:
    limits = get_limits()
    row = await get_job(paths.db, job_id)
    if not row:
        return
    upload_path = Path(row["upload_path"])
    enable_debug_paragraph = False
    enable_debug_window = False
    t0 = time.monotonic()
    try:
        LOGGER.info("Job %s started: %s", job_id, upload_path.name)
        _JOB_TASKS.setdefault(job_id, set())
        if _is_cancelled(job_id):
            await update_job(paths.db, job_id, status=JobStatus.cancelled, stage="cancelled", progress=100, error="CANCELLED")
            return
        await update_job(paths.db, job_id, status=JobStatus.running, stage="parse", progress=0, started_at=datetime.now(timezone.utc))
        
        # Check if it's a text file or PDF
        if upload_path.suffix.lower() == ".txt":
            # Parse text file directly
            text_content = upload_path.read_text(encoding="utf-8")
            para_infos = _split_into_paragraphs(text_content)
            paragraphs = []
            for idx, para_info in enumerate(para_infos):
                paragraphs.append(Paragraph(
                    text=para_info.text,
                    page=1,  # Text files are treated as single page
                    paragraph_index=idx,
                    bbox=None
                ))
            page_count = 1
            if limits["max_paragraphs"] > 0 and len(paragraphs) > limits["max_paragraphs"]:
                raise ValueError(f"Text produced > {limits['max_paragraphs']} paragraphs, exceeds MAX_PARAGRAPHS={limits['max_paragraphs']}.")
        else:
            # Parse PDF file
            paragraphs, page_count = parse_pdf_to_paragraphs(
                upload_path,
                max_paragraphs=limits["max_paragraphs"],
            )
        paragraphs = _normalize_paragraphs(paragraphs)
        LOGGER.info("Job %s [timing] parse done in %.1fs — pages=%s paragraphs=%s", job_id, time.monotonic() - t0, page_count, len(paragraphs))
        await update_job(
            paths.db,
            job_id,
            stage="stage1",
            progress=5,
            page_count=page_count,
            paragraph_count=len(paragraphs),
        )

        # Provider selection: allow per-job overrides from the UI (preferred),
        # then fall back to environment variables.
        base_url: Optional[str] = None
        message_mode = "string"
        overrides = llm_overrides or {}
        provider = (overrides.get("provider") or "").strip().lower()
        api_key = (overrides.get("api_key") or "").strip() or (os.environ.get("NEBIUS_API_KEY") or os.environ.get("OPENAI_API_KEY") or "")
        chosen_model = (overrides.get("model") or "").strip()
        override_base_url = (overrides.get("base_url") or "").strip()

        if provider == "openai":
            base_url = None
            message_mode = "string"
        else:
            provider = "nebius"
            base_url = override_base_url or os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
            # Nebius TokenFactory expects content as a list of text parts (matches their examples).
            message_mode = "parts"

        if not api_key:
            raise RuntimeError("API key required. Provide it in the UI (preferred) or set NEBIUS_API_KEY / OPENAI_API_KEY.")
        mode = (overrides.get("mode") or os.environ.get("CLASSIFIER_MODE", "uk_two_stage")).strip().lower()
        prompt_version = (overrides.get("prompt_version") or os.environ.get("REDFLAG_PROMPT_VERSION", "v1")).strip().lower()
        if mode:
            if mode in ("uk_two_stage", "uk"):
                use_uk_redflags = True
                single_classifier = True
            elif mode in ("single_legacy", "legacy", "v1", "old", "single"):
                use_uk_redflags = False
                single_classifier = True
            else:
                use_uk_redflags = False
                single_classifier = os.environ.get("SINGLE_CLASSIFIER", "1") == "1"
        else:
            use_uk_redflags = prompt_version in ("uk", "uk_v2", "v2_uk", "uk2")
            single_classifier = os.environ.get("SINGLE_CLASSIFIER", "1") == "1"
        # Load category definitions from CSV
        # To edit categories, modify:
        # - Two-stage mode: data/category_definitions_new.csv
        # - Legacy mode: data/category_descriptions.csv
        categories_csv = DATA_DIR / ("category_definitions_new.csv" if use_uk_redflags else "category_descriptions.csv")
        categories_version = "uk_tenancy_v1" if use_uk_redflags else "cuad_v1_41_from_csv"
        cats_payload = load_categories_payload(
            csv_path=categories_csv,
            json_path=CONFIGS_DIR / "categories.json",
            version=categories_version,
        )
        fewshot_enabled = os.environ.get("ENABLE_FEWSHOT", "0") == "1"
        fewshot_k = int(os.environ.get("FEWSHOT_MAX_PER_CATEGORY", "2"))
        fewshot_window = int(os.environ.get("FEWSHOT_CONTEXT_WINDOW", "240"))
        default_fs_dir = DATA_DIR / ("fewshot_redflags" if single_classifier else "fewshot_by_category")
        fewshot_dir = Path(os.environ.get("FEWSHOT_DIR", str(default_fs_dir)))
        only_fewshot_category_ids = (
            {
                "termination_for_convenience",
                "uncapped_liability",
                "irrevocable_or_perpetual_license",
                "most_favored_nation",
                "audit_rights",
                "ip_ownership_assignment",
            }
            if single_classifier
            else None
        )
        uk_fewshot_text = ""
        if use_uk_redflags and single_classifier:
            # Load few-shot examples from plain text file
            # To edit few-shot examples, modify: data/fewshot_uk_redflags.txt
            # Path can be overridden via UK_FEWSHOT_PATH environment variable
            uk_fewshot_path = Path(os.environ.get("UK_FEWSHOT_PATH", str(DATA_DIR / "fewshot_uk_redflags.txt")))
            if uk_fewshot_path.exists():
                uk_fewshot_text = uk_fewshot_path.read_text(encoding="utf-8").strip()
            fewshot_enabled = bool(uk_fewshot_text)
            fewshot = {}
        else:
            fewshot = load_or_build_fewshot(
                data_dir=DATA_DIR,
                categories_payload=cats_payload,
                enabled=fewshot_enabled,
                max_examples_per_category=fewshot_k,
                context_window=fewshot_window,
                out_dir=fewshot_dir,
                only_category_ids=only_fewshot_category_ids,
        )
        cuad_path = resolve_cuad_json_path(DATA_DIR)
        cuad_meta = cuad_file_meta(cuad_path) if cuad_path.exists() else {"path": str(cuad_path)}

        temperature = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
        seed_env = os.environ.get("LLM_SEED", "").strip()
        seed = int(seed_env) if seed_env else 1
        LOGGER.info(
            "Job %s LLM config: temperature=%.1f, seed=%s",
            job_id, temperature, seed if seed is not None else "none",
        )
        if provider == "openai":
            default_stage1 = os.environ.get("OPENAI_MODEL_STAGE1", "gpt-5-nano")
            default_stage2 = os.environ.get("OPENAI_MODEL_STAGE2", "gpt-5-mini")
        else:
            default_stage1 = os.environ.get("NEBIUS_MODEL_STAGE1") or os.environ.get("OPENAI_MODEL_STAGE1", "gpt-5-nano")
            default_stage2 = (
                os.environ.get("NEBIUS_MODEL_STAGE2")
                or os.environ.get("NEBIUS_MODEL")
                or os.environ.get("OPENAI_MODEL_STAGE2", "gpt-5-mini")
            )
        resolved_stage2 = chosen_model or default_stage2
        if use_uk_redflags:
            # Use the same model for both steps to avoid provider/model mismatches.
            default_stage1 = resolved_stage2
        llm = LLMRunner(
            LLMConfig(
                api_key=api_key,
                base_url=base_url,
                model_stage1=default_stage1,
                model_stage2=resolved_stage2,
                max_concurrency=limits["max_llm_concurrency"],
                stage1_include_descriptions=os.environ.get("STAGE1_INCLUDE_DESCRIPTIONS", "0") == "1",
                message_content_mode=message_mode,
                temperature=temperature,
                seed=seed,
            ),
            cats_payload,
            fewshot_by_category=fewshot,
            uk_fewshot_text=uk_fewshot_text,
        )

        enable_debug_paragraph = bool(int(row.get("debug_paragraph_enabled") or 0))
        enable_debug_window = bool(int(row.get("debug_window_enabled") or 0))
        # Backwards compat: if old jobs only have debug_enabled=1, enable both.
        if bool(int(row.get("debug_enabled") or 0)) and not (enable_debug_paragraph or enable_debug_window):
            enable_debug_paragraph = True
            enable_debug_window = True
        stage1_max_categories = int(os.environ.get("STAGE1_MAX_CATEGORIES", "10"))
        t_router = float(os.environ.get("T_ROUTER", "0.3"))
        router_topk = int(os.environ.get("ROUTER_TOPK_PROBS", "15"))
        debug_traces: list[dict] = []
        debug_by_idx: dict[int, dict] = {}

        if enable_debug_paragraph:
            for p in paragraphs:
                is_heading_para = _is_heading_paragraph(p.text)
                trace = {
                    "page": p.page,
                    "paragraph_index": p.paragraph_index,
                    "paragraph_text": p.text,
                    "section_path": "",
                    "kind": "heading" if is_heading_para else "paragraph",
                    "note": (
                        "Heading-only paragraph (section title). Not sent to the classifier."
                        if is_heading_para
                        else None
                    ),
                    "covered_by_windows": [],
                    "stage1": {"categories": []},
                    "stage1_raw": "",
                    # Note: in SINGLE_CLASSIFIER mode we primarily use window-level traces.
                    # We keep paragraph traces for parser transparency and section-path debugging.
                    "stage1_error": "SKIPPED_HEADING" if is_heading_para else None,
                    "stage2_error": "SKIPPED_HEADING" if is_heading_para else None,
                    "stage2_runs": [],
                }
                debug_traces.append(trace)
                debug_by_idx[p.paragraph_index] = trace

        await update_job(paths.db, job_id, stage="stage1", progress=10)

        # Compute section paths (Article/Section hierarchy) and build sections
        path_by_idx: dict[int, str] = {}
        stack: list[Heading] = []
        for p in paragraphs:
            h = parse_heading(p.text, max_len=160)
            if h is not None:
                while stack and stack[-1].level >= h.level:
                    stack.pop()
                stack.append(h)
            path_by_idx[p.paragraph_index] = _heading_stack_to_path(stack)
            if enable_debug_paragraph:
                t = debug_by_idx.get(p.paragraph_index)
                if t is not None:
                    t["section_path"] = path_by_idx[p.paragraph_index]

        # Build sections (heading-to-heading)
        # Filter out garbage paragraphs (signatures, dates, very short lines) - they shouldn't go to LLM
        filtered_paragraphs = _filter_garbage_paragraphs(paragraphs)
        sections: list[list[Paragraph]] = []
        cur: list[Paragraph] = []

        def _is_top_level_section_heading(h: Optional[Heading]) -> bool:
            if h is None:
                return False
            # Article-level headings are always top-level.
            if h.level == 0:
                return True
            # Section/Clause/§ headings: only split if the number has no dot.
            if h.level == 1:
                label = (h.label or "").strip()
                m = re.search(r"(?:Section|Clause|§)\s*(\d+(?:\.\d+)*)", label, re.IGNORECASE)
                if m:
                    return "." not in m.group(1)
                return True
            # Pure numeric headings like "1." or "2" are level 2 in our parser.
            if h.level == 2:
                label = (h.label or "").strip()
                return "." not in label
            return False
        
        # Paragraph starts with "N. Title" then body. Body starts with capital: after " ", ". ", or " N.M ".
        heading_with_body_pattern = re.compile(
            r"^\s*(\d+(?:\.\d+)*)\s*\.?\s+([A-Z][A-Za-z\s]{2,}?)\s+(?:\.\s+|\d+\.\d+\s+)?[A-Z]",
            re.IGNORECASE
        )

        for p in filtered_paragraphs:
            para_text = (p.text or "").strip()
            heading = parse_heading(para_text, max_len=160)
            is_section_heading = _is_top_level_section_heading(heading)

            # Paragraph starts with heading + body: "N. Title. Body" or "N. Title N.M Body"
            m = heading_with_body_pattern.match(para_text)
            starts_with_heading = False
            if m:
                num_part = m.group(1)
                title_part = m.group(2).strip()
                if title_part.endswith('.'):
                    title_part = title_part[:-1].strip()
                words = title_part.split()[:10]
                if words:
                    uppercase_words = sum(1 for w in words if w.isupper() and len(w) > 1)
                    titlecase_words = sum(1 for w in words if w and w[0].isupper() and not w.isupper())
                    uppercase_ratio = uppercase_words / len(words) if words else 0
                    titlecase_ratio = titlecase_words / len(words) if words else 0
                    is_short_heading = len(words) <= 5
                    is_uppercase_heading = uppercase_ratio >= 0.5
                    is_title_case_short = (2 <= len(words) <= 4 and titlecase_ratio >= 0.5)
                    is_heading_like = (is_uppercase_heading or is_short_heading or is_title_case_short)
                    starts_with_heading = is_heading_like and num_part.count(".") == 0
            
            if (is_section_heading or starts_with_heading) and cur:
                sections.append(cur)
                cur = [p]
            else:
                cur.append(p)
        if cur:
            sections.append(cur)

        # index -> (section_i, pos_in_section)
        section_by_idx: dict[int, tuple[int, int]] = {}
        for si, sec in enumerate(sections):
            for pi, p in enumerate(sec):
                section_by_idx[p.paragraph_index] = (si, pi)

        # Section-owned heading and path (authoritative for context; one section = one heading)
        section_heading_by_i: dict[int, Optional[Paragraph]] = {}
        section_path_by_i: dict[int, str] = {}
        for si, sec in enumerate(sections):
            h = sec[0] if sec and _is_heading_paragraph(sec[0].text) else None
            section_heading_by_i[si] = h
            section_path_by_i[si] = path_by_idx.get(h.paragraph_index, "") if h else ""

        window_tokens = int(os.environ.get("STAGE1_WINDOW_TOKENS", "1100"))
        stride_tokens = int(os.environ.get("STAGE1_STRIDE_TOKENS", "300"))
        edge_frac = float(os.environ.get("STAGE1_TRIGGER_EDGE_FRAC", "0.15"))
        llm_max_retries = int(os.environ.get("LLM_MAX_RETRIES", "3"))
        llm_retry_sleep = float(os.environ.get("LLM_RETRY_SLEEP", "0.8"))
        quote_repair_enabled = os.environ.get("LLM_QUOTE_REPAIR", "1") == "1"
        max_chars = _approx_chars_limit()

        def approx_size(ps: list[Paragraph]) -> int:
            return sum(len((p.text or "")) + 40 for p in ps)

        window_traces: list[dict] = []
        failed_windows: list[dict] = []

        if single_classifier:
            # Single-classifier mode: each window is classified directly into red-flag findings.
            await update_job(paths.db, job_id, stage="processing", progress=55)

            findings: list[Finding] = []
            window_findings: list[tuple[str, int, Finding]] = []

            def recommend_for(cat: str) -> str:
                if cat == "Uncapped Liability":
                    return "Confirm whether liability is capped and whether carve-outs are acceptable."
                if cat == "Termination For Convenience":
                    return "Assess whether termination without cause is acceptable; consider adding constraints/fees."
                if cat == "Irrevocable Or Perpetual License":
                    return "Review whether perpetual/irrevocable rights are intended; consider term/termination limits."
                if cat == "Most Favored Nation":
                    return "Evaluate MFN obligations; consider narrowing scope and defining comparators."
                if cat == "Audit Rights":
                    return "Check audit scope, frequency, confidentiality, and cost allocation."
                return "Review the clause for compliance and propose a tenant-fair revision."

            async def classify_window(
                sec_body: list[Paragraph], start_i: int, end_i: int, *, window_id: str,
                main_start: Optional[int] = None, main_end: Optional[int] = None,
                sec_heading: Optional[Paragraph] = None, sec_path: str = "",
                all_sec_paras: Optional[list[Paragraph]] = None,  # Full section for quote search across boundaries
            ) -> None:
                # Check cancellation at the very start
                if _is_cancelled(job_id):
                    return
                
                chunk_paras = sec_body[start_i:end_i]
                
                # Skip windows that contain only signatures/contact info (not real content)
                # Check if all paragraphs in chunk are signatures/contact info
                import re
                signature_patterns = [
                    re.compile(r"^\s*By:\s*", re.IGNORECASE),
                    re.compile(r"^\s*By:\s+.*By:\s+", re.IGNORECASE),  # Multiple "By:" lines
                    re.compile(r"^\s*/s/\s*", re.IGNORECASE),
                    re.compile(r"^\s*Signed:\s*", re.IGNORECASE),
                    re.compile(r"^\s*Signature:\s*", re.IGNORECASE),
                    re.compile(r"^\s*Attn:\s+", re.IGNORECASE),
                    re.compile(r"^\s*Attention:\s+", re.IGNORECASE),
                    re.compile(r"^\s*If\s+to\s+", re.IGNORECASE),
                    re.compile(r"^\s*With\s+a\s+copy\s+to:\s*", re.IGNORECASE),
                ]
                
                # Check if chunk contains only signatures/contact info
                all_signatures = True
                meaningful_text = ""
                for p in chunk_paras:
                    text = (p.text or "").strip()
                    if not text:
                        continue
                    is_sig = any(pattern.match(text) for pattern in signature_patterns)
                    if not is_sig:
                        all_signatures = False
                        meaningful_text += text + " "
                
                # If chunk is only signatures/contact info and has no meaningful content, skip it
                if all_signatures and len(meaningful_text.strip()) < 20:
                    LOGGER.debug(f"Window {window_id}: Skipping signature/contact info chunk")
                    return
                
                pages = sorted({int(p.page) for p in chunk_paras})
                page_min = pages[0] if pages else None
                page_max = pages[-1] if pages else None
                # Paragraphs are already normalized via _normalize_paragraphs, use as-is
                body_raw = "\n\n".join(p.text for p in chunk_paras).strip()
                # Prepend section heading so LLM always sees which section the chunk belongs to
                if sec_heading and (sec_heading.text or "").strip():
                    text_chunk_raw = f"[Section: {(sec_heading.text or '').strip()}]\n\n{body_raw}"
                else:
                    text_chunk_raw = body_raw
                # Text is already normalized, use as-is
                text_chunk = text_chunk_raw
                if not text_chunk:
                    return
                if _is_cancelled(job_id):
                    return
                
                spath = sec_path or ""
                if any(p.paragraph_index == 48 for p in chunk_paras):
                    LOGGER.info(f"Window {window_id}: section_path='{spath}', paragraphs={[p.paragraph_index for p in chunk_paras]}")
                if enable_debug_paragraph:
                    for p in chunk_paras:
                        t = debug_by_idx.get(p.paragraph_index)
                        if t is not None:
                            lst = t.get("covered_by_windows")
                            if isinstance(lst, list) and window_id not in lst:
                                lst.append(window_id)

                if enable_debug_window:
                    _set_live_trace(
                        job_id,
                        {
                            "phase": "calling_llm",
                            "ts": time.time(),
                            "window_id": window_id,
                            "section_path": spath,
                            "paragraph_indices": [p.paragraph_index for p in chunk_paras],
                            "page_min": page_min,
                            "page_max": page_max,
                            "page_count": page_count,
                            "processing": True,
                            "context_prefix": spath,
                            "text_chunk": text_chunk[:6000],
                        },
                    )
                out = None
                raw = ""
                prompt = None
                repair_logs: list[dict] = []
                final_findings: list[dict] = []
                for attempt in range(llm_max_retries):
                    try:
                        if use_uk_redflags:
                            out, raw, prompt = await llm.classify_redflags_for_chunk_v2(
                                text_chunk=text_chunk,
                                section_path=spath,
                                return_prompt=enable_debug_window,
                            )
                        else:
                            out, raw, prompt = await llm.classify_redflags_for_chunk(
                                text_chunk=text_chunk,
                                section_path=spath,
                                return_prompt=enable_debug_window,
                            )
                        # Log summary of LLM response
                        findings_count = len(out.findings or [])
                        if findings_count > 0:
                            LOGGER.info(
                                f"Window {window_id}: LLM returned {findings_count} findings. "
                                f"Categories: {[f.category for f in (out.findings or [])]}"
                            )
                        else:
                            LOGGER.debug(
                                f"Window {window_id}: LLM returned 0 findings"
                            )
                        break
                    except Exception as exc:
                        if _is_cancelled(job_id):
                            return
                        if attempt < llm_max_retries - 1:
                            LOGGER.warning(
                                "Window %s: LLM call failed (%s/%s), retrying: %s",
                                window_id,
                                attempt + 1,
                                llm_max_retries,
                                str(exc)[:2000],
                            )
                            await asyncio.sleep(llm_retry_sleep * (attempt + 1))
                            continue
                        if enable_debug_window:
                            failed_windows.append(
                                {
                                    "window_id": window_id,
                                    "section_path": spath,
                                    "paragraph_indices": [p.paragraph_index for p in chunk_paras],
                                    "error": str(exc)[:2000],
                                }
                            )
                        raise
                if _is_cancelled(job_id):
                    return

                chunk_for_llm = text_chunk
                if enable_debug_window:
                    prompt_user = (prompt or {}).get("user", "")
                    prompt_system = (prompt or {}).get("system", "")
                    sections = (prompt or {}).get("sections") if isinstance(prompt, dict) else None
                    router_info = (prompt or {}).get("router") if isinstance(prompt, dict) else None
                    # Fall back to extracting minimal parts if sections are not present.
                    ctx = ""
                    fewshot = ""
                    defs = ""
                    if isinstance(sections, dict):
                        ctx = str(sections.get("context_prefix") or "")
                        chunk_for_llm = str(sections.get("text_chunk") or "")
                        fewshot = str(sections.get("fewshot_examples") or "")
                        defs = str(sections.get("category_definitions") or "")
                    head_n = 9000
                    tail_n = 9000
                    prompt_user_head = prompt_user[:head_n]
                    prompt_user_tail = prompt_user[-tail_n:] if len(prompt_user) > head_n else ""
                    window_traces.append(
                        {
                            "window_id": window_id,
                            "section_path": spath,
                            "paragraph_indices": [p.paragraph_index for p in chunk_paras],
                            "text_chunk": text_chunk[:8000],
                            "text_chunk_raw": text_chunk_raw[:8000],
                            "prompt_system": prompt_system[:2000],
                            "prompt_user_len": len(prompt_user),
                            "prompt_user_head": prompt_user_head,
                            "prompt_user_tail": prompt_user_tail,
                            "prompt_parts": {
                                "context_prefix": ctx,
                                "text_chunk": chunk_for_llm[:8000],
                                "fewshot": fewshot[:8000],
                                "category_definitions": defs[:4000],
                            },
                            "router": router_info,
                            "raw_output": (raw or "")[:8000],
                            "final_findings": final_findings,
                            "quote_repairs": repair_logs,
                            "parsed": out.model_dump(),
                        }
                    )
                    _set_live_trace(
                        job_id,
                        {
                            "phase": "llm_returned",
                            "ts": time.time(),
                            "window_id": window_id,
                            "section_path": spath,
                            "paragraph_indices": [p.paragraph_index for p in chunk_paras],
                            "page_min": page_min,
                            "page_max": page_max,
                            "page_count": page_count,
                            "processing": False,
                            "prompt_system": (prompt_system or "")[:2000],
                            "prompt_user_len": len(prompt_user or ""),
                            "context_prefix": ctx or spath,
                            "category_definitions": defs[:4000],
                            "fewshot": fewshot[:8000],
                            "text_chunk": chunk_for_llm[:8000] or text_chunk[:8000],
                            "router": router_info,
                            "raw_output": (raw or "")[:6000],
                            "final_findings": final_findings,
                            "quote_repairs": repair_logs,
                            "parsed": out.model_dump(),
                        },
                    )

                # Process all findings (no limit)
                findings_to_process = out.findings
                
                for j, f in enumerate(findings_to_process):
                    # Map model-provided severity_of_consequences (0..3) to low/medium/high.
                    sev_score = int(getattr(f.risk_assessment, "severity_of_consequences", 1) or 1)
                    if sev_score >= 2:
                        sev = "high"
                    elif sev_score == 1:
                        sev = "medium"
                    else:
                        sev = "low"
                    loc_para = None
                    evidence_quote = f.quote or ""
                    if not f.quote:
                        # LLM did not return an evidence quote at all.
                        # Skip this finding - without a quote, we can't highlight specific text,
                        # and it's likely a false positive or the finding is not well-grounded.
                        LOGGER.warning(
                            f"Window {window_id}, finding {j+1} ({f.category}): "
                            f"✗ SKIPPING - LLM did not provide a quote. Cannot highlight specific evidence."
                        )
                        continue
                    else:
                        # Try exact match first (paragraphs are already normalized)
                        found_quote = False
                        for p in chunk_paras:
                            para_text = p.text or ""
                            if f.quote and f.quote in para_text:
                                loc_para = p
                                evidence_quote = f.quote
                                found_quote = True
                                break
                        
                        if not found_quote:
                            # Try normalized match to handle whitespace differences
                            quote_norm = _normalize_for_match(f.quote or "")
                            if quote_norm and len(quote_norm) > 20:  # Only for substantial quotes
                                for p in chunk_paras:
                                    para_text_norm = _normalize_for_match(p.text or "")
                                    if quote_norm in para_text_norm:
                                        # Found in normalized text - extract actual quote from original paragraph
                                        # This handles cases where quote is part of a longer sentence
                                        para_text = p.text or ""
                                        quote_words = quote_norm.split()
                                        
                                        # Try to find quote by matching key words (at least 5 words for reliability)
                                        if len(quote_words) >= 5:
                                            # Use first 5 words to find start position
                                            search_start = " ".join(quote_words[:5]).lower()
                                            para_lower = para_text.lower()
                                            start_pos = para_lower.find(search_start)
                                            
                                            if start_pos >= 0:
                                                # Found start, now find end using last 5 words
                                                search_end = " ".join(quote_words[-5:]).lower()
                                                end_pos = para_lower.find(search_end, start_pos)
                                                
                                                if end_pos > start_pos:
                                                    # Extract actual quote from paragraph
                                                    actual_end = end_pos + len(search_end)
                                                    extracted_quote = para_text[start_pos:actual_end].strip()
                                                    
                                                    # Verify extracted quote is reasonable (at least 50% of original length)
                                                    if len(extracted_quote) >= len(f.quote or "") * 0.5:
                                                        loc_para = p
                                                        evidence_quote = extracted_quote
                                                        found_quote = True
                                                        LOGGER.debug(
                                                            f"Window {window_id}, finding {j+1}: "
                                                            f"Extracted quote from paragraph using word matching"
                                                        )
                                                        break
                                        
                                        # If word-based extraction failed, try simpler approach:
                                        # Find quote by matching first and last significant words
                                        if not found_quote and len(quote_words) >= 3:
                                            # Use first 3 and last 3 words
                                            first_words = " ".join(quote_words[:3]).lower()
                                            last_words = " ".join(quote_words[-3:]).lower()
                                            para_lower = para_text.lower()
                                            
                                            start_pos = para_lower.find(first_words)
                                            if start_pos >= 0:
                                                end_pos = para_lower.find(last_words, start_pos)
                                                if end_pos > start_pos:
                                                    actual_end = end_pos + len(last_words)
                                                    extracted_quote = para_text[start_pos:actual_end].strip()
                                                    if len(extracted_quote) >= len(f.quote or "") * 0.5:
                                                        loc_para = p
                                                        evidence_quote = extracted_quote
                                                        found_quote = True
                                                        LOGGER.debug(
                                                            f"Window {window_id}, finding {j+1}: "
                                                            f"Extracted quote using simplified word matching"
                                                        )
                                                        break
                                        
                                        # Final fallback: if quote found in normalized text but extraction failed,
                                        # use original LLM quote (it's better than highlighting entire paragraph)
                                        if not found_quote:
                                            loc_para = p
                                            evidence_quote = f.quote  # Use original quote from LLM
                                            found_quote = True
                                            LOGGER.debug(
                                                f"Window {window_id}, finding {j+1}: "
                                                f"Using original LLM quote (found in normalized text but extraction failed)"
                                            )
                                            break
                        
                        # If quote not found in individual paragraphs, try searching in combined window text.
                        # This handles cases where the quote spans multiple paragraphs (separated by "\n\n").
                        if not found_quote and f.quote and chunk_paras:
                            separator = "\n\n"
                            body_raw = separator.join((p.text or "") for p in chunk_paras).strip()

                            # Pre-compute character spans of each paragraph inside body_raw.
                            raw_spans: list[tuple[Paragraph, int, int]] = []
                            cur_pos = 0
                            for idx_p, p in enumerate(chunk_paras):
                                t = p.text or ""
                                start = cur_pos
                                end = start + len(t)
                                raw_spans.append((p, start, end))
                                if idx_p < len(chunk_paras) - 1:
                                    cur_pos = end + len(separator)
                                else:
                                    cur_pos = end

                            # 1) Exact search in combined raw text.
                            if f.quote in body_raw:
                                quote_pos = body_raw.find(f.quote)
                                if quote_pos >= 0:
                                    for para_obj, start, end in raw_spans:
                                        if start <= quote_pos < end:
                                            loc_para = para_obj
                                            evidence_quote = f.quote
                                            found_quote = True
                                            break

                            # 2) Normalized search in combined text to handle whitespace differences.
                            if not found_quote:
                                para_norms: list[str] = [
                                    _normalize_for_match(p.text or "") for p in chunk_paras
                                ]
                                body_raw_norm = " ".join(para_norms)
                                quote_norm = _normalize_for_match(f.quote or "")
                                if quote_norm and quote_norm in body_raw_norm:
                                    quote_pos = body_raw_norm.find(quote_norm)
                                    if quote_pos >= 0:
                                        norm_spans: list[tuple[Paragraph, int, int]] = []
                                        cur_pos = 0
                                        for idx_p, (p, p_norm) in enumerate(zip(chunk_paras, para_norms)):
                                            start = cur_pos
                                            end = start + len(p_norm)
                                            norm_spans.append((p, start, end))
                                            if idx_p < len(para_norms) - 1:
                                                cur_pos = end + 1  # one space between normalized paragraphs
                                            else:
                                                cur_pos = end

                                        for para_obj, start, end in norm_spans:
                                            if start <= quote_pos < end:
                                                loc_para = para_obj
                                                evidence_quote = f.quote  # Use original quote from LLM
                                                found_quote = True
                                                break
                        
                        if not found_quote and quote_repair_enabled and f.quote:
                            search_paras = list(chunk_paras)
                            if all_sec_paras:
                                if start_i > 0 and start_i <= len(all_sec_paras):
                                    prev_para = all_sec_paras[start_i - 1]
                                    if prev_para not in chunk_paras:
                                        search_paras.insert(0, prev_para)
                                if end_i < len(all_sec_paras):
                                    next_para = all_sec_paras[end_i]
                                    if next_para not in chunk_paras:
                                        search_paras.append(next_para)
                            
                            for p in search_paras:
                                try:
                                    repaired = await llm.repair_quote_from_paragraph(
                                        paragraph_text=p.text or "", target_quote=f.quote
                                    )
                                    if repaired:
                                        para_text_norm = _normalize_for_match(p.text or "")
                                        repaired_norm = _normalize_for_match(repaired)
                                        original_norm = _normalize_for_match(f.quote or "")
                                        # Only accept repaired quote if:
                                        # 1. It's found in paragraph text
                                        # 2. It's not much longer than original (max 2x length to avoid "babbling")
                                        # 3. It's not much shorter than original (min 0.5x length to avoid truncation)
                                        if (repaired_norm and repaired_norm in para_text_norm and
                                            len(repaired) <= len(f.quote or "") * 2 and
                                            len(repaired) >= len(f.quote or "") * 0.5):
                                            loc_para = p
                                            evidence_quote = repaired
                                            found_quote = True
                                            repair_logs.append(
                                                {
                                                    "finding_index": j + 1,
                                                    "category": f.category,
                                                    "original_quote": f.quote,
                                                    "repaired_quote": repaired,
                                                    "matched": True,
                                                    "method": "llm_repair_search",
                                                }
                                            )
                                            break
                                        else:
                                            LOGGER.warning(
                                                f"Window {window_id}, finding {j+1} ({f.category}): "
                                                f"LLM repair returned quote that does not appear in paragraph text. Ignoring repair."
                                            )
                                except Exception as exc:
                                    LOGGER.debug(
                                        f"Window {window_id}, finding {j+1} ({f.category}): "
                                        f"LLM repair failed for paragraph {p.paragraph_index}: {str(exc)[:200]}"
                                    )
                                    continue
                        
                        if not found_quote:
                            # Last attempt: try to find quote by matching key words even if exact match fails
                            # This handles cases where quote is part of a longer sentence with different punctuation
                            quote_norm = _normalize_for_match(f.quote or "")
                            if quote_norm and len(quote_norm) > 30:  # Only for substantial quotes
                                quote_words = quote_norm.split()
                                # Use first 5-7 words to find the paragraph containing the quote
                                if len(quote_words) >= 5:
                                    search_phrase = " ".join(quote_words[:min(7, len(quote_words))])
                                    for p in chunk_paras:
                                        para_text_norm = _normalize_for_match(p.text or "")
                                        if search_phrase in para_text_norm:
                                            # Found paragraph - try to extract actual quote
                                            para_text = p.text or ""
                                            para_lower = para_text.lower()
                                            
                                            # Find start using first 5 words
                                            first_5 = " ".join(quote_words[:5]).lower()
                                            start_pos = para_lower.find(first_5)
                                            
                                            if start_pos >= 0:
                                                # Find end using last 5 words
                                                last_5 = " ".join(quote_words[-5:]).lower()
                                                end_pos = para_lower.find(last_5, start_pos)
                                                
                                                if end_pos > start_pos:
                                                    actual_end = end_pos + len(last_5)
                                                    extracted_quote = para_text[start_pos:actual_end].strip()
                                                    
                                                    # Verify it's reasonable (at least 60% of original length)
                                                    if len(extracted_quote) >= len(f.quote or "") * 0.6:
                                                        loc_para = p
                                                        evidence_quote = extracted_quote
                                                        found_quote = True
                                                        LOGGER.info(
                                                            f"Window {window_id}, finding {j+1}: "
                                                            f"✓ Found quote using key word matching: extracted {len(extracted_quote)} chars"
                                                        )
                                                        break
                            
                            if not found_quote:
                                LOGGER.warning(
                                    f"Window {window_id}, finding {j+1} ({f.category}): "
                                    f"✗ SKIPPING - Could not locate quote in paragraphs after ALL search methods: "
                                    f"{f.quote[:100] if f.quote else 'EMPTY'}. "
                                    f"To avoid highlighting entire paragraphs without specific evidence, this finding is skipped."
                                )
                                continue  # Skip this finding - don't use fallback that highlights entire paragraph

                    cat_id = llm.resolve_category_id(f.category)
                    cat_name = llm.category_name_by_id.get(cat_id, f.category)
                    reasoning_bullets = "\n".join(f"- {b}" for b in f.reasoning)
                    what = reasoning_bullets or f.explanation
                    revision_expl = (f.recommended_revision.revision_explanation or "").strip()
                    revised_clause = (f.recommended_revision.revised_clause or "").strip()
                    
                    if not loc_para:
                        if chunk_paras:
                            loc_para = chunk_paras[0]
                        else:
                            LOGGER.error(
                                f"Window {window_id}, finding {j+1} ({f.category}): "
                                f"✗ CRITICAL: No loc_para and no chunk_paras! Cannot create finding. "
                                f"Quote: {f.quote[:100] if f.quote else 'EMPTY'}"
                            )
                            continue  # Skip this finding
                    
                    if not evidence_quote:
                        # Final safeguard: prefer the original LLM quote if present.
                        # If the quote is empty, fall back to a short snippet from the paragraph
                        # instead of the entire paragraph to avoid over-broad evidence.
                        if f.quote:
                            evidence_quote = f.quote
                        elif loc_para:
                            para_text = (loc_para.text or "").strip()
                            evidence_quote = para_text[:300]
                        else:
                            evidence_quote = ""
                    
                    # Skip findings without evidence quotes - if LLM couldn't provide a quote,
                    # it's likely a false positive or the finding is not well-grounded.
                    # We don't want to highlight entire paragraphs without specific evidence.
                    if not evidence_quote or not evidence_quote.strip():
                        LOGGER.warning(
                            f"Window {window_id}, finding {j+1} ({f.category}): "
                            f"✗ SKIPPING - no evidence quote available. LLM did not provide quote and "
                            f"we could not locate it in paragraphs."
                        )
                        continue
                    
                    if loc_para and evidence_quote and quote_repair_enabled:
                        para_text = loc_para.text or ""
                        if _normalize_for_match(evidence_quote) not in _normalize_for_match(para_text):
                            try:
                                repaired = await llm.repair_quote_from_paragraph(
                                    paragraph_text=para_text, target_quote=evidence_quote
                                )
                                if repaired:
                                    para_text_norm = _normalize_for_match(para_text)
                                    repaired_norm = _normalize_for_match(repaired)
                                    original_norm = _normalize_for_match(evidence_quote)
                                    # Only accept repaired quote if:
                                    # 1. It's found in paragraph text
                                    # 2. It's not much longer than original (max 2x length to avoid "babbling")
                                    # 3. It's not much shorter than original (min 0.5x length to avoid truncation)
                                    if (repaired_norm and repaired_norm in para_text_norm and
                                        len(repaired) <= len(evidence_quote) * 2 and
                                        len(repaired) >= len(evidence_quote) * 0.5):
                                        repair_logs.append(
                                            {
                                                "finding_index": j + 1,
                                                "category": f.category,
                                                "original_quote": evidence_quote,
                                                "repaired_quote": repaired,
                                                "matched": True,
                                            }
                                        )
                                        evidence_quote = repaired
                                    else:
                                        repair_logs.append(
                                            {
                                                "finding_index": j + 1,
                                                "category": f.category,
                                                "original_quote": evidence_quote,
                                                "repaired_quote": repaired,
                                                "matched": False,
                                                "error": "Repaired quote not found in paragraph (normalized)",
                                            }
                                        )
                                        LOGGER.warning(
                                            f"Window {window_id}, finding {j+1} ({f.category}): "
                                            f"LLM repair returned quote that does not appear in paragraph text. Ignoring repair."
                                        )
                                else:
                                    repair_logs.append(
                                        {
                                            "finding_index": j + 1,
                                            "category": f.category,
                                            "original_quote": evidence_quote,
                                            "repaired_quote": "",
                                            "matched": False,
                                        }
                                    )
                            except Exception as exc:
                                repair_logs.append(
                                    {
                                        "finding_index": j + 1,
                                        "category": f.category,
                                        "original_quote": evidence_quote,
                                        "repaired_quote": "",
                                        "matched": False,
                                        "error": str(exc)[:200],
                                    }
                                )
                                LOGGER.warning(
                                    "Quote repair failed for window %s: %s", window_id, str(exc)[:200]
                                )
                    
                    original_refs = f.legal_references or []
                    valid_refs = _filter_valid_legal_references(original_refs)
                    
                    # If ALL references are invalid, this may indicate LLM hallucination.
                    # Option 1: Skip the finding entirely (controlled by env var)
                    # Option 2: Try to regenerate legal references once (controlled by env var, default: enabled)
                    skip_if_all_refs_invalid = os.environ.get("SKIP_FINDING_IF_ALL_REFS_INVALID", "0") == "1"
                    regenerate_if_all_invalid = os.environ.get("REGENERATE_LEGAL_REFS_IF_ALL_INVALID", "1") == "1"
                    
                    if len(original_refs) > 0 and len(valid_refs) == 0:
                        invalid_refs = original_refs
                        
                        if skip_if_all_refs_invalid:
                            LOGGER.warning(
                                f"Window {window_id}, finding {j+1} ({f.category}): "
                                f"✗ SKIPPING - ALL legal references are invalid (hallucinated?): {invalid_refs}"
                            )
                            continue  # Skip this finding
                        
                        # Try to regenerate legal references once
                        if regenerate_if_all_invalid:
                            try:
                                LOGGER.info(
                                    f"Window {window_id}, finding {j+1} ({f.category}): "
                                    f"Attempting to regenerate legal references (invalid: {invalid_refs})"
                                )
                                regenerated_refs = await llm.regenerate_legal_references(
                                    finding=f,
                                    text_chunk=text_chunk,
                                    section_path=spath,
                                )
                                if regenerated_refs:
                                    valid_refs = regenerated_refs
                                    LOGGER.info(
                                        f"Window {window_id}, finding {j+1} ({f.category}): "
                                        f"✓ Regenerated {len(regenerated_refs)} valid legal reference(s): {regenerated_refs}"
                                    )
                                else:
                                    LOGGER.warning(
                                        f"Window {window_id}, finding {j+1} ({f.category}): "
                                        f"⚠️ Regeneration failed or returned no valid references. "
                                        f"Keeping finding but without legal references."
                                    )
                            except Exception as exc:
                                LOGGER.warning(
                                    f"Window {window_id}, finding {j+1} ({f.category}): "
                                    f"Failed to regenerate legal references: {exc}. "
                                    f"Keeping finding but without legal references."
                                )
                        else:
                            LOGGER.warning(
                                f"Window {window_id}, finding {j+1} ({f.category}): "
                                f"⚠️ ALL legal references are invalid (hallucinated?): {invalid_refs}. "
                                f"Keeping finding but without legal references."
                            )
                    elif len(original_refs) > len(valid_refs):
                        invalid_refs = [r for r in original_refs if r not in valid_refs]
                        LOGGER.warning(
                            f"Window {window_id}, finding {j+1} ({f.category}): "
                            f"Filtered out {len(invalid_refs)} invalid legal reference(s): {invalid_refs}"
                        )
                    
                    new_finding = Finding(
                        finding_id=f"{window_id}_f{j+1}",
                        category_id=cat_id,
                        category_name=cat_name,
                        severity=sev,  # type: ignore[arg-type]
                        confidence=0.8,
                        issue_title=f.category,
                        what_is_wrong=what,
                        explanation=f.explanation,
                        recommendation=revision_expl or recommend_for(f.category),
                        evidence_quote=evidence_quote,
                        is_unfair=bool(f.is_unfair),
                        legal_references=valid_refs[:6],
                        possible_consequences=f.possible_consequences,
                        risk_assessment=f.risk_assessment,
                        consequences_category=f.consequences_category,
                        risk_category=f.risk_category,
                        revised_clause=revised_clause,
                        revision_explanation=revision_expl,
                        suggested_follow_up=f.suggested_follow_up,
                        paragraph_bbox=loc_para.bbox if loc_para else None,
                        location={"page": int(loc_para.page) if loc_para else 0, "paragraph_index": int(loc_para.paragraph_index) if loc_para else 0},
                    )
                    para_idx = int(loc_para.paragraph_index) if loc_para else 0
                    window_findings.append((window_id, para_idx, new_finding))
                    final_findings.append(
                        {
                            "finding_index": j + 1,
                            "category": new_finding.category_name,
                            "evidence_quote": new_finding.evidence_quote,
                        }
                    )

            # Count windows with the same logic as the loop below (for accurate time estimate)
            _n_windows = 0
            for _si, _sec in enumerate(sections):
                _sec_body = [p for p in _sec if not _is_heading_paragraph(p.text)]
                if not _sec_body:
                    continue
                _starts: list[int] = [0]
                while True:
                    _prev = _starts[-1]
                    _need = stride_tokens
                    _i = _prev
                    while _i < len(_sec_body) and _need > 0:
                        _need -= _approx_tokens(_sec_body[_i].text)
                        _i += 1
                    if _i >= len(_sec_body):
                        break
                    _starts.append(_i)
                _last_main_s, _last_main_e = -1, -1
                for _s in _starts:
                    _t = 0
                    _e = _s
                    while _e < len(_sec_body) and _t < window_tokens:
                        _t += _approx_tokens(_sec_body[_e].text)
                        _e += 1
                    if _e <= _s:
                        _e = min(len(_sec_body), _s + 1)
                    if _last_main_s >= 0 and _last_main_e >= 0 and _s >= _last_main_s and _e <= _last_main_e:
                        continue
                    _main_block = _sec_body[_s:_e]
                    _joined = "\n\n".join(p.text for p in _main_block)
                    _near_start, _near_end = _find_trigger_edges(_joined, edge_frac)
                    _exp_s, _exp_e = _extend_window_by_stride(
                        _sec_body, _s, _e, stride_tokens=stride_tokens, extend_left=_near_start, extend_right=_near_end
                    )
                    _exp_s = max(0, min(_exp_s, len(_sec_body)))
                    _exp_e = max(_exp_s, min(_exp_e, len(_sec_body)))
                    _trimmed = _exp_e
                    for _ii in range(_exp_s, _exp_e):
                        if _is_signature_paragraph((_sec_body[_ii].text or "").strip()):
                            _trimmed = _ii
                            break
                    _exp_e = _trimmed
                    if _exp_e <= _exp_s:
                        _exp_s, _exp_e = _s, _e
                        if _exp_e <= _exp_s:
                            _exp_e = min(_exp_s + 1, len(_sec_body))
                    _expanded = _sec_body[_exp_s:_exp_e]
                    if _expanded:
                        _expanded = [p for p in _expanded if not _is_heading_paragraph(p.text)]
                        if _expanded:
                            _first_si = section_by_idx.get(_expanded[0].paragraph_index)
                            if _first_si:
                                _filtered = []
                                for _p in _expanded:
                                    _ps = section_by_idx.get(_p.paragraph_index)
                                    if _ps and _ps[0] == _first_si[0]:
                                        _filtered.append(_p)
                                    else:
                                        break
                                _expanded = _filtered if _filtered else _sec_body[_s:_e]
                            if not _expanded:
                                _expanded = _sec_body[_s:_e] if _s < len(_sec_body) else []
                    _start_i = _exp_s
                    _end_i = _exp_s + len(_expanded) if _expanded else _exp_s
                    _total_size = approx_size(_expanded) if _expanded else 0
                    if _total_size > max_chars and _expanded:
                        _sub_start = _start_i
                        while _sub_start < _end_i:
                            _sub_paras: list[Paragraph] = []
                            _sub_end = _sub_start
                            while _sub_end < _end_i and approx_size(_sub_paras + [_sec_body[_sub_end]]) <= max_chars:
                                _sub_paras.append(_sec_body[_sub_end])
                                _sub_end += 1
                            if not _sub_paras:
                                _sub_end = min(_end_i, _sub_start + 1)
                            _n_windows += 1
                            if _sub_end >= _end_i:
                                break
                            _sub_start = max(_sub_end - 1, _sub_start + 1)
                    else:
                        _n_windows += 1
                    _last_main_s, _last_main_e = _s, _e
            if _n_windows > 0:
                _t_after_count = time.monotonic()
                _calls_per_window = 2 if not single_classifier else 1
                _concurrency = limits["max_llm_concurrency"]
                _estimated_sec = (_n_windows * _calls_per_window * 90) / _concurrency
                _est_min = max(1, int(_estimated_sec / 60) + (1 if _estimated_sec % 60 > 0 else 0) + 1)  # +1 min margin
                await update_job(paths.db, job_id, estimated_minutes=_est_min)
                LOGGER.info("Job %s [timing] window count done in %.1fs — %s windows, estimated nearly %s min", job_id, _t_after_count - t0, _n_windows, _est_min)

            window_tasks: list[asyncio.Task] = []
            for si, sec in enumerate(sections):
                    # Check cancellation before processing each section
                    if _is_cancelled(job_id):
                        # Cancel all existing tasks
                        for t in window_tasks:
                            if not t.done():
                                t.cancel()
                        await update_job(paths.db, job_id, status=JobStatus.cancelled, stage="cancelled", progress=100, error="CANCELLED")
                        if enable_debug_window:
                            _clear_live_trace(job_id)
                        return
                    
                    sec_body = [p for p in sec if not _is_heading_paragraph(p.text)]
                    if not sec_body:
                        continue
                    sec_heading = section_heading_by_i.get(si)
                    sec_path = section_path_by_i.get(si, "")

                    starts: list[int] = [0]
                    while True:
                        prev = starts[-1]
                        need = stride_tokens
                        i = prev
                        while i < len(sec_body) and need > 0:
                            need -= _approx_tokens(sec_body[i].text)
                            i += 1
                        if i >= len(sec_body):
                            break
                        starts.append(i)

                    last_main_s = -1
                    last_main_e = -1
                    for s in starts:
                        t = 0
                        e = s
                        while e < len(sec_body) and t < window_tokens:
                            t += _approx_tokens(sec_body[e].text)
                            e += 1
                        if e <= s:
                            e = min(len(sec_body), s + 1)
                        # Skip fully-contained windows to avoid redundant duplicates
                        if last_main_s >= 0 and last_main_e >= 0:
                            if s >= last_main_s and e <= last_main_e:
                                continue
                        main_block = sec_body[s:e]

                        joined = "\n\n".join(p.text for p in main_block)
                        near_start, near_end = _find_trigger_edges(joined, edge_frac)
                        exp_s, exp_e = _extend_window_by_stride(
                            sec_body, s, e, stride_tokens=stride_tokens, extend_left=near_start, extend_right=near_end
                        )
                        # CRITICAL: Ensure expansion stays within sec_body boundaries (one section only)
                        # sec_body already contains only paragraphs from the current section (heading removed)
                        # So we just need to clamp exp_s and exp_e to [0, len(sec_body)]
                        exp_s = max(0, min(exp_s, len(sec_body)))
                        exp_e = max(exp_s, min(exp_e, len(sec_body)))
                        
                        # Before creating expanded, optionally stop at obvious signature/contact blocks
                        # inside the same section to avoid dragging long address blocks into windows.
                        trimmed_exp_e = exp_e
                        for i in range(exp_s, exp_e):
                            p = sec_body[i]
                            para_text = (p.text or "").strip()
                            
                            # Check if it's address/contact info / signature block – these often
                            # appear before new sections and don't contain useful content for LLM.
                            if _is_signature_paragraph(para_text):
                                trimmed_exp_e = i
                                break
                        
                        exp_e = trimmed_exp_e
                        
                        # Ensure exp_e is still valid after trimming.
                        # If trimming made exp_e <= exp_s, fall back to using just the main block (s:e)
                        # to avoid losing content entirely.
                        if exp_e <= exp_s:
                            exp_s = s
                            exp_e = e
                            if exp_e <= exp_s:
                                exp_e = min(exp_s + 1, len(sec_body))
                        
                        expanded = sec_body[exp_s:exp_e]
                        
                        # Final safety check: remove any headings that might have slipped in
                        if expanded:
                            expanded = [p for p in expanded if not _is_heading_paragraph(p.text)]
                            
                            # Verify all paragraphs belong to the same section (si)
                            if expanded:
                                first_para_section = section_by_idx.get(expanded[0].paragraph_index)
                                if first_para_section:
                                    first_si, _ = first_para_section
                                    # Filter to only paragraphs from the same section
                                    filtered_expanded = []
                                    for p in expanded:
                                        para_section = section_by_idx.get(p.paragraph_index)
                                        if para_section and para_section[0] == first_si:
                                            filtered_expanded.append(p)
                                        else:
                                            # Hit a different section, stop here
                                            break
                                    expanded = filtered_expanded
                                    
                                    if len(expanded) < len(sec_body[exp_s:exp_e]):
                                        LOGGER.debug(
                                            f"Window {window_id}: Trimmed expanded window from {len(sec_body[exp_s:exp_e])} "
                                            f"to {len(expanded)} paragraphs to stay within section {first_si} boundaries"
                                        )
                                    
                                    # If filtering removed all paragraphs, fall back to main block
                                    if not expanded:
                                        LOGGER.warning(
                                            f"Window {window_id}: Section filtering removed all paragraphs. "
                                            f"Falling back to main block s={s}, e={e}"
                                        )
                                        expanded = sec_body[s:e]
                                        if not expanded and s < len(sec_body):
                                            expanded = [sec_body[s]]

                        # Use the trimmed expanded window, not the original exp_s:exp_e
                        # Check cancellation before creating new tasks
                        if _is_cancelled(job_id):
                            # Cancel all existing tasks
                            for t in window_tasks:
                                if not t.done():
                                    t.cancel()
                            await update_job(paths.db, job_id, status=JobStatus.cancelled, stage="cancelled", progress=100, error="CANCELLED")
                            if enable_debug_window:
                                _clear_live_trace(job_id)
                            return
                        
                        start_i = exp_s
                        end_i = exp_s + len(expanded) if expanded else exp_s
                        window_id = f"w_{sec_body[s].paragraph_index:04d}_{sec_body[min(e-1, len(sec_body)-1)].paragraph_index:04d}"

                        # If the expanded window is larger than max_chars, split it into
                        # multiple overlapping subwindows instead of trimming away content.
                        total_size = approx_size(expanded)
                        if total_size > max_chars and expanded:
                            overlap = 1  # overlap in paragraphs between subwindows
                            part_index = 0
                            # Work in indices relative to sec_body
                            sub_start = start_i
                            while sub_start < end_i:
                                sub_paras: list[Paragraph] = []
                                sub_end = sub_start
                                while sub_end < end_i and approx_size(sub_paras + [sec_body[sub_end]]) <= max_chars:
                                    sub_paras.append(sec_body[sub_end])
                                    sub_end += 1
                                if not sub_paras:
                                    # Ensure progress even if a single paragraph is very large
                                    sub_end = min(end_i, sub_start + 1)
                                sub_window_id = f"{window_id}_p{part_index:02d}"
                                t = asyncio.create_task(
                                    classify_window(
                                        sec_body,
                                        sub_start,
                                        sub_end,
                                        window_id=sub_window_id,
                                        main_start=s,
                                        main_end=e,
                                        sec_heading=sec_heading,
                                        sec_path=sec_path,
                                        all_sec_paras=sec_body,  # Full section for cross-paragraph quote search
                                    )
                                )
                                _register_job_task(job_id, t)
                                window_tasks.append(t)
                                part_index += 1
                                if sub_end >= end_i:
                                    break
                                # Move start forward with a small overlap to avoid gaps.
                                sub_start = max(sub_end - overlap, sub_start + 1)
                        else:
                            t = asyncio.create_task(
                                classify_window(
                                    sec_body,
                                    start_i,
                                    end_i,
                                    window_id=window_id,
                                    main_start=s,
                                    main_end=e,
                                    sec_heading=sec_heading,
                                    sec_path=sec_path,
                                    all_sec_paras=sec_body,  # Pass full section for cross-paragraph quote search
                                )
                            )
                            _register_job_task(job_id, t)
                            window_tasks.append(t)
                        last_main_s = s
                        last_main_e = e

            done = 0
            succeeded = 0
            failed = 0
            pending = set(window_tasks)
            t_llm_start = time.monotonic()
            total_windows = len(window_tasks)
            LOGGER.info("Job %s [timing] LLM phase starting — %s windows", job_id, total_windows)
            while pending:
                # Check cancellation before waiting
                if _is_cancelled(job_id):
                    # Cancel all pending tasks immediately
                    for t in pending:
                        if not t.done():
                            t.cancel()
                    # Wait for cancellations to propagate
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                    await update_job(paths.db, job_id, status=JobStatus.cancelled, stage="cancelled", progress=100, error="CANCELLED")
                    if enable_debug_window:
                        _clear_live_trace(job_id)
                    return
                
                # Wait for at least one task to complete
                done_tasks, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                
                # Process completed tasks
                for task in done_tasks:
                    findings_before = len(window_findings)
                    try:
                        await task
                        succeeded += 1
                        findings_after = len(window_findings)
                        if findings_after > findings_before:
                            LOGGER.info(
                                f"Window task completed: added {findings_after - findings_before} findings. "
                                f"Total findings now: {findings_after}"
                            )
                    except asyncio.CancelledError:
                        # Task was cancelled, ignore
                        pass
                    except Exception as exc:
                        failed += 1
                        if enable_debug_window:
                            _set_live_trace(job_id, {"phase": "error", "ts": time.time(), "error": str(exc)[:2000]})
                        # Don't raise - continue processing other tasks
                        LOGGER.warning(f"Window task failed: {exc}")
                    done += 1
                    if done % 5 == 0 or done == total_windows:
                        elapsed = time.monotonic() - t_llm_start
                        LOGGER.info("Job %s [timing] LLM progress %s/%s windows in %.1fs (%.1fs per window so far)", job_id, done, total_windows, elapsed, elapsed / done if done else 0)
                    if window_tasks:
                        prog = 55 + int(40 * (done / max(1, len(window_tasks))))
                        await update_job(paths.db, job_id, stage="processing", progress=min(95, prog))

            if window_tasks:
                LOGGER.info("Job %s [timing] LLM phase done in %.1fs — %s windows total", job_id, time.monotonic() - t_llm_start, total_windows)
            if window_tasks and succeeded == 0:
                raise RuntimeError(
                    "Error. The most likely cause is an invalid API key. "
                    "If you changed Base URL or Model, verify they are correct."
                )
            # Build a deterministic findings list (order does not depend on task completion).
            if window_findings:
                window_findings.sort(key=lambda t: (t[1], t[0], t[2].category_id))
                findings = [t[2] for t in window_findings]
            else:
                findings = []
            # Dedupe once after collecting all findings
            findings = dedupe_findings(findings)

        LOGGER.info(f"Job {job_id}: Total findings before dedupe: {len(findings)}")
        if findings:
            LOGGER.info(
                f"Job {job_id}: Findings before dedupe: "
                f"{[(f.category_name, f.evidence_quote[:60] if f.evidence_quote else 'EMPTY') for f in findings]}"
            )
        # Final dedupe pass (in case findings were added from different code paths)
        findings = dedupe_findings(findings)
        LOGGER.info(f"Job {job_id}: Total findings after dedupe: {len(findings)}")
        if findings:
            LOGGER.info(
                f"Job {job_id}: Findings after dedupe: "
                f"{[(f.category_name, f.evidence_quote[:60] if f.evidence_quote else 'EMPTY') for f in findings]}"
            )
        summary = compute_summary(findings)
        LOGGER.info("Job %s findings=%s risk_score=%s", job_id, len(findings), summary.risk_score)
        # Log findings details for debugging
        if len(findings) < 3:
            LOGGER.warning(
                f"Job {job_id}: Only {len(findings)} findings found. Expected at least 3. "
                f"Findings: {[(f.category_name, len(f.evidence_quote or '')) for f in findings]}"
            )
        else:
            LOGGER.info(
                f"Job {job_id}: Found {len(findings)} findings. "
                f"Categories: {[f.category_name for f in findings[:10]]}"
            )
        # Keep debug output stable and easy to read: sort by document order.
        if enable_debug_window and window_traces:
            def _win_key(w: dict) -> tuple[int, str]:
                idxs = w.get("paragraph_indices") or []
                try:
                    first = int(min(idxs)) if idxs else 10**9
                except Exception:
                    first = 10**9
                return (first, str(w.get("window_id") or ""))

            window_traces.sort(key=_win_key)
        result = ResultPayload(
            job_id=job_id,
            document=ResultDocument(
                filename=row.get("filename") or upload_path.name,
                page_count=int(page_count),
                paragraph_count=len(paragraphs),
            ),
            summary=summary,
            findings=findings,
            meta=ResultMeta(
                categories_version=cats_payload.get("version", "cuad_subset_v1"),
                prompt_version=(
                    "p_uk_two_stage" if (single_classifier and use_uk_redflags) else ("p_single_legacy" if single_classifier else "p_other")
                ),
                models=(
                    {"stage_1": llm.cfg.model_stage1, "stage_2": llm.cfg.model_stage2}
                    if (single_classifier and use_uk_redflags)
                    else (
                        {"classifier": llm.cfg.model_stage2}
                        if single_classifier
                        else {"stage_1": llm.cfg.model_stage1, "stage_2": llm.cfg.model_stage2}
                    )
                ),
                debug=(
                    {
                        "enabled_paragraph": enable_debug_paragraph,
                        "enabled_window": enable_debug_window,
                        "paragraph_traces": debug_traces,
                        "window_traces": window_traces,
                        "failed_windows": failed_windows if enable_debug_window else [],
                        "cuad_qa_json": cuad_meta,
                        "fewshot_only_category_ids": sorted(list(only_fewshot_category_ids)) if only_fewshot_category_ids else None,
                        "fewshot": {
                            "enabled": bool(fewshot_enabled),
                            "dir": str(fewshot_dir),
                            "max_per_category": fewshot_k,
                            "context_window": fewshot_window,
                        },
                    }
                    if (enable_debug_paragraph or enable_debug_window)
                    else None
                ),
            ),
        )
        out_path = job_result_path(paths, job_id)
        out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        await update_job(paths.db, job_id, status=JobStatus.done, stage="done", progress=100, result_path=out_path)
        if enable_debug_window:
            _clear_live_trace(job_id)
        _CANCELLED.discard(job_id)
        _JOB_TASKS.pop(job_id, None)
        LOGGER.info(
            "Job %s [timing] total %.1fs — done (temperature=%.1f, seed=%s, findings=%s)",
            job_id, time.monotonic() - t0, temperature, seed if seed is not None else "none", len(findings),
        )
    except Exception as e:
        LOGGER.exception("Job %s failed", job_id)
        if enable_debug_window:
            _set_live_trace(job_id, {"phase": "failed", "ts": time.time(), "error": str(e)[:2000]})
        await update_job(paths.db, job_id, status=JobStatus.failed, stage="failed", progress=100, error=str(e))
        if enable_debug_window:
            _clear_live_trace(job_id)
        _CANCELLED.discard(job_id)
        _JOB_TASKS.pop(job_id, None)

