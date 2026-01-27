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
from .models import (
    Finding,
    JobStatus,
    JobStatusResponse,
    Paragraph,
    ResultDocument,
    ResultMeta,
    ResultPayload,
    Stage1Category,
    UploadResponse,
)
from .parser import parse_pdf_to_paragraphs
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

security = HTTPBasic()

def _auth_enabled() -> bool:
    return os.environ.get("ENABLE_AUTH", "1") == "1"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def get_limits() -> dict:
    return {
        "max_pages": _env_int("MAX_PAGES", 20),
        "max_paragraphs": _env_int("MAX_PARAGRAPHS", 200),
        "max_file_mb": _env_int("MAX_FILE_MB", 0),
        "ttl_hours": _env_int("TTL_HOURS", 24),
        "max_llm_concurrency": _env_int("MAX_LLM_CONCURRENCY", 8),
        "max_findings_per_chunk": _env_int("MAX_FINDINGS_PER_CHUNK", 5),
    }


def require_basic_auth(creds: HTTPBasicCredentials = Depends(security)) -> None:
    if not _auth_enabled():
        return
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
    example_path = STATIC_DIR / "example_contract.pdf"
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


def _set_live_trace(job_id: str, payload: dict) -> None:
    _LIVE_TRACE[job_id] = payload


def _clear_live_trace(job_id: str) -> None:
    _LIVE_TRACE.pop(job_id, None)

def _is_cancelled(job_id: str) -> bool:
    return job_id in _CANCELLED


def _is_heading_paragraph(text: str) -> bool:
    return parse_heading(text, max_len=160) is not None


def _is_signature_paragraph(text: str) -> bool:
    """
    Check if a paragraph is likely a signature line, date, or name-only line.
    Such paragraphs should be filtered out before LLM processing as they're not contract content.
    """
    if not text:
        return False
    t = text.strip()
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


def _build_stage2_context(
    *,
    paragraphs: list[Paragraph],
    idx: int,
    section_by_idx: dict[int, tuple[int, int]],
    sections: list[list[Paragraph]],
    window: int,
    max_chars: int,
) -> str:
    """
    Build a compact context string around an evidence paragraph within its section.
    Includes section heading (if present) + neighbors (prev/next).
    """
    loc = section_by_idx.get(idx)
    if not loc:
        return ""
    sec_i, pos = loc
    sec = sections[sec_i]
    start = max(0, pos - window)
    end = min(len(sec), pos + window + 1)

    blocks: list[Paragraph] = []
    if sec and _is_heading_paragraph(sec[0].text):
        blocks.append(sec[0])
    for p in sec[start:end]:
        if p.paragraph_index == idx:
            continue
        blocks.append(p)

    seen = set()
    uniq: list[Paragraph] = []
    for p in blocks:
        if p.paragraph_index in seen:
            continue
        seen.add(p.paragraph_index)
        uniq.append(p)

    parts: list[str] = []
    used = 0
    for p in uniq:
        line = f"[¶{p.paragraph_index} p.{p.page}] {p.text}".strip()
        add = len(line) + 2
        if used and used + add > max_chars:
            break
        parts.append(line)
        used += add
    return "\n\n".join(parts).strip()


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
    return JSONResponse(
        content={
            "provider": provider,
            "base_url": base_url,
            "models": {
                "stage_1": os.environ.get("OPENAI_MODEL_STAGE1", "Qwen/Qwen3-30B-A3B-Thinking-2507"),
                "stage_2": os.environ.get("OPENAI_MODEL_STAGE2", "Qwen/Qwen3-30B-A3B-Thinking-2507"),
            },
            "limits": get_limits(),
        }
    )


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

    try:
        import pdfplumber

        with pdfplumber.open(str(upload_path)) as pdf:
            page_count = len(pdf.pages)
        if page_count > limits["max_pages"]:
            upload_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail=f"PDF has {page_count} pages, exceeds MAX_PAGES={limits['max_pages']}.",
            )
    except HTTPException:
        raise
    except Exception:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Failed to read PDF. Is it a valid, non-encrypted PDF?")

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
    
    llm_overrides = {
        "provider": (provider or "").strip().lower(),
        "api_key": final_api_key,
        "base_url": (base_url or "").strip(),
        "model": (model or "").strip(),
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
    upload_path = row.get("upload_path")
    if not upload_path or not Path(upload_path).exists():
        raise HTTPException(status_code=404, detail="PDF not found")
    filename = row.get("filename") or "document.pdf"
    return FileResponse(path=upload_path, media_type="application/pdf", filename=filename)


@app.get("/job/{job_id}/annotated", response_class=HTMLResponse)
async def job_annotated(job_id: str, _: None = Depends(require_basic_auth)) -> HTMLResponse:
    return FileResponse(str(TEMPLATES_DIR / "annotated.html"))


@app.get("/job/{job_id}/annotated-data")
async def job_annotated_data(job_id: str, _: None = Depends(require_basic_auth)) -> JSONResponse:
    row = await get_job(paths.db, job_id)
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    upload_path = row.get("upload_path")
    result_path = row.get("result_path")
    if not upload_path or not Path(upload_path).exists():
        raise HTTPException(status_code=404, detail="PDF not found")
    if not result_path or not Path(result_path).exists():
        raise HTTPException(status_code=404, detail="Result not found")

    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    findings = result.get("findings", [])
    # Log filtering for debugging
    total_findings = len(findings)
    unfair_findings = [f for f in findings if f.get("is_unfair") is not False]
    filtered_count = total_findings - len(unfair_findings)
    if filtered_count > 0:
        LOGGER.info(f"Filtered {filtered_count} findings with is_unfair=False (out of {total_findings} total)")
    unfair_findings = dedupe_findings([Finding(**f) for f in unfair_findings])
    unfair_findings_dict = [f.model_dump() for f in unfair_findings]
    limits = get_limits()
    paragraphs, page_count = parse_pdf_to_paragraphs(
        Path(upload_path),
        max_pages=limits["max_pages"],
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
    upload_path = row.get("upload_path")
    result_path = row.get("result_path")
    if not upload_path or not Path(upload_path).exists():
        raise HTTPException(status_code=404, detail="PDF not found")
    if not result_path or not Path(result_path).exists():
        raise HTTPException(status_code=404, detail="Result not found")

    out_path = paths.results / f"{job_id}_annotated.pdf"
    try:
        result = json.loads(Path(result_path).read_text(encoding="utf-8"))
        findings = result.get("findings", [])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read result: {exc}")

    limits = get_limits()
    paragraphs, _page_count = parse_pdf_to_paragraphs(
        Path(upload_path),
        max_pages=limits["max_pages"],
        max_paragraphs=limits["max_paragraphs"],
    )
    paragraphs = _normalize_paragraphs(paragraphs)
    build_annotated_text_pdf(paragraphs=paragraphs, findings=findings, out_path=out_path)
    return FileResponse(path=out_path, media_type="application/pdf", filename=f"annotated_{job_id}.pdf")


async def process_job(job_id: str, llm_overrides: Optional[dict] = None) -> None:
    limits = get_limits()
    row = await get_job(paths.db, job_id)
    if not row:
        return
    upload_path = Path(row["upload_path"])
    enable_debug_paragraph = False
    enable_debug_window = False
    try:
        LOGGER.info("Job %s started: %s", job_id, upload_path.name)
        if _is_cancelled(job_id):
            await update_job(paths.db, job_id, status=JobStatus.cancelled, stage="cancelled", progress=100, error="CANCELLED")
            return
        await update_job(paths.db, job_id, status=JobStatus.running, stage="parse", progress=0, started_at=datetime.now(timezone.utc))
        paragraphs, page_count = parse_pdf_to_paragraphs(
            upload_path,
            max_pages=limits["max_pages"],
            max_paragraphs=limits["max_paragraphs"],
        )
        paragraphs = _normalize_paragraphs(paragraphs)
        LOGGER.info("Job %s parsed: pages=%s paragraphs=%s", job_id, page_count, len(paragraphs))
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
        cats_payload = load_categories_payload(
            csv_path=DATA_DIR / "category_descriptions.csv",
            json_path=CONFIGS_DIR / "categories.json",
        )

        single_classifier = os.environ.get("SINGLE_CLASSIFIER", "1") == "1"
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
        LOGGER.info("Job %s LLM config: temperature=%.1f, max_findings_per_chunk=%d", job_id, temperature, limits.get("max_findings_per_chunk", 5))
        llm = LLMRunner(
            LLMConfig(
                api_key=api_key,
                base_url=base_url,
                model_stage1=os.environ.get("OPENAI_MODEL_STAGE1", "gpt-5-nano"),
                model_stage2=chosen_model or os.environ.get("OPENAI_MODEL_STAGE2", "gpt-5-mini"),
                max_concurrency=limits["max_llm_concurrency"],
                stage1_include_descriptions=os.environ.get("STAGE1_INCLUDE_DESCRIPTIONS", "0") == "1",
                message_content_mode=message_mode,
                temperature=temperature,
            ),
            cats_payload,
            fewshot_by_category=fewshot,
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
        
        # Pattern to detect headings that start a paragraph but have body text after (e.g., "5.8. Title. Body...")
        heading_with_body_pattern = re.compile(
            r"^\s*(\d+(?:\.\d+)*)\s*\.?\s+([A-Z][A-Za-z\s]{2,}?)\s*\.\s+[A-Z]",
            re.IGNORECASE
        )
        
        for p in filtered_paragraphs:
            # Check if paragraph is a heading (entire paragraph)
            is_heading = _is_heading_paragraph(p.text)
            
            # Also check if paragraph starts with a heading pattern (heading + body in same para)
            # This catches cases like "5.8. Nonpublic Information. The Distributor..."
            para_text = (p.text or "").strip()
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
                    starts_with_heading = (is_uppercase_heading or is_short_heading or is_title_case_short)
            
            if (is_heading or starts_with_heading) and cur:
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

        ctx_n = int(os.environ.get("STAGE1_CONTEXT_PARAGRAPHS", "5"))
        max_chars = _approx_chars_limit()

        para_to_stage1: dict[int, list[Stage1Category]] = {p.paragraph_index: [] for p in paragraphs}

        async def run_stage1_block(main_block: list[Paragraph], context_block: list[Paragraph]) -> None:
            main_block = [p for p in main_block if len((p.text or "").split()) >= 5 and len((p.text or "").strip()) >= 30]
            if not main_block:
                return
            block_path = path_by_idx.get(main_block[0].paragraph_index, "")
            try:
                if enable_debug:
                    out, raw = await llm.stage1_for_block_with_raw(
                        main_paragraphs=main_block,
                        context_paragraphs=context_block,
                        section_path=block_path,
                        t_router=t_router,
                        topk_probs=router_topk,
                        max_categories=stage1_max_categories,
                    )
                else:
                    out = await llm.stage1_for_block(
                        main_paragraphs=main_block,
                        context_paragraphs=context_block,
                        section_path=block_path,
                        t_router=t_router,
                        topk_probs=router_topk,
                        max_categories=stage1_max_categories,
                    )
                    raw = ""
                for cat in out.categories:
                    for idx in cat.evidence_paragraph_indices:
                        if idx in para_to_stage1:
                            para_to_stage1[idx].append(cat)
                            if enable_debug:
                                t = debug_by_idx.get(idx)
                                if t is not None:
                                    t["stage1"] = {"categories": [c.model_dump() for c in para_to_stage1[idx]]}
                                    if raw:
                                        t["stage1_raw"] = (t.get("stage1_raw", "") + "\n\n" + raw)[:4000]
                                    # keep router probs visible in debug
                                    t["stage1_router"] = {
                                        "t_router": t_router,
                                        "unclear_prob": out.unclear_prob,
                                        "category_probs": [p.model_dump() for p in out.category_probs],
                                    }
                # Mark empty categories as skipped
                if enable_debug:
                    for p in main_block:
                        if not para_to_stage1[p.paragraph_index]:
                            t = debug_by_idx.get(p.paragraph_index)
                            if t is not None and t.get("stage2_error") is None:
                                t["stage2_error"] = "SKIPPED_NO_STAGE1_CATEGORIES"
            except Exception as exc:
                LOGGER.warning("Stage1 block failed: %s", str(exc))
                if enable_debug:
                    for p in main_block:
                        t = debug_by_idx.get(p.paragraph_index)
                        if t is not None:
                            t["stage1_error"] = str(exc)[:2000]
                            t["stage1_raw"] = str(exc)[:4000]
                            t["stage2_error"] = "SKIPPED_DUE_TO_STAGE1_ERROR"

        window_tokens = int(os.environ.get("STAGE1_WINDOW_TOKENS", "1100"))
        stride_tokens = int(os.environ.get("STAGE1_STRIDE_TOKENS", "300"))
        edge_frac = float(os.environ.get("STAGE1_TRIGGER_EDGE_FRAC", "0.15"))

        def approx_size(ps: list[Paragraph]) -> int:
            return sum(len((p.text or "")) + 40 for p in ps)

        window_traces: list[dict] = []

        if single_classifier:
            # Single-classifier mode: each window is classified directly into up to 2 red-flag findings.
            await update_job(paths.db, job_id, stage="processing", progress=55)

            findings: list[Finding] = []

            severity_by_category = {
                "Termination For Convenience": "medium",
                "Uncapped Liability": "high",
                "Irrevocable Or Perpetual License": "medium",
                "Most Favored Nation": "medium",
                "Audit Rights": "low",
                "Ip Ownership Assignment": "high",
            }

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
                return "Review IP assignment scope and consider carve-outs/limitations."

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
                max_findings = limits.get("max_findings_per_chunk", 5)
                out, raw, prompt = await llm.classify_redflags_for_chunk(
                    text_chunk=text_chunk,
                    section_path=spath,
                    max_findings=max_findings,
                    return_prompt=enable_debug_window,
                )
                if _is_cancelled(job_id):
                    return

                if enable_debug_window:
                    prompt_user = (prompt or {}).get("user", "")
                    prompt_system = (prompt or {}).get("system", "")
                    sections = (prompt or {}).get("sections") if isinstance(prompt, dict) else None
                    # Fall back to extracting minimal parts if sections are not present.
                    ctx = ""
                    chunk_for_llm = ""
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
                            "raw_output": (raw or "")[:8000],
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
                            "raw_output": (raw or "")[:6000],
                            "parsed": out.model_dump(),
                        },
                    )

                # Log raw findings for debugging (only raw output, no text_chunk)
                LOGGER.info(
                    f"Window {window_id}: LLM returned {len(out.findings)} findings. "
                    f"Raw output length={len(raw)}"
                )
                if not out.findings:
                    LOGGER.warning(
                        f"Window {window_id}: LLM returned EMPTY findings! "
                        f"Temperature={llm.cfg.temperature}, max_findings={max_findings}, "
                        f"chunk_length={len(chunk_for_llm)}, section_path={spath}"
                    )
                    if raw:
                        LOGGER.warning(f"Window {window_id} raw output (EMPTY FINDINGS): {raw[:2000]}")
                else:
                    if raw:
                        LOGGER.info(f"Window {window_id} raw output: {raw[:2000]}")
                    for j, f in enumerate(out.findings):
                        LOGGER.info(
                            f"Window {window_id}, finding {j+1}: category={f.category}, "
                            f"quote_length={len(f.quote or '')}, quote_preview={f.quote[:150] if f.quote else 'EMPTY'}, "
                            f"is_unfair={f.is_unfair}"
                        )
                
                if len(out.findings) > max_findings:
                    LOGGER.warning(
                        f"Window {window_id}: LLM returned {len(out.findings)} findings, but max_findings={max_findings}. "
                        f"Truncating to {max_findings} findings."
                    )
                
                # Process findings (limit to max_findings)
                findings_to_process = out.findings[:max_findings]
                LOGGER.info(
                    f"Window {window_id}: Processing {len(findings_to_process)} findings "
                    f"(out of {len(out.findings)} returned by LLM, max_findings={max_findings})"
                )
                
                for j, f in enumerate(findings_to_process):
                    sev = severity_by_category.get(f.category, "medium")
                    loc_para = None
                    evidence_quote = f.quote or ""
                    if not f.quote:
                        LOGGER.warning(
                            f"Window {window_id}, finding {j+1} ({f.category}): Empty quote, will use entire first paragraph as quote"
                        )
                        if chunk_paras:
                            loc_para = chunk_paras[0]
                            evidence_quote = (loc_para.text or "").strip()
                    else:
                        # Paragraphs are already normalized via _normalize_paragraphs.
                        # LLM receives normalized text, so its quote should also be normalized.
                        # Try exact match first
                        found_quote = False
                        for p in chunk_paras:
                            # Paragraphs are already normalized, use as-is
                            para_text = p.text or ""
                            # LLM quote should also be normalized (LLM receives normalized text)
                            if f.quote and f.quote in para_text:
                                loc_para = p
                                evidence_quote = f.quote
                                found_quote = True
                                break
                        
                        if not found_quote:
                            # Try case-insensitive match
                            quote_lower = (f.quote or "").lower()
                            if quote_lower:
                                for p in chunk_paras:
                                    para_text_lower = (p.text or "").lower()
                                    if quote_lower in para_text_lower:
                                        loc_para = p
                                        evidence_quote = f.quote
                                        found_quote = True
                                        break
                        
                        if not found_quote:
                            # Try searching in adjacent paragraphs if quote spans across paragraph boundaries
                            # First, try to find quote start or end in chunk_paras
                            quote_start_words = f.quote.split()[:5] if f.quote and len(f.quote.split()) >= 5 else []
                            quote_end_words = f.quote.split()[-5:] if f.quote and len(f.quote.split()) >= 5 else []
                            
                            if quote_start_words or quote_end_words:
                                # Search in extended context: chunk_paras + adjacent paragraphs from section
                                search_paras = list(chunk_paras)
                                if all_sec_paras:
                                    # Add previous paragraph if exists
                                    if start_i > 0 and start_i <= len(all_sec_paras):
                                        prev_para = all_sec_paras[start_i - 1]
                                        if prev_para not in chunk_paras:
                                            search_paras.insert(0, prev_para)
                                    # Add next paragraph if exists
                                    if end_i < len(all_sec_paras):
                                        next_para = all_sec_paras[end_i]
                                        if next_para not in chunk_paras:
                                            search_paras.append(next_para)
                                
                                # Try to find quote spanning multiple paragraphs
                                for p in search_paras:
                                    para_text = (p.text or "").lower()
                                    # Check if quote start or end is in this paragraph
                                    quote_start_lower = " ".join(quote_start_words).lower() if quote_start_words else ""
                                    quote_end_lower = " ".join(quote_end_words).lower() if quote_end_words else ""
                                    
                                    if (quote_start_lower and quote_start_lower in para_text) or \
                                       (quote_end_lower and quote_end_lower in para_text):
                                        # Found start or end, use this paragraph
                                        loc_para = p
                                        evidence_quote = f.quote
                                        found_quote = True
                                        LOGGER.info(
                                            f"Window {window_id}, finding {j+1} ({f.category}): "
                                            f"Found quote start/end in paragraph {p.paragraph_index}, using it."
                                        )
                                        break
                        
                        if not found_quote:
                            # Try fuzzy match: look for key words from quote
                            if f.quote and len(f.quote) > 20:
                                quote_words = set(w.lower() for w in f.quote.split() if len(w) > 3)
                                best_match = None
                                best_score = 0
                                # Search in extended context if available
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
                                    # Paragraphs are already normalized
                                    para_words = set(w.lower() for w in (p.text or "").split() if len(w) > 3)
                                    # Count matching significant words
                                    matches = len(quote_words & para_words)
                                    if matches > best_score and matches >= len(quote_words) * 0.5:  # At least 50% of words match
                                        best_score = matches
                                        best_match = p
                                
                                if best_match:
                                    loc_para = best_match
                                    evidence_quote = f.quote
                                    LOGGER.info(
                                        f"Window {window_id}, finding {j+1} ({f.category}): "
                                        f"Used fuzzy match to locate quote (matched {best_score}/{len(quote_words)} words). "
                                        f"Using original LLM quote - frontend will attempt case-insensitive match."
                                    )
                                else:
                                    LOGGER.warning(
                                        f"Window {window_id}, finding {j+1} ({f.category}): "
                                        f"Could not locate quote in paragraphs: {f.quote[:100]}. "
                                        f"Using entire first paragraph as quote."
                                    )
                                    if chunk_paras:
                                        loc_para = chunk_paras[0]
                                        evidence_quote = (loc_para.text or "").strip() or f.quote
                            else:
                                LOGGER.warning(
                                    f"Window {window_id}, finding {j+1} ({f.category}): "
                                    f"Could not locate quote in paragraphs: {f.quote[:100] if f.quote else 'EMPTY'}. "
                                    f"Using entire first paragraph as quote."
                                )
                                if chunk_paras:
                                    loc_para = chunk_paras[0]
                                    evidence_quote = (loc_para.text or "").strip() or f.quote

                    cat_id = llm.redflag_category_id_by_name.get(f.category, "").strip() or f.category.lower().replace(" ", "_")
                    reasoning_bullets = "\n".join(f"- {b}" for b in f.reasoning)
                    what = reasoning_bullets or f.explanation
                    revision_expl = (f.recommended_revision.revision_explanation or "").strip()
                    revised_clause = (f.recommended_revision.revised_clause or "").strip()
                    
                    if not loc_para:
                        if chunk_paras:
                            loc_para = chunk_paras[0]
                            if not evidence_quote:
                                evidence_quote = (loc_para.text or "").strip() or f.quote or ""
                            LOGGER.warning(
                                f"Window {window_id}, finding {j+1} ({f.category}): No location found, using first paragraph of chunk as fallback"
                            )
                        else:
                            LOGGER.error(
                                f"Window {window_id}, finding {j+1} ({f.category}): No paragraphs in chunk, cannot assign location."
                            )
                            findings.append(
                                Finding(
                                    finding_id=f"{window_id}_f{j+1}",
                                    category_id=cat_id,
                                    category_name=f.category,
                                    severity=sev,  # type: ignore[arg-type]
                                    confidence=0.8,
                                    issue_title=f.category,
                                    what_is_wrong=what,
                                    explanation=f.explanation,
                                    recommendation=revision_expl or recommend_for(f.category),
                                    evidence_quote=evidence_quote,
                                    is_unfair=bool(f.is_unfair),
                                    legal_references=list(f.legal_references or [])[:6],
                                    possible_consequences=f.possible_consequences,
                                    risk_assessment=f.risk_assessment,
                                    consequences_category=f.consequences_category,
                                    risk_category=f.risk_category,
                                    revised_clause=revised_clause,
                                    revision_explanation=revision_expl,
                                    suggested_follow_up=f.suggested_follow_up,
                                    paragraph_bbox=None,
                                    location={"page": 0, "paragraph_index": 0},
                                )
                            )
                            continue
                    
                    # Add finding with location
                    new_finding = Finding(
                        finding_id=f"{window_id}_f{j+1}",
                        category_id=cat_id,
                        category_name=f.category,
                        severity=sev,  # type: ignore[arg-type]
                        confidence=0.8,
                        issue_title=f.category,
                        what_is_wrong=what,
                        explanation=f.explanation,
                        recommendation=revision_expl or recommend_for(f.category),
                        evidence_quote=evidence_quote,
                        is_unfair=bool(f.is_unfair),
                        legal_references=list(f.legal_references or [])[:6],
                        possible_consequences=f.possible_consequences,
                        risk_assessment=f.risk_assessment,
                        consequences_category=f.consequences_category,
                        risk_category=f.risk_category,
                        revised_clause=revised_clause,
                        revision_explanation=revision_expl,
                        suggested_follow_up=f.suggested_follow_up,
                        paragraph_bbox=loc_para.bbox,
                        location={"page": int(loc_para.page), "paragraph_index": int(loc_para.paragraph_index)},
                    )
                    findings.append(new_finding)
                    LOGGER.info(
                        f"Window {window_id}, finding {j+1} ({f.category}): "
                        f"Added finding to list. Total findings so far: {len(findings)}. "
                        f"Quote preview: {evidence_quote[:80] if evidence_quote else 'EMPTY'}..."
                    )

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

                for s in starts:
                    t = 0
                    e = s
                    while e < len(sec_body) and t < window_tokens:
                        t += _approx_tokens(sec_body[e].text)
                        e += 1
                    if e <= s:
                        e = min(len(sec_body), s + 1)
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
                    
                    # CRITICAL: Before creating expanded, check for headings from different sections
                    # This prevents mixing paragraphs from different sections (e.g., 5.6 and 5.8)
                    # Get current section number for comparison
                    current_section_heading = section_heading_by_i.get(si)
                    current_num = None
                    if current_section_heading:
                        current_heading = parse_heading(current_section_heading.text or "", max_len=160)
                        if current_heading:
                            label = current_heading.label
                            num_match = re.search(r"(\d+(?:\.\d+)*)", label)
                            if num_match:
                                current_num = num_match.group(1)
                    
                    # Check paragraphs in exp_s:exp_e range for headings from different sections
                    # Pattern 1: "5.8. Title. Body..." - heading with period, then body starts with capital
                    heading_start_pattern1 = re.compile(
                        r"^\s*(\d+(?:\.\d+)*)\s*\.?\s+([A-Z][A-Za-z\s]{2,}?)\s*\.\s+[A-Z]",
                        re.IGNORECASE
                    )
                    # Pattern 2: "6. INTERPRETATION... 6.1 Assignment..." - heading, then subsection number
                    heading_start_pattern2 = re.compile(
                        r"^\s*(\d+(?:\.\d+)*)\s*\.?\s+([A-Z][A-Z\s]{5,}?)\s+(\d+\.\d+)",
                        re.IGNORECASE
                    )
                    
                    # Find the first paragraph that looks like a heading from a different section
                    # Also stop at address/contact info blocks that typically appear before new sections
                    trimmed_exp_e = exp_e
                    for i in range(exp_s, exp_e):
                        p = sec_body[i]
                        para_text = (p.text or "").strip()
                        
                        # Check if it's address/contact info - these often appear before new sections
                        # and shouldn't be in the current section's window
                        if _is_signature_paragraph(para_text):
                            # This looks like address/contact info - stop here to avoid mixing sections
                            trimmed_exp_e = i
                            LOGGER.warning(
                                f"Window {window_id}: Found address/contact info at paragraph {p.paragraph_index} "
                                f"(text: {para_text[:60]}...), trimming exp_e from {exp_e} to {trimmed_exp_e}"
                            )
                            break
                        
                        # Check if entire paragraph is a heading
                        if _is_heading_paragraph(para_text):
                            # This is a heading - stop here
                            trimmed_exp_e = i
                            LOGGER.warning(
                                f"Window {window_id}: Found heading at paragraph {p.paragraph_index}, trimming exp_e from {exp_e} to {trimmed_exp_e}"
                            )
                            break
                        
                        # Check if paragraph starts with heading pattern
                        m = heading_start_pattern1.match(para_text) or heading_start_pattern2.match(para_text)
                        if m:
                            num_part = m.group(1)
                            title_part = m.group(2).strip()
                            if title_part.endswith('.'):
                                title_part = title_part[:-1].strip()
                            
                            # Check if title looks like a heading
                            words = title_part.split()[:10]
                            if words:
                                uppercase_words = sum(1 for w in words if w.isupper() and len(w) > 1)
                                titlecase_words = sum(1 for w in words if w and w[0].isupper() and not w.isupper())
                                uppercase_ratio = uppercase_words / len(words) if words else 0
                                titlecase_ratio = titlecase_words / len(words) if words else 0
                                is_short_heading = len(words) <= 5
                                is_uppercase_heading = uppercase_ratio >= 0.5
                                is_title_case_short = (2 <= len(words) <= 4 and titlecase_ratio >= 0.5)
                                
                                if is_uppercase_heading or is_short_heading or is_title_case_short:
                                    # Compare section numbers
                                    if current_num:
                                        if num_part != current_num:
                                            is_subsection = num_part.startswith(current_num + ".")
                                            is_parent_section = current_num.startswith(num_part + ".")
                                            if not is_subsection and not is_parent_section:
                                                # Different section - stop here (don't include this paragraph)
                                                trimmed_exp_e = i
                                                LOGGER.warning(
                                                    f"Window {window_id}: Found different section '{num_part}' (current: '{current_num}') "
                                                    f"at paragraph {p.paragraph_index} (text: {para_text[:60]}...), "
                                                    f"trimming exp_e from {exp_e} to {trimmed_exp_e}"
                                                )
                                                break
                                    else:
                                        # No current section number, but found a heading - stop to be safe
                                        trimmed_exp_e = i
                                        LOGGER.warning(
                                            f"Window {window_id}: Found heading '{num_part}' but current section has no number, "
                                            f"trimming exp_e from {exp_e} to {trimmed_exp_e}"
                                        )
                                        break
                    
                    exp_e = trimmed_exp_e
                    
                    # Ensure exp_e is still valid after trimming
                    # If trimming made exp_e <= exp_s, fall back to using just the main block (s:e)
                    # This ensures we don't skip windows that contain important content
                    if exp_e <= exp_s:
                        LOGGER.warning(
                            f"Window {window_id}: exp_e ({exp_e}) <= exp_s ({exp_s}) after trimming. "
                            f"Falling back to main block s={s}, e={e}"
                        )
                        # Use main block instead of expanded window
                        exp_s = s
                        exp_e = e
                        # Ensure we have at least one paragraph
                        if exp_e <= exp_s:
                            exp_e = min(exp_s + 1, len(sec_body))
                    
                    expanded = sec_body[exp_s:exp_e]
                    while expanded and approx_size(expanded) > max_chars:
                        expanded = expanded[:-1]
                    
                    # Ensure expanded is not empty - if it is, use at least the main block
                    if not expanded:
                        LOGGER.warning(
                            f"Window {window_id}: expanded is empty after trimming. Using main block s={s}, e={e}"
                        )
                        expanded = sec_body[s:e]
                        if not expanded:
                            # Last resort: use at least one paragraph
                            if s < len(sec_body):
                                expanded = [sec_body[s]]
                            else:
                                LOGGER.warning(f"Window {window_id}: Cannot create window, skipping")
                                continue
                    
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
                    
                    # Log window info for debugging
                    if expanded:
                        para_indices = [p.paragraph_index for p in expanded]
                        LOGGER.debug(
                            f"Window {window_id}: section {si} ({sec_path}), "
                            f"expanded from {exp_s} to {exp_e} (len={len(expanded)}), "
                            f"paragraph indices: {para_indices[:5]}{'...' if len(para_indices) > 5 else ''}"
                        )
                    
                    window_tasks.append(asyncio.create_task(classify_window(
                        sec_body, start_i, end_i, window_id=window_id,
                        main_start=s, main_end=e,
                        sec_heading=sec_heading, sec_path=sec_path,
                        all_sec_paras=sec_body,  # Pass full section for cross-paragraph quote search
                    )))

            done = 0
            # Use asyncio.wait to check cancellation more frequently
            pending = set(window_tasks)
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
                    findings_before = len(findings)
                    try:
                        await task
                        findings_after = len(findings)
                        if findings_after > findings_before:
                            LOGGER.info(
                                f"Window task completed: added {findings_after - findings_before} findings. "
                                f"Total findings now: {findings_after}"
                            )
                    except asyncio.CancelledError:
                        # Task was cancelled, ignore
                        pass
                    except Exception as exc:
                        if enable_debug_window:
                            _set_live_trace(job_id, {"phase": "error", "ts": time.time(), "error": str(exc)[:2000]})
                        # Don't raise - continue processing other tasks
                        LOGGER.warning(f"Window task failed: {exc}")
                    done += 1
                    if window_tasks:
                        prog = 55 + int(40 * (done / max(1, len(window_tasks))))
                        await update_job(paths.db, job_id, stage="processing", progress=min(95, prog))

            findings = dedupe_findings(findings)
        else:
            # Legacy mode (stage1 router + stage2) kept for fallback.
            stage1_tasks = []
            for sec in sections:
                sec_body = [p for p in sec if not _is_heading_paragraph(p.text)]
                if not sec_body:
                    continue
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
                for s in starts:
                    t = 0
                    e = s
                    while e < len(sec_body) and t < window_tokens:
                        t += _approx_tokens(sec_body[e].text)
                        e += 1
                    if e <= s:
                        e = min(len(sec_body), s + 1)
                    main_block = sec_body[s:e]
                    ctx_start = max(0, s - ctx_n)
                    context_block = sec_body[ctx_start:s]
                    while context_block and approx_size(context_block) > (max_chars // 2):
                        context_block = context_block[1:]
                    joined = "\n\n".join(p.text for p in main_block)
                    near_start, near_end = _find_trigger_edges(joined, edge_frac)
                    exp_s, exp_e = _extend_window_by_stride(
                        sec_body, s, e, stride_tokens=stride_tokens, extend_left=near_start, extend_right=near_end
                    )
                    expanded = sec_body[exp_s:exp_e]
                    while expanded and approx_size(expanded) > max_chars:
                        expanded = expanded[:-1]
                    stage1_tasks.append(run_stage1_block(expanded, context_block))
            for coro in asyncio.as_completed(stage1_tasks):
                await coro
            LOGGER.info("Job %s stage1 complete", job_id)
            await update_job(paths.db, job_id, stage="stage2", progress=55)
            findings = []
            # (legacy stage2 code remains unchanged below)
            cat_map = {c["id"]: c["name"] for c in cats_payload.get("categories", [])}
            stage2_ctx_n = int(os.environ.get("STAGE2_CONTEXT_PARAGRAPHS", "3"))
            stage2_ctx_max_chars = int(os.environ.get("STAGE2_MAX_CONTEXT_CHARS", "6000"))
            async def stage2_one(p: Paragraph, s1: Stage1Category):
                ctx = _build_stage2_context(
                    paragraphs=paragraphs,
                    idx=p.paragraph_index,
                    section_by_idx=section_by_idx,
                    sections=sections,
                    window=stage2_ctx_n,
                    max_chars=stage2_ctx_max_chars,
                )
                spath = path_by_idx.get(p.paragraph_index, "")
                try:
                    if enable_debug:
                        out, raw = await llm.stage2_with_raw(p, s1, context_text=ctx, section_path=spath)
                        trace = debug_by_idx.get(p.paragraph_index)
                        if trace is not None:
                            trace["stage2_runs"].append(
                                {
                                    "category_id": s1.category_id,
                                    "stage1": s1.model_dump(),
                                    "stage2": out.model_dump(),
                                    "stage2_raw": (raw or "")[:4000],
                                    "stage2_error": None,
                                    "stage2_context": ctx[:4000],
                                    "section_path": spath,
                                }
                            )
                    else:
                        out = await llm.stage2(p, s1, context_text=ctx, section_path=spath)
                except Exception as exc:
                    LOGGER.warning("Stage2 failed for paragraph %s: %s", p.paragraph_index, str(exc))
                    if enable_debug:
                        trace = debug_by_idx.get(p.paragraph_index)
                        if trace is not None:
                            trace["stage2_runs"].append(
                                {
                                    "category_id": s1.category_id,
                                    "stage1": s1.model_dump(),
                                    "stage2": None,
                                    "stage2_raw": "",
                                    "stage2_error": str(exc)[:2000],
                                    "stage2_context": ctx[:4000],
                                    "section_path": spath,
                                }
                            )
                    return None
                finding_id = f"f_{p.paragraph_index:04d}_{s1.category_id}"
                return Finding(
                    finding_id=finding_id,
                    category_id=s1.category_id,
                    category_name=cat_map.get(s1.category_id, s1.category_id),
                    severity=out.severity,
                    confidence=float(out.confidence),
                    is_unfair=out.severity in ("medium", "high"),
                    issue_title=out.issue_title,
                    what_is_wrong=out.what_is_wrong,
                    explanation=out.what_is_wrong,
                    recommendation=out.recommendation,
                    evidence_quote=out.evidence_quote,
                    legal_references=[],
                    possible_consequences="",
                    risk_assessment={"severity_of_consequences": 0, "degree_of_legal_violation": 0},
                    consequences_category="Nothing",
                    risk_category="Nothing",
                    revised_clause="",
                    revision_explanation="",
                    suggested_follow_up="",
                    paragraph_bbox=p.bbox,
                    location={"page": p.page, "paragraph_index": p.paragraph_index},
                )
            stage2_tasks = []
            para_by_idx = {p.paragraph_index: p for p in paragraphs}
            for idx, cats in para_to_stage1.items():
                if not cats:
                    continue
                p = para_by_idx.get(idx)
                if p is None:
                    continue
                for cat in cats:
                    stage2_tasks.append(stage2_one(p, cat))
            done = 0
            for coro in asyncio.as_completed(stage2_tasks):
                item = await coro
                done += 1
                if item is not None:
                    findings.append(item)
                if stage2_tasks:
                    prog = 55 + int(40 * (done / max(1, len(stage2_tasks))))
                    await update_job(paths.db, job_id, stage="stage2", progress=min(95, prog))

        LOGGER.info(f"Job {job_id}: Before dedupe: {len(findings)} findings")
        findings = dedupe_findings(findings)
        LOGGER.info(f"Job {job_id}: After dedupe: {len(findings)} findings")
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
                prompt_version="p_single_v2_uklaw" if single_classifier else "p_v1",
                models={"classifier": llm.cfg.model_stage2} if single_classifier else {"stage_1": llm.cfg.model_stage1, "stage_2": llm.cfg.model_stage2},
                debug=(
                    {
                        "enabled_paragraph": enable_debug_paragraph,
                        "enabled_window": enable_debug_window,
                        "paragraph_traces": debug_traces,
                        "window_traces": window_traces,
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
        LOGGER.info("Job %s done", job_id)
    except Exception as e:
        LOGGER.exception("Job %s failed", job_id)
        if enable_debug_window:
            _set_live_trace(job_id, {"phase": "failed", "ts": time.time(), "error": str(e)[:2000]})
        await update_job(paths.db, job_id, status=JobStatus.failed, stage="failed", progress=100, error=str(e))
        if enable_debug_window:
            _clear_live_trace(job_id)
        _CANCELLED.discard(job_id)

