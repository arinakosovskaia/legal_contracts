from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .headings import parse_heading, Heading

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from .aggregate import compute_summary, dedupe_findings
from .fewshot import cuad_file_meta, load_or_build_fewshot, resolve_cuad_json_path
from .llm import LLMConfig, LLMRunner, load_categories_payload
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


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def get_limits() -> dict:
    return {
        "max_pages": _env_int("MAX_PAGES", 20),
        "max_paragraphs": _env_int("MAX_PARAGRAPHS", 200),
        "max_file_mb": _env_int("MAX_FILE_MB", 15),
        "ttl_hours": _env_int("TTL_HOURS", 24),
        "max_llm_concurrency": _env_int("MAX_LLM_CONCURRENCY", 8),
    }


def require_basic_auth(creds: HTTPBasicCredentials = Depends(security)) -> None:
    user = os.environ.get("BASIC_AUTH_USER", "")
    pwd = os.environ.get("BASIC_AUTH_PASS", "")
    ok = secrets.compare_digest(creds.username or "", user) and secrets.compare_digest(creds.password or "", pwd)
    if not ok:
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Basic"})


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TEMPLATES_DIR = BASE_DIR / "templates"
CONFIGS_DIR = BASE_DIR / "configs"

paths = Paths(base=DATA_DIR)

app = FastAPI(title="Contract Red-Flag Detector (Demo)")


def _is_heading_paragraph(text: str) -> bool:
    return parse_heading(text, max_len=160) is not None


def _approx_chars_limit() -> int:
    # Conservative default; override via env if needed.
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
    # Extend right by ~stride_tokens worth of paragraphs
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
    # collapse duplicates that can happen with odd formatting
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

    # include heading if it's the first element in section and looks like heading
    blocks: list[Paragraph] = []
    if sec and _is_heading_paragraph(sec[0].text):
        blocks.append(sec[0])
    for p in sec[start:end]:
        if p.paragraph_index == idx:
            continue
        blocks.append(p)

    # de-dup by paragraph_index
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
    # Preserve headers (esp. WWW-Authenticate) so browsers show Basic Auth prompt.
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
    _: None = Depends(require_basic_auth),
) -> UploadResponse:
    limits = get_limits()
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    raw = await file.read()
    max_bytes = limits["max_file_mb"] * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(status_code=400, detail=f"File too large. Max {limits['max_file_mb']} MB.")

    job_id = secrets.token_hex(12)
    upload_path = job_upload_path(paths, job_id, file.filename or "upload.pdf")
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(raw)

    # Fast pre-check: page limit (so we can fail fast on /upload).
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

    await create_job(paths.db, job_id, file.filename or "upload.pdf", upload_path)
    background.add_task(process_job, job_id)

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


async def process_job(job_id: str) -> None:
    limits = get_limits()
    row = await get_job(paths.db, job_id)
    if not row:
        return
    upload_path = Path(row["upload_path"])
    try:
        LOGGER.info("Job %s started: %s", job_id, upload_path.name)
        await update_job(paths.db, job_id, status=JobStatus.running, stage="parse", progress=0, started_at=datetime.now(timezone.utc))
        paragraphs, page_count = parse_pdf_to_paragraphs(
            upload_path,
            max_pages=limits["max_pages"],
            max_paragraphs=limits["max_paragraphs"],
        )
        LOGGER.info("Job %s parsed: pages=%s paragraphs=%s", job_id, page_count, len(paragraphs))
        await update_job(
            paths.db,
            job_id,
            stage="stage1",
            progress=5,
            page_count=page_count,
            paragraph_count=len(paragraphs),
        )

        # Provider selection (demo): prefer Nebius if configured, else OpenAI.
        base_url: Optional[str] = None
        message_mode = "string"
        api_key = os.environ.get("NEBIUS_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if os.environ.get("NEBIUS_API_KEY"):
            base_url = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
            # Nebius TokenFactory expects content as a list of text parts (matches their examples).
            message_mode = "parts"
        if not api_key:
            raise RuntimeError("NEBIUS_API_KEY or OPENAI_API_KEY is required")
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
        # In single-classifier mode we only need these 6 categories (faster to build and smaller prompts).
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

        llm = LLMRunner(
            LLMConfig(
                api_key=api_key,
                base_url=base_url,
                model_stage1=os.environ.get("OPENAI_MODEL_STAGE1", "gpt-5-nano"),
                model_stage2=os.environ.get("OPENAI_MODEL_STAGE2", "gpt-5-mini"),
                max_concurrency=limits["max_llm_concurrency"],
                stage1_include_descriptions=os.environ.get("STAGE1_INCLUDE_DESCRIPTIONS", "0") == "1",
                message_content_mode=message_mode,
            ),
            cats_payload,
            fewshot_by_category=fewshot,
        )

        enable_debug = os.environ.get("ENABLE_DEBUG_TRACES", "0") == "1"
        stage1_max_categories = int(os.environ.get("STAGE1_MAX_CATEGORIES", "10"))
        t_router = float(os.environ.get("T_ROUTER", "0.3"))
        router_topk = int(os.environ.get("ROUTER_TOPK_PROBS", "15"))
        debug_traces: list[dict] = []
        debug_by_idx: dict[int, dict] = {}

        # Initialize debug traces for EVERY paragraph up-front.
        if enable_debug:
            for p in paragraphs:
                trace = {
                    "page": p.page,
                    "paragraph_index": p.paragraph_index,
                    "paragraph_text": p.text,
                    "section_path": "",
                    "stage1": {"categories": []},
                    "stage1_raw": "",
                    "stage1_error": None,
                    "stage2_error": None,
                    "stage2_runs": [],
                }
                debug_traces.append(trace)
                debug_by_idx[p.paragraph_index] = trace

        # Stage 1 (default): run per SECTION (heading-to-heading).
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
            if enable_debug:
                t = debug_by_idx.get(p.paragraph_index)
                if t is not None:
                    t["section_path"] = path_by_idx[p.paragraph_index]

        # Build sections (heading-to-heading)
        sections: list[list[Paragraph]] = []
        cur: list[Paragraph] = []
        for p in paragraphs:
            if _is_heading_paragraph(p.text) and cur:
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

        # For each section, run stage1 on a block. If too large, chunk and add previous-N context.
        ctx_n = int(os.environ.get("STAGE1_CONTEXT_PARAGRAPHS", "5"))
        max_chars = _approx_chars_limit()

        para_to_stage1: dict[int, list[Stage1Category]] = {p.paragraph_index: [] for p in paragraphs}

        async def run_stage1_block(main_block: list[Paragraph], context_block: list[Paragraph]) -> None:
            # Skip tiny paras in the main block by filtering at source.
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
                # Assign categories to paragraphs based on evidence indices
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

        # Stage1 (CUAD-style): within each structural section, run sliding windows over approx tokens.
        window_tokens = int(os.environ.get("STAGE1_WINDOW_TOKENS", "1100"))
        stride_tokens = int(os.environ.get("STAGE1_STRIDE_TOKENS", "300"))
        edge_frac = float(os.environ.get("STAGE1_TRIGGER_EDGE_FRAC", "0.15"))

        def approx_size(ps: list[Paragraph]) -> int:
            return sum(len((p.text or "")) + 40 for p in ps)

        window_traces: list[dict] = []

        if single_classifier:
            # Single-classifier mode: each window is classified directly into up to 2 red-flag findings.
            await update_job(paths.db, job_id, stage="classify", progress=55)

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

            async def classify_window(sec_body: list[Paragraph], start_i: int, end_i: int, *, window_id: str) -> None:
                chunk_paras = sec_body[start_i:end_i]
                text_chunk = "\n\n".join(p.text for p in chunk_paras).strip()
                if not text_chunk:
                    return
                spath = path_by_idx.get(chunk_paras[0].paragraph_index, "")
                out, raw, prompt = await llm.classify_redflags_for_chunk(
                    text_chunk=text_chunk,
                    section_path=spath,
                    max_findings=2,
                    return_prompt=enable_debug,
                )

                if enable_debug:
                    prompt_user = (prompt or {}).get("user", "")
                    prompt_system = (prompt or {}).get("system", "")
                    window_traces.append(
                        {
                            "window_id": window_id,
                            "section_path": spath,
                            "paragraph_indices": [p.paragraph_index for p in chunk_paras],
                            "text_chunk": text_chunk[:8000],
                            "prompt_system": prompt_system[:2000],
                            "prompt_user": prompt_user[:20000],
                            "raw_output": (raw or "")[:8000],
                            "parsed": out.model_dump(),
                        }
                    )

                for j, f in enumerate(out.findings):
                    sev = severity_by_category.get(f.category, "medium")
                    # locate evidence paragraph
                    loc_para = None
                    for p in chunk_paras:
                        if f.evidence in p.text:
                            loc_para = p
                            break
                    if loc_para is None:
                        loc_para = chunk_paras[0]

                    cat_id = llm.redflag_category_id_by_name.get(f.category, "").strip() or f.category.lower().replace(" ", "_")
                    what = "\n".join(f"- {b}" for b in f.reasoning)
                    findings.append(
                        Finding(
                            finding_id=f"{window_id}_f{j+1}",
                            category_id=cat_id,
                            category_name=f.category,
                            severity=sev,  # type: ignore[arg-type]
                            confidence=0.8,
                            issue_title=f.category,
                            what_is_wrong=what,
                            recommendation=recommend_for(f.category),
                            evidence_quote=f.evidence,
                            location={"page": int(loc_para.page), "paragraph_index": int(loc_para.paragraph_index)},
                        )
                    )

            window_tasks = []
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

                    joined = "\n\n".join(p.text for p in main_block)
                    near_start, near_end = _find_trigger_edges(joined, edge_frac)
                    exp_s, exp_e = _extend_window_by_stride(
                        sec_body, s, e, stride_tokens=stride_tokens, extend_left=near_start, extend_right=near_end
                    )
                    expanded = sec_body[exp_s:exp_e]
                    while expanded and approx_size(expanded) > max_chars:
                        expanded = expanded[:-1]

                    start_i = exp_s
                    end_i = exp_e
                    window_id = f"w_{sec_body[s].paragraph_index:04d}_{sec_body[min(e-1, len(sec_body)-1)].paragraph_index:04d}"
                    window_tasks.append(classify_window(sec_body, start_i, end_i, window_id=window_id))

            done = 0
            for coro in asyncio.as_completed(window_tasks):
                await coro
                done += 1
                if window_tasks:
                    prog = 55 + int(40 * (done / max(1, len(window_tasks))))
                    await update_job(paths.db, job_id, stage="classify", progress=min(95, prog))

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
                    issue_title=out.issue_title,
                    what_is_wrong=out.what_is_wrong,
                    recommendation=out.recommendation,
                    evidence_quote=out.evidence_quote,
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

        findings = dedupe_findings(findings)
        summary = compute_summary(findings)
        LOGGER.info("Job %s findings=%s risk_score=%s", job_id, len(findings), summary.risk_score)
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
                prompt_version="p_single_v1" if single_classifier else "p_v1",
                models={"classifier": llm.cfg.model_stage2} if single_classifier else {"stage_1": llm.cfg.model_stage1, "stage_2": llm.cfg.model_stage2},
                debug=(
                    {
                        "enabled": enable_debug,
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
                    if enable_debug
                    else None
                ),
            ),
        )
        out_path = job_result_path(paths, job_id)
        out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        await update_job(paths.db, job_id, status=JobStatus.done, stage="done", progress=100, result_path=out_path)
        LOGGER.info("Job %s done", job_id)
    except Exception as e:
        LOGGER.exception("Job %s failed", job_id)
        await update_job(paths.db, job_id, status=JobStatus.failed, stage="failed", progress=100, error=str(e))

