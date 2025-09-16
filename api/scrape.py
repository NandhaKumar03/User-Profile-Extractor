# profile_scraper_llm.py
import os
import re
import json
import asyncio
import logging
import textwrap
from typing import Optional, List, Dict, Any
from urllib.parse import urljoin, urlparse
import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, AnyUrl

from fastapi.middleware.cors import CORSMiddleware # Added this line
from dotenv import load_dotenv
load_dotenv()

# Optional playwright import
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    PLAYWRIGHT_AVAILABLE = False

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("profile-scraper-llm")

app = FastAPI(title="Profile Scraper (LLM)", version="1.0")

load_dotenv()

origins = [
    "http://localhost:3000",  # Your frontend's address
    "http://127.0.0.1:3000",  # Another common address for localhost
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------- Configuration (tweak these if you change model) ----------
#DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_MODEL = "deepseek-r1-distill-llama-70b"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "gsk_YOUR_KEY_HERE")  # replace if you must hardcode
# Model token capacity (approx). For llama-3.1-8b-instant your logs showed 6000 tokens limit -> use 6000
MODEL_TOKEN_LIMIT = 6000
MAX_COMPLETION_TOKENS = 1024   # tokens reserved for model completion
TOKEN_SAFETY_MARGIN = 200      # extra margin
# chars per token heuristic (rough): use 4 chars per token (English average)
CHARS_PER_TOKEN = 4

DEFAULT_MAX_INPUT_CHARS = 32000  # fallback char cap if compute fails
HTTP_TIMEOUT = 40

SOCIAL_DOMAINS = [
    "linkedin.com", "twitter.com", "x.com", "facebook.com", "instagram.com",
    "github.com", "gitlab.com", "stackoverflow.com", "medium.com",
    "angel.co", "behance.net", "dribbble.com", "youtube.com", "t.me", "discord.com"
]

# ---------- Request model ----------
class ScrapeRequest(BaseModel):
    url: AnyUrl

# ---------- Helpers ----------
def pick_user_agent() -> str:
    return "profile-scraper/1.0 (+https://example.com)"

def absolutize(url: str, base: str) -> str:
    try:
        return urljoin(base, url)
    except Exception:
        return url

def sanitize_for_prompt(s: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", s)

# ---------- System prompt & example (used to estimate system size) ----------
SYSTEM_INSTRUCTIONS = textwrap.dedent("""
    You are a JSON-only extractor. Output EXACTLY one valid JSON object and NOTHING ELSE.
    The JSON must match this schema exactly (use null for missing fields):
    {
      "ok": true,
      "url": "string",
      "profiles": [
        {
          "name": "string or null",
          "title": "string or null",
          "email": "string or null",
          "image": "string or null",
          "bio": "string or null",
          "socials": [ { "platform": "string or null", "url": "string" } ],
          "source": "string or null"
        }
      ]
    }

    Rules:
    1) Map image URL to the person's name only if confident.
    2) Only include social/profile URLs that clearly belong to the person (prefer URLs containing the name).
    3) Set platform based on domain when possible otherwise null.
    4) Return empty profiles array if none found.
    5) Never output markdown fences or text outside the JSON.
""").strip()

EXAMPLE_USER = textwrap.dedent("""
    EXAMPLE_PAGE_TEXT:
    Dennis Woodside - CEO
    [IMG: https://example.com/images/dennis.jpg]
    LinkedIn: https://www.linkedin.com/in/dennis-woodside

    EXPECTED_JSON:
    {
      "ok": true,
      "url": "https://example.com/people",
      "profiles": [
        {
          "name": "Dennis Woodside",
          "title": "CEO",
          "email": null,
          "image": "https://example.com/images/dennis.jpg",
          "bio": null,
          "socials": [
            { "platform": "linkedin", "url": "https://www.linkedin.com/in/dennis-woodside" }
          ],
          "source": null
        }
      ]
    }
""").strip()

FULL_SYSTEM = SYSTEM_INSTRUCTIONS + "\n\n" + "SHORT_EXAMPLE:\n" + EXAMPLE_USER
FULL_SYSTEM_CHARLEN = len(FULL_SYSTEM)

# ---------- Fetching ----------
async def fetch_via_httpx(url: str, timeout: int = HTTP_TIMEOUT) -> (str, str):
    headers = {"User-Agent": pick_user_agent()}
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        return str(resp.url), resp.text

async def render_with_playwright(url: str, max_wait: int = 8) -> (str, str):
    if not PLAYWRIGHT_AVAILABLE:
        raise RuntimeError("Playwright not installed.")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(user_agent=pick_user_agent())
        await page.goto(url, timeout=max(30000, max_wait * 1000))
        try:
            await page.wait_for_load_state("networkidle", timeout=max(5000, max_wait * 1000))
        except Exception:
            await asyncio.sleep(min(max_wait, 5))
        content = await page.content()
        final = page.url
        await browser.close()
        return final, content

async def fetch_page(url: str, render: bool = False) -> (str, str):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")
    if render and PLAYWRIGHT_AVAILABLE:
        try:
            return await render_with_playwright(url)
        except Exception as e:
            logger.warning("Playwright render failed: %s. Falling back to httpx.", e)
    return await fetch_via_httpx(url)

# ---------- Ordered text extraction ----------
BLOCK_TAGS = ["article", "section", "div", "p", "li", "header", "main", "figure", "blockquote",
              "h1", "h2", "h3", "h4", "h5", "h6", "td", "th"]

def extract_ordered_text_and_markers(html: str, base_url: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    pieces = []
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        pieces.append(f"PAGE_TITLE: {title_tag.string.strip()}")
    dtag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property":"og:description"})
    if dtag and dtag.get("content"):
        pieces.append(f"PAGE_DESCRIPTION: {dtag.get('content').strip()}")
    for script in soup.find_all("script", type=lambda v: v and "ld+json" in v):
        text = script.string or script.get_text() or ""
        if text and len(text.strip()) > 10:
            pieces.append(f"JSON-LD: {text.strip()}")

    body = soup.body
    if not body:
        full_text = soup.get_text(" ", strip=True)
        pieces.append(full_text)
        return sanitize_for_prompt("\n\n".join(pieces))

    for tag in body.find_all(BLOCK_TAGS, recursive=True):
        if tag.has_attr("hidden"):
            continue
        text = tag.get_text(" ", strip=True)
        if not text:
            continue
        markers = []
        for img in tag.find_all("img"):
            src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
            if src:
                markers.append(f"[IMG: {absolutize(src, base_url)}]")
            else:
                alt = img.get("alt")
                if alt:
                    markers.append(f"[IMG_ALT: {alt}]")
        for a in tag.find_all("a", href=True):
            href = a.get("href").strip()
            if not href or href.startswith("javascript:") or href.startswith("mailto:"):
                continue
            try:
                abs_href = absolutize(href, base_url)
            except Exception:
                abs_href = href
            domain = urlparse(abs_href).netloc.lower()
            for sd in SOCIAL_DOMAINS:
                if sd in domain:
                    markers.append(f"[LINK: {abs_href}]")
                    break
        block_parts = [text]
        if markers:
            seen = set()
            unique_markers = []
            for m in markers:
                if m not in seen:
                    seen.add(m)
                    unique_markers.append(m)
            block_parts.append(" ".join(unique_markers))
        block_text = " ".join(block_parts).strip()
        if len(block_text) >= 3:
            pieces.append(block_text)

    if not pieces:
        pieces.append(body.get_text(" ", strip=True))
    joined = "\n\n".join(pieces)
    return sanitize_for_prompt(joined)

# ---------- Groq LLM call ----------
async def call_groq_llm(prompt_text: str, model: str = DEFAULT_MODEL, max_tokens: int = MAX_COMPLETION_TOKENS) -> str:
    from groq import Groq
    full_system = FULL_SYSTEM
    def sync_call():
        client = Groq(api_key=GROQ_API_KEY, max_retries=0) # Added max_retries=0
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": full_system},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.0,
            max_completion_tokens=max_tokens,
            top_p=1,
            stream=False,
        )
        return completion
    completion = await asyncio.to_thread(sync_call)
    try:
        choice = completion.choices[0]
        msg = getattr(choice, "message", None)
        if msg:
            reply = getattr(msg, "content", "") or ""
        else:
            reply = getattr(choice, "text", "") or str(choice)
    except Exception:
        reply = str(completion)
    reply = (reply or "").strip()
    logger.debug("LLM reply length=%s", len(reply))
    return reply

# ---------- Robust parsing & helpers ----------
def fix_trailing_commas(s: str) -> str:
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*\]", "]", s)
    return s

def safe_parse_json_reply(text: str) -> Optional[dict]:
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    cleaned = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    best = None
    for m in re.finditer(r"\{", s):
        start = m.start()
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start:i+1]
                    try:
                        json.loads(candidate)
                        if best is None or len(candidate) > len(best):
                            best = candidate
                    except Exception:
                        try:
                            json.loads(fix_trailing_commas(candidate))
                            if best is None or len(candidate) > len(best):
                                best = candidate
                        except Exception:
                            pass
                    break
    if best:
        try:
            return json.loads(best)
        except Exception:
            try:
                return json.loads(fix_trailing_commas(best))
            except Exception:
                return None
    try:
        return json.loads(fix_trailing_commas(cleaned))
    except Exception:
        return None

def extract_profiles_from_truncated_reply(reply: str) -> List[Dict[str, Any]]:
    if not reply or not isinstance(reply, str):
        return []
    profiles = []
    m = re.search(r'"profiles"\s*:\s*\[', reply)
    if not m:
        return []
    s = reply[m.end():]
    L = len(s)
    i = 0
    while True:
        start = s.find('{', i)
        if start == -1:
            break
        depth = 0
        found_complete = False
        for j in range(start, L):
            ch = s[j]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = s[start:j+1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            profiles.append(obj)
                        found_complete = True
                        i = j + 1
                        break
                    except Exception:
                        try:
                            obj = json.loads(fix_trailing_commas(candidate))
                            if isinstance(obj, dict):
                                profiles.append(obj)
                            found_complete = True
                            i = j + 1
                            break
                        except Exception:
                            i = j + 1
                            found_complete = True
                            break
        if not found_complete:
            break
    return profiles

def dedupe_profiles(profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for p in profiles:
        name = (p.get("name") or "").strip().lower()
        title = (p.get("title") or "").strip().lower()
        image = (p.get("image") or "").strip()
        first_social = ""
        socials = p.get("socials") or []
        if isinstance(socials, list) and socials:
            first = socials[0]
            if isinstance(first, dict):
                first_social = (first.get("url") or "").strip()
            elif isinstance(first, str):
                first_social = first.strip()
        key = (name, title, image, first_social)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out

def validate_and_normalize_schema(obj: dict) -> Dict[str, Any]:
    out = {"ok": False, "url": None, "profiles": []}
    if not isinstance(obj, dict):
        return out
    ok = obj.get("ok")
    out["ok"] = bool(ok) if ok is not None else True
    out["url"] = obj.get("url") or None
    profiles = obj.get("profiles") or []
    if not isinstance(profiles, list):
        profiles = []
    norm_profiles = []
    for p in profiles:
        if not isinstance(p, dict):
            continue
        np = {
            "name": p.get("name") if p.get("name") is not None else None,
            "title": p.get("title") if p.get("title") is not None else None,
            "email": p.get("email") if p.get("email") is not None else None,
            "image": p.get("image") if p.get("image") is not None else None,
            "bio": p.get("bio") if p.get("bio") is not None else None,
            "socials": [],
            "source": p.get("source") if p.get("source") is not None else None,
        }
        socials = p.get("socials") or []
        if isinstance(socials, list):
            for s in socials:
                if isinstance(s, dict) and s.get("url"):
                    np["socials"].append({
                        "platform": s.get("platform") if s.get("platform") is not None else None,
                        "url": s.get("url")
                    })
                elif isinstance(s, str):
                    np["socials"].append({"platform": None, "url": s})
        norm_profiles.append(np)
    out["profiles"] = norm_profiles
    return out

# ---------- Token/char estimate helpers ----------
def estimate_tokens_from_chars(char_count: int) -> int:
    return max(1, int(char_count / CHARS_PER_TOKEN))

def estimate_chars_from_tokens(token_count: int) -> int:
    return int(token_count * CHARS_PER_TOKEN)

# ---------- Endpoint ----------
@app.post("/api/scrape")
async def api_scrape(payload: ScrapeRequest):
    url = str(payload.url)
    render = False
    model = DEFAULT_MODEL

    try:
        final_url, html = await fetch_page(url, render=render)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Error fetching page: {e}")
    except Exception as e:
        logger.exception("Fetch error")
        raise HTTPException(status_code=500, detail=str(e))

    ordered_text = extract_ordered_text_and_markers(html, base_url=final_url)
    if not ordered_text:
        return JSONResponse(status_code=200, content={"ok": True, "url": final_url, "profiles": []})

    # Compute allowed input chars per call:
    allowed_input_tokens = max(200, MODEL_TOKEN_LIMIT - MAX_COMPLETION_TOKENS - TOKEN_SAFETY_MARGIN)
    # account for system prompt + example length (approx chars)
    allowed_chars = estimate_chars_from_tokens(allowed_input_tokens) - FULL_SYSTEM_CHARLEN
    # safety lower bound
    if allowed_chars < 2000:
        allowed_chars = min(DEFAULT_MAX_INPUT_CHARS, 2000)

    logger.info("Computed allowed_chars per call=%s (model_tokens=%s, system_len=%s chars)",
                allowed_chars, MODEL_TOKEN_LIMIT, FULL_SYSTEM_CHARLEN)

    # Split into coherent parts near double-newline boundaries
    parts: List[str] = []
    text = ordered_text
    L = len(text)
    if L <= allowed_chars:
        parts = [text]
    else:
        i = 0
        while i < L:
            end = min(i + allowed_chars, L)
            if end < L:
                cut = text.rfind("\n\n", i, end)
                if cut == -1 or cut <= i:
                    cut = end
                else:
                    end = cut
            else:
                end = L
            parts.append(text[i:end])
            i = end
            while i < L and text[i] in ("\n", "\r"):
                i += 1

    all_profiles: List[Dict[str, Any]] = []

    for idx, part in enumerate(parts, start=1):
        if idx > 2:  # Add this condition to limit to the first two parts
            break
        prompt = (
            "Extract all person/user profiles from the following text. "
            "Return ONLY a single JSON object exactly matching this schema:\n\n"
            + FULL_SYSTEM + "\n\n"
            f"PAGE_TEXT (part {idx}):\n\n"
        )
        prompt_text = prompt + part

        # call LLM
        try:
            reply = await call_groq_llm(prompt_text, model=model, max_tokens=MAX_COMPLETION_TOKENS)
        except Exception as e:
            logger.exception("LLM call failed for part %s: %s", idx, e)
            # if API says request too large, fallback to splitting this part more aggressively:
            # split part into halves and process each half (simple recursion-ish)
            if "Request too large" in str(e) or "Request too large" in getattr(e, "args", [""])[0] or "413" in str(e):
                # split into two and continue
                half = len(part) // 2
                subparts = [part[:half], part[half:]]
                for sub_idx, sp in enumerate(subparts, start=1):
                    try:
                        reply_sp = await call_groq_llm(FULL_SYSTEM + "\n\nPAGE_TEXT:\n\n" + sp,
                                                      model=model, max_tokens=MAX_COMPLETION_TOKENS)
                        parsed_sp = safe_parse_json_reply(reply_sp)
                        if parsed_sp and isinstance(parsed_sp, dict) and isinstance(parsed_sp.get("profiles", None), list):
                            for p in parsed_sp.get("profiles", []):
                                if isinstance(p, dict):
                                    all_profiles.append(p)
                            continue
                        truncated_sp = extract_profiles_from_truncated_reply(reply_sp)
                        for p in truncated_sp:
                            if isinstance(p, dict):
                                all_profiles.append(p)
                    except Exception as ee:
                        logger.warning("Subpart LLM failed: %s", ee)
                continue
            else:
                continue

        logger.info("LLM reply length (part %s) = %s", idx, len(reply or ""))

        # robust parse
        parsed = safe_parse_json_reply(reply)
        if parsed and isinstance(parsed, dict) and isinstance(parsed.get("profiles", None), list):
            for p in parsed.get("profiles", []):
                if isinstance(p, dict):
                    all_profiles.append(p)
            continue

        truncated_profiles = extract_profiles_from_truncated_reply(reply)
        if truncated_profiles:
            for p in truncated_profiles:
                if isinstance(p, dict):
                    all_profiles.append(p)
            continue

        # fallback: short prompt asking only for profiles (retry once)
        try:
            fallback_prompt = (
                "Return ONLY a JSON object with the key 'profiles' mapping to an array of person objects "
                "matching the schema. If none found, return {\"profiles\": []}.\n\n"
                f"PAGE_TEXT (part {idx}) (truncated to safe length):\n\n"
            )
            fallback_reply = await call_groq_llm(fallback_prompt + part[:8000],
                                                 model=model, max_tokens=MAX_COMPLETION_TOKENS)
            parsed_fb = safe_parse_json_reply(fallback_reply)
            if parsed_fb and isinstance(parsed_fb, dict) and isinstance(parsed_fb.get("profiles", None), list):
                for p in parsed_fb.get("profiles", []):
                    if isinstance(p, dict):
                        all_profiles.append(p)
                continue
            # extract truncated objects from fallback reply
            fb_truncated = extract_profiles_from_truncated_reply(fallback_reply)
            for p in fb_truncated:
                if isinstance(p, dict):
                    all_profiles.append(p)
        except Exception as e:
            logger.warning("Fallback LLM call failed for part %s: %s", idx, e)
            continue

    # dedupe, normalize, return
    all_profiles = dedupe_profiles(all_profiles)
    final_obj = {"ok": True, "url": final_url, "profiles": all_profiles}
    normalized = validate_and_normalize_schema(final_obj)
    if not normalized.get("url"):
        normalized["url"] = final_url
    return JSONResponse(status_code=200, content=normalized)

@app.get("/")
async def root():
    return {"message": "Profile Scraper LLM. POST /api/scrape with JSON {url}"}