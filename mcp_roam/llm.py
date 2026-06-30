"""LLM integration via Ollama for note enhancement."""

import json
import os
import re
import urllib.request
from dataclasses import dataclass


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "granite3.3:latest")
CHUNK_WORDS = 500


@dataclass(frozen=True)
class EnhancementResult:
    summary: str
    themes: str
    timestamps: str


@dataclass(frozen=True)
class ClaimResult:
    claims: list[str]
    evidence: list[str]


def _generate(prompt: str, model: str = "", timeout: int = 300) -> str:
    model = model or OLLAMA_MODEL
    url = f"http://{OLLAMA_HOST}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_ctx": 4096},
    }).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read()).get("response", "").strip()


def _chunk_text(text: str, max_words: int = CHUNK_WORDS) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i : i + max_words])
        for i in range(0, len(words), max_words)
    ]


def _map_chunk(chunk: str, idx: int, total: int) -> str:
    prompt = (
        f"Summarize this segment (part {idx}/{total}) of a lecture/note:\n\n"
        f"{chunk}\n\n"
        f"Respond in 2-3 sentences capturing the key ideas."
    )
    return _generate(prompt)


def _reduce(summaries: list[str], title: str) -> str:
    joined = "\n\n".join(
        f"Part {i+1}: {s}" for i, s in enumerate(summaries)
    )
    prompt = (
        f'You are analyzing: "{title}".\n\n'
        f"Below are summaries of each segment:\n\n{joined}\n\n"
        f"Produce:\n"
        f"## SUMMARY\n<3-5 sentences>\n\n"
        f"## THEMES\n<comma-separated key themes as short phrases>"
    )
    return _generate(joined, timeout=120) if len(summaries) <= 1 else _generate(
        f'Analyze: "{title}".\n\nSegment summaries:\n\n{joined}\n\n'
        f"Respond:\n## SUMMARY\n<3-5 sentence summary>\n## THEMES\n<comma-separated themes>",
        timeout=600,
    )


def enhance_content(content: str, title: str = "") -> EnhancementResult:
    text = re.sub(r"\s+", " ", content).strip()
    chunks = _chunk_text(text)

    if len(chunks) <= 1:
        raw = _generate(
            f'Summarize this note "{title}":\n\n{text[:2000]}\n\n'
            f"Respond:\n## SUMMARY\n<3-5 sentences>\n## THEMES\n<comma-separated themes>"
        )
    else:
        summaries = []
        for i, chunk in enumerate(chunks):
            summaries.append(_map_chunk(chunk, i + 1, len(chunks)))
        raw = _reduce(summaries, title)

    summary = ""
    themes = ""
    s = raw.find("## SUMMARY")
    t = raw.find("## THEMES")
    if s != -1:
        summary = raw[s + 10 : t if t != -1 else len(raw)].strip()
    if t != -1:
        themes = raw[t + 9 :].strip()

    return EnhancementResult(summary=summary, themes=themes, timestamps="")


def extract_claims(
    content: str,
    title: str = "",
    max_claims: int = 10,
) -> ClaimResult:
    """Extract atomic claims and evidence from note content via Ollama.

    Uses map-reduce for long content: chunks → per-chunk claims → merge.
    """
    text = re.sub(r"\s+", " ", content).strip()
    chunks = _chunk_text(text)

    if len(chunks) <= 1:
        raw = _generate(
            f'Extract key claims from "{title}":\n\n{text[:3000]}\n\n'
            f"List up to {max_claims} atomic, self-contained claims.\n"
            f"Each claim should be a single sentence that stands alone.\n"
            f"Respond:\n## CLAIMS\n- claim 1\n- claim 2\n..."
        )
    else:
        per_chunk: list[str] = []
        for i, chunk in enumerate(chunks):
            res = _generate(
                f"Extract 3-5 key atomic claims from this segment (part {i+1}/{len(chunks)}) "
                f'of "{title}":\n\n{chunk}\n\n'
                f"Each claim must be self-contained.\n"
                f"Respond:\n- claim 1\n- claim 2\n..."
            )
            per_chunk.append(res)

        joined = "\n\n".join(per_chunk)
        raw = _generate(
            f"Below are candidate claims extracted from segments of \"{title}\":\n\n"
            f"{joined}\n\n"
            f"Merge, deduplicate, and select the top {max_claims} most important "
            f"self-contained claims.\n"
            f"Respond:\n## CLAIMS\n- claim 1\n- claim 2\n..."
        )

    claims: list[str] = []
    evidence: list[str] = []

    claim_line_re = re.compile(r'^(?:[-*]\s+|\d+[.)]\s+)(.+)')
    claim_prefix_re = re.compile(
        r'^(?:\*\*Claim\s*\d+\*\*\s*:\s*|Claim\s*:\s*)', re.IGNORECASE
    )

    in_claims = False
    for line in raw.split("\n"):
        stripped = line.strip()
        if stripped.startswith("## CLAIMS"):
            in_claims = True
            continue
        if stripped.startswith("## "):
            in_claims = False
            continue
        if in_claims:
            m = claim_line_re.match(stripped)
            if m:
                text = claim_prefix_re.sub("", m.group(1)).strip()
                if text:
                    claims.append(text)

    if not claims:
        for line in raw.split("\n"):
            stripped = line.strip()
            m = claim_line_re.match(stripped)
            if m:
                text = claim_prefix_re.sub("", m.group(1)).strip()
                if text:
                    claims.append(text)

    return ClaimResult(claims=claims[:max_claims], evidence=evidence)
