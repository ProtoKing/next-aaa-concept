"""LLM and embedding clients for Part B, configured entirely from environment variables.

Three roles, each with its own model:

    GENERATOR   writes the film concepts            e.g. gpt-6-sol
    BASELINE    the LLM-as-predictor backtest       e.g. gpt-5-mini (needs an old knowledge cutoff)
    EMBEDDING   plot embeddings (novelty, features) e.g. text-embedding-3-large

Settings (in the environment or in a `.env` file at the project root). The shared
connection uses the OpenAI SDK's own variable names, so with an OpenAI key you
only set the key and the three model names:

    OPENAI_API_KEY           the key, shared by every role
    OPENAI_BASE_URL          optional; default https://api.openai.com/v1 (see "Providers")
    OPENAI_API_VERSION       optional; only for the classic Azure "api-version" style

    GENERATOR_MODEL          a role is switched on by naming its model;
    BASELINE_MODEL           leave one empty to switch it off
    EMBEDDING_MODEL          (empty EMBEDDING_MODEL = free local TF-IDF embeddings)
    BASELINE_KNOWLEDGE_CUTOFF   YYYY-MM; the backtest only uses films released after it

    <ROLE>_API_KEY / <ROLE>_BASE_URL / <ROLE>_API_VERSION   optional per-role overrides,
                             to mix providers (e.g. GENERATOR_BASE_URL=https://api.anthropic.com/v1)
    <ROLE>_MAX_TOKENS        optional reply cap (default: none; 16000 for the Claude API)
    ANTHROPIC_WORKSPACE_ID   optional; only for Claude keys not tied to a workspace
    LLM_MODE                 auto (default) | mock | live, see "Modes"

Providers (all through the standard OpenAI client):
    * OpenAI: leave OPENAI_BASE_URL empty.
    * Azure OpenAI v1 API: OPENAI_BASE_URL=https://<res>.openai.azure.com
      (we add /openai/v1); the model names are your deployment names.
    * Classic Azure: also set OPENAI_API_VERSION -> AzureOpenAI client.
    * Claude API: https://api.anthropic.com/v1 (Anthropic's OpenAI-compatible layer).
      It ignores JSON mode (we parse JSON from the text anyway), has no embeddings,
      and newer Claude models reject temperature < 1 (we then drop temperature).
    * Any other OpenAI-compatible server (vLLM, Ollama, OpenRouter): a base URL ending in /v1.

Modes (LLM_MODE):
    auto (default)  live where a role is configured, mock otherwise (with a warning)
    mock            always offline: deterministic fake concepts, for testing the plumbing
    live            fail loudly if the generator is not configured

Embeddings without EMBEDDING_MODEL use a local TF-IDF + SVD model (scikit-learn),
so the novelty filter and text features always work.

Every LLM call is logged (prompt, response, token usage; never keys) to
outputs/partb/llm_log.jsonl: your audit trail and the "what does the prompt look
like" appendix slide.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import zlib
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from cookiecutter import zipfile

logger = logging.getLogger(__name__)

ROLES = ("GENERATOR", "BASELINE", "EMBEDDING")
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_dotenv(path: Path) -> list[str]:
    """Minimal .env reader (KEY=VALUE lines). Existing environment variables win.

    Within the file, a later line overrides an earlier one, and empty values are
    ignored, so a left-over blank placeholder (e.g. `GENERATOR_API_KEY=`) never
    hides a filled-in line further down.
    """
    if not path.exists():
        return []
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.split(" #")[0].strip().strip('"').strip("'")
        if key and value:
            values[key] = value
    loaded = [k for k in values if k not in os.environ]
    for k in loaded:
        os.environ[k] = values[k]
    return loaded


@dataclass(frozen=True)
class Endpoint:
    role: str
    base_url: str
    api_key: str = field(repr=False)  # never printed
    model: str
    api_version: str | None = None
    max_tokens: int | None = None
    workspace_id: str | None = None

    @property
    def is_anthropic(self) -> bool:
        return "api.anthropic.com" in self.base_url

    @classmethod
    def from_env(cls, role: str) -> "Endpoint | None":
        """<ROLE>_* settings, falling back to the shared OPENAI_* ones. None = role switched off."""
        def get(name: str) -> str | None:
            return os.environ.get(f"{role}_{name}") or os.environ.get(f"OPENAI_{name}") or None

        model = os.environ.get(f"{role}_MODEL")
        key = get("API_KEY")
        if not (model and key):
            return None
        base_url = get("BASE_URL") or OPENAI_DEFAULT_BASE_URL
        raw_max = os.environ.get(f"{role}_MAX_TOKENS")
        max_tokens = int(raw_max) if raw_max else (16000 if "api.anthropic.com" in base_url else None)
        return cls(role, base_url, key, model, get("API_VERSION"), max_tokens,
                   os.environ.get("ANTHROPIC_WORKSPACE_ID") or None)


def make_openai_client(ep: Endpoint):
    """OpenAI-compatible client for an endpoint (see "Providers" in the module docstring)."""
    from openai import AzureOpenAI, OpenAI

    if ep.api_version:
        return AzureOpenAI(azure_endpoint=ep.base_url, api_key=ep.api_key, api_version=ep.api_version)
    base = ep.base_url.rstrip("/")
    if not re.search(r"/v\d+$", base):  # a bare Azure resource URL, or the Claude host without /v1
        base += "/v1" if ep.is_anthropic else "/openai/v1"
    headers = {"anthropic-workspace-id": ep.workspace_id} if ep.workspace_id else None
    return OpenAI(base_url=base + "/", api_key=ep.api_key, default_headers=headers)


def extract_json(text: str) -> dict:
    """Parse JSON from a model reply, tolerating ```json fences or text around it."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in the reply.")
    return json.loads(text[start:end + 1])


class _CallLog:
    def __init__(self, path: Path | None):
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Chat models
# ---------------------------------------------------------------------------

class ChatLLM:
    """A live chat model that returns JSON.

    Robust to the usual differences between model families:
      * models that reject `temperature` (OpenAI reasoning models; Claude 4.7+ below 1.0)
        -> retried without it. Any 400 while temperature is set triggers this, because
        providers word the error differently.
      * deployments without JSON mode -> retried without it, JSON parsed from text
      * rate limits / timeouts -> exponential backoff
      * malformed JSON -> one repair request
      * a reply cut off at max_tokens -> a warning (raise <ROLE>_MAX_TOKENS)
    """

    is_mock = False

    def __init__(self, ep: Endpoint, log_path: Path | None = None, max_attempts: int = 5):
        self.ep = ep
        self.name = f"{ep.role}:{ep.model}"
        self.client = make_openai_client(ep)
        self.log = _CallLog(log_path)
        self.max_attempts = max_attempts
        self._use_temperature = True
        self._use_json_mode = True
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._lock = threading.Lock()

    def _create(self, messages: list[dict], temperature: float | None):
        import openai

        for attempt in range(self.max_attempts):
            kwargs = {"model": self.ep.model, "messages": messages}
            if temperature is not None and self._use_temperature:
                kwargs["temperature"] = temperature
            if self._use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if self.ep.max_tokens:  # max_completion_tokens: accepted by OpenAI reasoning models and Claude
                kwargs["max_completion_tokens"] = self.ep.max_tokens
            try:
                resp = self.client.chat.completions.create(**kwargs)
                if getattr(resp.choices[0], "finish_reason", None) == "length":
                    logger.warning("%s: reply hit max_tokens and was cut off; raise "
                                   "%s_MAX_TOKENS.", self.name, self.ep.role)
                return resp
            except openai.BadRequestError as exc:
                msg = str(exc).lower()
                sent_temp = "temperature" in kwargs
                sent_json = "response_format" in kwargs
                json_err = "response_format" in msg or "json" in msg
                if sent_temp and ("temperature" in msg or not (sent_json and json_err)):
                    logger.info("%s rejected the request with temperature=%s; retrying without "
                                "temperature (%s).", self.name, temperature, msg[:160])
                    self._use_temperature = False
                    continue
                if sent_json:
                    logger.info("%s has no JSON mode; retrying without it.", self.name)
                    self._use_json_mode = False
                    continue
                raise
            except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                    openai.InternalServerError) as exc:
                wait = min(60, 2 ** attempt * 2)
                logger.warning("%s: %s; retrying in %ds.", self.name, type(exc).__name__, wait)
                time.sleep(wait)
        raise RuntimeError(f"{self.name}: giving up after {self.max_attempts} attempts.")

    def complete_json(self, messages: list[dict], temperature: float | None = None,
                      purpose: str = "") -> dict:
        resp = self._create(messages, temperature)
        text = resp.choices[0].message.content or ""
        self._count(resp)
        try:
            data = extract_json(text)
        except (ValueError, json.JSONDecodeError):
            repair = messages + [{"role": "assistant", "content": text},
                                 {"role": "user", "content": "That was not valid JSON. "
                                  "Return only the JSON object, nothing else."}]
            resp = self._create(repair, temperature)
            text = resp.choices[0].message.content or ""
            self._count(resp)
            data = extract_json(text)
        with self._lock:
            self.log.write({"model": self.name, "purpose": purpose, "temperature": temperature,
                        "messages": messages, "response": text})
        return data

    def _count(self, resp) -> None:
        u = getattr(resp, "usage", None)
        with self._lock:
            self.usage["calls"] += 1
            if u is not None:
                self.usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
                self.usage["completion_tokens"] += getattr(u, "completion_tokens", 0)


class MockLLM:
    """Offline stand-in with the same interface. Produces schema-valid, varied,
    deterministic output so the whole pipeline (and the tests) run without keys.
    Its concepts are random recombinations: they do NOT get better with feedback,
    so never present mock results."""

    is_mock = True
    NOUNS = ["signal", "harbor", "frontier", "archive", "colony", "verdict", "orchard", "tide", "relay",
             "lantern", "glacier", "circuit", "mirage", "citadel", "meridian", "ember", "border", "island",
             "election", "plague", "kingdom", "robot", "ocean", "prison", "circus", "desert", "train",
             "festival", "satellite", "vineyard", "monastery", "casino", "reactor", "carnival", "embassy"]
    VERBS = ["must outwit", "races to stop", "uncovers", "is forced to protect", "teams up with",
             "infiltrates", "hunts", "escapes from", "inherits", "falls for", "betrays", "rebuilds"]
    HEROES = ["a disgraced pilot", "an estranged daughter", "a retired detective", "a young engineer",
              "a reluctant heir", "a field medic", "a smuggler", "a translator", "two sisters", "a chef",
              "a boxer", "a monk", "a journalist", "a nurse", "an android", "a teacher", "a widow"]

    def __init__(self, role: str = "GENERATOR", log_path: Path | None = None):
        self.name = f"{role}:mock"
        self.log = _CallLog(log_path)
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def complete_json(self, messages: list[dict], temperature: float | None = None,
                      purpose: str = "") -> dict:
        prompt = "\n".join(m["content"] for m in messages)
        rng = np.random.default_rng(zlib.crc32(prompt.encode()) + self.usage["calls"])
        self.usage["calls"] += 1
        if purpose.startswith("predict"):
            data = {"p_aaa_hit": float(np.clip(rng.beta(1, 12), 0, 1)), "reason": "mock"}
        else:
            data = {"concepts": [self._concept(prompt, rng) for _ in range(self._n(prompt))]}
        self.log.write({"model": self.name, "purpose": purpose, "messages": messages,
                        "response": json.dumps(data)})
        return data

    @staticmethod
    def _n(prompt: str) -> int:
        m = re.search(r"(?:Propose|Write) (\d+)", prompt)
        return int(m.group(1)) if m else 4

    def _concept(self, prompt: str, rng) -> dict:
        genres_line = re.search(r"Allowed genres: (.+)", prompt)
        genres = [g.strip() for g in genres_line.group(1).split(",")] if genres_line else ["Drama"]
        directors = sorted(set(re.findall(r"\b(D\d{2})\b", prompt))) or ["D01"]
        actors = sorted(set(re.findall(r"\b(A\d{2})\b", prompt))) or ["A01", "A02"]
        lead = re.search(r"Lead genre for this batch: (\w[\w-]*)", prompt)
        chosen = list(rng.choice(genres, size=min(2, len(genres)), replace=False))
        if lead and lead.group(1) in genres and lead.group(1) not in chosen:
            chosen[0] = lead.group(1)
        n1, n2, n3, n4 = rng.choice(self.NOUNS, 4, replace=False)
        hero, ally = rng.choice(self.HEROES, 2, replace=False)
        verb, verb2 = rng.choice(self.VERBS, 2, replace=False)
        logline = f"{hero.capitalize()} {verb} the {n1} before the {n2} falls."
        return {
            "title": f"The {n1.capitalize()} {n2.capitalize()}",
            "logline": logline,
            "synopsis": f"{logline} With {ally}, who {verb2} the {n3}, the trail leads to a {n4}.",
            "genres": chosen,
            "runtime_min": int(rng.integers(100, 165)),
            "budget_usd": int(rng.integers(100, 300)) * 1_000_000,
            "director_id": str(rng.choice(directors)),
            "lead_ids": [str(a) for a in rng.choice(actors, size=2, replace=False)],
            "comparables": ["(mock)", "(mock)"],
            "why_now": "mock",
        }


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class APIEmbeddings:
    """Embeddings from an OpenAI-compatible endpoint, cached on disk by text hash."""

    is_local = False

    def __init__(self, ep: Endpoint, cache_dir: Path, batch_size: int = 256):
        self.ep = ep
        self.name = f"api:{ep.model}"
        self.client = make_openai_client(ep)
        self.batch_size = batch_size
        self.cache_path = cache_dir / f"embeddings__{re.sub(r'[^A-Za-z0-9_.-]', '_', ep.model)}.npz"
        self._cache: dict[str, np.ndarray] = {}
        if self.cache_path.exists():
            with np.load(self.cache_path) as z:
                self._cache = {k: z[k] for k in z.files}
        self.usage = {"calls": 0, "texts": 0}

        if self.cache_path.exists():
            try:
                with np.load(self.cache_path) as z:
                    self._cache = {k: z[k] for k in z.files}
            except (zipfile.BadZipFile, OSError, ValueError, EOFError) as exc:
                bad = self.cache_path.with_name(self.cache_path.name + ".corrupt")
                self.cache_path.replace(bad)
                logger.warning("Embedding cache unreadable (%s): moved to %s. Missing embeddings "
                               "will be re-fetched; run the recovery cell to rescue it.", exc, bad.name)

    def fit(self, texts) -> "APIEmbeddings":
        return self  # nothing to fit

    def embed(self, texts: list[str]) -> np.ndarray:
        import openai

        keys = [f"h{zlib.crc32(t.encode()):08x}{len(t)}" for t in texts]
        todo = sorted({(k, t) for k, t in zip(keys, texts) if k not in self._cache})
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i:i + self.batch_size]
            for attempt in range(5):
                try:
                    resp = self.client.embeddings.create(model=self.ep.model,
                                                         input=[t for _, t in batch])
                    break
                except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError) as exc:
                    wait = min(60, 2 ** attempt * 2)
                    logger.warning("Embeddings: %s; retrying in %ds.", type(exc).__name__, wait)
                    time.sleep(wait)
            else:
                raise RuntimeError("Embeddings: giving up after 5 attempts.")
            for (k, _), item in zip(batch, resp.data):
                self._cache[k] = np.asarray(item.embedding, dtype=np.float32)
            self.usage["calls"] += 1
            self.usage["texts"] += len(batch)
        if todo:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            # np.savez_compressed(self.cache_path, **self._cache)
            self._save_cache()
        return _normalize(np.vstack([self._cache[k] for k in keys]))

    def _save_cache(self) -> None:
        tmp = self.cache_path.with_name(self.cache_path.stem + ".tmp.npz")
        np.savez(tmp, **self._cache)
        os.replace(tmp, self.cache_path)   # atomic swap: a crash mid-save keeps the old cache

class LocalEmbeddings:
    """Offline fallback: TF-IDF + truncated SVD (latent semantic analysis).
    Weaker than a neural embedding but needs no API; fitted on the film plots."""

    is_local = True

    def __init__(self, n_components: int = 256, random_state: int = 42):
        self.name = "local:tfidf-svd"
        self.n_components = n_components
        self.random_state = random_state
        self._pipe = None
        self.usage = {"calls": 0, "texts": 0}

    def fit(self, texts: list[str]) -> "LocalEmbeddings":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline

        vec = TfidfVectorizer(stop_words="english", min_df=2, max_features=50_000, sublinear_tf=True)
        n_terms = len(vec.fit(texts).vocabulary_)
        k = max(2, min(self.n_components, n_terms - 1, len(texts) - 1))
        self._pipe = make_pipeline(vec, TruncatedSVD(n_components=k, random_state=self.random_state))
        self._pipe.fit(texts)
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._pipe is None:
            raise RuntimeError("LocalEmbeddings.fit() must be called first (on the film plots).")
        self.usage["texts"] += len(texts)
        return _normalize(self._pipe.transform(texts).astype(np.float32))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

@dataclass
class Clients:
    generator: object
    judge: object
    embeddings: object
    mode: str

    def describe(self) -> dict:
        return {"mode": self.mode, "generator": self.generator.name, "judge": self.judge.name,
                "embeddings": self.embeddings.name}

    def usage(self) -> dict:
        return {"generator": self.generator.usage, "judge": self.judge.usage,
                "embeddings": self.embeddings.usage}


def get_clients(project_root: Path, cache_dir: Path, log_path: Path | None,
                mode: str | None = None) -> Clients:
    """Build the three clients from the environment (and .env), per LLM_MODE."""
    loaded = load_dotenv(project_root / ".env")
    if loaded:
        logger.info("Loaded %d settings from .env", len(loaded))
    mode = (mode or os.environ.get("LLM_MODE", "auto")).lower()
    if mode not in ("auto", "mock", "live"):
        raise ValueError(f"LLM_MODE must be auto, mock or live, not {mode!r}")

    eps = {r: (None if mode == "mock" else Endpoint.from_env(r)) for r in ROLES}
    if mode == "live" and eps["GENERATOR"] is None:
        raise RuntimeError("LLM_MODE=live but the generator is not configured: set OPENAI_API_KEY "
                           "(or GENERATOR_API_KEY) and GENERATOR_MODEL.")
    gen = ChatLLM(eps["GENERATOR"], log_path) if eps["GENERATOR"] else MockLLM("GENERATOR", log_path)
    judge = ChatLLM(eps["BASELINE"], log_path) if eps["BASELINE"] else MockLLM("BASELINE", log_path)
    emb = APIEmbeddings(eps["EMBEDDING"], cache_dir) if eps["EMBEDDING"] else LocalEmbeddings()
    for role, client in (("generator", gen), ("LLM baseline", judge)):
        if getattr(client, "is_mock", False):
            logger.warning("%s: MOCK mode (no key or model set). Results are placeholders.", role)
    for ep in eps.values():
        if ep:
            logger.info("%-9s -> %s at %s", ep.role, ep.model, ep.base_url)  # never the key
    resolved = "mock" if gen.is_mock else "live"
    return Clients(generator=gen, judge=judge, embeddings=emb, mode=resolved)
