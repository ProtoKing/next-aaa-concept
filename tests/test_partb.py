"""Tests for Part B: talent records, features, judge, prompts, LLM client, loop, pipeline."""

import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pandas as pd
import pytest

from nextaaa.config import PartBConfig
from nextaaa.partb import dataset, llm, loop, prompts, scorer, talent
from nextaaa.pipeline_v2 import run_part_a_v2


# ---------------------------------------------------------------------------
# Talent track records: no leakage from the future
# ---------------------------------------------------------------------------

def _toy_labels():
    return pd.DataFrame({
        "tconst": ["t1", "t2", "t3", "t4"], "startYear": [2000, 2001, 2001, 2005],
        "genres": ["Action", "Action,Sci-Fi", "Drama", "Action"],
        "aud_hit": [True, False, True, True], "has_finance": [True, True, True, True],
        "is_aaa_hit": [True, False, False, True], "audience_score": [90.0, 40.0, 80.0, 95.0],
    })


def _toy_credits():
    return pd.DataFrame({"tconst": ["t1", "t2", "t3", "t4"], "nconst": ["d1", "d1", "d2", "d1"],
                         "category": ["director"] * 4, "billing_rank": [1, 1, 1, 1]})


def test_asof_records_use_only_earlier_years():
    labels, credits = _toy_labels(), _toy_credits()
    pf = talent.person_film_table(labels, credits, leads_per_film=2)
    rec = talent.asof_records(pf, labels, a=2.0).set_index(["nconst", "year"])
    assert rec.loc[("d1", 2000), "prior_films"] == 0          # nothing before the first film
    assert rec.loc[("d1", 2001), "prior_aaa_hits"] == 1       # t1 counts for 2001...
    assert rec.loc[("d1", 2005), "prior_aaa_hits"] == 1       # ...but t4's own hit never counts for t4
    assert rec.loc[("d1", 2005), "prior_films"] == 2
    feats = talent.film_track_features(pf, rec.reset_index())
    assert feats.loc["t4", "dir_prior_aaa_hits"] == 1
    # smoothed rate: (1 + 2 * base) / (2 + 2), never the raw 50%
    base = labels["is_aaa_hit"].mean()
    assert feats.loc["t4", "dir_aaa_rate"] == pytest.approx((1 + 2 * base) / 4)


def test_records_as_of_for_concepts_include_everything_before():
    labels, credits = _toy_labels(), _toy_credits()
    pf = talent.person_film_table(labels, credits, 2)
    now = talent.records_as_of(pf, labels, 2006, 2.0).set_index("nconst")
    assert now.loc["d1", "prior_aaa_hits"] == 2 and now.loc["d1", "last_year"] == 2005


# ---------------------------------------------------------------------------
# Prompts and validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [(150_000_000, 150_000_000), ("$150M", 150_000_000),
                                          ("150 million", 150_000_000), ("1.2B", 1_200_000_000),
                                          ("abc", None)])
def test_money_parsing(raw, expected):
    assert prompts._to_int(raw) == expected


def test_validate_concept():
    bcfg = PartBConfig()
    good = {"title": "X", "logline": "A story.", "synopsis": "Longer.", "genres": ["sci-fi", "Nonsense"],
            "runtime_min": 300, "budget_usd": "$180M", "director_id": "d01", "lead_ids": ["A01", "A02"]}
    clean, problems = prompts.validate_concept(good, ["Sci-Fi", "Drama"], {"D01"}, {"A01", "A02"}, bcfg)
    assert clean["genres"] == ["Sci-Fi"] and clean["director_id"] == "D01"
    assert clean["runtime_min"] == bcfg.max_runtime and any("clipped" in p for p in problems)
    bad = {**good, "director_id": "D99"}
    assert prompts.validate_concept(bad, ["Sci-Fi"], {"D01"}, {"A01", "A02"}, bcfg)[0] is None
    same_leads = {**good, "lead_ids": ["A01", "A01"]}
    assert prompts.validate_concept(same_leads, ["Sci-Fi"], {"D01"}, {"A01", "A02"}, bcfg)[0] is None


def test_extract_json_tolerates_fences():
    assert llm.extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.extract_json('{"concepts": []} trailing') == {"concepts": []}


def test_refine_prompt_lists_lowest_first():
    hist = [{"title": t, "objective": o, "p_aaa_hit": o, "p_lo": o, "p_hi": o, "genres": ["Drama"],
             "budget_usd": 1e8, "runtime_min": 120, "director_id": "D01", "lead_ids": ["A01", "A02"],
             "drivers_text": "+ budget (+0.01)", "in_distribution": True}
            for t, o in (("High", 0.3), ("Low", 0.05))]
    msgs = prompts.refine_prompt("EVIDENCE", "Allowed genres: Drama", 4, "Drama", hist, [])
    text = msgs[1]["content"]
    assert text.index('"Low"') < text.index('"High"') and "Propose 4 NEW" in text


def test_novelty_threshold_ignores_exact_duplicates():
    rng = np.random.default_rng(0)
    e = rng.normal(size=(200, 16))
    e = np.vstack([e, e[:20]])            # 20 exact duplicates
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    assert loop.nn_similarity_threshold(e, 99) < 0.999


# ---------------------------------------------------------------------------
# A fake OpenAI-compatible server: exercises the real ChatLLM / embeddings code
# ---------------------------------------------------------------------------

class _FakeOpenAI(BaseHTTPRequestHandler):
    calls: list = []
    headers_seen: list = []
    vague_errors = False  # True: reject temperature with an error that never says "temperature"

    def log_message(self, *args):
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeOpenAI.calls.append(body)
        _FakeOpenAI.headers_seen.append({k.lower(): v for k, v in self.headers.items()})
        if self.path.endswith("/embeddings"):
            data = [{"object": "embedding", "index": i, "embedding": [float(len(t) % 7), 1.0, 0.5]}
                    for i, t in enumerate(body["input"])]
            return self._send(200, {"object": "list", "data": data, "model": body["model"],
                                    "usage": {"prompt_tokens": 1, "total_tokens": 1}})
        if "temperature" in body and _FakeOpenAI.vague_errors:  # behave like a provider with other wording
            return self._send(400, {"error": {"message": "Invalid request: value out of range",
                                              "type": "invalid_request_error"}})
        if "temperature" in body:  # behave like a reasoning model
            return self._send(400, {"error": {"message": "Unsupported value: 'temperature' does not support 1.0",
                                              "type": "invalid_request_error", "param": "temperature",
                                              "code": "unsupported_value"}})
        content = '```json\n{"concepts": [], "ok": true}\n```'
        return self._send(200, {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                                "choices": [{"index": 0, "finish_reason": "stop",
                                             "message": {"role": "assistant", "content": content}}],
                                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}})


@pytest.fixture()
def fake_server():
    server = HTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _FakeOpenAI.calls, _FakeOpenAI.headers_seen, _FakeOpenAI.vague_errors = [], [], False
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()
    server.server_close()


def test_claude_style_endpoint(fake_server, tmp_path, monkeypatch):
    """Claude API quirks: temperature < 1 rejected with non-OpenAI wording; max_tokens; workspace header."""
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    _FakeOpenAI.vague_errors = True
    ep = llm.Endpoint("BASELINE", fake_server, "k", "claude-x", max_tokens=1234, workspace_id="wrkspc_1")
    chat = llm.ChatLLM(ep, log_path=None)
    out = chat.complete_json([{"role": "user", "content": "hi"}], temperature=0.0, purpose="predict")
    assert out == {"concepts": [], "ok": True}
    assert chat._use_temperature is False and chat._use_json_mode is True  # dropped the right thing
    assert _FakeOpenAI.calls[-1]["max_completion_tokens"] == 1234
    assert _FakeOpenAI.headers_seen[-1]["anthropic-workspace-id"] == "wrkspc_1"
    chat.client.close()


LLM_VARS = [f"{r}_{n}" for r in ("GENERATOR", "BASELINE", "EMBEDDING", "OPENAI")
            for n in ("MODEL", "API_KEY", "BASE_URL", "API_VERSION", "MAX_TOKENS")] + ["ANTHROPIC_WORKSPACE_ID"]


@pytest.fixture()
def clean_llm_env(monkeypatch):
    for k in LLM_VARS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_env_openai_shared_key(clean_llm_env):
    """With an OpenAI key, only OPENAI_API_KEY and the model names are needed."""
    m = clean_llm_env
    assert llm.Endpoint.from_env("GENERATOR") is None                 # nothing set -> off (mock)
    m.setenv("OPENAI_API_KEY", "sk-test")
    assert llm.Endpoint.from_env("GENERATOR") is None                 # key but no model -> off
    m.setenv("GENERATOR_MODEL", "gpt-6-sol")
    m.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
    gen, emb = llm.Endpoint.from_env("GENERATOR"), llm.Endpoint.from_env("EMBEDDING")
    assert gen.base_url == emb.base_url == "https://api.openai.com/v1" and gen.api_key == "sk-test"
    assert gen.model == "gpt-6-sol" and gen.max_tokens is None and gen.api_version is None
    assert llm.Endpoint.from_env("BASELINE") is None                  # no BASELINE_MODEL -> off
    c = llm.make_openai_client(gen)
    assert str(c.base_url) == "https://api.openai.com/v1/"
    c.close()


def test_env_per_role_override_claude(clean_llm_env):
    """Mix providers: OpenAI shared, Claude for the generator only."""
    m = clean_llm_env
    m.setenv("OPENAI_API_KEY", "sk-openai")
    m.setenv("GENERATOR_MODEL", "claude-sonnet-5-5")
    m.setenv("GENERATOR_BASE_URL", "https://api.anthropic.com")
    m.setenv("GENERATOR_API_KEY", "sk-ant-test")
    m.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
    gen, emb = llm.Endpoint.from_env("GENERATOR"), llm.Endpoint.from_env("EMBEDDING")
    assert gen.is_anthropic and gen.api_key == "sk-ant-test" and gen.max_tokens == 16000
    assert emb.api_key == "sk-openai" and not emb.is_anthropic
    c = llm.make_openai_client(gen)
    assert str(c.base_url) == "https://api.anthropic.com/v1/"         # not Azure's /openai/v1
    c.close()


def test_live_client_against_fake_server(fake_server, tmp_path, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    ep = llm.Endpoint("GENERATOR", fake_server, "test-key", "my-deployment")
    chat = llm.ChatLLM(ep, log_path=tmp_path / "log.jsonl")
    out = chat.complete_json([{"role": "user", "content": "hi"}], temperature=1.0, purpose="generate")
    assert out == {"concepts": [], "ok": True}
    assert chat._use_temperature is False                       # learned to drop temperature
    assert _FakeOpenAI.calls[-1]["model"] == "my-deployment"
    assert _FakeOpenAI.calls[-1]["response_format"] == {"type": "json_object"}
    assert chat.usage["prompt_tokens"] == 11
    log = (tmp_path / "log.jsonl").read_text()
    assert "test-key" not in log and "generate" in log         # audit log, never the key

    emb = llm.APIEmbeddings(llm.Endpoint("EMBEDDING", fake_server, "k", "emb"), tmp_path)
    v = emb.embed(["abc", "abcdefgh", "abc"])
    assert v.shape == (3, 3) and np.allclose(np.linalg.norm(v, axis=1), 1)
    n_calls = len(_FakeOpenAI.calls)
    emb2 = llm.APIEmbeddings(llm.Endpoint("EMBEDDING", fake_server, "k", "emb"), tmp_path)
    emb2.embed(["abc"])
    assert len(_FakeOpenAI.calls) == n_calls                  # served from the disk cache
    for client in (chat.client, emb.client, emb2.client):
        client.close()


def test_endpoint_styles_azure(clean_llm_env):
    from openai import AzureOpenAI
    c = llm.make_openai_client(llm.Endpoint("GENERATOR", "https://res.openai.azure.com", "k", "d"))
    assert str(c.base_url).rstrip("/") == "https://res.openai.azure.com/openai/v1"
    c.close()
    m = clean_llm_env
    m.setenv("OPENAI_API_KEY", "k")
    m.setenv("OPENAI_BASE_URL", "https://res.openai.azure.com")
    m.setenv("OPENAI_API_VERSION", "2024-10-21")
    m.setenv("GENERATOR_MODEL", "my-gpt-deployment")
    c = llm.make_openai_client(llm.Endpoint.from_env("GENERATOR"))
    assert isinstance(c, AzureOpenAI)
    c.close()


def test_dotenv_blank_placeholder_does_not_hide_value(tmp_path, monkeypatch):
    for k in ("NEXTAAA_T_KEY", "NEXTAAA_T_URL"):
        monkeypatch.delenv(k, raising=False)
    (tmp_path / ".env").write_text("NEXTAAA_T_KEY=\nNEXTAAA_T_URL=https://a/v1\n# NEXTAAA_T_KEY=commented\n"
                                   "NEXTAAA_T_KEY=sk-real  # my key\nexport NEXTAAA_T_URL=https://b/v1\n")
    llm.load_dotenv(tmp_path / ".env")
    import os
    assert os.environ["NEXTAAA_T_KEY"] == "sk-real" and os.environ["NEXTAAA_T_URL"] == "https://b/v1"
    for k in ("NEXTAAA_T_KEY", "NEXTAAA_T_URL"):
        monkeypatch.delenv(k, raising=False)


def test_dotenv_does_not_override(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text('# comment\nNEXTAAA_TEST_A="from file"\nNEXTAAA_TEST_B=file\n')
    monkeypatch.setenv("NEXTAAA_TEST_B", "from env")
    monkeypatch.delenv("NEXTAAA_TEST_A", raising=False)
    llm.load_dotenv(tmp_path / ".env")
    import os
    assert os.environ["NEXTAAA_TEST_A"] == "from file" and os.environ["NEXTAAA_TEST_B"] == "from env"


# ---------------------------------------------------------------------------
# End to end on synthetic data with the mock LLM
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def partb_run(v2_paths, v2_cfg, tmp_path_factory):
    from nextaaa.partb.pipeline import run_part_b
    run_part_a_v2(v2_paths, v2_cfg, make_plots=False)
    paths = replace(v2_paths, output_dir=tmp_path_factory.mktemp("partb_out"))
    bcfg = replace(PartBConfig(), n_rounds=2, n_islands=2, concepts_per_call=4, n_bootstrap=2,
                   llm_baseline_max_films=30, shortlist_min_prior_films=1)
    summary = run_part_b(paths, v2_cfg, bcfg, mode="mock")
    return paths, bcfg, summary


def test_part_b_pipeline_outputs(partb_run):
    paths, bcfg, summary = partb_run
    out = paths.output_dir / "partb"
    assert summary["clients"]["mode"] == "mock"
    for f in ("pitch_cards.md", "final_portfolio.json", "summary.json", "llm_log.jsonl",
              "tables/backtest_test_metrics.csv", "tables/concepts_all.csv",
              "figures/01_backtest_pr_curves.png", "figures/05_concept_map.png"):
        assert (out / f).exists(), f
    concepts = pd.read_csv(out / "tables" / "concepts_all.csv")
    assert set(concepts["arm"]) == {"loop", "no_feedback"}
    assert summary["usage"]["generator"]["calls"] == 2 * bcfg.n_islands * bcfg.n_rounds
    scored = concepts[concepts["status"] == "scored"]
    assert scored["p_aaa_hit"].between(0, 1).all() and (scored["p_lo"] <= scored["p_hi"] + 1e-9).all()
    assert "MOCK MODE" in (out / "pitch_cards.md").read_text()
    assert summary["backtest"]["chosen"] in ("logistic", "hgb", "tabpfn")


def test_concept_and_film_features_share_columns(partb_run):
    paths, *_ = partb_run
    films = pd.read_csv(paths.output_dir / "partb" / "tables" / "features_films.csv", index_col=0)
    assert not films.filter(like="src_").isna().any().any()
    assert {"log_budget", "dir_aaa_rate", "lead_aaa_rate", "src_original"} <= set(films.columns)


def test_dotenv_later_lines_and_blank_placeholders(tmp_path, monkeypatch):
    """A blank placeholder early in .env must not hide the real value further down."""
    for k in ("NEXTAAA_T_KEY", "NEXTAAA_T_URL", "NEXTAAA_T_SHELL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NEXTAAA_T_SHELL", "from-shell")
    (tmp_path / ".env").write_text("NEXTAAA_T_KEY=\nNEXTAAA_T_URL=https://a/v1\n# NEXTAAA_T_KEY=commented\n"
                                   "NEXTAAA_T_KEY=sk-real  # my key\nexport NEXTAAA_T_URL=https://b/v1\n"
                                   "NEXTAAA_T_SHELL=from-file\n")
    llm.load_dotenv(tmp_path / ".env")
    import os
    assert os.environ["NEXTAAA_T_KEY"] == "sk-real"
    assert os.environ["NEXTAAA_T_URL"] == "https://b/v1"
    assert os.environ["NEXTAAA_T_SHELL"] == "from-shell"   # the shell still wins
    for k in ("NEXTAAA_T_KEY", "NEXTAAA_T_URL"):
        monkeypatch.delenv(k, raising=False)
