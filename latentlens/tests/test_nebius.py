import json

import pytest
import requests

from src import nebius_client as nb

KEY = "sk-test-SECRET-123"


class FakeResp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code, self._body, self.text = status, body, text or json.dumps(body or {})

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeSession:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []

    def post(self, url, **kw):
        self.calls.append(("POST", url, kw))
        if self.exc:
            raise self.exc
        return self.resp

    def get(self, url, **kw):
        self.calls.append(("GET", url, kw))
        if self.exc:
            raise self.exc
        return self.resp


CFG = nb.NebiusConfig(api_key=KEY)
RESULTS = {"run_id": "r1", "status": "complete", "config": {"seed": 0, "correlation": 0.9},
           "summary": {"A_baseline": {"id_acc_mean": 0.95, "shifted_acc_mean": 0.4, "gap_mean": 0.55,
                                      "id_confusion": [[1]], "n_seeds": 1}}}


def test_missing_key_is_graceful(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.setattr(nb, "load_dotenv", lambda *a, **k: None)
    cfg = nb.get_config()
    assert not cfg.configured and cfg.redacted()["api_key_set"] is False
    with pytest.raises(nb.NebiusError, match="NEBIUS_API_KEY"):
        nb.chat([{"role": "user", "content": "hi"}], cfg=cfg)


def test_request_construction_and_secret_hygiene():
    s = FakeSession(FakeResp(200, {"choices": [{"message": {"content": "<think>x</think>Hello"}}]}))
    out = nb.ask(RESULTS, "Summarize", history=[{"role": "user", "content": "earlier"}], cfg=CFG, session=s)
    assert out == "Hello"
    method, url, kw = s.calls[0]
    assert url == "https://api.tokenfactory.nebius.com/v1/chat/completions"
    assert kw["headers"]["Authorization"] == f"Bearer {KEY}"
    body = kw["json"]
    assert body["model"] == "nvidia/Nemotron-3_5-Lightning"
    roles = [m["role"] for m in body["messages"]]
    assert roles[0] == "system" and roles[-1] == "user" and body["messages"][-1]["content"] == "Summarize"
    blob = json.dumps(body)
    assert KEY not in blob  # key only in header
    assert "0.95" in blob and "id_confusion" not in blob  # compact results, no raw matrices
    assert "Never invent" in body["messages"][0]["content"]


@pytest.mark.parametrize("status,needle", [(401, "credentials"), (403, "credentials"), (404, "NEBIUS_MODEL"),
                                           (429, "rate limit"), (500, "HTTP 500")])
def test_http_errors_are_mapped_without_leaking_key(status, needle):
    s = FakeSession(FakeResp(status, {"error": f"bad {KEY}"}, text=f"bad {KEY}"))
    with pytest.raises(nb.NebiusError) as e:
        nb.chat([{"role": "user", "content": "x"}], cfg=CFG, session=s)
    assert needle in str(e.value) and KEY not in str(e.value)


def test_network_timeout_and_malformed_response():
    with pytest.raises(nb.NebiusError, match="timed out"):
        nb.chat([{"role": "user", "content": "x"}], cfg=CFG, session=FakeSession(exc=requests.Timeout()))
    with pytest.raises(nb.NebiusError, match="Network error") as e:
        nb.chat([{"role": "user", "content": "x"}], cfg=CFG, session=FakeSession(exc=requests.ConnectionError(KEY)))
    assert KEY not in str(e.value)
    with pytest.raises(nb.NebiusError, match="Unexpected response"):
        nb.chat([{"role": "user", "content": "x"}], cfg=CFG, session=FakeSession(FakeResp(200, {"nope": 1})))


def test_list_models_parses_ids():
    s = FakeSession(FakeResp(200, {"data": [{"id": "nvidia/Nemotron-3_5-Lightning"}, {"id": "other"}]}))
    assert nb.list_models(CFG, session=s) == ["nvidia/Nemotron-3_5-Lightning", "other"]
    assert s.calls[0][1].endswith("/v1/models")


def test_dotenv_loader(tmp_path, monkeypatch):
    monkeypatch.delenv("NEBIUS_MODEL", raising=False)
    p = tmp_path / ".env"
    p.write_text("# c\nNEBIUS_MODEL=nvidia/nemotron-3-super-120b-a12b\n")
    nb.load_dotenv(p)
    assert nb.get_config().model == "nvidia/nemotron-3-super-120b-a12b"
    monkeypatch.delenv("NEBIUS_MODEL", raising=False)


def test_strip_reasoning_variants():
    assert nb._strip_reasoning("<think>a</think>Answer") == "Answer"
    assert nb._strip_reasoning("hidden reasoning</think>Answer") == "Answer"  # opening tag missing
    assert nb._strip_reasoning("Plain answer") == "Plain answer"


def test_pick_model_prefers_configured_then_nemotron():
    ids = ["meta/llama", "nvidia/nemotron-3-super-120b-a12b", "nvidia/Nemotron-3_5-Lightning"]
    assert nb.pick_model(ids, "nvidia/Nemotron-3_5-Lightning") == "nvidia/Nemotron-3_5-Lightning"
    assert nb.pick_model(ids[:2], "retired/model") == "nvidia/nemotron-3-super-120b-a12b"
    assert nb.pick_model(["meta/llama"], "retired/model") is None


def test_empty_answer_due_to_token_limit_gives_clear_error():
    body = {"choices": [{"message": {"content": "<think>long</think>"}, "finish_reason": "length"}]}
    with pytest.raises(nb.NebiusError, match="ran out of tokens"):
        nb.chat([{"role": "user", "content": "x"}], cfg=CFG, session=FakeSession(FakeResp(200, body)))
