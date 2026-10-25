"""Headless UI tests using Streamlit's AppTest (no browser, no network)."""
import pytest
from streamlit.testing.v1 import AppTest

import src.experiment as ex
import src.nebius_client as nb

APP = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def empty_env(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "RESULTS_ROOT", tmp_path)
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.setattr(nb, "load_dotenv", lambda *a, **k: None)
    return tmp_path


def test_app_starts_with_no_results_and_no_api_key(empty_env):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown) + " ".join(w.value for w in at.warning) + " ".join(i.value for i in at.info)
    assert "NEBIUS_API_KEY is not set" in text
    assert "Run an experiment first" in text


def test_run_experiment_from_ui_then_views_render(empty_env):
    at = AppTest.from_file(APP, default_timeout=180).run()
    # shrink the run so the test is fast
    labels = [s.label for s in at.slider]
    at.slider[labels.index("Number of seeds")].set_value(1)
    at.slider[labels.index("Epochs")].set_value(2)
    at.select_slider[0].set_value(300)  # training images
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    runs = ex.list_runs(empty_env)
    assert len(runs) == 1
    res = ex.load_results(runs[0], empty_env)
    assert res["status"] == "complete" and res["config"]["epochs"] == 2 and res["config"]["n_train"] == 300
    # comparison + explorer tabs rendered with data (no exceptions, tables present)
    assert len(at.dataframe) >= 1
    assert not at.error
