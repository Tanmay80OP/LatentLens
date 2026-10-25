import csv
import io
import json

from src.experiment import ExperimentConfig, run_experiment
from src.report import plain_language_summary, to_csv, to_json, to_markdown


def _results(tmp_path):
    return run_experiment(ExperimentConfig(n_train=90, n_val=45, n_test=60, epochs=1, batch_size=32),
                          results_root=tmp_path, save_models=False)


def test_exports(tmp_path):
    res = _results(tmp_path)
    rows = list(csv.DictReader(io.StringIO(to_csv(res))))
    assert len(rows) == 3 and "shifted_accuracy" in rows[0] and "robustness_gap" in rows[0]
    assert json.loads(to_json(res))["run_id"] == res["run_id"]
    md = to_markdown(res)
    assert "Measured results" in md and "Configuration" in md and res["run_id"] in md
    assert any("single seed" in s or "Only one seed" in s for s in plain_language_summary(res))


def test_markdown_for_failed_run_shows_error():
    md = to_markdown({"run_id": "x", "status": "failed", "error": "boom", "config": {}, "runs": []})
    assert "boom" in md
