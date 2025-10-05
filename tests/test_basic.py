import os
from pathlib import Path

def test_structure():
    assert Path('data').exists()
    assert Path('src').exists()


def test_metrics_json_after_train():
    # This will only pass after running training; skip if not present.
    if Path('reports/metrics.json').exists():
        import json
        data = json.loads(Path('reports/metrics.json').read_text())
        assert isinstance(data, list)
        assert len(data) >= 1
