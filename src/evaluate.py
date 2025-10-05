import json
from pathlib import Path
import pandas as pd


def compare():
    metrics_file = Path('reports/metrics.json')
    if not metrics_file.exists():
        print('No metrics file found.')
        return
    metrics = json.loads(metrics_file.read_text())
    df = pd.DataFrame(metrics)
    df.to_markdown('reports/metrics_table.md', index=False)
    print(df)
    if set(['baseline','engineered']).issubset(df['experiment']):
        base_rmse = df[df['experiment']=='baseline']['rmse'].values[0]
        eng_rmse = df[df['experiment']=='engineered']['rmse'].values[0]
        improvement = (base_rmse - eng_rmse) / base_rmse * 100
        summary = f"RMSE improvement: {improvement:.2f}% (baseline={base_rmse:.2f}, engineered={eng_rmse:.2f})"
        Path('reports/summary.txt').write_text(summary)
        print(summary)

if __name__ == '__main__':
    compare()
