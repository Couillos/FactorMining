from pathlib import Path
import tempfile, os
from factor_mining.reporting.pareto_export import export_pareto
from factor_mining.reporting.csv_export import export_diagnostics
from factor_mining.reporting.plots import plot_pareto_3d, plot_ic_decay


def test_export_pareto_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        export_pareto([], tmpdir)
        assert (Path(tmpdir) / "pareto_front.csv").exists()


def test_export_diagnostics():
    with tempfile.TemporaryDirectory() as tmpdir:
        export_diagnostics([{"name": "test", "value": 1}], tmpdir)
        path = Path(tmpdir) / "diagnostics.csv"
        assert path.exists()
        import pandas as pd
        df = pd.read_csv(path)
        assert len(df) == 1


def test_plot_pareto_3d():
    with tempfile.TemporaryDirectory() as tmpdir:
        output = os.path.join(tmpdir, "test.png")
        plot_pareto_3d([], output)
        assert os.path.exists(output)


def test_plot_ic_decay():
    with tempfile.TemporaryDirectory() as tmpdir:
        output = os.path.join(tmpdir, "ic_decay.png")
        plot_ic_decay({1: 0.1, 7: 0.05, 30: 0.01}, output)
        assert os.path.exists(output)
