from pathlib import Path

from streamlit.testing.v1 import AppTest

from quant_stack.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_cli_backtest_runs(capsys):
    code = main(
        [
            "backtest",
            "--config",
            str(ROOT / "configs/paper_trading.yaml"),
            "--provider",
            "synthetic",
            "--end",
            "2024-06-28",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "synthetic" in out.lower() and "Sharpe" in out and "Benchmark" in out


def test_dashboard_runs_and_uses_real_results(monkeypatch):
    monkeypatch.setenv("QUANT_STACK_CONFIG_DIR", str(ROOT / "configs"))
    at = AppTest.from_file(
        str(ROOT / "src/quant_stack/dashboard/app.py"), default_timeout=120
    ).run()
    assert not at.exception
    assert any("SYNTHETIC" in i.value for i in at.info)
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert not at.error
    shown = " ".join(m.label for m in at.metric)
    assert "Fills" in shown and "Kill-switch threshold" in shown


def test_dashboard_command_targets_the_packaged_app():
    from quant_stack.cli import dashboard_command

    cmd = dashboard_command("0.0.0.0", 9000)
    assert cmd[1:4] == ["-m", "streamlit", "run"]
    assert Path(cmd[4]).name == "app.py" and Path(cmd[4]).exists()
    assert "9000" in cmd and "0.0.0.0" in cmd


def test_dashboard_renders_three_charts_after_a_run(monkeypatch):
    monkeypatch.setenv("QUANT_STACK_CONFIG_DIR", str(ROOT / "configs"))
    at = AppTest.from_file(
        str(ROOT / "src/quant_stack/dashboard/app.py"), default_timeout=120
    ).run()
    at.button[0].click().run()
    assert not at.exception
    charts = [e for e in at.main if e.type == "vega_lite_chart"]
    assert len(charts) == 3  # equity, drawdown, weights
    assert any("first trade" in c.value for c in at.caption)
