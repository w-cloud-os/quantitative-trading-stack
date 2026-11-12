"""Streamlit research dashboard. Everything shown comes from a real simulation run; nothing here
is hard-coded, and no live orders can be placed from this app."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from quant_stack.backtest.runner import run_reference_backtest
from quant_stack.config import load_config
from quant_stack.data.providers import get_provider

CONFIG_DIR = Path(os.environ.get("QUANT_STACK_CONFIG_DIR", "configs"))

st.set_page_config(page_title="Quant Stack: paper-trading research", page_icon="📈", layout="wide")
st.title("Quant Stack: paper-trading research dashboard")
st.warning("Educational backtesting only. Not financial advice. This app cannot place live orders.")


presets = sorted([*CONFIG_DIR.glob("*.yaml"), *CONFIG_DIR.glob("*.yml")])
if not presets:
    st.error(f"No presets found in {CONFIG_DIR.resolve()}")
    st.stop()

st.sidebar.header("Setup")
preset = st.sidebar.selectbox("Strategy preset", [p.name for p in presets])
provider_name = st.sidebar.selectbox("Data provider", ["synthetic", "yfinance"])
seed = st.sidebar.number_input(
    "Synthetic seed", value=42, step=1, disabled=provider_name != "synthetic"
)
end_date = st.sidebar.date_input("End date", value=pd.Timestamp.today())

try:
    cfg = load_config(CONFIG_DIR / preset)
except Exception as exc:  # show config errors in the UI instead of a stack trace
    st.error(f"Invalid preset {preset}: {exc}")
    st.stop()

with st.sidebar.expander("Preset parameters"):
    st.json(cfg.model_dump())

if provider_name == "synthetic":
    st.info("Using SYNTHETIC random prices: results say nothing about real markets.")
else:
    st.info("Yahoo Finance data (research use only; needs `pip install .[yfinance]` and internet).")

if st.button("Run backtest", type="primary"):
    provider = get_provider(
        provider_name, **({"seed": int(seed)} if provider_name == "synthetic" else {})
    )
    try:
        with st.spinner("Fetching data, optimizing walk-forward, simulating..."):
            st.session_state["report"] = run_reference_backtest(cfg, provider, end_date)
    except Exception as exc:
        st.session_state.pop("report", None)
        st.error(f"Backtest failed: {exc}")

report = st.session_state.get("report")
if report is None:
    st.caption("Pick a preset and press 'Run backtest'.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Performance", "Risk and kill-switch", "Rebalance decisions"])

with tab1:
    start = report.evaluation_start
    equity = pd.DataFrame(
        {
            "Strategy": report.strategy.equity.loc[start:],
            "Equal-weight benchmark": report.benchmark.equity.loc[start:],
        }
    )
    st.line_chart(equity / equity.iloc[0])
    st.caption(f"Measured from the first trade on {start.date()}; the idle warm-up is excluded.")
    metrics = pd.DataFrame(
        {"Strategy": report.strategy_metrics, "Benchmark": report.benchmark_metrics}
    )
    st.dataframe(metrics.style.format("{:.3f}"), width="stretch")
    s = report.strategy
    c1, c2, c3 = st.columns(3)
    c1.metric("Fills", len(s.fills))
    c2.metric("Fees paid", f"${s.fees:,.0f}")
    c3.metric("Traded notional", f"${s.traded_notional:,.0f}")

with tab2:
    limit = cfg.execution.risk_limits.max_drawdown_pct
    eq = report.strategy.equity.loc[report.evaluation_start :]
    dd = eq / eq.cummax() - 1.0
    st.line_chart(dd.rename("Drawdown"))
    st.metric("Kill-switch threshold", f"{limit:.1%}")
    if report.strategy.halted_on is not None:
        st.error(
            f"In this simulation the kill-switch TRIPPED on {report.strategy.halted_on.date()}: "
            "all positions were flattened and trading halted for the rest of the run."
        )
    else:
        st.success("In this simulation the kill-switch did not trip.")
    st.caption("These are results of the backtest above, not a live monitor.")

with tab3:
    rows = pd.DataFrame(
        {
            "confidence": [d.confidence for d in report.schedule],
            "optimizer": [d.method for d in report.schedule],
        },
        index=[d.date for d in report.schedule],
    )
    weights = pd.DataFrame([d.weights for d in report.schedule], index=rows.index)
    st.area_chart(weights, stack=True)
    st.dataframe(pd.concat([rows, weights.round(3)], axis=1), width="stretch")
