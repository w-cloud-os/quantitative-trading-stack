"""Portfolio desk: skfolio Mean-Risk optimization driven entirely by ``OptimizationConfig``."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from skfolio import RiskMeasure
from skfolio.exceptions import SkfolioError
from skfolio.moments import DenoiseCovariance, EmpiricalCovariance, EmpiricalMu, ShrunkMu
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.prior import EmpiricalPrior

from quant_stack.config import OptimizationConfig
from quant_stack.optimization.blend import benchmark_weights

log = logging.getLogger(__name__)


class PortfolioOptimizer:
    def __init__(self, cfg: OptimizationConfig, sectors: dict[str, str] | None = None) -> None:
        self.cfg = cfg
        self.sectors = sectors or {}
        self.last_method = "none"

    def build_model(
        self,
        assets: list[str],
        objective: ObjectiveFunction = ObjectiveFunction.MAXIMIZE_RATIO,
        previous_weights: pd.Series | None = None,
    ) -> MeanRisk:
        c = self.cfg
        prior = EmpiricalPrior(
            mu_estimator=ShrunkMu() if c.shrink_mu else EmpiricalMu(),
            covariance_estimator=DenoiseCovariance()
            if c.denoise_covariance
            else EmpiricalCovariance(),
        )
        kwargs: dict = {}
        present = {a: s for a, s in self.sectors.items() if a in assets}
        constraints = [f"{s} <= {v}" for s, v in c.sector_max.items() if s in present.values()]
        constraints += [f"{s} >= {v}" for s, v in c.sector_min.items() if s in present.values()]
        if constraints:
            # skfolio expects {asset: [group, ...]}; the reverse mapping is silently ignored.
            kwargs["groups"] = {a: [s] for a, s in present.items()}
            kwargs["linear_constraints"] = constraints
        if previous_weights is not None and c.max_weight_change is not None:
            kwargs["previous_weights"] = previous_weights.reindex(assets).fillna(0.0).to_dict()
            kwargs["max_turnover"] = c.max_weight_change  # per-asset cap in skfolio
        return MeanRisk(
            objective_function=objective,
            risk_measure=getattr(RiskMeasure, c.risk_measure),
            prior_estimator=prior,
            min_weights=c.min_asset_weight,
            max_weights=c.max_asset_weight,
            l2_coef=c.l2_coef,
            **kwargs,
        )

    def optimize(
        self, returns: pd.DataFrame, previous_weights: pd.Series | None = None
    ) -> pd.Series:
        """Fallback chain: max-ratio -> min-risk -> min-risk without the turnover cap -> equal
        weight. ``last_method`` records which step produced the result."""
        assets = list(returns.columns)
        attempts = [
            ("max_ratio", ObjectiveFunction.MAXIMIZE_RATIO, previous_weights),
            ("min_risk", ObjectiveFunction.MINIMIZE_RISK, previous_weights),
            ("min_risk_no_turnover_cap", ObjectiveFunction.MINIMIZE_RISK, None),
        ]
        for label, objective, prev in attempts:
            try:
                model = self.build_model(assets, objective, prev)
                model.fit(returns)
                w = pd.Series(np.asarray(model.weights_, dtype=float), index=assets).clip(lower=0.0)
                self.last_method = label
                return w / w.sum()
            except (ValueError, SkfolioError) as exc:
                log.info("optimizer step %s failed (%s)", label, str(exc)[:80])
        self.last_method = "benchmark"
        return benchmark_weights(assets)
