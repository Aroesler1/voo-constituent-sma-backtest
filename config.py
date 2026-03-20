"""Configuration objects and loaders for constituent-level VOO backtests."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


@dataclass
class BacktestConfig:
    """Runtime configuration for the full backtest pipeline."""

    # Data
    START_DATE: str = "1993-01-29"
    END_DATE: str = "today"
    VOO_INCEPTION: str = "2010-09-09"
    TICKERS: list[str] = field(default_factory=lambda: ["SPY", "VOO"])
    CACHE_DIR: str = "./data_cache"
    PRIMARY_PRICE_SOURCE: str = "crsp"

    # Strategy core
    STRATEGY_MODE: str = "constituent_sma"
    SIGNAL_TYPE: str = "level"
    SMA_LENGTH: int = 200  # retained for compatibility
    SMA_LENGTH_DAYS: int = 200
    ENTRY_BAND_BPS: float = 0.0
    EXIT_BAND_BPS: float = 0.0
    EXECUTION_TIMING: str = "next_open"
    ALLOW_SHORT: bool = False

    # Rebalance policy
    REBALANCE_DEFAULT: str = "semi_monthly"
    REBALANCE_SWEEP_VALUES: list[str] = field(
        default_factory=lambda: ["daily", "weekly", "semi_monthly", "monthly"]
    )

    # Universe policy
    UNIVERSE_POST_2019_SOURCE: str = "sec_proxy"
    UNIVERSE_PRE_2019_SOURCE: str = "sp500_public_history"
    PRE2019_PROXY_CUTOFF: str = "2019-10-01"
    CAPITAL_DEPLOYMENT: str = "equal_weight_active_full_invest"
    MIN_ACTIVE_NAMES: int = 1
    HOLDINGS_LAG_BUSINESS_DAYS: int = 45

    # Free-source file/url settings
    SEC_VOO_PROXY_CSV: str = "./data/universe/sec_voo_holdings_proxy.csv"
    SP500_MEMBERSHIP_CSV: str = "./data/universe/sp500_membership_history.csv"
    SEC_VOO_PROXY_URL: str | None = None
    SP500_MEMBERSHIP_URL: str | None = None

    # Quality controls
    PROXY_FIDELITY_REPORT: bool = True
    PROXY_TRACKING_ERROR_HIGH: float = 0.03
    PROXY_TRACKING_ERROR_MEDIUM: float = 0.07

    # Allocation
    FULLY_ALLOCATED: bool = True
    ALLOCATION_STYLE: str = "equal_weight_active"

    # Costs
    COMMISSION_PER_TRADE: float = 0.0
    COMMISSION_PER_SHARE: float = 0.0
    MIN_COMMISSION_PER_ORDER: float = 0.0
    SLIPPAGE_BPS: float = 1.0
    ENABLE_ENHANCED_COST_MODEL: bool = True
    SPREAD_MODEL: str = "corwin_schultz"
    SPREAD_FLOOR_BPS: float = 1.0
    IMPACT_MODEL: str = "sqrt_participation"
    IMPACT_COEF: float = 0.02
    ADV_LOOKBACK_DAYS: int = 20
    VOL_LOOKBACK_DAYS: int = 20
    EXPLICIT_FEE_BPS: float = 0.0
    ASSUMED_ADV_USD_PER_NAME: float = 50_000_000.0
    RETAIL_ACCOUNT_MODE: bool = True
    REBALANCE_BUFFER_BPS: float = 25.0
    MIN_TRADE_NOTIONAL_USD: float = 250.0
    SPREAD_CAP_BPS: float = 25.0
    MIN_PARTICIPATION_FOR_IMPACT: float = 0.005
    ENFORCE_INVESTABILITY_FILTER: bool = True
    MIN_PRICE_TO_TRADE: float = 5.0
    MIN_ADV_USD_TO_TRADE: float = 5_000_000.0
    USE_FRACTIONAL_SHARES: bool = True
    RETAIL_EXECUTION_ACCOUNT_SIZE_USD: float = 100_000.0
    OPEN_AUCTION_SLIPPAGE_BPS: float = 3.0
    INCLUDE_REGULATORY_FEES: bool = True
    FINRA_TAF_PER_SHARE: float = 0.000195
    FINRA_TAF_MAX_PER_TRADE: float = 9.79

    # Cash
    CASH_RATE_ANNUAL: float = 0.02
    USE_FRED_CASH_RATE: bool = False
    USE_DYNAMIC_CASH_RATE: bool = True
    CASH_RATE_SOURCE: str = "DGS3MO"
    CASH_RATE_FALLBACK_SOURCE: str = "DTB3"
    CASH_RATE_DAY_COUNT: str = "ACT/360"
    CASH_RATE_AS_OF_DATE: str = "today"

    # Reporting
    INITIAL_CAPITAL: float = 1_000_000.0
    OUTPUT_DIR: str = "./output"
    SMA_SWEEP_VALUES: list[int] = field(default_factory=lambda: [150, 175, 200, 225, 250])

    # Validation tolerances
    VENDOR_DISCREPANCY_THRESHOLD_BPS: float = 5.0
    VENDOR_METRIC_TOLERANCE_CAGR_BPS: float = 10.0
    VENDOR_METRIC_TOLERANCE_SHARPE: float = 0.05

    # API keys
    CRSP_API_KEY: str | None = None
    CRSP_USERNAME: str | None = None
    WRDS_USERNAME: str | None = None
    WRDS_PASSWORD: str | None = None
    EODHD_API_KEY: str | None = None
    FRED_API_KEY: str | None = None
    CRSP_BATCH_SIZE: int = 200

    # Snapshot/reproducibility
    FREEZE_SNAPSHOTS: bool = True
    SNAPSHOT_AS_OF_UTC: str = "now"
    SNAPSHOT_LOCK_ID: str | None = None
    ALLOW_LIVE_FETCH_WHEN_NO_SNAPSHOT: bool = True

    # Benchmark source
    SP500_TR_SOURCE: str = "SPY_PROXY"

    def validate(self) -> None:
        """Validate internal consistency of configuration."""
        if self.PRIMARY_PRICE_SOURCE not in {"crsp", "eodhd"}:
            raise ValueError("PRIMARY_PRICE_SOURCE must be 'crsp' or 'eodhd'.")
        if self.STRATEGY_MODE not in {"constituent_sma"}:
            raise ValueError("STRATEGY_MODE must be 'constituent_sma'.")
        if self.SIGNAL_TYPE not in {"level", "cross"}:
            raise ValueError("SIGNAL_TYPE must be 'level' or 'cross'.")
        if self.EXECUTION_TIMING not in {"next_open", "same_close"}:
            raise ValueError("EXECUTION_TIMING must be 'next_open' or 'same_close'.")
        valid_sched = {"daily", "weekly", "semi_monthly", "monthly"}
        if self.REBALANCE_DEFAULT not in valid_sched:
            raise ValueError(f"REBALANCE_DEFAULT must be one of {sorted(valid_sched)}.")
        if any(freq not in valid_sched for freq in self.REBALANCE_SWEEP_VALUES):
            raise ValueError("REBALANCE_SWEEP_VALUES contains unsupported frequency.")
        if self.UNIVERSE_POST_2019_SOURCE not in {"sec_proxy"}:
            raise ValueError("UNIVERSE_POST_2019_SOURCE must be 'sec_proxy'.")
        if self.UNIVERSE_PRE_2019_SOURCE not in {"sp500_public_history"}:
            raise ValueError("UNIVERSE_PRE_2019_SOURCE must be 'sp500_public_history'.")
        if self.CAPITAL_DEPLOYMENT not in {"equal_weight_active_full_invest"}:
            raise ValueError("CAPITAL_DEPLOYMENT must be 'equal_weight_active_full_invest'.")
        if self.MIN_ACTIVE_NAMES < 1:
            raise ValueError("MIN_ACTIVE_NAMES must be >= 1.")
        if self.HOLDINGS_LAG_BUSINESS_DAYS < 0:
            raise ValueError("HOLDINGS_LAG_BUSINESS_DAYS must be >= 0.")
        if self.ALLOCATION_STYLE not in {"equal_weight_active"}:
            raise ValueError("ALLOCATION_STYLE must be 'equal_weight_active'.")
        if self.CASH_RATE_DAY_COUNT not in {"ACT/360", "ACT/365"}:
            raise ValueError("CASH_RATE_DAY_COUNT must be 'ACT/360' or 'ACT/365'.")
        if self.CRSP_BATCH_SIZE < 1:
            raise ValueError("CRSP_BATCH_SIZE must be >= 1.")
        if self.REBALANCE_BUFFER_BPS < 0:
            raise ValueError("REBALANCE_BUFFER_BPS must be >= 0.")
        if self.MIN_TRADE_NOTIONAL_USD < 0:
            raise ValueError("MIN_TRADE_NOTIONAL_USD must be >= 0.")
        if self.SPREAD_CAP_BPS <= 0:
            raise ValueError("SPREAD_CAP_BPS must be > 0.")
        if self.MIN_PARTICIPATION_FOR_IMPACT < 0:
            raise ValueError("MIN_PARTICIPATION_FOR_IMPACT must be >= 0.")
        if self.MIN_PRICE_TO_TRADE < 0:
            raise ValueError("MIN_PRICE_TO_TRADE must be >= 0.")
        if self.MIN_ADV_USD_TO_TRADE < 0:
            raise ValueError("MIN_ADV_USD_TO_TRADE must be >= 0.")
        if self.RETAIL_EXECUTION_ACCOUNT_SIZE_USD <= 0:
            raise ValueError("RETAIL_EXECUTION_ACCOUNT_SIZE_USD must be > 0.")
        if self.OPEN_AUCTION_SLIPPAGE_BPS < 0:
            raise ValueError("OPEN_AUCTION_SLIPPAGE_BPS must be >= 0.")
        if self.FINRA_TAF_PER_SHARE < 0:
            raise ValueError("FINRA_TAF_PER_SHARE must be >= 0.")
        if self.FINRA_TAF_MAX_PER_TRADE < 0:
            raise ValueError("FINRA_TAF_MAX_PER_TRADE must be >= 0.")
        if self.USE_DYNAMIC_CASH_RATE and not self.FRED_API_KEY and self.CASH_RATE_ANNUAL is None:
            raise ValueError(
                "USE_DYNAMIC_CASH_RATE=True requires FRED_API_KEY or a non-null CASH_RATE_ANNUAL fallback."
            )
        if self.SNAPSHOT_LOCK_ID and not self.FREEZE_SNAPSHOTS:
            raise ValueError("SNAPSHOT_LOCK_ID requires FREEZE_SNAPSHOTS=True.")
        if not self.has_crsp_credentials() and not self.EODHD_API_KEY:
            raise ValueError("Provide CRSP/WRDS credentials or an EODHD_API_KEY.")

    def has_crsp_credentials(self) -> bool:
        """Return whether a WRDS/CRSP connection is plausibly available."""
        username = self.WRDS_USERNAME or self.CRSP_USERNAME or os.getenv("PGUSER")
        password = self.WRDS_PASSWORD or self.CRSP_API_KEY or os.getenv("PGPASSWORD")
        return bool(username and (password or Path.home().joinpath(".pgpass").exists()))


def resolve_date(value: str) -> str:
    """Resolve date placeholders to ISO date string."""
    if value.lower() == "today":
        return date.today().isoformat()
    return value


def resolve_utc_timestamp(value: str) -> str:
    """Resolve timestamp placeholders to UTC ISO timestamp."""
    if value.lower() == "now":
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return value


def get_periods() -> dict[str, tuple[str, str]]:
    """Return named regime periods with dynamic end-date resolution."""
    return {
        "Dot-Com Bust": ("2000-03-24", "2002-10-09"),
        "Pre-GFC Bull": ("2002-10-10", "2007-10-09"),
        "GFC": ("2007-10-10", "2009-03-09"),
        "Post-GFC Recovery": ("2009-03-10", "2019-12-31"),
        "COVID Crash": ("2020-01-01", "2020-03-23"),
        "COVID Recovery": ("2020-03-24", "2021-12-31"),
        "2022 Bear": ("2022-01-01", "2022-12-31"),
        "Post-2022": ("2023-01-01", resolve_date("today")),
    }


def load_config() -> BacktestConfig:
    """Load runtime configuration from defaults and environment."""
    cfg = BacktestConfig(
        CRSP_API_KEY=os.getenv("CRSP_API_KEY"),
        CRSP_USERNAME=os.getenv("CRSP_USERNAME"),
        WRDS_USERNAME=os.getenv("WRDS_USERNAME"),
        WRDS_PASSWORD=os.getenv("WRDS_PASSWORD"),
        EODHD_API_KEY=os.getenv("EODHD_API_KEY"),
        FRED_API_KEY=os.getenv("FRED_API_KEY"),
    )
    cfg.END_DATE = resolve_date(cfg.END_DATE)
    cfg.CASH_RATE_AS_OF_DATE = resolve_date(cfg.CASH_RATE_AS_OF_DATE)
    cfg.PRE2019_PROXY_CUTOFF = resolve_date(cfg.PRE2019_PROXY_CUTOFF)
    cfg.SNAPSHOT_AS_OF_UTC = resolve_utc_timestamp(cfg.SNAPSHOT_AS_OF_UTC)
    cfg.validate()
    return cfg


def config_hash(config: BacktestConfig) -> str:
    """Compute a stable hash for config values (excluding secrets)."""
    payload: dict[str, Any] = dataclasses.asdict(config)
    payload.pop("CRSP_API_KEY", None)
    payload.pop("CRSP_USERNAME", None)
    payload.pop("WRDS_USERNAME", None)
    payload.pop("WRDS_PASSWORD", None)
    payload.pop("EODHD_API_KEY", None)
    payload.pop("FRED_API_KEY", None)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
