"""Every login must be explicit, single-attempt and safe to fail offline."""
import builtins
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loader import _get_wrds_connection


@pytest.mark.parametrize("value", [None, "0", "true", "1 ", ""])
def test_disabled_before_vendor_import(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("WRDS_DUO_READY", raising=False)
    else:
        monkeypatch.setenv("WRDS_DUO_READY", value)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name not in {"wrds", "sqlalchemy"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(RuntimeError, match="WRDS_DUO_READY=1"):
        _get_wrds_connection(username=None, password=None)


def _fake_vendor(monkeypatch):
    monkeypatch.setenv("WRDS_DUO_READY", "1")
    client, engine = SimpleNamespace(), Mock()
    constructor = Mock(return_value=client)
    factory = Mock(return_value=engine)
    vendor = SimpleNamespace(Connection=constructor, sql=SimpleNamespace(
        WRDS_POSTGRES_HOST="example.invalid", WRDS_POSTGRES_PORT=1,
        WRDS_POSTGRES_DB="fixture", WRDS_CONNECT_ARGS={"connect_timeout": 2},
    ))
    monkeypatch.setitem(sys.modules, "wrds", vendor)
    monkeypatch.setitem(sys.modules, "sqlalchemy", SimpleNamespace(create_engine=factory))
    monkeypatch.setitem(sys.modules, "sqlalchemy.engine", SimpleNamespace(
        URL=SimpleNamespace(create=Mock(return_value="fixture-url"))))
    monkeypatch.setitem(sys.modules, "sqlalchemy.pool", SimpleNamespace(NullPool=object))
    monkeypatch.setattr(builtins, "input", Mock(side_effect=AssertionError("no prompt")))
    return client, constructor, engine, factory


def test_authorized_success_opens_once_without_autoconnect(monkeypatch):
    client, constructor, engine, factory = _fake_vendor(monkeypatch)
    result = _get_wrds_connection(username=None, password=None)
    assert result is client
    constructor.assert_called_once_with(autoconnect=False, verbose=False)
    engine.connect.assert_called_once_with()
    assert result.connection is engine.connect.return_value
    assert factory.call_args.kwargs["poolclass"] is object
    engine.dispose.assert_not_called()


def test_failure_does_not_retry_prompt_or_expose_driver_payload(monkeypatch):
    _, _, engine, _ = _fake_vendor(monkeypatch)
    engine.connect.side_effect = RuntimeError("sensitive driver payload")
    with pytest.raises(RuntimeError, match="one attempt") as caught:
        _get_wrds_connection(username=None, password=None)
    assert "sensitive" not in str(caught.value)
    assert caught.value.__suppress_context__
    engine.connect.assert_called_once_with()
    engine.dispose.assert_called_once_with()


def test_ciz_uses_the_same_guard():
    import crsp_v2
    assert crsp_v2._get_wrds_connection is _get_wrds_connection


@pytest.mark.parametrize("stage", ["constructor", "engine", "cleanup"])
def test_setup_and_cleanup_errors_are_sanitized(monkeypatch, stage):
    _, constructor, engine, factory = _fake_vendor(monkeypatch)
    if stage == "constructor":
        constructor.side_effect = RuntimeError("sensitive setup")
    elif stage == "engine":
        factory.side_effect = RuntimeError("sensitive URL")
    else:
        engine.connect.side_effect = RuntimeError("sensitive driver")
        engine.dispose.side_effect = RuntimeError("sensitive cleanup")
    with pytest.raises(RuntimeError, match="at most one attempt") as caught:
        _get_wrds_connection(username=None, password=None)
    assert "sensitive" not in str(caught.value)
    assert caught.value.__suppress_context__
    assert engine.connect.call_count <= 1


def test_credentials_are_passed_structurally(monkeypatch):
    _, constructor, _, _ = _fake_vendor(monkeypatch)
    from sqlalchemy.engine import URL
    _get_wrds_connection(username="fixture-user", password="fixture-password")
    constructor.assert_called_once_with(autoconnect=False, verbose=False,
                                       wrds_username="fixture-user", wrds_password="fixture-password")
    assert URL.create.call_args.kwargs["username"] == "fixture-user"
    assert URL.create.call_args.kwargs["password"] == "fixture-password"


def test_legacy_offline_miss_never_reaches_connection_helper(monkeypatch, tmp_path):
    import data_loader

    monkeypatch.setenv("BACKTEST_OFFLINE", "1")
    monkeypatch.setenv("BACKTEST_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        data_loader,
        "_get_wrds_connection",
        Mock(side_effect=AssertionError("connection helper reached")),
    )
    assert data_loader.fetch_crsp_batch_prices(
        ["UNRESOLVED"], "2020-01-01", "2020-01-31"
    ) == {}


def test_ciz_offline_miss_never_reaches_connection_helper(monkeypatch, tmp_path):
    import crsp_v2
    from config import BacktestConfig

    monkeypatch.setattr(
        crsp_v2,
        "_get_wrds_connection",
        Mock(side_effect=AssertionError("connection helper reached")),
    )
    cfg = BacktestConfig(CACHE_DIR=str(tmp_path), OFFLINE_MODE=True)
    assert crsp_v2.fetch_crsp_v2_batch_prices(
        ["UNRESOLVED"],
        "2020-01-01",
        "2020-01-31",
        config=cfg,
    ) == {}
