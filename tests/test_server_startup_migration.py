"""Startup ordering tests for the Java PVC data migration in ``run_server``."""

import asyncio
import socket
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pytest
from hypercorn.config import Config

from trustyai_service import main
from trustyai_service.main import run_server


def _free_port() -> int:
    """Return a port that was free on loopback a moment ago."""
    with closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _use_plaintext_tls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Select optional TLS with no serving material, i.e. plaintext mode."""
    monkeypatch.setenv("TLS_REQUIRED", "false")
    monkeypatch.setenv("TLS_POLICY", "system")
    monkeypatch.setenv("TLS_CERT_FILE", str(tmp_path / "missing.crt"))
    monkeypatch.setenv("TLS_KEY_FILE", str(tmp_path / "missing.key"))


def _use_free_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PORT", str(_free_port()))
    monkeypatch.setenv("HEALTH_PORT", str(_free_port()))
    monkeypatch.setenv("SSL_PORT", str(_free_port()))


@pytest.mark.asyncio
async def test_run_server_migrates_data_after_probes_and_before_the_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Probes are up while data migrates, and the API starts only afterwards."""
    _use_plaintext_tls(monkeypatch, tmp_path)
    _use_free_ports(monkeypatch)
    events: list[str] = []

    async def fake_serve(app: object, _config: Config) -> None:
        events.append("health serving" if app is main.health_app else "api serving")

    async def fake_migration() -> None:
        # A real migration yields to the event loop while it works.
        await asyncio.sleep(0)
        events.append("migration")

    with (
        patch("trustyai_service.main.serve", fake_serve),
        patch("trustyai_service.main.run_startup_migrations", fake_migration),
    ):
        await run_server()

    assert events == ["health serving", "migration", "api serving"]


@pytest.mark.asyncio
async def test_run_server_does_not_start_the_api_when_migration_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed migration ends startup instead of serving without data."""
    _use_plaintext_tls(monkeypatch, tmp_path)
    _use_free_ports(monkeypatch)
    served: list[object] = []

    async def fake_serve(app: object, _config: Config) -> None:
        served.append(app)

    async def failing_migration() -> None:
        msg = "migration failed"
        raise RuntimeError(msg)

    with (
        patch("trustyai_service.main.serve", fake_serve),
        patch("trustyai_service.main.run_startup_migrations", failing_migration),
        pytest.raises(RuntimeError, match="migration failed"),
    ):
        await run_server()

    assert main.app not in served


@pytest.mark.asyncio
async def test_run_server_still_starts_when_the_migration_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Without the optional dependencies the service starts and says so."""
    _use_plaintext_tls(monkeypatch, tmp_path)
    _use_free_ports(monkeypatch)

    with (
        patch("trustyai_service.main.serve") as serve_mock,
        patch("trustyai_service.main.JAVA_PVC_MIGRATION_AVAILABLE", new=False),
        patch.object(main.logger, "warning") as warning_mock,
    ):
        await run_server()

    assert serve_mock.call_count == 2
    assert "unavailable" in warning_mock.call_args.args[0]
