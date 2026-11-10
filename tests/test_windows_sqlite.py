from __future__ import annotations

import hashlib
import io
import tarfile
import time
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

import windows_sqlite


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.HTTPError(
            "https://example.org/python", 500, "server error", Message(), None
        ),
        urllib.error.URLError(ConnectionResetError("connection reset")),
        TimeoutError("read timed out"),
    ],
    ids=["http-500", "connection-reset", "read-timeout"],
)
@pytest.mark.parametrize("outcome", ["recover", "exhaust", "bad-size", "bad-hash"])
def test_acquire_retries_transient_download_failures_without_weakening_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception, outcome: str
) -> None:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        member = tarfile.TarInfo("python/python.exe")
        member.size = len(b"interpreter")
        archive.addfile(member, io.BytesIO(b"interpreter"))
    data = payload.getvalue()
    config = tmp_path / "variants.toml"
    config.write_text(
        '[base]\nrelease = "test"\n[interpreters.windows]\n'
        'filename = "python.tar.gz"\n'
        f'size = {len(data)}\nsha256 = "{hashlib.sha256(data).hexdigest()}"\n'
        '[interpreters.windows.sqlite]\nversion = "test"\n'
    )
    monkeypatch.setattr(windows_sqlite.sys, "platform", "win32")
    monkeypatch.setattr(windows_sqlite, "build", lambda output: {"version": "test"})
    monkeypatch.setattr(windows_sqlite, "install", lambda directory, python: None)
    attempts = []
    waits = []

    class Response(io.BytesIO):
        def read(self, *args: object, **kwargs: object) -> bytes:
            if len(attempts) == 1 or outcome == "exhaust":
                raise failure
            return super().read()

    def open_url(url: str, *, timeout: int) -> io.BytesIO:
        attempts.append(url)
        assert timeout == 60
        if len(attempts) == 1 and not isinstance(failure, TimeoutError):
            raise failure
        received = data
        if outcome == "bad-size":
            received = data[:-1]
        elif outcome == "bad-hash":
            received = bytes([data[0] ^ 1]) + data[1:]
        return Response(received)

    monkeypatch.setattr(windows_sqlite.urllib.request, "urlopen", open_url)
    monkeypatch.setattr(time, "sleep", waits.append)
    output = tmp_path / "runtime"
    if outcome == "recover":
        python = windows_sqlite.acquire(config, output)
        assert python == output / "python/python.exe"
        assert python.read_bytes() == b"interpreter"
    elif outcome == "exhaust":
        with pytest.raises(type(failure)) as caught:
            windows_sqlite.acquire(config, output)
        assert caught.value is failure
    else:
        with pytest.raises(ValueError, match="Windows interpreter size/hash mismatch"):
            windows_sqlite.acquire(config, output)
    assert len(attempts) == (3 if outcome == "exhaust" else 2)
    assert waits == ([1, 2] if outcome == "exhaust" else [1])
    if outcome != "recover":
        assert not output.exists()


def test_acquire_does_not_retry_http_client_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "variants.toml"
    config.write_text(
        '[base]\nrelease = "test"\n[interpreters.windows]\n'
        'filename = "python.tar.gz"\n[interpreters.windows.sqlite]\n'
    )
    monkeypatch.setattr(windows_sqlite.sys, "platform", "win32")
    failure = urllib.error.HTTPError(
        "https://example.org/python", 404, "not found", Message(), None
    )
    attempts = []

    def open_url(url: str, *, timeout: int) -> io.BytesIO:
        attempts.append(url)
        raise failure

    monkeypatch.setattr(windows_sqlite.urllib.request, "urlopen", open_url)
    with pytest.raises(urllib.error.HTTPError) as caught:
        windows_sqlite.acquire(config, tmp_path / "runtime")
    assert caught.value is failure
    assert len(attempts) == 1
