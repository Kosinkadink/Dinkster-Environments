"""Disposable channel authority for publication and retention fixtures."""

from __future__ import annotations

import hashlib
from typing import cast

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from dinkster_supervisor import environment_channel_signatures

KEY = Ed25519PrivateKey.generate()
PUBLIC = KEY.public_key().public_bytes_raw()
KEYS = {"sha256:" + hashlib.sha256(PUBLIC).hexdigest(): PUBLIC}


def channel_json(document: dict, channel: str = "nightly") -> bytes:
    return environment_channel_signatures.sign_channel(document, channel, KEY)


def read_channel(data: bytes, channel: str = "nightly") -> dict:
    return cast(
        dict, environment_channel_signatures.verify_channel(data, channel, keys=KEYS)
    )


@pytest.fixture(autouse=True)
def signed_channels(monkeypatch):
    monkeypatch.setattr(environment_channel_signatures, "CHANNEL_KEYS", KEYS)
