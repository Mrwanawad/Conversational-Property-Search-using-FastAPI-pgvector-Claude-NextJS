"""Langfuse tracing (no-op when keys are not configured). One trace per turn, one span per stage."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from propsearch.config import get_settings

log = logging.getLogger(__name__)


class _NoopObservation:
    def update(self, **_: Any) -> None:
        pass


@lru_cache(maxsize=1)
def _client():
    s = get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(
            public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key, host=s.langfuse_host
        )
    except Exception as e:
        log.warning("langfuse disabled: %s", e)
        return None


def enabled() -> bool:
    return _client() is not None


@contextlib.contextmanager
def observe(name: str, as_type: str = "span", **kwargs: Any) -> Iterator[Any]:
    client = _client()
    if client is None:
        yield _NoopObservation()
        return
    try:
        cm = client.start_as_current_observation(name=name, as_type=as_type, **kwargs)
    except Exception as e:
        log.debug("langfuse span failed: %s", e)
        yield _NoopObservation()
        return
    with cm as obs:
        yield obs


def flush() -> None:
    client = _client()
    if client is not None:
        client.flush()
