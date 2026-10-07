# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable, Iterator
from typing import Literal
import gc
import logging

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.gc_monitor import GCMonitor, GCMonitorConfig

_GC_LOGGER = "lmcache.v1.mp_observability.gc_monitor"

_Probe = tuple[
    Callable[[Literal["start", "stop"], dict[str, int]], object], list[int | None]
]


@pytest.fixture()
def gc_probe(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> Iterator[_Probe]:
    seen: list[int | None] = []
    real_get_objects = gc.get_objects

    def _spy(generation: int | None = None) -> list[object]:
        seen.append(generation)
        return real_get_objects(generation)

    monkeypatch.setattr(gc, "get_objects", _spy)
    logger = logging.getLogger(_GC_LOGGER)
    monkeypatch.setattr(logger, "propagate", True)
    caplog.set_level(logging.INFO, logger=_GC_LOGGER)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    monitor = GCMonitor(GCMonitorConfig(enabled=True, min_pause_ms=0.0, top_objects=3))
    monitor.install()
    try:
        hooks = [cb for cb in gc.callbacks if getattr(cb, "__self__", None) is monitor]
        assert len(hooks) == 1, hooks
        yield hooks[0], seen
    finally:
        monitor.uninstall()
        if gc_was_enabled:
            gc.enable()


@pytest.mark.parametrize("generation", (0, 1))
def test_young_collections_never_enumerate_objects(
    gc_probe: _Probe,
    caplog: pytest.LogCaptureFixture,
    generation: int,
) -> None:
    hook, seen = gc_probe
    hook("start", {"generation": generation})
    hook(
        "stop",
        {"generation": generation, "collected": 12, "uncollectable": 0},
    )

    assert seen == []
    (message,) = [m for m in caplog.messages if m.startswith("GC gen")]
    assert message.startswith(f"GC gen{generation} ")
    assert message.endswith("collected=12 uncollectable=0")


def test_full_collection_logs_type_breakdown(
    gc_probe: _Probe,
    caplog: pytest.LogCaptureFixture,
) -> None:
    hook, seen = gc_probe
    hook("start", {"generation": 2})
    hook("stop", {"generation": 2, "collected": 100, "uncollectable": 1})

    assert seen == [2]
    (message,) = [m for m in caplog.messages if m.startswith("GC gen")]
    marker = "collected=100 uncollectable=1 "
    assert marker in message
    tokens = message.split(marker, 1)[1].split()
    assert len(tokens) == 3
    assert all(token.rpartition("=")[2].isdigit() for token in tokens)
