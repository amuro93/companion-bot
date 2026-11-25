"""Selectable chat backends for the companion bot (the /model switch).

The router is a plain callable so CompanionChat keeps one injection point.
The selection is persisted to a small JSON file beside the bot's database;
persistence failures never block a chat turn.
"""
from __future__ import annotations

import json
from pathlib import Path


class ModelRouter:
    """Call through to the currently selected backend."""

    def __init__(self, backends: dict, order=None, *, state_path=None, default=None):
        if not isinstance(backends, dict) or not backends:
            raise ValueError("at least one backend is required")
        self._backends = dict(backends)
        labels = list(order) if order is not None else list(backends)
        if len(labels) != len(self._backends) or set(labels) != set(self._backends):
            raise ValueError("order must list every backend exactly once")
        self._order = labels
        self._state_path = Path(state_path) if state_path is not None else None
        self._current = default if default in self._backends else labels[0]
        self._load()

    def _load(self):
        if self._state_path is None:
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        label = data.get("current") if isinstance(data, dict) else None
        if label in self._backends:
            self._current = label

    def _save(self):
        if self._state_path is None:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps({"current": self._current}, ensure_ascii=False),
                                        encoding="utf-8")
        except Exception:
            pass

    @property
    def current(self) -> str:
        return self._current

    def labels(self) -> list[str]:
        return list(self._order)

    def switch(self, label: str) -> str:
        if label not in self._backends:
            raise ValueError("unknown model choice")
        self._current = label
        self._save()
        return label

    def __call__(self, messages, **kwargs):
        return self._backends[self._current](messages, **kwargs)
