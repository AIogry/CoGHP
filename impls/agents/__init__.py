"""Resolve algorithms without importing an unrelated baseline implementation."""

from collections.abc import Mapping
from importlib import import_module


class _AgentRegistry(Mapping):
    _entries = {
        "coghp": ("agents.coghp", "CoGHPAgent"),
        "navigation": ("agents.navigation", "NavigationAgent"),
        "hrm_action_coghp": ("agents.navigation", "NavigationAgent"),
        "navigation_coghp": ("agents.navigation", "NavigationAgent"),
    }

    def __getitem__(self, name):
        module, cls = self._entries[name]
        return getattr(import_module(module), cls)

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)


agents = _AgentRegistry()
