"""Shared pytest configuration for the pallatom test suite."""

from hypothesis import settings

settings.register_profile(
    "deterministic",
    deadline=None,
    derandomize=True,
    database=None,
    max_examples=50,
)
settings.register_profile(
    "explore",
    deadline=None,
    database=None,
    max_examples=1000,
)
settings.load_profile("deterministic")
