"""Legacy setup.py for editable install compatibility.

This exists only to support `pip install -e .` on environments where
the wheel package is not available. The canonical metadata lives in pyproject.toml.
"""
from setuptools import setup, find_packages

setup(
    name="aegis-router",
    version="0.1.0",
    packages=find_packages(include=["aegis_router*"]),
    python_requires=">=3.10",
)
