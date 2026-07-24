"""Minimal litellm stub for Docker builds.
Provides only the interfaces used by aegis_router,
plus a CLI entry point for supervisord to run.
"""
from setuptools import setup, find_packages

setup(
    name="litellm",
    version="1.40.0",
    packages=find_packages(),
    install_requires=[],
    entry_points={
        "console_scripts": [
            "litellm=litellm.cli:main",
        ],
    },
)
