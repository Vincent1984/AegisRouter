"""Minimal routellm stub for Docker builds.
Provides only the interfaces used by aegis_router.
"""
from setuptools import setup, find_packages

setup(
    name="routellm",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[],
)
