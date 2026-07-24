"""Minimal clawvault stub for Docker builds.
aegis_router has its own ClawVault implementation in aegis_router.clawvault.
This just satisfies the package requirement.
"""
from setuptools import setup, find_packages

setup(
    name="clawvault",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
)
