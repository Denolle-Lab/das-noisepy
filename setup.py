#!/usr/bin/env python3
"""
Setup script for noisepy-das
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="noisepy-das",
    version="0.1.0",
    author="Denolle Lab",
    description="Tools for working with DAS data using NoisePy concepts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Denolle-Lab/das-noisepy",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "jupyter", "matplotlib"],
    },
    project_urls={
        "Bug Reports": "https://github.com/Denolle-Lab/das-noisepy/issues",
        "Source": "https://github.com/Denolle-Lab/das-noisepy",
    },
)