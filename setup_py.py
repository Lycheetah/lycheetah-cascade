"""
Setup configuration for Lycheetah × CASCADE
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="lycheetah-cascade",
    version="2.0.1",
    author="Lycheetah",
    author_email="your.email@example.com",  # Update this
    description="Multi-dimensional signature verification for the Lycheetah Trinity Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lycheetah-cascade",  # Update this
    project_urls={
        "Bug Reports": "https://github.com/yourusername/lycheetah-cascade/issues",
        "Source": "https://github.com/yourusername/lycheetah-cascade",
        "Documentation": "https://github.com/yourusername/lycheetah-cascade/wiki",
    },
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords="nlp signature-verification brand-monitoring text-analysis semantic-analysis",
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "validation": [
            "scikit-learn>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lycheetah=lycheetah.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
