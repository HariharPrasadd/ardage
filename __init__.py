"""
ardage - arXiv Dataset Generator

A Python package for searching academic papers via Semantic Scholar,
downloading PDFs from arXiv, and converting them to clean Markdown format with parallel processing.
"""

from .core import (
    search_papers,
    download_papers,
    convert_papers,
    download_and_convert,
)

__version__ = "0.1.0"
__all__ = [
    "search_papers",
    "download_papers", 
    "convert_papers",
    "download_and_convert",
]