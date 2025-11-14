from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ardage",
    version="0.1.0",
    author="Harihar Prasad",
    author_email="hariharprasad2006@gmail.com",
    description="arXiv Dataset Generator - Download and convert academic papers to Markdown",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HariharPrasadd/ardage",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "requests",
        "pymupdf4llm",
        "arxiv",
        "pymupdf",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "ardage=ardage.cli:cli_main",
        ],
    },
)