# ardage

**ARxiv DAtaset GEnerator** - A Python library and CLI tool for building academic paper datasets from arXiv.

Search papers via Semantic Scholar, download PDFs from arXiv, and convert to clean Markdown format with parallel processing, suitable for LLM training, RAG systems, or research analysis.

## Installation

```bash
pip install ardage
```

## Quick Start

### CLI Usage

```bash
# Interactive mode
ardage

# Direct command
ardage -q "machine learning" -n 100 -c 50

# Save only markdown (delete PDFs after conversion)
ardage -q "transformers" -n 50 --delete-pdfs
```

### Library Usage

```python
from ardage import search_papers, download_papers, convert_papers

# Search for papers
papers = search_papers(
    query="machine learning",
    num_papers=100,
    min_citations=50,
    year_range="2020-"
)

# Download PDFs (returns paper IDs: [1, 2, 3, ...])
downloaded = download_papers(
    papers,
    output_dir="data/pdfs",
    max_workers=8
)

# Convert to markdown (paper IDs correspond to paper1.pdf, paper2.md, etc.)
converted = convert_papers(
    downloaded,
    pdf_dir="data/pdfs",
    md_dir="data/md",
    max_workers=4,
    timeout=120
)

print(f"Successfully converted {len(converted)} papers")
```

## API Reference

### Core Functions

#### `search_papers(query, num_papers=150, min_citations=50, year_range="2016-")`

Search for papers via Semantic Scholar API.

**Parameters:**
- `query` (str): Search query (e.g., "deep learning", "computer vision")
- `num_papers` (int): Maximum papers to return (default: 150)
- `min_citations` (int): Minimum citation count filter (default: 50)
- `year_range` (str): Year range in format "YYYY-" or "YYYY-YYYY" (default: "2016-")

**Returns:**
- `list`: Paper metadata dictionaries sorted by citation count, filtered for arXiv availability

**Example:**
```python
papers = search_papers("neural networks", num_papers=50, year_range="2020-2023")
print(f"Found {len(papers)} papers")
```

---

#### `download_papers(papers, output_dir="data/pdfs", max_workers=8, show_progress=True)`

Download PDFs from arXiv in parallel.

**Parameters:**
- `papers` (list): List of paper dictionaries from `search_papers()`
- `output_dir` (str): Directory to save PDFs (default: "data/pdfs")
- `max_workers` (int): Number of concurrent downloads (default: 8)
- `show_progress` (bool): Display progress bar (default: True)

**Returns:**
- `list[int]`: Successfully downloaded paper IDs (e.g., `[1, 2, 3, ...]` corresponding to `paper1.pdf`, `paper2.pdf`, etc.)

**Example:**
```python
papers = search_papers("reinforcement learning")
downloaded = download_papers(papers, output_dir="my_papers/pdfs", max_workers=16)
# downloaded = [1, 2, 3, 5, 7, ...]  (some may fail, so not always sequential)
```

---

#### `convert_papers(paper_ids, pdf_dir="data/pdfs", md_dir="data/md", max_workers=4, timeout=120, show_progress=True)`

Convert PDFs to Markdown with timeout protection for hung processes.

**Parameters:**
- `paper_ids` (list): List of paper IDs to convert (e.g., `[1, 2, 3]`)
- `pdf_dir` (str): Directory containing PDFs (default: "data/pdfs")
- `md_dir` (str): Directory to save markdown files (default: "data/md")
- `max_workers` (int): Number of concurrent conversions (default: 4)
- `timeout` (int): Timeout per conversion in seconds (default: 120)
- `show_progress` (bool): Display progress bar (default: True)

**Returns:**
- `list[int]`: Successfully converted paper IDs (e.g., `[1, 3, 5, ...]` - creates `paper1.md`, `paper3.md`, etc.)

**Example:**
```python
converted = convert_papers(
    downloaded,
    pdf_dir="data/pdfs",
    md_dir="data/md",
    max_workers=8,
    timeout=90
)
# converted = [1, 2, 3, 4, ...]  (subset of downloaded if some conversions fail)
```

---

#### `download_and_convert(query, num_papers=150, min_citations=50, year_range="2016-", output_dir="data", download_workers=8, conversion_workers=4, conversion_timeout=120, show_progress=True, keep_pdfs=True)`

All-in-one convenience function for the complete pipeline.

**Parameters:**
- `query` (str): Search query
- `num_papers` (int): Maximum papers to process (default: 150)
- `min_citations` (int): Minimum citation filter (default: 50)
- `year_range` (str): Year range (default: "2016-")
- `output_dir` (str): Base output directory (default: "data")
- `download_workers` (int): Concurrent downloads (default: 8)
- `conversion_workers` (int): Concurrent conversions (default: 4)
- `conversion_timeout` (int): Timeout per conversion in seconds (default: 120)
- `show_progress` (bool): Display progress bars (default: True)
- `keep_pdfs` (bool): Keep PDF files after conversion (default: True)

**Returns:**
- `dict`: Contains:
  - `papers`: List of paper metadata from search
  - `downloaded`: List of paper IDs that were successfully downloaded (e.g., `[1, 2, 3, ...]`)
  - `converted`: List of paper IDs that were successfully converted (e.g., `[1, 2, 4, ...]`)

**Example:**
```python
result = download_and_convert(
    query="natural language processing",
    num_papers=200,
    min_citations=100,
    year_range="2022-2024",
    output_dir="nlp_dataset",
    keep_pdfs=False  # Delete PDFs after conversion
)

print(f"Papers found: {len(result['papers'])}")
print(f"Downloaded: {len(result['downloaded'])}")
print(f"Converted: {len(result['converted'])}")
```

## CLI Reference

```
usage: ardage [-h] [-q QUERY] [-n NUM_PAPERS] [-c MIN_CITATIONS] 
              [-y YEAR_RANGE] [-o OUTPUT_DIR] [-w WORKERS] 
              [-p PROCESSORS] [-t TIMEOUT] [--delete-pdfs]

options:
  -h, --help            Show help message
  -q, --query           Search query
  -n, --num-papers      Number of papers (default: 150)
  -c, --min-citations   Minimum citations (default: 50)
  -y, --year-range      Year range (default: "2016-")
  -o, --output-dir      Output directory (default: data)
  -w, --workers         Download workers (default: 8)
  -p, --processors      Conversion processors (default: 4)
  -t, --timeout         Conversion timeout in seconds (default: 120)
  --delete-pdfs         Delete PDFs after conversion
```

**Examples:**
```bash
# Basic search
ardage -q "machine learning" -n 100

# Recent papers only
ardage -q "large language models" -y "2023-" -c 100

# High-throughput processing
ardage -q "computer vision" -n 500 -w 20 -p 8

# Save disk space
ardage -q "robotics" -n 200 --delete-pdfs
```

## Advanced Usage

### Custom Processing Pipeline

```python
from ardage import search_papers, download_papers, convert_papers
import json

# Search and filter
papers = search_papers("quantum computing", num_papers=200, min_citations=100)

# Custom filtering
high_impact = [p for p in papers if p['citationCount'] > 500]
print(f"High-impact papers: {len(high_impact)}")

# Download subset
downloaded = download_papers(high_impact[:50], output_dir="quantum/pdfs")

# Convert with custom settings
converted = convert_papers(
    downloaded,
    pdf_dir="quantum/pdfs",
    md_dir="quantum/md",
    max_workers=8,
    timeout=180  # Longer timeout for complex papers
)

# Save metadata
metadata = {
    'query': 'quantum computing',
    'total_found': len(papers),
    'downloaded': len(downloaded),
    'converted': len(converted),
    'papers': papers
}

with open('quantum/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

### Batch Processing Multiple Queries

```python
from ardage import download_and_convert

queries = [
    "transformer architectures",
    "attention mechanisms",
    "self-supervised learning"
]

for query in queries:
    safe_name = query.replace(" ", "_")
    result = download_and_convert(
        query=query,
        num_papers=100,
        output_dir=f"datasets/{safe_name}",
        keep_pdfs=False
    )
    print(f"{query}: {len(result['converted'])} papers converted")
```

### Integration with Data Processing

```python
from ardage import download_and_convert
import pathlib

# Generate dataset
result = download_and_convert(
    query="neural architecture search",
    num_papers=100,
    keep_pdfs=False
)

# Process markdown files
md_dir = pathlib.Path("data/md")
for md_file in md_dir.glob("*.md"):
    content = md_file.read_text()
    
    # Your processing here
    tokens = content.split()
    print(f"{md_file.name}: {len(tokens)} tokens")
```

## Output Structure

```
data/
├── pdfs/
│   ├── paper1.pdf
│   ├── paper2.pdf
│   └── ...
└── md/
    ├── paper1.md
    ├── paper2.md
    └── ...
```

Papers are numbered sequentially based on citation count (highest first).

## Performance Tuning

### Download Speed

Increase concurrent downloads for faster throughput:
```python
downloaded = download_papers(papers, max_workers=20)  # Default: 8
```

Recommended: 8-20 workers depending on network and rate limits.

### Conversion Speed

Match CPU cores for optimal conversion performance:
```python
import multiprocessing as mp
converted = convert_papers(
    downloaded,
    max_workers=mp.cpu_count()  # Use all available cores
)
```

### Timeout Settings

Adjust timeout based on paper complexity:
```python
# Quick timeout for simple papers
converted = convert_papers(downloaded, timeout=60)

# Longer timeout for complex/large papers
converted = convert_papers(downloaded, timeout=180)
```

## Technical Notes

- **Conversion success rate**: Typically 95%+ success rate for standard academic papers
- **Timeout protection**: Hung conversions are automatically killed via process termination
- **Rate limiting**: arXiv enforces ~1 request per 3 seconds (handled automatically)
- **Multiprocessing**: Uses `spawn` method for cross-platform compatibility
- **Memory**: Each conversion worker runs in isolated process to prevent memory leaks

## Common Use Cases

### Building LLM Training Datasets
```python
result = download_and_convert(
    query="machine learning",
    num_papers=1000,
    min_citations=100,
    keep_pdfs=False  # Save disk space
)
```

### Research Literature Review
```python
papers = search_papers("federated learning", year_range="2023-")
downloaded = download_papers(papers)
# Keep PDFs for manual review, markdown for text analysis
```

### RAG System Knowledge Base
```python
# Get high-quality recent papers
result = download_and_convert(
    query="retrieval augmented generation",
    min_citations=50,
    year_range="2022-",
    keep_pdfs=False
)
# Use markdown files for vector embeddings
```

## Requirements

- Python 3.7+
- Dependencies: `requests`, `pymupdf4llm`, `arxiv`, `pymupdf`, `tqdm`

## License

MIT License - see LICENSE file for details.

## Contributing

Issues and pull requests welcome at [github.com/HariharPrasadd/ardage](https://github.com/HariharPrasadd/ardage)