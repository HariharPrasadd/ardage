# ardage

**arXiv Dataset Generator** - A Python library and CLI tool for building academic paper datasets from arXiv.

Search papers via Semantic Scholar, download PDFs from arXiv, and convert to clean Markdown format suitable for LLM training, RAG systems, or research analysis.

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

# Custom output directories
ardage -q "transformers" -n 50 --pdf-dir my_pdfs --md-dir my_markdown

# Save only markdown (delete PDFs after conversion)
ardage -q "deep learning" -n 50 --delete-pdfs
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

# Download PDFs to specific directory
downloaded = download_papers(
    papers,
    output_dir="my_pdfs",
    max_workers=8
)

# Convert to markdown with custom directories
converted = convert_papers(
    downloaded,
    pdf_dir="my_pdfs",
    md_dir="my_markdown",
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
- `output_dir` (str): Directory to save PDFs - exact path (default: "data/pdfs")
- `max_workers` (int): Number of concurrent downloads (default: 8)
- `show_progress` (bool): Display progress bar (default: True)

**Returns:**
- `list[int]`: Successfully downloaded paper IDs (e.g., `[1, 2, 3, ...]` corresponding to `paper1.pdf`, `paper2.pdf`, etc.)

**Example:**
```python
papers = search_papers("reinforcement learning")
downloaded = download_papers(papers, output_dir="my_pdfs", max_workers=16)
# PDFs saved to: my_pdfs/paper1.pdf, my_pdfs/paper2.pdf, ...
```

---

#### `convert_papers(paper_ids, pdf_dir="data/pdfs", md_dir="data/md", max_workers=4, timeout=120, show_progress=True)`

Convert PDFs to Markdown with timeout protection for hung processes.

**Parameters:**
- `paper_ids` (list): List of paper IDs to convert (e.g., `[1, 2, 3]`)
- `pdf_dir` (str): Directory containing PDFs - exact path (default: "data/pdfs")
- `md_dir` (str): Directory to save markdown files - exact path (default: "data/md")
- `max_workers` (int): Number of concurrent conversions (default: 4)
- `timeout` (int): Timeout per conversion in seconds (default: 120)
- `show_progress` (bool): Display progress bar (default: True)

**Returns:**
- `list[int]`: Successfully converted paper IDs (e.g., `[1, 3, 5, ...]` - creates `paper1.md`, `paper3.md`, etc.)

**Example:**
```python
converted = convert_papers(
    downloaded,
    pdf_dir="my_pdfs",
    md_dir="converted",
    max_workers=8,
    timeout=90
)
# Markdown saved to: converted/paper1.md, converted/paper2.md, ...
```

---

#### `download_and_convert(query, num_papers=150, min_citations=50, year_range="2016-", output_dir="data", pdf_dir=None, md_dir=None, download_workers=8, conversion_workers=4, conversion_timeout=120, show_progress=True, keep_pdfs=True)`

All-in-one convenience function for the complete pipeline.

**Parameters:**
- `query` (str): Search query
- `num_papers` (int): Maximum papers to process (default: 150)
- `min_citations` (int): Minimum citation filter (default: 50)
- `year_range` (str): Year range (default: "2016-")
- `output_dir` (str): Base output directory, only used if pdf_dir/md_dir not specified (default: "data")
- `pdf_dir` (str): PDF output directory - exact path (default: None, uses `<output_dir>/pdfs`)
- `md_dir` (str): Markdown output directory - exact path (default: None, uses `<output_dir>/md`)
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
# Using base directory (creates subdirectories)
result = download_and_convert(
    query="natural language processing",
    num_papers=200,
    min_citations=100,
    year_range="2022-2024",
    output_dir="nlp_dataset"
)
# PDFs: nlp_dataset/pdfs/
# Markdown: nlp_dataset/md/

# Using exact paths
result = download_and_convert(
    query="computer vision",
    num_papers=100,
    pdf_dir="papers/pdfs",
    md_dir="papers/markdown",
    keep_pdfs=False  # Delete PDFs after conversion
)

print(f"Papers found: {len(result['papers'])}")
print(f"Downloaded: {len(result['downloaded'])}")
print(f"Converted: {len(result['converted'])}")
```

## CLI Reference

```
usage: ardage [-h] [-q QUERY] [-n NUM_PAPERS] [-c MIN_CITATIONS] 
              [-y YEAR_RANGE] [-o OUTPUT_DIR] [--pdf-dir PDF_DIR]
              [--md-dir MD_DIR] [-w WORKERS] [-p PROCESSORS] 
              [-t TIMEOUT] [--delete-pdfs]

options:
  -h, --help            Show help message
  -q, --query           Search query
  -n, --num-papers      Number of papers (default: 150)
  -c, --min-citations   Minimum citations (default: 50)
  -y, --year-range      Year range (default: "2016-")
  -o, --output-dir      Base output directory (default: data)
  --pdf-dir             PDF output directory - exact path (default: data/pdfs)
  --md-dir              Markdown output directory - exact path (default: data/md)
  -w, --workers         Download workers (default: 8)
  -p, --processors      Conversion processors (default: 4)
  -t, --timeout         Conversion timeout in seconds (default: 120)
  --delete-pdfs         Delete PDFs after conversion
```

**Examples:**
```bash
# Basic search (outputs to data/pdfs and data/md)
ardage -q "machine learning" -n 100

# Recent papers only
ardage -q "large language models" -y "2023-" -c 100

# Custom base directory (creates subdirectories)
ardage -q "robotics" -o my_papers
# Outputs to: my_papers/pdfs/ and my_papers/md/

# Exact output paths
ardage -q "computer vision" --pdf-dir downloads --md-dir converted
# Outputs to: downloads/ and converted/

# High-throughput processing
ardage -q "transformers" -n 500 -w 20 -p 8

# Save disk space (delete PDFs after conversion)
ardage -q "deep learning" -n 200 --delete-pdfs
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

# Download to custom directory
downloaded = download_papers(high_impact[:50], output_dir="quantum/pdfs")

# Convert with custom settings
converted = convert_papers(
    downloaded,
    pdf_dir="quantum/pdfs",
    md_dir="quantum/markdown",
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
        pdf_dir=f"datasets/{safe_name}/pdfs",
        md_dir=f"datasets/{safe_name}/markdown",
        keep_pdfs=False
    )
    print(f"{query}: {len(result['converted'])} papers converted")
```

### Using Individual Functions for Maximum Control

```python
from ardage import search_papers, download_papers, convert_papers

# Step 1: Search
papers = search_papers("deep learning", num_papers=100, year_range="2023-")

# Step 2: Download to specific location
downloaded = download_papers(
    papers,
    output_dir="project/raw_pdfs",
    max_workers=16
)

# Step 3: Process only successful downloads
print(f"Processing {len(downloaded)} papers...")

# Step 4: Convert with custom paths
converted = convert_papers(
    downloaded,
    pdf_dir="project/raw_pdfs",
    md_dir="project/processed_markdown",
    max_workers=8,
    timeout=60
)

# Step 5: Clean up PDFs if needed
import shutil
if len(converted) > 0:
    shutil.rmtree("project/raw_pdfs")
    print("Cleaned up PDF files")
```

### Integration with Data Processing

```python
from ardage import download_and_convert
import pathlib

# Generate dataset
result = download_and_convert(
    query="neural architecture search",
    num_papers=100,
    pdf_dir="papers",
    md_dir="markdown",
    keep_pdfs=False
)

# Process markdown files
md_dir = pathlib.Path("markdown")
for md_file in md_dir.glob("*.md"):
    content = md_file.read_text()
    
    # Your processing here
    tokens = content.split()
    print(f"{md_file.name}: {len(tokens)} tokens")
```

## Output Structure

### Default Structure (using `output_dir`)
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

### Custom Structure (using `--pdf-dir` and `--md-dir`)
```
my_project/
├── downloads/
│   ├── paper1.pdf
│   ├── paper2.pdf
│   └── ...
└── converted/
    ├── paper1.md
    ├── paper2.md
    └── ...
```

Papers are numbered sequentially based on citation count (highest first).

## Performance Tuning

### Download Speed

Increase concurrent downloads for faster throughput:
```python
downloaded = download_papers(papers, output_dir="pdfs", max_workers=20)  # Default: 8
```

Recommended: 8-20 workers depending on network and rate limits.

### Conversion Speed

Match CPU cores for optimal conversion performance:
```python
import multiprocessing as mp
converted = convert_papers(
    downloaded,
    pdf_dir="pdfs",
    md_dir="markdown",
    max_workers=mp.cpu_count()  # Use all available cores
)
```

### Timeout Settings

Adjust timeout based on paper complexity:
```python
# Quick timeout for simple papers
converted = convert_papers(downloaded, pdf_dir="pdfs", md_dir="md", timeout=60)

# Longer timeout for complex/large papers
converted = convert_papers(downloaded, pdf_dir="pdfs", md_dir="md", timeout=180)
```

## Technical Notes

- **Conversion success rate**: Typically 95%+ success rate for standard academic papers
- **Timeout protection**: Hung conversions are automatically killed via process termination
- **Rate limiting**: arXiv enforces ~1 request per 3 seconds (handled automatically)
- **Multiprocessing**: Uses `spawn` method for cross-platform compatibility
- **Memory**: Each conversion worker runs in isolated process to prevent memory leaks
- **File naming**: Papers are saved as `paper1.pdf`, `paper2.md`, etc., numbered by citation rank

## Common Use Cases

### Building LLM Training Datasets
```python
result = download_and_convert(
    query="machine learning",
    num_papers=1000,
    min_citations=100,
    pdf_dir="training_data/pdfs",
    md_dir="training_data/markdown",
    keep_pdfs=False  # Save disk space
)
```

### Research Literature Review
```python
papers = search_papers("federated learning", year_range="2023-")
downloaded = download_papers(papers, output_dir="literature/pdfs")
converted = convert_papers(downloaded, pdf_dir="literature/pdfs", md_dir="literature/markdown")
# Keep PDFs for manual review, markdown for text analysis
```

### RAG System Knowledge Base
```python
# Get high-quality recent papers
result = download_and_convert(
    query="retrieval augmented generation",
    min_citations=50,
    year_range="2022-",
    md_dir="rag_knowledge_base",
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