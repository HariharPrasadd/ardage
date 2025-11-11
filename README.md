# arXiv Paper Downloader & Converter

Downloads academic papers from arXiv via Semantic Scholar search and converts them to Markdown.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python arxiv_downloader.py
```

Edit these variables in `main()` to customize:

```python
query = "machine learning"    # Search query
number_of_papers = 150        # Max papers to download
min_citations = 50            # Minimum citation filter
year_range = "2016-"          # Year range: "2020-", "2018-2022", etc.
```

Output structure:
```
data/
├── pdfs/    # Downloaded PDFs
└── md/      # Converted Markdown files
```

## Performance Tuning

```python
# Download speed (line ~70)
ThreadPoolExecutor(max_workers=8)  # Increase to 16-20 for faster downloads

# Conversion speed (line ~85)
ThreadPoolExecutor(max_workers=4)  # Set to mp.cpu_count() to use all cores

# Conversion timeout (line ~82)
convert_with_timeout(paper_id, timeout=120)  # Reduce to 60-90 if needed
```

## Notes

- Expect 70-90% conversion success rate (some PDFs fail due to format issues)
- Papers must be available on arXiv (auto-filtered from Semantic Scholar results)
- Conversions that hang are automatically killed after timeout