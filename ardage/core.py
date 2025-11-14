"""
Core functionality for ardage - search, download, and convert papers
"""

import requests
import os
import pymupdf4llm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pathlib
import arxiv
import pymupdf
import shutil


def search_papers(query, num_papers=150, min_citations=50, year_range="2016-"):
    """
    Search for papers via Semantic Scholar API
    
    Args:
        query: Search query string (e.g., "machine learning")
        num_papers: Maximum number of papers to return (default: 150)
        min_citations: Minimum citation count filter (default: 50)
        year_range: Year range like "2020-" or "2018-2022" (default: "2016-")
    
    Returns:
        List of paper dictionaries with metadata, sorted by citation count
        Returns empty list if no papers found or error occurs
    """
    url = "http://api.semanticscholar.org/graph/v1/paper/search/bulk"
    query_params = {
        "query": query,
        "fields": "title,url,citationCount,externalIds",
        "year": year_range,
        "minCitationCount": min_citations,
    }
    
    try:
        response = requests.get(url, params=query_params)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching papers: {e}")
        return []
    
    # Filter for arXiv papers only, sort by citations
    papers = response.json()["data"]
    papers.sort(key=lambda paper: paper["citationCount"], reverse=True)
    papers = [paper for paper in papers if "ArXiv" in paper.get("externalIds", {})]
    papers = papers[:min(num_papers, len(papers))]
    
    return papers


def _download_single_paper(paper_id, paper, output_dir):
    """Helper function to download a single paper"""
    try:
        arxiv_id = paper["externalIds"].get("ArXiv")
        paper_obj = next(arxiv.Client().results(arxiv.Search(id_list=[arxiv_id])))
        paper_obj.download_pdf(dirpath=output_dir, filename=f"paper{paper_id}.pdf")
        
        # Skip files under 10KB (likely errors)
        if os.path.getsize(f"{output_dir}/paper{paper_id}.pdf") < 10000:
            return None
        return paper_id
    except Exception:
        return None


def download_papers(papers, output_dir="data/pdfs", max_workers=8, show_progress=True):
    """
    Download papers from arXiv in parallel
    
    Args:
        papers: List of paper dictionaries from search_papers()
        output_dir: Directory to save PDFs (default: "data/pdfs")
        max_workers: Number of concurrent downloads (default: 8)
        show_progress: Show progress bar (default: True)
    
    Returns:
        List of successfully downloaded paper IDs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded_papers = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        download_futures = {
            executor.submit(_download_single_paper, i + 1, papers[i], output_dir): i + 1
            for i in range(len(papers))
        }
        
        iterator = as_completed(download_futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(download_futures), desc="Downloading")
        
        for future in iterator:
            try:
                paper_id = future.result(timeout=60)
                if paper_id is not None:
                    downloaded_papers.append(paper_id)
            except Exception:
                pass
    
    return downloaded_papers


def _convert_worker(paper_id, pdf_dir, md_dir, result_queue):
    """Worker process: convert PDF to markdown"""
    try:
        pymupdf.TOOLS.mupdf_display_errors(False)
        pymupdf.TOOLS.mupdf_display_warnings(False)
        pdf_path = f"{pdf_dir}/paper{paper_id}.pdf"
        md_path = f"{md_dir}/paper{paper_id}.md"
        md_text = pymupdf4llm.to_markdown(pdf_path)
        pathlib.Path(md_path).write_bytes(md_text.encode())
        result_queue.put(('success', paper_id))
    except Exception as e:
        result_queue.put(('error', paper_id, str(e)))


def _convert_with_timeout(paper_id, pdf_dir, md_dir, timeout=120):
    """Run conversion in separate process, kill if it hangs"""
    result_queue = mp.Queue()
    process = mp.Process(target=_convert_worker, args=(paper_id, pdf_dir, md_dir, result_queue))
    process.start()
    process.join(timeout=timeout)
    
    # Kill hung process
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
        return None
    
    if not result_queue.empty():
        result = result_queue.get()
        if result[0] == 'success':
            return result[1]
    return None


def convert_papers(paper_ids, pdf_dir="data/pdfs", md_dir="data/md", 
                   max_workers=4, timeout=120, show_progress=True):
    """
    Convert PDFs to Markdown in parallel with timeout protection
    
    Args:
        paper_ids: List of paper IDs to convert
        pdf_dir: Directory containing PDFs (default: "data/pdfs")
        md_dir: Directory to save markdown files (default: "data/md")
        max_workers: Number of concurrent conversions (default: 4)
        timeout: Timeout per conversion in seconds (default: 120)
        show_progress: Show progress bar (default: True)
    
    Returns:
        List of successfully converted paper IDs
    """
    os.makedirs(md_dir, exist_ok=True)
    
    converted_papers = []
    conversion_lock = mp.Lock()
    
    def convert_wrapper(paper_id):
        result = _convert_with_timeout(paper_id, pdf_dir, md_dir, timeout=timeout)
        if result is not None:
            with conversion_lock:
                converted_papers.append(result)
        return result
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        conversion_futures = {
            executor.submit(convert_wrapper, pid): pid 
            for pid in paper_ids
        }
        
        iterator = as_completed(conversion_futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(conversion_futures), desc="Converting")
        
        for future in iterator:
            try:
                future.result(timeout=timeout + 5)
            except Exception:
                pass
    
    return converted_papers


def download_and_convert(query, num_papers=150, min_citations=50, year_range="2016-",
                        output_dir="data", download_workers=8, conversion_workers=4,
                        conversion_timeout=120, show_progress=True, keep_pdfs=True):
    """
    All-in-one function: search, download, and convert papers
    
    Args:
        query: Search query string
        num_papers: Maximum number of papers (default: 150)
        min_citations: Minimum citation filter (default: 50)
        year_range: Year range like "2020-" or "2018-2022" (default: "2016-")
        output_dir: Base output directory (default: "data")
        download_workers: Concurrent downloads (default: 8)
        conversion_workers: Concurrent conversions (default: 4)
        conversion_timeout: Timeout per conversion in seconds (default: 120)
        show_progress: Show progress bars (default: True)
        keep_pdfs: Keep PDF files after conversion (default: True)
    
    Returns:
        Dictionary with keys:
            - 'papers': List of paper metadata
            - 'downloaded': List of downloaded paper IDs
            - 'converted': List of converted paper IDs
    """
    pdf_dir = f"{output_dir}/pdfs"
    md_dir = f"{output_dir}/md"
    
    # Search
    if show_progress:
        print("Searching for papers...")
    papers = search_papers(query, num_papers, min_citations, year_range)
    
    if not papers:
        if show_progress:
            print("No papers found matching criteria")
        return {'papers': [], 'downloaded': [], 'converted': []}
    
    if show_progress:
        print(f"Found {len(papers)} papers on arXiv\n")
    
    # Download
    downloaded = download_papers(papers, pdf_dir, download_workers, show_progress)
    if show_progress:
        print(f"\nDownloaded {len(downloaded)} papers")
    
    # Convert
    converted = convert_papers(downloaded, pdf_dir, md_dir, conversion_workers, 
                              conversion_timeout, show_progress)
    if show_progress:
        print(f"\nConverted {len(converted)}/{len(downloaded)} papers")
    
    # Clean up PDFs if requested
    if not keep_pdfs:
        if os.path.exists(pdf_dir):
            shutil.rmtree(pdf_dir)
            if show_progress:
                print(f"\nDeleted PDF files from {pdf_dir}")
    
    return {
        'papers': papers,
        'downloaded': downloaded,
        'converted': converted
    }