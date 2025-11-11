import requests
import os
import pymupdf4llm
import sys
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pathlib
import arxiv
import pymupdf
import time


def download_paper(paper_id, paper):
    """Download PDF from arXiv, skip files under 10KB"""
    try:
        arxiv_id = paper["externalIds"].get("ArXiv")
        paper = next(arxiv.Client().results(arxiv.Search(id_list=[f"{arxiv_id}"])))
        paper.download_pdf(dirpath="data/pdfs", filename=f"paper{paper_id}.pdf")
        if os.path.getsize(f"data/pdfs/paper{paper_id}.pdf") < 10000:
            return None
        return paper_id
    except Exception as e:
        return None


def _convert_worker(paper_id, result_queue):
    """Worker process: convert PDF to markdown"""
    try:
        pymupdf.TOOLS.mupdf_display_errors(False)
        pymupdf.TOOLS.mupdf_display_warnings(False)
        pdf_path = f"data/pdfs/paper{paper_id}.pdf"
        md_path = f"data/md/paper{paper_id}.md"
        md_text = pymupdf4llm.to_markdown(pdf_path)
        pathlib.Path(md_path).write_bytes(md_text.encode())
        result_queue.put(('success', paper_id))
    except Exception as e:
        result_queue.put(('error', paper_id, str(e)))


def convert_with_timeout(paper_id, timeout=120):
    """Run conversion in separate process, kill if it hangs"""
    result_queue = mp.Queue()
    process = mp.Process(target=_convert_worker, args=(paper_id, result_queue))
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


def main():
    # Search parameters
    query = "machine learning"              # Search query
    number_of_papers = 150                  # Max papers (1-1000)
    min_citations = 50                      # Min citation filter (0+)
    year_range = "2016-"                    # Year range: "2020-", "2018-2022", etc.
    
    url = "http://api.semanticscholar.org/graph/v1/paper/search/bulk"
    query_params = {
        "query": f"{query}",
        "fields": "title,url,citationCount,externalIds",
        "year": f"{year_range}",
        "minCitationCount": min_citations,
    }
    
    # Fetch papers from Semantic Scholar
    print("Searching for papers...")
    try:
        response = requests.get(url, params=query_params)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching papers: {e}")
        sys.exit(1)
    
    # Filter for arXiv papers, sort by citations
    papers = response.json()["data"]
    papers.sort(key=lambda paper: paper["citationCount"], reverse=True)
    papers = [paper for paper in papers if "ArXiv" in paper.get("externalIds", {})]
    papers = papers[:min(number_of_papers, len(papers))]
    
    os.makedirs("data/pdfs", exist_ok=True)
    os.makedirs("data/md", exist_ok=True)
    
    # Download PDFs in parallel (8 concurrent downloads)
    downloaded_papers = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        download_futures = {executor.submit(download_paper, i + 1, papers[i]): i + 1 
                           for i in range(len(papers))}
        for future in tqdm(as_completed(download_futures), total=len(download_futures), desc="Downloading"):
            try:
                paper_id = future.result(timeout=60)
                if paper_id is not None:
                    downloaded_papers.append(paper_id)
            except Exception:
                pass
    
    print(f"\nDownloaded {len(downloaded_papers)} papers")
    
    # Convert PDFs to markdown (4 concurrent processes)
    converted_papers = []
    conversion_lock = mp.Lock()
    
    def convert_wrapper(paper_id):
        result = convert_with_timeout(paper_id, timeout=120)
        if result is not None:
            with conversion_lock:
                converted_papers.append(result)
        return result
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        conversion_futures = {executor.submit(convert_wrapper, pid): pid 
                             for pid in downloaded_papers}
        for future in tqdm(as_completed(conversion_futures), total=len(conversion_futures), desc="Converting"):
            try:
                future.result(timeout=125)
            except Exception:
                pass
    
    print(f"\nConverted {len(converted_papers)}/{len(downloaded_papers)} papers")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()