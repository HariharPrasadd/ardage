#!/usr/bin/env python3
"""
CLI interface for ardage
"""

import argparse
import sys
import multiprocessing as mp
from . import search_papers, download_papers, convert_papers


def get_user_input():
    """Interactive prompts for search parameters"""
    print("\n=== ardage - arXiv Dataset Generator ===\n")
    
    query = input("Search query (e.g., 'machine learning'): ").strip()
    if not query:
        print("Error: Query cannot be empty")
        sys.exit(1)
    
    while True:
        try:
            num_papers = input("Number of papers (default: 150): ").strip()
            num_papers = int(num_papers) if num_papers else 150
            if num_papers < 1:
                print("Error: Must be at least 1")
                continue
            break
        except ValueError:
            print("Error: Enter a valid number")
    
    while True:
        try:
            min_citations = input("Minimum citations (default: 50): ").strip()
            min_citations = int(min_citations) if min_citations else 50
            if min_citations < 0:
                print("Error: Cannot be negative")
                continue
            break
        except ValueError:
            print("Error: Enter a valid number")
    
    while True:
        year_range = input("Year range (e.g., '2020-' or '2018-2022', default: '2016-'): ").strip()
        year_range = year_range if year_range else "2016-"
        if '-' not in year_range:
            print("Error: Format should be 'YYYY-' or 'YYYY-YYYY'")
            continue
        break
    
    print(f"\nSearching for '{query}' papers...")
    print(f"Papers: {num_papers} | Min citations: {min_citations} | Years: {year_range}\n")
    
    return query, num_papers, min_citations, year_range


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Download and convert academic papers from arXiv to Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ardage                                      # Interactive mode
  ardage -q "machine learning" -n 100         # Quick download (to data/pdfs and data/md)
  ardage -q "transformers" -n 50 -c 100 -y "2020-2023"
  ardage -q "computer vision" -o my_papers    # Outputs to my_papers/pdfs and my_papers/md
  ardage -q "deep learning" --delete-pdfs     # Delete PDFs after conversion
  ardage -q "NLP" --pdf-dir my_pdfs --md-dir my_markdown  # Exact paths
        """
    )
    
    parser.add_argument('-q', '--query', help='Search query')
    parser.add_argument('-n', '--num-papers', type=int, help='Number of papers (default: 150)')
    parser.add_argument('-c', '--min-citations', type=int, help='Minimum citations (default: 50)')
    parser.add_argument('-y', '--year-range', help='Year range, e.g., "2020-" or "2018-2022" (default: "2016-")')
    parser.add_argument('-o', '--output-dir', default='data', help='Base output directory (default: data)')
    parser.add_argument('--pdf-dir', help='PDF output directory (default: data/pdfs)')
    parser.add_argument('--md-dir', help='Markdown output directory (default: data/md)')
    parser.add_argument('-w', '--workers', type=int, default=8, help='Download workers (default: 8)')
    parser.add_argument('-p', '--processors', type=int, default=4, help='Conversion processors (default: 4)')
    parser.add_argument('-t', '--timeout', type=int, default=120, help='Conversion timeout in seconds (default: 120)')
    parser.add_argument('--delete-pdfs', action='store_true', help='Delete PDF files after conversion')
    
    args = parser.parse_args()
    
    # Interactive mode if no query provided
    if args.query is None:
        query, num_papers, min_citations, year_range = get_user_input()
    else:
        query = args.query
        num_papers = args.num_papers if args.num_papers else 150
        min_citations = args.min_citations if args.min_citations else 50
        year_range = args.year_range if args.year_range else "2016-"
        print(f"\nSearching for '{query}' papers...")
        print(f"Papers: {num_papers} | Min citations: {min_citations} | Years: {year_range}\n")
    
    # Search
    papers = search_papers(query, num_papers, min_citations, year_range)
    
    if not papers:
        print("No papers found matching criteria")
        sys.exit(1)
    
    print(f"Found {len(papers)} papers on arXiv\n")
    
    # Determine output directories - use exact paths if specified
    if args.pdf_dir:
        pdf_dir = args.pdf_dir
    else:
        pdf_dir = f"{args.output_dir}/pdfs"
    
    if args.md_dir:
        md_dir = args.md_dir
    else:
        md_dir = f"{args.output_dir}/md"
    
    # Download
    downloaded = download_papers(papers, pdf_dir, args.workers, show_progress=True)
    print(f"\nDownloaded {len(downloaded)} papers")
    
    # Convert
    converted = convert_papers(downloaded, pdf_dir, md_dir, args.processors, 
                              args.timeout, show_progress=True)
    print(f"\nConverted {len(converted)}/{len(downloaded)} papers")
    
    # Delete PDFs if requested
    if args.delete_pdfs:
        import shutil
        import os
        if os.path.exists(pdf_dir):
            shutil.rmtree(pdf_dir)
            print(f"Deleted PDF files from {pdf_dir}/")
    
    # Summary
    print(f"\nDataset saved to:")
    if not args.delete_pdfs:
        print(f"  PDFs: {pdf_dir}/")
    print(f"  Markdown: {md_dir}/")


def cli_main():
    """Entry point that sets multiprocessing start method"""
    mp.set_start_method('spawn', force=True)
    main()


if __name__ == "__main__":
    cli_main()