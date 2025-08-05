# --------------------------------------------------------------------------
# --- 1. IMPORTS
# --------------------------------------------------------------------------
import urllib.error
from http.client import RemoteDisconnected
import socket
import datetime
import time
import requests
import json
from datetime import timedelta
import os
import pathlib
import random

# New libraries to replace the old method
import arxiv
import semanticscholar as s2

# --------------------------------------------------------------------------
# --- 2. HELPER FUNCTIONS FOR FINDING CODE
# --------------------------------------------------------------------------

def find_code_from_arxiv(arxiv_result):
    """
    Checks for a code link directly in the arXiv result object.
    Authors can link their code directly on the abstract page.
    
    @param arxiv_result: An arxiv.Result object.
    @return: The URL to the code repository if found, otherwise None.
    """
    # The arxiv library stores links in arxiv_result.links.
    # We are looking for a link with title 'code' or common repo domains.
    for link in arxiv_result.links:
        if 'github.com' in link.href or 'gitlab.com' in link.href:
            return link.href
        if hasattr(link, 'title') and link.title and link.title.lower() == 'code':
            return link.href
    return None

def find_code_from_semantic_scholar(arxiv_id):
    """
    Uses the Semantic Scholar API to find the official code repository.
    Requires the 'semanticscholar' library.

    @param arxiv_id: The arXiv ID of the paper (e.g., '2305.12345').
    @return: The URL to the code repository if found, otherwise None.
    """
    try:
        # Use the s2.paper() method with the arXiv ID prefix.
        # It's good practice to include a timeout.
        paper = s2.paper(f'ARXIV:{arxiv_id}', timeout=20, fields=['url', 'title', 'tldr', 'codeOfficial'])
        # The 'codeOfficial' field contains the info we need.
        if paper and paper.get('codeOfficial'):
            repo_url = paper['codeOfficial'].get('url')
            if repo_url and isinstance(repo_url, str) and repo_url.startswith('http'):
                print(f"  [Semantic Scholar] Found code for {arxiv_id}: {repo_url}")
                return repo_url
    except (requests.exceptions.RequestException, s2.errors.S2ApiException, Exception) as e:
        # Handle potential API errors gracefully.
        print(f"  [Semantic Scholar] Warning: API call failed for {arxiv_id}. Error: {e}")
    return None

# --------------------------------------------------------------------------
# --- 3. MAIN LOGIC FUNCTIONS
# --------------------------------------------------------------------------

def get_daily_code(DateToday, cats):
    """
    Fetches papers from arXiv for a specific day and finds their associated code repositories
    using a cascade strategy (arXiv API first, then Semantic Scholar API).

    @param DateToday: str (in "YYYY-MM-DD" format)
    @param cats: dict of categories to search
    @return: A dictionary with the date as key and found papers/code as value.
    """
    content = dict()
    output = dict() # Using a dict to store unique papers by ID

    # --- Part 1: Fetch papers from arXiv ---
    for k, v in cats.items():
        # Create a search query by combining sub-categories
        query = ' OR '.join([f'cat:{cat}' for cat in v])
        
        # Use the arxiv library's search functionality
        search = arxiv.Search(
            query=query,
            max_results=1000,  # Fetch a larger number to filter by day
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        print(f"Searching arXiv for categories {v} for date {DateToday}...")
        
        try:
            # Filter results by the specific day
            target_date = datetime.datetime.strptime(DateToday, "%Y-%m-%d").date()
            for result in search.results():
                published_date = result.published.date()
                if published_date == target_date:
                    arxiv_id = result.get_short_id()
                    if arxiv_id not in output:
                        # Store the full result object for later use
                        output[arxiv_id] = result
            
            print(f"Successfully fetched and filtered {len(output)} unique papers from arXiv for {DateToday}.")
        except Exception as e:
            print(f"Error: An unexpected error occurred searching arXiv for {DateToday}: {e}")
            continue  # Move to the next category if search fails

        # Add a polite delay between searching different primary categories
        print(f"Waiting a moment after processing category {k}...")
        time.sleep(5)

    # --- Part 2: Check each paper for code ---
    cnt = 0
    papers_to_check = list(output.values())
    print(f"\nChecking {len(papers_to_check)} papers for associated code...")

    for paper_result in papers_to_check:
        _id = paper_result.get_short_id()
        paper_title = " ".join(paper_result.title.split())
        paper_url = paper_result.pdf_url
        repo_url = None

        print(f"Processing paper: {_id} - {paper_title[:60]}...")

        # Strategy 1: Check the arXiv object directly (most reliable)
        repo_url = find_code_from_arxiv(paper_result)
        if repo_url:
            print(f"  [arXiv] Found code directly from arXiv metadata: {repo_url}")
        else:
            # Strategy 2: Fallback to Semantic Scholar API
            print(f"  [Fallback] No direct link on arXiv. Querying Semantic Scholar for {_id}...")
            repo_url = find_code_from_semantic_scholar(_id)
            # Add a small delay between Semantic Scholar API calls to be polite
            time.sleep(random.uniform(1.0, 2.0))

        # If a repo URL was found by any method, format and add it
        if repo_url:
            cnt += 1
            repo_name = repo_url.split("/")[-1] if '/' in repo_url else repo_url
            # The final format is kept the same for compatibility with your markdown function
            content[_id] = f"|[{paper_title}]({paper_url})|[{repo_name}]({repo_url})|\n"
        else:
            print(f"  -> No official code found for {_id}")

    data = {DateToday: content}
    print(f"\nFinished processing for {DateToday}. Found {cnt} papers with official code.")
    return data

def update_daily_json(filename, data_all):
    """
    Reads a JSON file, updates it with new data, and writes it back.
    """
    m = {}
    try:
        with open(filename, "r") as f:
            content = f.read()
            if content:
                m = json.loads(content)
    except FileNotFoundError:
        print(f"File {filename} not found. A new one will be created.")

    # Update the dictionary with new data
    for data in data_all:
        m.update(data)

    # Save data back to the JSON file
    with open(filename, "w") as f:
        json.dump(m, f, indent=4, sort_keys=True) # Use indent for readability
    print(f"Updated {filename} successfully.")

def ensure_archive_dirs():
    """
    Create archive directory structure if it doesn't exist.
    """
    archive_base = "archives"
    current_year = datetime.date.today().year
    
    # Create base archives dir
    pathlib.Path(archive_base).mkdir(exist_ok=True)
    
    # Create directories for all years from 2023 to current year
    # Starting from a fixed year like 2023 makes it more robust
    for year in range(2023, current_year + 1):
        year_dir = os.path.join(archive_base, str(year))
        pathlib.Path(year_dir).mkdir(exist_ok=True)
        
        # Create month markdown files if they don't exist
        for month in range(1, 13):
            month_file = os.path.join(year_dir, f"{month:02d}.md")
            if not os.path.exists(month_file):
                with open(month_file, "w") as f:
                    f.write(f"# {datetime.date(year, month, 1).strftime('%B %Y')} Archive\n\n")

def json_to_md(filename):
    """
    Convert JSON data to markdown files, including a main README and monthly archives.
    """
    try:
        with open(filename, "r") as f:
            content = f.read()
            if not content:
                data = {}
            else:
                data = json.loads(content)
    except FileNotFoundError:
        print(f"Error: {filename} not found. Cannot generate markdown.")
        return

    # Ensure archive structure exists before writing
    ensure_archive_dirs()
    
    # Group entries by year and month
    entries_by_month = {}
    latest_entries = []
    today = datetime.date.today()
    # Display last 14 days in README for more content
    recent_days_limit = today - timedelta(days=14) 
    
    # Sort dates from newest to oldest
    sorted_days = sorted(data.keys(), reverse=True)

    for day in sorted_days:
        day_date = datetime.datetime.strptime(day, "%Y-%m-%d").date()
        year_month_str = day_date.strftime("%Y/%m") # e.g., "2025/08"
        
        # Collect entries for the main README's "Latest Updates"
        if day_date > recent_days_limit:
            latest_entries.append((day, data[day]))
            
        # Group all entries by month for the archives
        if year_month_str not in entries_by_month:
            entries_by_month[year_month_str] = []
        entries_by_month[year_month_str].append((day, data[day]))
    
    # --- Update main README.md ---
    with open("README.md", "w", encoding="utf-8") as f:
        # Write header and overview
        f.write("# Daily ArXiv HCI\n\n")
        f.write("A curated collection of arXiv papers with open-source implementations, specifically focusing on Human-Computer Interaction (cs.HC) ")
        f.write("and related fields like Computer Graphics (cs.GR), Computer Vision (cs.CV), etc. This repository aims to serve researchers and practitioners ")
        f.write("in HCI by providing easy access to papers that come with their source code implementations.\n\n")
        f.write("## Latest Updates\n\n")
        f.write(f"*Last updated: {today.strftime('%Y-%m-%d')}*\n\n")
        
        # Write table header for latest updates
        f.write("| Date | Paper | Code |\n")
        f.write("|---|---|---|\n")
        
        # Write latest entries to the table
        for day, day_content in latest_entries:
            if not day_content: continue
            # Sort papers within a day for consistent output
            for paper_id in sorted(day_content.keys()):
                # The value now already contains the full markdown table row format
                # We just need to add the date
                f.write(f"| {day} {day_content[paper_id]}")
        f.write("\n")
        
        # Write archive links
        f.write("## Archives\n\n")
        for year_month in sorted(entries_by_month.keys(), reverse=True):
            year, month = year_month.split("/")
            month_name = datetime.date(int(year), int(month), 1).strftime("%B")
            f.write(f"- [{month_name} {year}](archives/{year}/{int(month):02d}.md)\n")

    # --- Update monthly archive files ---
    for year_month, entries in entries_by_month.items():
        year, month = year_month.split("/")
        archive_file = f"archives/{year}/{int(month):02d}.md"
        
        with open(archive_file, "w", encoding="utf-8") as f:
            month_name_full = datetime.date(int(year), int(month), 1).strftime('%B %Y')
            f.write(f"# {month_name_full} Archive\n\n")
            f.write("[Back to README](../../README.md)\n\n")
            f.write("| Date | Paper | Code |\n")
            f.write("|---|---|---|\n")

            # Entries are already grouped by month, sort by day within the month
            for day, day_content in sorted(entries, key=lambda x: x[0], reverse=True):
                if not day_content: continue
                # Sort papers within a day for consistent output
                for paper_id in sorted(day_content.keys()):
                    f.write(f"| {day} {day_content[paper_id]}")
            f.write("\n")
            
    print("Finished generating markdown files.")

# --------------------------------------------------------------------------
# --- 4. SCRIPT EXECUTION
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # The number of past days to check for papers.
    # Set to 7 to check the last week.
    DAYS_TO_FETCH = 7
    
    # Define the categories you are interested in.
    # The key is a general name, the value is a list of specific arXiv categories.
    CATEGORIES = {
        "cs": ["cs.HC", "cs.GR", "cs.MM", "cs.CV", "cs.AI", "cs.RO"]
    }
    
    JSON_FILENAME = "daily.json"
    
    today = datetime.date.today()
    data_all = []

    print("==================================================")
    print(f"Starting daily arXiv fetch for the last {DAYS_TO_FETCH} days.")
    print("==================================================")
    
    for i in range(DAYS_TO_FETCH):
        # We go from newest to oldest: yesterday, the day before, etc.
        day_to_check = str(today - timedelta(days=i + 1))
        print(f"\n--- Processing Date: {day_to_check} ---")
        daily_data = get_daily_code(day_to_check, CATEGORIES)
        data_all.append(daily_data)

    print("\n--- Finalizing Data ---")
    update_daily_json(JSON_FILENAME, data_all)
    json_to_md(JSON_FILENAME)
    print("\nScript finished successfully.")
