import urllib.error
from http.client import RemoteDisconnected
import socket

import arxivscraper
import datetime
import time
import requests
import json
from datetime import timedelta
import os
import pathlib
import random


def get_daily_code(DateToday,cats):
    """
    @param DateToday: str
    @param cats: dict
    @return paper_with_code: dict
    """
    from_day = until_day = DateToday
    content = dict()
    # content
    output = dict()

    MAX_RETRIES = 3 # Number of retries
    BASE_RETRY_DELAY = 15 # Base seconds to wait before retry
    
    for k,v in cats.items():
        scraper = arxivscraper.Scraper(category=k, date_from=from_day,date_until=until_day,filters={'categories':v})
        print(f"Attempting to scrape category {k} with filters {v} for date {DateToday}...")

        retries = 0
        success = False
        while retries < MAX_RETRIES and not success:
            try:
                #print(f"Scrape attempt {retries + 1}/{MAX_RETRIES}...") # Optional: more verbose logging
                tmp = scraper.scrape()
                print(f"Successfully scraped category {k} for {DateToday}.")
                if isinstance(tmp, list):
                    for item in tmp:
                        if item["id"] not in output:
                            output[item["id"]] = item
                success = True # Mark as success

            except (ConnectionResetError, urllib.error.URLError, RemoteDisconnected, socket.timeout, TimeoutError) as e: # Catch specific network errors
                retries += 1
                print(f"Warning: Attempt {retries}/{MAX_RETRIES} failed for category {k} on {DateToday}. Error: {e}")
                if retries >= MAX_RETRIES:
                    print(f"Error: Max retries reached for category {k} on {DateToday}. Skipping this category.")
                    break # Stop retrying for this category
                # Exponential backoff with jitter
                wait_time = BASE_RETRY_DELAY * (2 ** (retries - 1)) + random.uniform(0, 5)
                print(f"Waiting {wait_time:.2f} seconds before next retry...")
                time.sleep(wait_time)
            except Exception as e: # Catch any other unexpected error during scrape
                 print(f"Error: An unexpected error occurred scraping {k} for {DateToday}: {e}")
                 # Decide if you want to retry for other errors or just break
                 break # Breaking here for unexpected errors

        # Add a mandatory delay *between* scraping different primary categories (k)
        # This helps even if retries weren't needed for the previous category
        print(f"Waiting mandatory delay after processing category {k}...")
        time.sleep(45) # Use a reasonable base delay between categories (e.g., 45-60s)

    base_url = "https://arxiv.paperswithcode.com/api/v0/papers/"
    cnt = 0

    papers_to_check = list(output.items()) # Create a list to iterate over safely

    print(f"\nChecking {len(papers_to_check)} papers against PapersWithCode API...")

    for k_id, v_data in papers_to_check:
        #print(f"Checking paper ID: {k_id}") # Optional: more verbose logging
        _id = v_data["id"]
        paper_title = " ".join(v_data["title"].split())
        paper_url = v_data["url"]
        paper_date = v_data.get("published", DateToday)
        if isinstance(paper_date, datetime.datetime):
            paper_date = paper_date.strftime("%Y-%m-%d")

        url = base_url + _id
        retries_pwc = 0
        success_pwc = False
        while retries_pwc < MAX_RETRIES and not success_pwc:
            try:
                response = requests.get(url, timeout=20) # Add a timeout to requests
                response.raise_for_status() # Check for HTTP errors (4xx, 5xx)
                r = response.json()

                if "official" in r and r["official"] and isinstance(r["official"], dict) and "url" in r["official"]:
                    cnt += 1
                    repo_url = r["official"]["url"]
                    # Basic check to avoid malformed URLs (simple version)
                    if isinstance(repo_url, str) and repo_url.startswith('http'):
                        repo_name = repo_url.split("/")[-1] if '/' in repo_url else repo_url
                        content[_id] = f"|[{paper_title}]({paper_url})|[{repo_name}]({repo_url})|\n"
                        #print(f" Found code for {_id}: {repo_url}") # Optional logging
                    else:
                        print(f"Warning: Malformed repo URL found for {_id}: {repo_url}")

                success_pwc = True # Mark as success

            except requests.exceptions.ConnectionError as e:
                retries_pwc += 1
                print(f"Warning: PapersWithCode API connection error for {_id} (Attempt {retries_pwc}/{MAX_RETRIES}): {e}")
            except requests.exceptions.Timeout as e:
                 retries_pwc += 1
                 print(f"Warning: PapersWithCode API timeout for {_id} (Attempt {retries_pwc}/{MAX_RETRIES}): {e}")
            except requests.exceptions.RequestException as e: # Catch other request errors (like HTTP errors)
                print(f"Error: PapersWithCode API request failed for {_id}: {e}")
                if response is not None and 400 <= response.status_code < 500:
                     print(f"Client error ({response.status_code}), likely paper not found or bad request. Skipping.")
                     break # Don't retry client errors usually
                retries_pwc += 1 # Retry server errors or unknown request errors
                print(f"Retrying attempt {retries_pwc}/{MAX_RETRIES}...")

            except json.JSONDecodeError as e:
                 print(f"Error: Failed to decode JSON response from PapersWithCode for {_id}: {e}")
                 # This might indicate an API issue or non-JSON response, usually best to skip
                 break # Don't retry JSON errors
            except Exception as e:
                 print(f"Error: Unexpected error checking PapersWithCode for {_id}: {e}")
                 break # Don't retry other unexpected errors

            if not success_pwc and retries_pwc < MAX_RETRIES:
                 wait_time_pwc = BASE_RETRY_DELAY * (2 ** (retries_pwc - 1)) + random.uniform(0, 3)
                 print(f"Waiting {wait_time_pwc:.2f} seconds before next PapersWithCode API retry...")
                 time.sleep(wait_time_pwc)
            elif not success_pwc:
                 print(f"Error: Max retries reached for PapersWithCode API check for {_id}. Skipping.")

        # Add a small delay *between* PapersWithCode API calls to be nice
        time.sleep(random.uniform(0.5, 1.5)) # Small random delay

    data = {DateToday: content}
    print(f"Found {cnt} papers with official code for {DateToday}.")
    return data

def update_daily_json(filename,data_all):
    with open(filename,"r") as f:
        content = f.read()
        if not content:
            m = {}
        else:
            m = json.loads(content)
    
    #将datas更新到m中
    for data in data_all:
        m.update(data)

    # save data to daily.json

    with open(filename,"w") as f:
        json.dump(m,f)
    



def ensure_archive_dirs():
    """Create archive directory structure if it doesn't exist"""
    archive_base = "archives"
    current_year = datetime.date.today().year
    
    # Create base archives dir
    pathlib.Path(archive_base).mkdir(exist_ok=True)
    
    # Create directories for all years from 2021 to current year
    for year in range(2021, current_year + 1):
        year_dir = os.path.join(archive_base, str(year))
        pathlib.Path(year_dir).mkdir(exist_ok=True)
        
        # Create month directories if they don't exist
        for month in range(1, 13):
            month_file = os.path.join(year_dir, f"{month:02d}.md")
            if not os.path.exists(month_file):
                with open(month_file, "w") as f:
                    f.write(f"# {datetime.date(year, month, 1).strftime('%B %Y')} Archive\n\n")

def json_to_md(filename):
    """
    Convert JSON data to markdown files with archives
    @param filename: str
    @return None
    """
    with open(filename, "r") as f:
        content = f.read()
        if not content:
            data = {}
        else:
            data = json.loads(content)
    
    # Ensure archive structure exists
    ensure_archive_dirs()
    
    # Group entries by year and month
    entries_by_month = {}
    latest_entries = []
    today = datetime.date.today()
    week_ago = today - timedelta(days=7)
    
    for day in sorted(data.keys(), reverse=True):
        day_date = datetime.datetime.strptime(day, "%Y-%m-%d").date()
        year_month = day_date.strftime("%Y/%m")
        
        # Collect entries for archives
        if day_date > week_ago:
            latest_entries.append((day, data[day]))
            
        if year_month not in entries_by_month:
            entries_by_month[year_month] = []
        entries_by_month[year_month].append((day, data[day]))
    
    # Update main README.md
    with open("README.md", "w") as f:
        # Write header and overview
        f.write("# Daily ArXiv HCI\n\n") 
        f.write("A curated collection of arXiv papers with open-source implementations, specifically focusing on Human-Computer Interaction (cs.HC) ")
        f.write("and related fields like Computer Graphics (cs.GR), Computer Vision (cs.CV), etc. This repository aims to serve researchers and practitioners ")
        f.write("in HCI by providing easy access to papers that come with their source code implementations.\n\n") # Updated description
        f.write("## Overview\n")
        f.write("This project automatically tracks and analyzes papers from relevant HCI categories ") # Updated description
        f.write("on arXiv daily using GitHub Actions. It specifically identifies ")
        f.write("and catalogs papers that have released their source code, making it easier for researchers ")
        f.write("in HCI and related areas to find implementable research work.\n\n") # Updated description
        f.write("The main features include:\n")
        f.write("- Daily updates of papers with open-source implementations\n")
        f.write("- Focus on Human-Computer Interaction and related research\n") # Updated description
        f.write("- Automatic tracking and organization\n\n")
        # --- END ---
        
        # Write latest updates
        f.write("## Latest Updates \n")
        yymm = f"{str(today.year)[2:]}{today.month:02d}"
        f.write("|date|paper|code|\n" + "|---|---|---|\n")
        for day, day_content in latest_entries:
            if not day_content:
                continue
            for k, v in day_content.items():
                if k.startswith(yymm):
                    f.write(f"|{k}{v}")
        f.write("\n")
        
        # Write archive links
        f.write("\n## Archives\n")
        for year_month in sorted(entries_by_month.keys(), reverse=True):
            year, month = year_month.split("/")
            month_name = datetime.date(int(year), int(month), 1).strftime("%B")
            f.write(f"- [{month_name} {year}](archives/{year}/{int(month):02d}.md)\n")
    
    # Update archive files
    for year_month, entries in entries_by_month.items():
        year, month = year_month.split("/")
        archive_file = f"archives/{year}/{int(month):02d}.md"
        yymm = f"{year[2:]}{month}"
        with open(archive_file, "w") as f:
            f.write(f"# {datetime.date(int(year), int(month), 1).strftime('%B %Y')} Archive\n\n")
            f.write("[Back to README](../../README.md)\n\n")
            f.write("|date|paper|code|\n" + "|---|---|---|\n")
            for day, day_content in entries:
                if not day_content:
                    continue
                for k, v in day_content.items():
                    if k.startswith(yymm):
                        f.write(f"|{k}{v}")
            f.write("\n")
    
    print("Finished generating markdown files")

if __name__ == "__main__":

    DateToday = datetime.date.today()
    N = 3
    data_all = []
    for i in range(1,N):
        day = str(DateToday + timedelta(-i))
        # you can add the categories in cats
        cats = {
        "cs": ["cs.HC", "cs.GR"]
    }
        data = get_daily_code(day,cats)
        data_all.append(data)
    update_daily_json("daily.json",data_all)
    json_to_md("daily.json")
