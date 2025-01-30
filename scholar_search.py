import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

def scholar_and_pubmed_search(search_term):
    """
    Scrapes the first 5 pages of Google Scholar for a given search term to collect titles.
    Then, for each title, queries PubMed and attempts to extract the abstract, URL, and authors.
    
    :param search_term: String representing the query (e.g., 'microbial motility in porous media').
    :return: A list of dictionaries, where each dictionary has:
             {
               'title': ...,
               'abstract': ...,
               'url': ...,
               'authors': [...]
             }
    """
    
    # --- Step 1: Fetch titles from Google Scholar ---
    base_scholar_url = "https://scholar.google.com/scholar"
    all_titles = []
    
    # We will collect titles from 5 pages: start=0,10,20,30,40
    for page in range(5):
        start = page * 10
        params = {
            'q': search_term,
            'hl': 'en',
            'as_sdt': '0,29',
            'start': start
        }
        
        response = requests.get(base_scholar_url, params=params)
        if response.status_code != 200:
            print(f"Warning: Scholar request failed on page {page+1} with status {response.status_code}")
            continue
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Titles appear in <h3 class="gs_rt">; they typically contain an <a>
        for h3 in soup.select('h3.gs_rt'):
            link = h3.select_one('a')
            if link:
                title = link.get_text(separator=" ", strip=True)
                all_titles.append(title)
    
    # --- Step 2: For each title, query PubMed and parse the first result ---
    results = []
    for title in all_titles:
        pubmed_data = fetch_pubmed_info(title)
        results.append({
            'title': title,
            'abstract': pubmed_data.get('abstract'),
            'url': pubmed_data.get('url'),
            'authors': pubmed_data.get('authors')
        })
    
    return results

import requests
from bs4 import BeautifulSoup
import re

import requests
from bs4 import BeautifulSoup
import re

def fetch_pubmed_info(title):
    """
    Given a paper title, search for it on PubMed and try to extract:
      - Abstract
      - PubMed URL
      - Authors (no digits, no duplicates)
      - Publication Year
      - Journal Name (if available)
    
    :param title: Paper title string
    :return: Dict with keys:
        {
          'abstract': str or None,
          'url': str or None,
          'authors': list of str,
          'year': str or None,
          'journal': str or None
        }
    """
    base_pubmed_url = "https://pubmed.ncbi.nlm.nih.gov/"
    params = {
        'term': title
    }
    
    # Step 1: Search PubMed
    response = requests.get(base_pubmed_url, params=params)
    if response.status_code != 200:
        return {
            'abstract': None,
            'url': None,
            'authors': [],
            'year': None,
            'journal': None
        }
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Step 2: Find the first result link (classically a.docsum-title)
    first_result_link = soup.select_one('a.docsum-title')
    if not first_result_link:
        return {
            'abstract': None,
            'url': None,
            'authors': [],
            'year': None,
            'journal': None
        }
    
    # Step 3: Build the article URL
    article_url = "https://pubmed.ncbi.nlm.nih.gov" + first_result_link.get('href', '')
    
    # Step 4: Fetch the article page
    article_response = requests.get(article_url)
    if article_response.status_code != 200:
        return {
            'abstract': None,
            'url': article_url,
            'authors': [],
            'year': None,
            'journal': None
        }
    
    article_soup = BeautifulSoup(article_response.text, "html.parser")
    
    # Step 5: Extract the abstract
    abstract_tag = article_soup.select_one('div.abstract-content')
    abstract = abstract_tag.get_text(strip=True) if abstract_tag else None
    
    # Step 6: Extract authors
    author_tags = article_soup.select('div.authors-list span.authors-list-item')
    authors = []
    for tag in author_tags:
        raw_author = tag.get_text(strip=True)
        
        # Remove digits
        no_digits = re.sub(r'\d+', '', raw_author)
        
        # Strip trailing commas, periods, or extra whitespace
        cleaned_author = no_digits.strip(",. ")
        authors.append(cleaned_author)
    
    # Deduplicate authors while preserving order
    seen = set()
    unique_authors = []
    for author in authors:
        if author not in seen:
            seen.add(author)
            unique_authors.append(author)
    
    # Simpler year extraction - look for first 4-digit number in URL or page content
    year = None
    year_match = re.search(r'/20\d{2}/', article_url)  # Check URL first
    if not year_match:
        # Check entire page content for a year between 1900-2024
        year_match = re.search(r'(?:19|20)\d{2}', str(article_soup))
    if year_match:
        year = year_match.group(0).strip('/')
    
    # Step 8: Extract journal name
    # PubMed often includes journal info in:
    #   meta[name="citation_journal_title"] or meta[name="citation_source"]
    possible_journal_names = ["citation_journal_title", "citation_source"]
    journal = None
    for jmeta in possible_journal_names:
        meta_tag_journal = article_soup.select_one(f'meta[name="{jmeta}"]')
        if meta_tag_journal and meta_tag_journal.get("content"):
            journal = meta_tag_journal["content"].strip()
            break
    
    return {
        'title': title,
        'abstract': abstract,
        'url': article_url,
        'authors': unique_authors,
        'year': year,
        'journal': journal
    }

