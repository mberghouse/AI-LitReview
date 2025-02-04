import asyncio
import aiohttp
from bs4 import BeautifulSoup
import re

async def fetch_pubmed_info(query_title, session):
    """
    Given a paper title (query), search for it on PubMed and extract:
      - Official Title (from the PubMed page)
      - Abstract (if not found, the paper is skipped)
      - PubMed URL
      - Authors (cleaned, no digits, no duplicates)
      - Publication Year (using meta tags and a fallback)
      - Journal Name (if available)

    Returns a dict with keys:
      {
        'title': str or None,
        'abstract': str,
        'url': str or None,
        'authors': list of str,
        'year': str or None,
        'journal': str or None
      }
    or None if no abstract is found.
    """
    base_pubmed_url = "https://pubmed.ncbi.nlm.nih.gov/"
    params = {'term': query_title}

    # Search PubMed using the query title.
    async with session.get(base_pubmed_url, params=params) as response:
        text = await response.text()

    soup = BeautifulSoup(text, features='xml')
    print(soup)
    # Locate the first search result link (commonly with class 'docsum-title')
    first_result_link = soup.select_one('a.docsum-title')
    if not first_result_link:
        return None

    article_url = "https://pubmed.ncbi.nlm.nih.gov" + first_result_link.get('href', '')

    # Fetch the article page.
    async with session.get(article_url) as art_response:
        art_text = await art_response.text()

    article_soup = BeautifulSoup(art_text, "html.parser")

    # Extract the official PubMed title using meta tag.
    citation_title_meta = article_soup.select_one('meta[name="citation_title"]')
    pubmed_title = (citation_title_meta["content"].strip() 
                    if citation_title_meta and citation_title_meta.get("content") 
                    else None)

    # Extract the abstract; if not found, skip this paper.
    abstract_tag = article_soup.select_one('div.abstract-content')
    abstract = abstract_tag.get_text(strip=True) if abstract_tag else None
    if not abstract:
        return None

    # Extract authors and clean them (remove digits, trailing punctuation, and duplicates).
    author_tags = article_soup.select('div.authors-list span.authors-list-item')
    authors = []
    for tag in author_tags:
        raw_author = tag.get_text(strip=True)
        no_digits = re.sub(r'\d+', '', raw_author)
        cleaned_author = no_digits.strip(",. ")
        authors.append(cleaned_author)
    # Deduplicate authors while preserving order.
    seen = set()
    unique_authors = []
    for author in authors:
        if author not in seen:
            seen.add(author)
            unique_authors.append(author)

    # Attempt to extract the publication year using meta tags.
    possible_meta_names = ["citation_publication_date", "citation_date"]
    year = None
    for meta_name in possible_meta_names:
        meta_tag = article_soup.select_one(f'meta[name="{meta_name}"]')
        if meta_tag and meta_tag.get("content"):
            match = re.search(r'(\d{4})', meta_tag["content"])
            if match:
                year = match.group(1)
                break
    # Fallback: try to find the year in a div with class 'cit'.
    if not year:
        cit_tag = article_soup.select_one('div.cit')
        if cit_tag:
            match = re.search(r'(\d{4})', cit_tag.get_text())
            if match:
                year = match.group(1)

    # Extract the journal name from common meta tags.
    possible_journal_names = ["citation_journal_title", "citation_source"]
    journal = None
    for jmeta in possible_journal_names:
        meta_tag_journal = article_soup.select_one(f'meta[name="{jmeta}"]')
        if meta_tag_journal and meta_tag_journal.get("content"):
            journal = meta_tag_journal["content"].strip()
            break

    doi = ""
    # if article_soup.find('articleid', idtype='pubmed'):
    #     pubmed_id = article_soup.find('articleid', idtype='pubmed').text
    if article_soup.find('elocationid', eidtype='doi'):
        doi = article_soup.find('elocationid', eidtype='doi').text

    return {
        'title': pubmed_title,
        'abstract': abstract,
        'url': article_url,
        'authors': unique_authors,
        'year': year,
        'journal': journal,
        'doi': doi
    }

async def scholar_and_pubmed_search(search_term):
    """
    Scrapes the first 3 pages of Google Scholar for a given search term to collect titles.
    Then, for each title, concurrently queries PubMed and extracts the desired information.

    Returns a list of dictionaries, each containing:
      {
        'title': (Google Scholar title),
        'pubmed_title': (Official PubMed title),
        'abstract': str,
        'url': str,
        'authors': list of str,
        'year': str,
        'journal': str
      }
    Papers with no abstract are skipped.
    """
    base_scholar_url = "https://scholar.google.com/scholar"
    all_titles = []
    # Use a browser-like User-Agent.
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/103.0.0.0 Safari/537.36"
        )
    }
    async with aiohttp.ClientSession(headers=headers) as session:
        # Fetch Google Scholar pages sequentially.
        for page in range(4):
            start = page * 10
            params = {
                'q': search_term,
                'hl': 'en',
                'as_sdt': '0,29',
                'start': start
            }
            async with session.get(base_scholar_url, params=params) as response:
                text = await response.text()
            soup = BeautifulSoup(text, "html.parser")
            for h3 in soup.select('h3.gs_rt'):
                link = h3.select_one('a')
                if link:
                    title = link.get_text(separator=" ", strip=True)
                    if title not in all_titles:
                        all_titles.append(title)

        # For each Google Scholar title, query PubMed concurrently.
        pubmed_tasks = [fetch_pubmed_info(title, session) for title in all_titles]
        pubmed_results = await asyncio.gather(*pubmed_tasks)
    print(pubmed_results)
    results = []
    seen_abstracts = set()
    # Combine results; only include those with valid PubMed data.
    for original_title, pubmed_data in zip(all_titles, pubmed_results):
        if pubmed_data:
            authors_str = ", ".join(pubmed_data['authors']) if isinstance(pubmed_data['authors'], list) else pubmed_data['authors']
            normalized_abstract = pubmed_data['abstract'].strip() if pubmed_data['abstract'] else ""
            if normalized_abstract in seen_abstracts:
                continue
            else:
                seen_abstracts.add(normalized_abstract)
                results.append({
                'title': original_title,
                'pubmed_title': pubmed_data['title'],
                'abstract': pubmed_data['abstract'],
                'url': pubmed_data['url'],
                'authors': authors_str,
                'year': pubmed_data['year'],
                'journal': pubmed_data['journal'],
                'doi': pubmed_data['doi']
                })
    return results