from bs4 import BeautifulSoup
import requests
import urllib.request
import json
import calendar
import pandas as pd
import ssl
from typing import List, Union, Dict, Optional, Any, Tuple
import traceback

def strip_brackets(s: str) -> str:
    """Remove square brackets from a string."""
    return ''.join(char for char in s if char not in ['[', ']'])

class PubMedSearchAgent:
    def __init__(self):
        self._setup_ssl()
    
    def _setup_ssl(self):
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
    
    def search_pubmed(self, phrases: List[str], results_per_phrase: Union[int, Dict[str, int]] = 40, 
                     status_placeholder: Optional[Any] = None) -> pd.DataFrame:
        print("\nDEBUG - Starting PubMedSearchAgent.search_pubmed")
        print(f"DEBUG - Received phrases: {phrases}")
        

        all_articles = []
        
        for phrase in phrases:
            phrase = phrase.replace(" ", "+")
            base_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retmax={results_per_phrase}&sort=relevance&term={phrase}"

            print(f"\nDEBUG - Processing phrase: {phrase}")
            
            # Split into individual terms and join with AND
            #terms = phrase.lower().split()
            response = requests.get(base_url)

            
            data = response.json()
            idlist = data['esearchresult']['idlist']
            print(f"DEBUG - Found {len(idlist)} paper IDs")
            
            for paper_id in idlist:
                article_data = self._fetch_article_data(paper_id, status_placeholder)
                if article_data:
                    all_articles.append(article_data)
                    if status_placeholder:
                        status_placeholder.write(f"Found paper: {article_data[1][:100]}...")
                    

        # Create DataFrame with proper columns
        df = pd.DataFrame(all_articles, columns=[
            'authors', 'title', 'journal', 'volume', 'date', 'pubmed_id', 'doi', 'abstract'
        ])
        
        print(f"\nDEBUG - Final DataFrame shape: {df.shape}")
        if not df.empty:
            print("DEBUG - Sample titles:")
            print(df['title'].head())
        
        return df
    
    def _fetch_article_data(self, paper_id: str, status_placeholder: Optional[Any] = None) -> Optional[Tuple]:
        try:
            if status_placeholder:
                status_placeholder.write(f"Fetching details for paper ID: {paper_id}")
            
            url = f"http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id={paper_id}"
            r = requests.get(url)
            soup = BeautifulSoup(r.content, features="xml")
            return self._parse_article_data(soup, status_placeholder)
        
        except Exception as e:
            print(f"Error fetching article: {e}")
            return None
    
    def _parse_article_data(self, soup, status_placeholder: Optional[Any] = None) -> Optional[Tuple]:
        try:
            article = soup.find('article')
            journal = soup.find('journal')
            if not article or not journal:
                return None

            # Extract authors
            authors = ""
            authorlist = article.find('authorlist')
            if authorlist:
                author_count = len(authorlist.find_all('lastname'))
                for i, (last, init) in enumerate(zip(authorlist.find_all('lastname'), authorlist.find_all('initials'))):
                    authors += f"{init.text}. {last.text}"
                    if i == author_count - 2:
                        authors += " and "
                    elif i < author_count - 1:
                        authors += ", "

            # Extract title
            title = ""
            if article.find('articletitle'):
                title = strip_brackets(article.find('articletitle').text)

            # Extract journal info
            journal_title = journal.find('title').text if journal.find('title') else ""
            volume = journal.find('volume').text if journal.find('volume') else ""
            if volume and soup.find('issue'):
                volume += f"({soup.find('issue').text})"

            # Extract date
            date = ""
            journal_issue = journal.find('journalissue')
            if journal_issue:
                month = journal_issue.find('month')
                year = journal_issue.find('year')
                if month and year:
                    month_text = month.text
                    if len(month_text) < 3:
                        month_text = calendar.month_abbr[int(month_text)]
                    date = f"({month_text}. {year.text})"
                elif year:
                    date = f"({year.text})"

            # Extract IDs
            pubmed_id = ""
            doi = ""
            if article.find('articleid', idtype='pubmed'):
                pubmed_id = article.find('articleid', idtype='pubmed').text
            if article.find('elocationid', eidtype='doi'):
                doi = article.find('elocationid', eidtype='doi').text

            # Extract abstract
            abstract = ""
            if article.find('abstract'):
                abstract = article.find('abstract').get_text()

            if status_placeholder:
                status_placeholder.write(f"Found paper: {title[:100]}...")
            
            return (authors, title, journal_title, volume, date, pubmed_id, doi, abstract)
        
        except Exception as e:
            print(f"Error parsing article: {e}")
            return None 