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
            'authors', 'title', 'journal', 'date', 'pubmed_id', 'doi', 'abstract'
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
            article_data = self._parse_article_data(soup, status_placeholder)
            
            if article_data:
                print(f"DEBUG - Successfully parsed article {paper_id}")
                print(f"DEBUG - Title: {article_data[1][:100]}...")
            else:
                print(f"DEBUG - Failed to parse article {paper_id}")
            
            return article_data
        
        except Exception as e:
            print(f"Error fetching article {paper_id}: {e}")
            traceback.print_exc()
            return None
    
    def _parse_article_data(self, soup, status_placeholder: Optional[Any] = None) -> Optional[Tuple]:
        try:
            article = soup.find('PubmedArticle')
            if not article:
                print("DEBUG - No PubmedArticle found in XML")
                return None

            # Extract authors
            authors = []
            author_list = article.find('AuthorList')
            if author_list:
                for author in author_list.find_all('Author'):
                    lastname = author.find('LastName')
                    initials = author.find('Initials')
                    if lastname and initials:
                        authors.append(f"{initials.text}. {lastname.text}")
            authors_str = ", ".join(authors)

            # Extract title
            title = ""
            article_title = article.find('ArticleTitle')
            if article_title:
                title = article_title.text.strip()

            # Extract journal info
            journal = ""
            journal_title = article.find('Title')
            if journal_title:
                journal = journal_title.text.strip()

            # Extract date
            date = ""
            pub_date = article.find('PubDate')
            if pub_date:
                year = pub_date.find('Year')
                if year:
                    date = year.text

            # Extract DOI and PubMed ID
            doi = ""
            pubmed_id = ""
            article_ids = article.find_all('ArticleId')
            for id_elem in article_ids:
                if id_elem.get('IdType') == 'doi':
                    doi = id_elem.text
            pmid = article.find('PMID')
            if pmid:
                pubmed_id = pmid.text

            # Extract abstract
            abstract = ""
            abstract_text = article.find('Abstract')
            if abstract_text:
                abstract = " ".join([text.text for text in abstract_text.find_all('AbstractText')])

            return (authors_str, title, journal, date, pubmed_id, doi, abstract)

        except Exception as e:
            print(f"Error parsing article: {e}")
            traceback.print_exc()
            return None 