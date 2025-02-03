from bs4 import BeautifulSoup
import aiohttp
import calendar
import pandas as pd
import ssl
from typing import List, Union, Optional, Any, Tuple
import traceback
import asyncio
import urllib.parse

def strip_brackets(s: str) -> str:
    """Remove square brackets from a string."""
    return ''.join(char for char in s if char not in ['[', ']'])

class PubMedSearchAgent:
    def __init__(self):
        self._setup_ssl()
        # Limit concurrent efetch requests to 3.
        self.semaphore = asyncio.Semaphore(3)
    
    def _setup_ssl(self):
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

    async def _async_get_json(self, url: str, session: aiohttp.ClientSession, 
                              retries: int = 3, delay: float = 1.0) -> dict:
        """
        Helper method to GET a URL and return JSON.
        If the JSON contains an error (e.g. rate limit exceeded), wait and retry.
        """
        for attempt in range(1, retries + 1):
            async with session.get(url) as response:
                data = await response.json()
            if "error" in data:
                print(f"DEBUG - API rate limit exceeded (attempt {attempt}/{retries}). "
                      f"Response: {data}. Retrying after {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                return data
        # After all retries, return the last response.
        return data

    async def search_pubmed(self, phrases: List[str], results_per_phrase: Union[int, dict] = 40, 
                            status_placeholder: Optional[Any] = None) -> pd.DataFrame:
        """
        Asynchronously search PubMed for a list of phrases.
        
        :param phrases: List of phrases to search for.
        :param results_per_phrase: Number of results per phrase.
        :param status_placeholder: (Optional) Object with a .write() method for status messages.
        :return: A pandas DataFrame with columns:
                 ['authors', 'title', 'journal', 'volume', 'date', 'pubmed_id', 'doi', 'abstract']
        """
        print("\nDEBUG - Starting PubMedSearchAgent.search_pubmed")
        print(f"DEBUG - Received phrases: {phrases}")
        
        all_articles = []
        
        async with aiohttp.ClientSession() as session:
            for phrase in phrases:
                # Use proper URL encoding.
                phrase_encoded = urllib.parse.quote(phrase)
                base_url = (
                    f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
                    f"db=pubmed&retmode=json&retmax={results_per_phrase}&sort=relevance&term={phrase_encoded}"
                )
                print(f"\nDEBUG - Processing phrase: {phrase_encoded}")
                
                # Use our safe-get JSON helper.
                data = await self._async_get_json(base_url, session, retries=5, delay=1.5)
                
                # Check if the expected key exists.
                idlist = data.get('esearchresult', {}).get('idlist', [])
                if not idlist:
                    print(f"DEBUG - No paper IDs found for phrase {phrase_encoded}. Full response: {data}")
                    await asyncio.sleep(0.3)
                    continue

                print(f"DEBUG - Found {len(idlist)} paper IDs for phrase {phrase_encoded}")
                
                # Create tasks to fetch article details concurrently.
                tasks = [self._fetch_article_data(paper_id, session, status_placeholder) for paper_id in idlist]
                results = await asyncio.gather(*tasks)
                for res in results:
                    if res:
                        all_articles.append(res)
                        if status_placeholder:
                            status_placeholder.write(f"Found paper: {res[1][:100]}...")
                # Stagger the processing between phrases.
                await asyncio.sleep(0.3)
        
        # Create DataFrame with proper columns.
        df = pd.DataFrame(all_articles, columns=[
            'authors', 'title', 'journal',  'date',  'doi', 'abstract'
        ])
        
        print(f"\nDEBUG - Final DataFrame shape: {df.shape}")
        if not df.empty:
            print("DEBUG - Sample titles:")
            print(df['title'].head())
        
        return df
    
    async def _fetch_article_data(self, paper_id: str, session: aiohttp.ClientSession, 
                                  status_placeholder: Optional[Any] = None) -> Optional[Tuple]:
        """
        Asynchronously fetch article details from PubMed for a given paper ID.
        
        :param paper_id: PubMed paper ID.
        :param session: aiohttp ClientSession to use for HTTP requests.
        :param status_placeholder: (Optional) Object with a .write() method for status messages.
        :return: A tuple with article data or None if fetching/parsing fails.
        """
        try:
            if status_placeholder:
                status_placeholder.write(f"Fetching details for paper ID: {paper_id}")
            
            url = f"http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&id={paper_id}"
            # Use the semaphore to throttle concurrent efetch requests.
            async with self.semaphore:
                async with session.get(url) as r:
                    content = await r.read()
            
            # Use an XML parser to avoid warnings.
            soup = BeautifulSoup(content, features="xml")
            return self._parse_article_data(soup, status_placeholder)
        
        except Exception as e:
            print(f"Error fetching article for ID {paper_id}: {e}")
            traceback.print_exc()
            return None
    
    def _parse_article_data(self, soup, status_placeholder: Optional[Any] = None) -> Optional[Tuple]:
        """
        Parse the BeautifulSoup-parsed XML to extract article details.
        
        :param soup: BeautifulSoup object of the XML response.
        :param status_placeholder: (Optional) Object with a .write() method for status messages.
        :return: A tuple containing article details or None if parsing fails.
        """
        try:
            # efetch XML responses usually contain <PubmedArticle> as the root for each article.
            article = soup.find('PubmedArticle')
            if not article:
                return None
            
            # Journal info is usually within <Journal>.
            journal = article.find('Journal')
            if not journal:
                return None

            # Extract authors.
            authors = ""
            authorlist = article.find('AuthorList')
            if authorlist:
                author_elements = authorlist.find_all('Author')
                author_count = len(author_elements)
                for i, author in enumerate(author_elements):
                    last = author.find('LastName')
                    init = author.find('Initials')
                    if last and init:
                        authors += f"{init.text}. {last.text}"
                        if i == author_count - 2:
                            authors += " and "
                        elif i < author_count - 1:
                            authors += ", "

            # Extract title.
            title = ""
            articletitle = article.find('ArticleTitle')
            if articletitle:
                title = strip_brackets(articletitle.text)

            # Extract journal title.
            journal_title = ""
            journal_title_tag = journal.find('Title')
            if journal_title_tag:
                journal_title = journal_title_tag.text

            # Extract volume and issue.
            volume = ""
            journal_issue = journal.find('JournalIssue')
            if journal_issue:
                volume_tag = journal_issue.find('Volume')
                if volume_tag:
                    volume = volume_tag.text
                issue_tag = journal_issue.find('Issue')
                if issue_tag:
                    volume += f"({issue_tag.text})" if volume else f"({issue_tag.text})"

            # Extract date (year and month).
            date = ""
            if journal_issue:
                month = journal_issue.find('Month')
                year = journal_issue.find('Year')
                if month and year:
                    month_text = month.text
                    if len(month_text) < 3:
                        try:
                            month_text = calendar.month_abbr[int(month_text)]
                        except Exception:
                            month_text = month.text
                    date = f"({month_text}. {year.text})"
                elif year:
                    date = f"({year.text})"

            # Extract IDs.
            pubmed_id = ""
            doi = ""
            pmid_tag = article.find('PMID')
            if pmid_tag:
                pubmed_id = pmid_tag.text
            doi_tag = article.find('ArticleId', IdType='doi')
            if doi_tag:
                doi = doi_tag.text

            # Extract abstract.
            abstract = ""
            abstract_tag = article.find('Abstract')
            if abstract_tag:
                abstract_text = abstract_tag.find_all('AbstractText')
                abstract = " ".join([a.text for a in abstract_text]) if abstract_text else ""

            if status_placeholder:
                status_placeholder.write(f"Found paper: {title[:100]}...")
            
            return (authors, title, journal_title, date, doi, abstract)
        
        except Exception as e:
            print(f"Error parsing article: {e}")
            traceback.print_exc()
            return None

# Example asynchronous usage:
async def main():
    agent = PubMedSearchAgent()
    phrases = ["microbial motility", "genome editing"]
    df = await agent.search_pubmed(phrases, results_per_phrase=40)
    print(df.head())
    print(df.iloc[-10:])

if __name__ == "__main__":
    asyncio.run(main())
