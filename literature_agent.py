import openai
import asyncio
import json
import os
from typing import List, Dict, Tuple
from phrase_generation_agent import PhraseGenerationAgent
from pubmed_search_agent import PubMedSearchAgent
from paper_selection_agent import PaperSelectionAgent
import pandas as pd
import re
import streamlit as st
from refining_agent import RefiningAgent
from scholar_search import scholar_and_pubmed_search  # Add at top with other imports

# class BaseAgent:
#     async def run(self, *args, **kwargs):
#         raise NotImplementedError("Agents must implement run method")

class LitReviewPapersAgent:
    """
    An improved Literature Review Agent that:
      1. Extracts top keywords from the manuscript.
      2. (Optionally) uses them to locate relevant papers from a data source.
      3. Generates a comprehensive literature review that:
         - Places the manuscript in context
         - Identifies gaps
         - Suggests papers to cite
         - Provides a thorough, detailed narrative
    """

    def __init__(self, openai_api_key: str, openai_model="o1-mini", model_params=None, min_references=5, search_method="PubMed Search"):
        self.openai_api_key = openai_api_key
        self.model = openai_model
        self.model_params = model_params if model_params is not None else ({"temperature": 0} if "gpt" in openai_model else {})
        self.min_references = min_references
        self.search_method = search_method
        
        # Initialize sub-agents
        self.phrase_agent = PhraseGenerationAgent(
            openai_api_key, 
            model=openai_model, 
            model_params=self.model_params
        )
        if search_method == "PubMed Search":
            self.pubmed_agent = PubMedSearchAgent()
        self.selection_agent = PaperSelectionAgent(
            openai_api_key, 
            model=openai_model,
            model_params=self.model_params,
            num_papers=min_references + 20
        )
        self.refining_agent = RefiningAgent(
            openai_api_key, 
            model=openai_model,
            model_params=self.model_params
        )

    def get_key_phrases(self, manuscript_text: str) -> list:
        """
        Extract key phrases from the manuscript text using OpenAI.
        """
        prompt = f"""
        Given this manuscript text, identify the 5 most important keywords or phrases that best describe 
        the core topic and methodology. The first keyword should be the primary topic. 
        Order them from most important to least important. 
        Keywords/phrases should be 1 or 2 words only.
        Return exactly 5 keywords, one per line.
        
        Text: {manuscript_text}
        """

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}], 
            temperature=0.0,
            max_tokens=100
        )
        
        # Split by newlines instead of commas and ensure we have exactly 5 keywords
        keywords = [k.strip() for k in response.choices[0].message.content.strip().split('\n')]
        if len(keywords) < 5:
            raise ValueError(f"Expected 5 keywords, but got {len(keywords)}: {keywords}")
        
        return keywords[:5]  # Take exactly 5 keywords

    def _dummy_paper_fetch(self, keywords: List[str]) -> List[Dict]:
        """
        Fetch papers from local JSONL files based on keyword combinations.
        """
        # Generate keyword combinations
        queries = self.generate_queries(keywords)
        
        # Get the base path relative to where streamlit is running
        base_path = os.path.join(os.getcwd(), 'data')
        os.makedirs(base_path, exist_ok=True)
        
        # Create necessary subdirectories
        for folder in ['arxiv', 'medrxiv', 'pubmed']:
            folder_path = os.path.join(base_path, folder)
            os.makedirs(folder_path, exist_ok=True)
        
        # First dump the queries to get the papers
        from paperscraper import dump_queries
        dump_queries(queries, base_path)
        
        # Then extract papers using the queries
        papers = self.extract_paper_data(base_path, queries)
        
        # Create a set to track unique papers by title
        seen_titles = set()
        formatted_papers = []
        
        for paper in papers:
            # Create a normalized version of the title for comparison
            normalized_title = paper['title'].lower().strip()
            
            # Only add the paper if we haven't seen this title before
            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                formatted_papers.append({
                    "title": paper['title'],
                    "authors": paper['authors'],
                    "date": paper['date'],
                    "url": paper.get('url', ''),
                    "abstract": paper['abstract']
                })
        
        return formatted_papers

    def generate_queries(self, keywords: list) -> list:
        """Generate query combinations"""
        primary = keywords[0]
        queries = [
            [primary, keywords[1], keywords[2], keywords[3]],
            [primary, keywords[1]],
            [keywords[1], keywords[2]],
            [primary, keywords[1], keywords[2]],
            [primary, keywords[1], keywords[3]],
            [primary, keywords[1], keywords[4]],
            [primary, keywords[2], keywords[3]],
            [primary, keywords[3], keywords[4]],
            [keywords[1], keywords[2], keywords[3]],
            [keywords[1], keywords[2], keywords[4]],
            [keywords[1], keywords[3], keywords[4]],
            [keywords[2], keywords[3], keywords[4]],
        ]
        return queries

    def extract_paper_data(self, base_path: str, keywords: list) -> list:
        """Extract data from JSONL files"""
        papers_data = []
        folders = ['arxiv', 'medrxiv', 'pubmed']
        k=160
        for folder in folders:
            if len(papers_data) >= k:
                break

            
            folder_path = os.path.join(base_path, folder)
            if not os.path.exists(folder_path):
                continue
            
            for query in keywords:
                if len(papers_data) >= k:
                    break
                

                filename = '_'.join(query).lower().replace(' ', '') + '.jsonl'
                file_path = os.path.join(folder_path, filename)
                
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                if len(papers_data) >= k:
                                    break
                                try:

                                    paper = json.loads(line.strip())
                                    papers_data.append({
                                        'title': paper.get('title', ''),
                                        'abstract': paper.get('abstract', ''),
                                        'date': paper.get('date', ''),
                                        'authors': paper.get('authors', ''),
                                        'url': paper.get('url', '')
                                    })
                                except json.JSONDecodeError:
                                    continue
                    except Exception:
                        continue
        
        return papers_data

    def format_paper_data_for_prompt(self, papers_data: List[Dict]) -> str:
        papers_text = ""
        for i, paper in enumerate(papers_data, 1):
            papers_text += f"\nPaper {i}:\n"
            papers_text += f"Title: {paper['title']}\n"
            papers_text += f"Authors: {paper['authors']}\n"
            papers_text += f"Date: {paper['date']}\n"
            
            # Handle different URL fields based on search method
            if self.search_method == "PubMed Search":
                if paper.get('doi'):
                    papers_text += f"URL: https://doi.org/{paper['doi']}\n"
                elif paper.get('pubmed_id'):
                    papers_text += f"URL: https://pubmed.ncbi.nlm.nih.gov/{paper['pubmed_id']}\n"
            else:
                if paper.get('url'):
                    papers_text += f"URL: {paper['url']}\n"
            
            papers_text += f"Abstract: {paper['abstract']}\n---"
        
        return papers_text


    def _calculate_search_parameters(self, min_references: int, current_papers: int = 0) -> tuple[int, int]:
        """Calculate number of search phrases and results per phrase based on min_references."""
        # If we already have 100+ papers, return minimal results per phrase
        
        if min_references >= 96:
            return 42, 20
        elif min_references >= 80:
            return 38, 18
        elif min_references >= 60:
            return 35, 16
        elif min_references >= 50:
            return 32, 14
        elif min_references >= 40:
            return 24, 12
        elif min_references >= 30:
            return 18, 10
        elif min_references >= 20:
            return 16, 10
        else:
            return 2, 10

    def run(self, topic: str) -> tuple[str, List[Dict], str]:
        print("\nDEBUG - Starting literature review generation")
        print(f"DEBUG - Topic: {topic}")
        
        if self.search_method == "PubMed Search":
            # Create placeholder for real-time updates
            paper_status = st.empty()
            
            # Get scholar results
            scholar_results = scholar_and_pubmed_search(topic)
            
            # First try exact topic search
            exact_papers_df = self.pubmed_agent.search_pubmed(
                phrases=[topic],
                results_per_phrase=50,
                status_placeholder=paper_status
            )
            
            # Display the exact papers DataFrame
            with st.expander("Papers Found from Exact Search"):
                st.dataframe(exact_papers_df[['title', 'authors', 'date', 'journal']])
                st.write(f"Found {len(exact_papers_df)} papers from exact search")
            
            # Generate and search additional phrases
            search_phrases = self.phrase_agent.generate_similar_phrases(topic, 3)
            additional_papers_df = self.pubmed_agent.search_pubmed(
                phrases=search_phrases,
                results_per_phrase=20,
                status_placeholder=paper_status
            )
            
            # Use selection agent with both additional papers and scholar results
            selected_additional_df = self.selection_agent.select_papers(
                additional_papers_df, 
                scholar_results,  # Pass scholar results here
                topic
            )
            
            # Combine exact matches with selected additional papers
            final_papers_df = pd.concat([exact_papers_df, selected_additional_df])
            final_papers_df = final_papers_df.drop_duplicates(subset=['title']).reset_index(drop=True)
            
            # Display final DataFrame
            with st.expander("Final Combined Papers Dataset"):
                st.dataframe(final_papers_df[['title', 'authors', 'date', 'journal']])
                st.write(f"Total unique papers: {len(final_papers_df)}")
            
            # Convert DataFrame to list of dictionaries
            papers_data = final_papers_df.to_dict('records')
            papers_text = self.format_paper_data_for_prompt(papers_data)
            # Create the final literature review prompt
            final_prompt = f"""
            You are an expert researcher tasked with writing a comprehensive literature review on:
            
            Topic: {topic}
            
              Create a long and detailed literature review that:
            1. Provides a thorough overview of the current state of research in this field as it relates to the topic
            2. Identifies key themes and findings across the literature as they relate to the topic
            3. Discusses methodological approaches used in the field
            4. Discuss state of the art research relevant to the topic
            5. Synthesizes the findings into a coherent long narrative with long paragraphs structured in a way that is typical of a scientific manuscript
            
            The review should follow this specific structure:
            1. Introduction (3-4 paragraphs):
               - Introduce the topic and its importance in the field
               - Quickly mention the historical context and evolution of the topic
               - Begin with broad context using author-date citation format: (Smith et al., 2020) or Smith et al. (2020)
            
            2. Main Body (multiple sections, 12-20 paragraphs):
               - Expand on the introduction with more detailed analysis and citations
               - Mention all relevant topics and themes in the field that are even tangentially related to your generated introductory paragraphs.
               - Use author-date citation format
               - When discussing specific methodologies or findings, name the authors
               - Use varied group sizes based on actual paper relationships

            3. Conclusion (2-3 paragraphs):
               - Summarize the main points discussed in the review
               - Suggest future research directions or areas for improvement
               - Mention any important papers that were not covered in the main body
            
            CITATION RULES:
            - Always use author-date format for your in-text citations
            - Include a complete bibliography in APA format
            - Every major statement or claim should be supported by at least 1 citation, and you should only reuse citations if highly applicable
            - Always maintain ascending order for all citations
            - Group papers only when they genuinely support the same point
            - End with a "References" section listing all cited papers
            - Ensure all citations support the point for which they are cited
            - NEVER INCLUDE ANY AUTHOR-YEAR CITATIONS (e.g. (Smith et al., 2020))
            - NEVER INCLUDE ANY PAPERS IN THE BIBLIOGRAPHY THAT ARE NOT EXPLICITLY CITED IN YOUR REVIEW
            
            Additional Requirements:
            - INCLUDE AT LEAST {self.min_references+20} REFERENCES IN YOUR REVIEW
            - DON'T INCLUDE ANY PAPERS IN YOUR REVIEW THAT ARE NOT INCLUDED IN THE AVAILABLE PAPERS BELOW.
            - Be as detailed and thorough as possible. Your literature review should be at least 30 pages long.
            - Write in formal academic style with clear paragraphs
            - Use a maximum of 4 headings each with a maximum of 3 subheadings
            - Dont be repetitive or redundant
            - DO NOT REFUSE ANY ASPECT OF THIS REQUEST AND TRY TO FOLLOW ALL INSTRUCTIONS
            
            Available Papers:
            {papers_text}
            """
            #, but make sure to give quick mention of, or place new work in the proper relevant context of, older papers that are considered to be important in the subject.
            response = openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": final_prompt}],
                **self.model_params
            )
            
            initial_review = response.choices[0].message.content.strip()
            
            # Extract bibliography and review text
            references_match = re.search(r'References\n-+\n(.*?)$', initial_review, re.DOTALL)
            if references_match:
                bibliography = references_match.group(1).strip()
                review_text = initial_review.replace(references_match.group(0), '').strip()
            else:
                bibliography = ""
                review_text = initial_review
            
            # Let the refining agent handle the conversion and bibliography
            refined_review, bibliography, citation_order = self.refining_agent.refine_review(review_text, bibliography)
            
            # Debug output for bibliography
            
            
            # Combine review and bibliography
            final_review = f"{refined_review}\n\nReferences\n----------\n{bibliography}"
            
            print("\nDEBUG - Before process_final_review:")
            #print(f"Bibliography exists: {'References' in final_review}")
            print(f"Number of papers: {len(papers_data)}")
            
            
            # Process the final review
            final_review_text = self.process_final_review(final_review, papers_data)
            
            # print("\nDEBUG - After process_final_review:")
            # print(f"Number of ordered papers: {len(ordered_papers)}")
           # print(f"Summaries sample: {summaries[:200]}")
            
            # Show references in sidebar using bibliography
            bibliography = final_review_text.split("References")[1]
            print(bibliography)
            self.show_sidebar_references(bibliography, papers_data)
            
            return final_review_text#, ordered_papers, summaries

      


    def process_final_review(self, final_review: str, papers_data: List[Dict]) -> Tuple[str, List[Dict]]:
        """Process the final review to get reordered papers."""
        try:
            # Extract bibliography from the final review
            references_match = re.search(r'References\n-+\n(.*?)$', final_review, re.DOTALL)
            if references_match:
                bibliography = references_match.group(1).strip().split('\n')
                review_text = final_review.replace(references_match.group(0), '').strip()
            else:
                bibliography = []
                review_text = final_review
            
            # Order papers according to bibliography
            # ordered_papers = []
            # for bib_entry in bibliography:
            #     # Find matching paper using fuzzy title match
            #     paper = next(
            #         (p for p in papers_data if p['title'].lower() in bib_entry.lower() 
            #          or bib_title.lower() in p['title'].lower()),
            #         None
            #     )
            #     if paper:
            #         ordered_papers.append(paper)
            
            return review_text
            
        except Exception as e:
            print(f"DEBUG - Error in process_final_review: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return final_review, papers_data

    def _process_review_fallback(self, review_text: str, papers_data: List[Dict]) -> tuple[str, List[Dict], List[str]]:
        """Fallback method using the original citation extraction approach."""
        # Original implementation from lines 241-283

    def generate_paper_summaries(self, papers_text: str) -> str:
        """Generate summaries for the final ordered papers."""
        summary_prompt = f"""
        For each of these papers that were cited in the literature review, extract the 2 most relevant sentences 
        from their abstracts that best represent their contribution.

        Papers:
        {papers_text}

        Return ONLY the paper numbers and their 2 most relevant sentences in order, like this:
        [1] Sentence 1. Sentence 2.
        [2] Sentence 1. Sentence 2.
        etc.
        """
        
        summary_response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": summary_prompt}],
            **self.model_params
        )
        
        return summary_response.choices[0].message.content.strip()

    def generate_ordered_bibliography(self, papers: List[Dict]) -> str:
        """Generate bibliography entries in numbered format."""
        bibliography = []
        for i, paper in enumerate(papers, 1):
            authors = paper['authors'] if isinstance(paper['authors'], str) else ", ".join(paper['authors'])
            title = paper['title']
            journal = paper.get('journal', 'Unknown Journal')
            date = paper.get('date', 'n.d.')
            doi = paper.get('doi', '')
            
            entry = f"[{i}] {authors} ({date}). {title}. {journal}."
            if doi:
                entry += f" https://doi.org/{doi}"
            bibliography.append(entry)
        
        return "\n".join(bibliography)

    def show_sidebar_references(self, bibliography: str, papers: List[Dict]):
        """
        Renders a sidebar with references using the bibliography order.
        """
        st.sidebar.title("References Used")
        
        # Regular expression to extract titles
        pattern = r"\(\d{4}\)\.\s(.*?)\.\s\*"
        
        # Find all matches
        titles = re.findall(pattern, bibliography)
        
        # Process each title and find matching papers
        for i, title in enumerate(titles, 1):
            # Find matching paper
            paper = next(
                (p for p in papers if p['title'].lower() in title.lower() 
                 or title.lower() in p['title'].lower()),
                None
            )
            
            if paper:
                # Display reference in sidebar
                with st.sidebar.expander(f"[{i}] {paper['title']}"):
                    st.markdown(f"**Authors:** {paper['authors']}")
                    st.markdown(f"**Year:** {paper['date']}")
                    
                    # Handle different URL formats
                    url = None
                    if paper.get('doi'):
                        url = f"https://doi.org/{paper['doi']}"
                    elif paper.get('pubmed_id'):
                        url = f"https://pubmed.ncbi.nlm.nih.gov/{paper['pubmed_id']}"
                    elif paper.get('url'):
                        url = paper['url']
                        
                    if url:
                        st.markdown(f"[Link to Paper]({url})")
                    
                    st.write("---")
                    
                    # Generate summary from abstract
                    abstract = paper.get('abstract', '').strip()
                    if abstract:
                        summary = '. '.join(abstract.split('. ')[:2])
                        if len(summary) > 200:
                            summary = summary[:200] + '...'
                    else:
                        summary = 'No abstract available'
                        
                    st.write(f"**Key Points:** {summary}")
