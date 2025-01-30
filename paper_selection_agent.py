import openai
import pandas as pd
from typing import List, Dict
import streamlit as st
import datetime

class PaperSelectionAgent:
    def __init__(self, openai_api_key: str, model="gpt-4o-mini", model_params=None, num_papers: int = 40):
        openai.api_key = openai_api_key
        self.model = model
        self.model_params = model_params if model_params is not None else ({"temperature": 0} if "gpt" in model else {})
        self.num_papers = num_papers
    
    def select_papers(self, df: pd.DataFrame, scholar_results: List[Dict], topic: str) -> pd.DataFrame:
        if df.empty:
            st.warning("No papers found in the search results.")
            return df
        
        # Show original DataFrame
        with st.expander("Original Papers Dataset"):
            display_df = df[['title', 'authors', 'date', 'journal', 'abstract']]
            st.dataframe(display_df)
            st.write(f"Total papers found: {len(df)}")
        
        # Convert scholar results to DataFrame, excluding entries with missing data
        scholar_papers = []
        for result in scholar_results:
            if result['title'] and result['authors'] and result['abstract']:
                # Extract year from date string if available, otherwise use current year
                year = result.get('year')
                if not year:  # If year is None or empty string
                    year = 'No year given'
                
                scholar_papers.append({
                    'title': result['title'],
                    'authors': result['authors'],
                    'date': str(year),  # Convert to string to match PubMed format
                    'journal': result.get('journal', 'Unknown Journal'),
                    'abstract': result['abstract'],
                    'pubmed_id': result['url']
                })
        
        scholar_df = pd.DataFrame(scholar_papers)
        
        # Find exact matches in original df
        topic_words = set(topic.lower().split())
        exact_matches = df.apply(
            lambda row: (all(word in row['title'].lower() for word in topic_words) or 
                        all(word in row['abstract'].lower() for word in topic_words)),
            axis=1
        )
        
        # Split dataframe into exact matches and other papers
        exact_match_df = df[exact_matches]
        other_papers_df = df[~exact_matches]
        
        exact_match_count = len(exact_match_df)
        st.write(f"Found {exact_match_count} exact matches")
        
        # Calculate remaining papers needed
        remaining_papers = max(0, self.num_papers - exact_match_count)
        
        if remaining_papers > 0 and not other_papers_df.empty:
            # Format remaining papers for prompt
            papers_text = self._format_papers_for_prompt(other_papers_df)
            
            prompt = f"""
            You are a research expert selecting and ordering papers for a literature review on:
            
            Topic: {topic}
            
            Note: {exact_match_count} exact matches have already been selected.
            You are now ranking the remaining less exact matches.
            
            For each paper, assign a number based on these rules:
            - Return 0 if the paper should be excluded ONLY IF it's completely irrelevant to the topic
            - For papers to include, assign numbers from 1 to {remaining_papers} based on importance
            - 1 is most important, {remaining_papers} is least important
            - Use each number only once
            - You MUST assign non-zero numbers to AT LEAST {remaining_papers} papers
            - Papers from the last 10 years should be prioritized, but don't exclude older papers if they're important
            
            Return ONLY the row numbers and their assigned values in this format:
            row_index:assigned_value
            Example:
            5:1
            12:2
            3:3
            8:0
            etc.

            Consider:
            1. Direct relevance to the topic (most important criterion)
            2. Scientific impact and importance
            3. Recency (especially papers from the last 10 years)
            4. Methodological significance
            
            Papers:
            {papers_text}
            """
            
            response = openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **self.model_params
            )
            
            # Parse response and get selected papers
            content = response.choices[0].message.content.strip()
            assignments = {}
            for line in content.split('\n'):
                if ':' in line:
                    idx, val = line.strip().split(':')
                    if idx.isdigit() and val.isdigit():
                        idx, val = int(idx), int(val)
                        if 0 <= idx < len(other_papers_df) and 0 <= val <= remaining_papers:
                            assignments[idx] = val
            
            # Filter and sort selected papers
            selected_indices = [(idx, val) for idx, val in assignments.items() if val > 0]
            selected_indices.sort(key=lambda x: x[1])
            ordered_indices = [idx for idx, _ in selected_indices]
            selected_other_papers = other_papers_df.iloc[ordered_indices]
        else:
            selected_other_papers = pd.DataFrame(columns=df.columns)
        
        # Combine all sources and remove duplicates
        final_df = pd.concat([exact_match_df, selected_other_papers, scholar_df])
        final_df = final_df.drop_duplicates(subset=['title'], keep='first').reset_index(drop=True)
        
        # Show filtered DataFrame
        with st.expander("Selected Papers for Review (Ordered by Importance)"):
            display_filtered_df = final_df[['title', 'authors', 'date', 'journal', 'abstract']]
            st.dataframe(display_filtered_df)
            st.write(f"Papers selected: {len(final_df)} (including {exact_match_count} exact matches)")
        
        return final_df
    
    def _format_papers_for_prompt(self, df: pd.DataFrame) -> str:
        papers_text = ""
        for idx, row in df.iterrows():
            papers_text += f"\nPaper {idx}:\n"
            papers_text += f"Title: {row['title']}\n"
            papers_text += f"Authors: {row['authors']}\n"
            papers_text += f"Date: {row['date']}\n"
            papers_text += f"Abstract: {row['abstract']}...\n---\n"
        return papers_text 