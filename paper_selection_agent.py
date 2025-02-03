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
        if df.empty and not scholar_results:
            st.warning("No papers found in the search results.")
            return pd.DataFrame()
        
        # Ensure input DataFrame has consistent columns
        required_columns = ['title', 'authors', 'date', 'journal', 'abstract']
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''
        
        # Convert scholar results to DataFrame
        scholar_papers = []
        for result in scholar_results:
            if result and result.get('title') and result.get('abstract'):
                scholar_papers.append({
                    'title': result.get('title', ''),
                    'authors': result.get('authors', ''),
                    'date': result.get('year', ''),
                    'journal': result.get('journal', 'Unknown Journal'),
                    'abstract': result.get('abstract', ''),
                    'pubmed_id': result.get('url', '')
                })
        
        # Create scholar DataFrame with consistent columns
        scholar_df = pd.DataFrame(scholar_papers)
        for col in required_columns:
            if col not in scholar_df.columns:
                scholar_df[col] = ''
        
        # Create scholar DataFrame and combine with input DataFrame
        combined_df = pd.concat([df, scholar_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['title']).reset_index(drop=True)
        
        # Show original combined DataFrame
        with st.expander("Original Combined Papers Dataset"):
            display_df = combined_df[['title', 'authors', 'date', 'journal', 'abstract']]
            st.dataframe(display_df)
            st.write(f"Total papers found: {len(combined_df)}")
        
        # Find exact matches in combined df
        topic_words = set(topic.lower().split())
        exact_matches = combined_df.apply(
            lambda row: (all(word in row['title'].lower() for word in topic_words) or 
                        all(word in row['abstract'].lower() for word in topic_words)),
            axis=1
        )
        
        # Split dataframe into exact matches and other papers
        exact_match_df = combined_df[exact_matches]
        other_papers_df = combined_df[~exact_matches]
        
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
                line = line.strip()
                # Skip empty lines or lines without ':'
                if not line or ':' not in line:
                    continue
                # Split only on the first occurrence of ':'
                parts = line.split(':', 1)
                if len(parts) == 2:
                    idx, val = parts
                    # Only process if both parts are valid integers
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
        final_df = pd.concat([exact_match_df, selected_other_papers])
        final_df = final_df.drop_duplicates(subset=['title'], keep='first').reset_index(drop=True)
        
        # Ensure abstract column exists
        if 'abstract' not in final_df.columns:
            final_df['abstract'] = ''
        
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