import openai
import re
from typing import List, Dict, Tuple
import streamlit as st
import pandas as pd

class RefiningAgent:
    def __init__(self, openai_api_key: str, model="gpt-4o-mini", model_params=None):
        openai.api_key = openai_api_key
        self.model = model
        self.model_params = model_params if model_params is not None else ({"temperature": 0} if "gpt" in model else {})
    
    def extract_citations(self, text: str) -> List[int]:
        """Extract all citation numbers including complex groups."""
        citations = []
        # Match both simple ranges [X-Y] and complex groups [X, Y-Z, W]
        for group in re.finditer(r'\[([\d\s,\-]+)\]', text):
            citation_group = group.group(1)
            # Split by comma for complex groups
            for part in citation_group.split(','):
                part = part.strip()
                if '-' in part:
                    # Handle ranges
                    start, end = map(int, part.split('-'))
                    citations.extend(range(start, end + 1))
                else:
                    # Handle single numbers
                    citations.append(int(part))
        return citations

    def verify_citations(self, review_text: str, bibliography: str) -> bool:
        """Verify that all citations in the review match the bibliography."""
        citations = self.extract_citations(review_text)
        # Split bibliography on newlines and filter empty lines
        bib_entries = [entry for entry in bibliography.strip().split('\n') if entry.strip()]
        max_citation = len(bib_entries)
        print(f"Citations found: {citations}")
        print(f"Bibliography entries: {len(bib_entries)}")
        return all(1 <= c <= max_citation for c in citations)

    def refine_review(self, review_text: str, bibliography: str) -> Tuple[str, str, str]:
        """Convert author-date citations to numbered citations and enhance groupings."""
        
        refinement_prompt = f"""
        You are an expert academic editor. Your task is to convert this author-date citation style review into a numbered citation format.
        
        IMPORTANT INSTRUCTIONS:
        1. Convert ALL author-date citations (e.g., "Smith et al., 2020" or "(Jones, 2019)") to numbered format
        2. Group related citations using formats like [1-3] or [1, 3-5, 7]
        3. Include author mentions naturally (e.g., "Smith et al. [3] showed...")
        4. Ensure citations appear in ascending order throughout the text
        5. Generate a new references section at the end that:
           - Lists all cited works in order of first appearance
           - Uses numbered format: [1] Authors. (Year). Title. Journal.
           - Includes EVERY paper you cited in the review
        6. Only edit the primary text of the literature review if it is neccessary to improve the use of the citation for that sentence.
        7. Ensure that the formatting of your refined review remains the same as the original review, using markdown for all headings.
        
        Current review:
        {review_text}
        
        Available references:
        {bibliography}
        
        Return the complete review WITH a numbered "References" section at the end.
        """
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": refinement_prompt}],
            **self.model_params
        )
        
        refined_text = response.choices[0].message.content.strip()
        
        # Extract bibliography from the refined text
        references_match = re.search(r'References\n-+\n(.*?)$', refined_text, re.DOTALL)
        if references_match:
            new_bibliography = references_match.group(1).strip()
            review_text = refined_text.replace(references_match.group(0), '').strip()
        else:
            new_bibliography = bibliography
            review_text = refined_text
        
        # Extract final citation order
        final_citations = self.extract_citations(review_text)
        citation_order = ",".join(map(str, sorted(set(final_citations))))
        
        return review_text, new_bibliography, citation_order 