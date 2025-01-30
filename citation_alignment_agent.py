import openai
import re
from typing import Tuple

class CitationAlignmentAgent:
    def __init__(self, openai_api_key: str, model="gpt-4o-mini", model_params=None):
        openai.api_key = openai_api_key
        self.model = model
        self.model_params = model_params if model_params is not None else ({"temperature": 0} if "gpt" in model else {})
        
    def align_citations(self, review_text: str, bibliography: str) -> str:
        """Ensure citations in review text match the bibliography."""
        alignment_prompt = f"""
        You are a citation alignment expert. Your task is to ensure that all in-text citations 
        in the review match exactly with the bibliography entries, and that the content accurately 
        reflects the cited papers.

        Rules:
        1. DO NOT change the order of citations or bibliography entries
        2. DO NOT add or remove citations
        3. ONLY modify the review text if absolutely necessary to ensure accuracy with cited papers
        4. Ensure every citation number matches its corresponding bibliography entry
        5. Return the complete review with bibliography

        Review Text:
        {review_text}

        Bibliography:
        {bibliography}

        Return the complete review followed by the bibliography, with citations properly aligned.
        """
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": alignment_prompt}],
            **self.model_params
        )
        
        return response.choices[0].message.content.strip() 