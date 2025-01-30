import openai
from typing import List

class PhraseGenerationAgent:
    def __init__(self, openai_api_key: str, model="gpt-4o-mini", model_params=None):
        openai.api_key = openai_api_key
        self.model = model
        self.model_params = model_params if model_params is not None else ({"temperature": 0} if "gpt" in model else {})
        
    def generate_similar_phrases(self, topic: str, num_phrases: int = 20) -> List[str]:
        prompt = f"""
        Given this research topic, generate {num_phrases} alternative search phrases that would help find relevant papers.
        The phrases should be similar in meaning but use different terminology or focus on different aspects.
        Each phrase should be 3-6 words long. Try to include commonly used phrases within the field that are relevant to the topic.
        Try to use phrases that reflect different aspects of the main topic but still use common enough keywords to get results from the search.
        Return exactly {num_phrases} phrases, one per line.
        
        Research Topic: {topic}
        """
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **self.model_params
        )
        
        phrases = [p.strip() for p in response.choices[0].message.content.strip().split('\n')]
        # Replace spaces with + for URL compatibility
        phrases = [phrase.replace(' ', '+') for phrase in phrases]
        return phrases[:num_phrases]  # Ensure we return exactly num_phrases 