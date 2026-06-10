import numpy as np
from typing import List

class PerplexityScorer:
    """
    Module 2: LLM perplexity based anomaly detection.
    Computes token-level perplexity of documents under a reference language model.
    Poisoned documents (e.g. containing random strings, patterns, or unnatural text)
    often show significantly different perplexity profiles.
    """
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """
        Load reference language model and tokenizer.
        """
        pass

    def score(self, documents: List[str]) -> np.ndarray:
        """
        Calculate perplexity score for each document.
        """
        return np.zeros(len(documents))

    def predict(self, documents: List[str], threshold: float) -> List[bool]:
        """
        Predict whether each document is poisoned (True) or clean (False) based on perplexity.
        """
        scores = self.score(documents)
        return [score > threshold for score in scores]
