import re
from typing import List

class InstructionDetector:
    """
    Module 3: Instruction injection detector.
    Identifies hidden or explicit instruction patterns, prompt injection markers,
    or adversarial instruction structures within the text data.
    """
    def __init__(self, patterns: List[str] = None):
        # Common prompt injection / instruction patterns
        self.patterns = patterns or [
            r"(?i)ignore\s+(?:all\s+)?previous\s+instructions",
            r"(?i)system\s*prompt",
            r"(?i)you\s+must\s+(?:now\s+)?act\s+as",
            r"(?i)new\s+role",
            r"(?i)translate\s+this\s+and\s+execute",
            r"(?i)delete\s+all\s+files",
        ]

    def add_pattern(self, pattern: str):
        """
        Add a custom regular expression pattern to detect.
        """
        self.patterns.append(pattern)

    def score(self, documents: List[str]) -> List[float]:
        """
        Calculate injection score based on matched patterns and their intensities.
        """
        scores = []
        for doc in documents:
            match_count = 0
            for pattern in self.patterns:
                match_count += len(re.findall(pattern, doc))
            scores.append(float(match_count))
        return scores

    def predict(self, documents: List[str], threshold: float = 0.5) -> List[bool]:
        """
        Predict whether each document is poisoned (True) or clean (False).
        """
        scores = self.score(documents)
        return [score > threshold for score in scores]
