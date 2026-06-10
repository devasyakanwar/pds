import numpy as np
from typing import List, Dict, Any

class PDSAggregator:
    """
    Composite scoring / aggregation for PDS modules.
    Aggregates anomaly scores from multiple detection modules to output a final prediction.
    """
    def __init__(self, weights: Dict[str, float] = None):
        # Default weights for composite scoring
        self.weights = weights or {
            "embedding": 0.25,
            "perplexity": 0.25,
            "instruction": 0.25,
            "hubness": 0.25
        }

    def aggregate_scores(self, scores: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute weighted sum/aggregation of normalized scores.
        """
        n_samples = next(iter(scores.values())).shape[0]
        final_scores = np.zeros(n_samples)
        for key, val in scores.items():
            weight = self.weights.get(key, 0.0)
            final_scores += weight * val
        return final_scores

    def predict(self, aggregated_scores: np.ndarray, threshold: float = 0.5) -> List[bool]:
        """
        Make final prediction based on aggregated scores.
        """
        return [score > threshold for score in aggregated_scores]
