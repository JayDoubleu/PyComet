from dataclasses import dataclass


@dataclass
class ModelUsage:
    """Class to track LLM usage statistics"""

    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    calls: int = 0
    char_count: int = 0
