from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict

from pycomet.models import ModelUsage


@dataclass
class UsageMetricsCollector:
    """Collects and displays usage metrics for AI model tests."""

    usages: Dict[str, ModelUsage] = field(
        default_factory=lambda: defaultdict(ModelUsage)
    )
    skipped_models: set = field(default_factory=set)

    def add_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        char_count: int,
    ) -> None:
        usage = self.usages[model]
        usage.input_tokens += input_tokens
        usage.output_tokens += output_tokens
        usage.total_cost += cost
        usage.calls += 1
        usage.char_count += char_count

    def add_skipped(self, model: str) -> None:
        """Track skipped models"""
        self.skipped_models.add(model)

    def display_summary(self) -> None:
        if not self.usages and not self.skipped_models:
            print("\nNo tests were run!")
            return

        print("\n" + "=" * 120)
        print("📊 Model Usage Summary")
        print("=" * 120)

        if self.usages:
            headers = [
                "Model",
                "Calls",
                "Avg Chars",
                "Input Tokens",
                "Output Tokens",
                "Total Tokens",
                "Cost ($)",
            ]
            row_format = "{:<50} {:>6} {:>10.1f} {:>14} {:>14} {:>14} {:>8.4f}"
            header_format = "{:<50} {:>6} {:>10} {:>14} {:>14} {:>14} {:>8}"

            print(header_format.format(*headers))
            print("-" * 120)

            total_cost = 0.0
            for model, usage in sorted(self.usages.items()):
                model_display = model[:47] + "..." if len(model) > 50 else model
                total_tokens = usage.input_tokens + usage.output_tokens
                avg_chars = usage.char_count / usage.calls if usage.calls > 0 else 0
                print(
                    row_format.format(
                        model_display,
                        usage.calls,
                        avg_chars,
                        usage.input_tokens,
                        usage.output_tokens,
                        total_tokens,
                        usage.total_cost,
                    )
                )
                total_cost += usage.total_cost

            print("-" * 120)
            print(f"Total Cost: ${total_cost:.4f}")

        if self.skipped_models:
            print("\n⚠️  Skipped Models (missing API keys):")
            print("-" * 120)
            for model in sorted(self.skipped_models):
                print(f"- {model}")

        print("=" * 120 + "\n")
