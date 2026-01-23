"""Comparison models for before/after skill optimization analysis."""

from pydantic import BaseModel, Field
from typing import Optional
from .skill import SkillMetrics


class MetricDelta(BaseModel):
    """Change in a single metric between original and optimized skill."""
    
    metric_name: str
    original_value: float
    optimized_value: float
    absolute_change: float
    percent_change: float
    improved: bool


class ComparisonResult(BaseModel):
    """Full comparison between original and optimized skill."""
    
    skill_name: str
    original_version: str
    optimized_version: str
    
    original_metrics: SkillMetrics
    optimized_metrics: SkillMetrics
    
    deltas: list[MetricDelta] = Field(default_factory=list)
    
    overall_improvement: float = Field(
        0.0, 
        description="Weighted overall improvement percentage"
    )
    
    optimization_strategy: str = Field(
        "bootstrap_fewshot",
        description="Strategy used for optimization"
    )
    
    target_model: Optional[str] = Field(None, description="Model optimized for")
    
    def to_markdown_table(self) -> str:
        """Generate markdown comparison table."""
        lines = [
            f"## Skill: {self.skill_name}",
            "",
            "| Metric | Original | Optimized | Δ Change |",
            "|--------|----------|-----------|----------|"
        ]
        for delta in self.deltas:
            sign = "+" if delta.improved else ""
            lines.append(
                f"| {delta.metric_name} | {delta.original_value:.2f} | "
                f"{delta.optimized_value:.2f} | {sign}{delta.percent_change:.1f}% |"
            )
        lines.append("")
        lines.append(f"**Overall Improvement: {self.overall_improvement:.1f}%**")
        return "\n".join(lines)
    
    @classmethod
    def from_metrics(
        cls,
        skill_name: str,
        original: SkillMetrics,
        optimized: SkillMetrics,
        strategy: str = "bootstrap_fewshot",
        target_model: Optional[str] = None
    ) -> "ComparisonResult":
        """Create comparison from two SkillMetrics objects."""
        deltas = []
        
        # Compare each metric
        metrics_to_compare = [
            ("accuracy", original.accuracy, optimized.accuracy, True),
            ("format_score", original.format_score, optimized.format_score, True),
        ]
        
        if original.latency_ms and optimized.latency_ms:
            metrics_to_compare.append(
                ("latency_ms", original.latency_ms, optimized.latency_ms, False)
            )
        
        if original.token_usage and optimized.token_usage:
            metrics_to_compare.append(
                ("token_usage", float(original.token_usage), float(optimized.token_usage), False)
            )
        
        total_improvement = 0.0
        for name, orig, opt, higher_is_better in metrics_to_compare:
            if orig == 0:
                pct = 100.0 if opt > 0 else 0.0
            else:
                pct = ((opt - orig) / abs(orig)) * 100
            
            improved = (pct > 0) if higher_is_better else (pct < 0)
            
            deltas.append(MetricDelta(
                metric_name=name,
                original_value=orig,
                optimized_value=opt,
                absolute_change=opt - orig,
                percent_change=pct if higher_is_better else -pct,
                improved=improved
            ))
            
            # Weight accuracy higher
            weight = 2.0 if name == "accuracy" else 1.0
            total_improvement += (pct if higher_is_better else -pct) * weight
        
        overall = total_improvement / sum(2.0 if d.metric_name == "accuracy" else 1.0 for d in deltas)
        
        return cls(
            skill_name=skill_name,
            original_version="1.0.0",
            optimized_version="2.0.0",
            original_metrics=original,
            optimized_metrics=optimized,
            deltas=deltas,
            overall_improvement=overall,
            optimization_strategy=strategy,
            target_model=target_model
        )
