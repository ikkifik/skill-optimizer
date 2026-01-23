"""Optimization potential analyzer - estimates improvement ceiling before optimizing."""

from typing import Optional
from pydantic import BaseModel, Field
from app.models.skill import Skill, SkillMetrics
from app.pipeline.evaluator import BaselineEvaluator
import statistics


class OptimizationAnalysis(BaseModel):
    """Analysis of a skill's optimization potential."""
    
    skill_name: str
    baseline_score: float = Field(..., description="Current accuracy score")
    estimated_ceiling: float = Field(..., description="Theoretical maximum score")
    estimated_improvement: float = Field(..., description="Expected improvement percentage")
    
    variance: float = Field(0.0, description="Score variance across examples")
    weakest_areas: list[str] = Field(default_factory=list, description="Areas with most room for improvement")
    
    recommendation: str = Field(..., description="Optimization recommendation")
    recommended_strategy: str = Field("bootstrap_fewshot", description="Best optimization strategy")
    
    sample_size: int = Field(0, description="Number of examples analyzed")


class OptimizationPotentialAnalyzer:
    """
    Analyzes a skill to estimate optimization potential before running expensive optimization.
    
    This helps users understand:
    1. Current skill quality (baseline)
    2. How much improvement is theoretically possible
    3. Which aspects are weakest and need improvement
    4. Which optimization strategy to use
    """
    
    def __init__(self, skill: Skill, model: Optional[str] = None):
        self.skill = skill
        self.model = model
        self.evaluator = BaselineEvaluator(skill, model=model)
    
    def analyze(self) -> OptimizationAnalysis:
        """
        Run comprehensive analysis of optimization potential.
        
        Returns:
            OptimizationAnalysis with baseline scores and recommendations
        """
        # Run baseline evaluation
        baseline_metrics = self.evaluator.evaluate()
        
        # Calculate per-example scores for variance analysis
        per_example_scores = self._get_per_example_scores()
        
        variance = statistics.variance(per_example_scores) if len(per_example_scores) > 1 else 0.0
        
        # Estimate ceiling based on example quality and variance
        # Higher variance suggests more room for improvement via few-shot selection
        ceiling = self._estimate_ceiling(baseline_metrics.accuracy, variance)
        
        # Identify weak areas
        weak_areas = self._identify_weak_areas(baseline_metrics)
        
        # Determine recommendation
        recommendation, strategy = self._get_recommendation(
            baseline_metrics.accuracy,
            variance,
            len(self.skill.examples)
        )
        
        return OptimizationAnalysis(
            skill_name=self.skill.name,
            baseline_score=baseline_metrics.accuracy,
            estimated_ceiling=ceiling,
            estimated_improvement=((ceiling - baseline_metrics.accuracy) / max(baseline_metrics.accuracy, 0.01)) * 100,
            variance=variance,
            weakest_areas=weak_areas,
            recommendation=recommendation,
            recommended_strategy=strategy,
            sample_size=len(self.skill.examples)
        )
    
    def _get_per_example_scores(self) -> list[float]:
        """Calculate individual scores per example."""
        scores = []
        examples = self.evaluator.examples
        
        for example in examples:
            try:
                input_data = {k: getattr(example, k) for k in example.inputs()}
                prediction = self.evaluator.module(**input_data)
                score = self.evaluator.metric_fn(example, prediction)
                scores.append(score)
            except Exception:
                scores.append(0.0)
        
        return scores
    
    def _estimate_ceiling(self, baseline: float, variance: float) -> float:
        """
        Estimate theoretical maximum improvement.
        
        Higher variance suggests better examples exist that could be selected.
        Low baseline with high variance = high potential.
        """
        # Base ceiling from perfect score
        max_ceiling = 0.95  # Allow for some irreducible error
        
        # Variance contributes to potential (better examples may exist)
        variance_bonus = min(variance * 0.5, 0.2)
        
        # Gap from perfect
        gap = max_ceiling - baseline
        
        # Estimate achievable improvement (diminishing returns near ceiling)
        if baseline > 0.9:
            achievable = gap * 0.3  # Hard to improve near-perfect skills
        elif baseline > 0.7:
            achievable = gap * 0.5
        else:
            achievable = gap * 0.7 + variance_bonus
        
        return min(baseline + achievable, max_ceiling)
    
    def _identify_weak_areas(self, metrics: SkillMetrics) -> list[str]:
        """Identify which aspects need improvement."""
        weak = []
        
        if metrics.accuracy < 0.7:
            weak.append("task_accuracy")
        if metrics.format_score < 0.8:
            weak.append("output_format")
        if metrics.latency_ms and metrics.latency_ms > 2000:
            weak.append("response_time")
        if len(self.skill.examples) < 5:
            weak.append("training_examples")
        
        return weak
    
    def _get_recommendation(
        self,
        baseline: float,
        variance: float,
        num_examples: int
    ) -> tuple[str, str]:
        """Generate optimization recommendation and strategy."""
        
        if baseline > 0.9:
            return ("Skill is already highly optimized. Minor gains possible.", "mipro_v2")
        
        if num_examples < 3:
            return (
                "Add more training examples (5-10 recommended) before optimization.",
                "bootstrap_fewshot"
            )
        
        if variance > 0.1:
            return (
                "High variance suggests few-shot selection will help significantly. "
                "BootstrapFewShot recommended.",
                "bootstrap_fewshot"
            )
        
        if baseline < 0.5:
            return (
                "Low baseline suggests fundamental instruction issues. "
                "Try MIPROv2 for instruction optimization.",
                "mipro_v2"
            )
        
        return (
            "Good candidate for optimization. BootstrapFewShot should yield 10-20% improvement.",
            "bootstrap_fewshot"
        )
