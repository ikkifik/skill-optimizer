"""Baseline evaluator for measuring skill performance before optimization."""

import dspy
from typing import Callable, Optional
from app.models.skill import Skill, SkillMetrics
from app.pipeline.signature_generator import DSPySignatureGenerator
import time


class BaselineEvaluator:
    """
    Evaluates a skill's baseline performance on a test set.
    
    This provides the "before" metrics for comparison after optimization.
    """
    
    def __init__(
        self,
        skill: Skill,
        metric_fn: Optional[Callable] = None,
        model: Optional[str] = None
    ):
        self.skill = skill
        self.metric_fn = metric_fn or self._default_metric
        self.model = model
        
        # Generate DSPy components
        self.generator = DSPySignatureGenerator(skill)
        self.module = self.generator.generate_module()
        self.examples = self.generator.generate_examples()
    
    def _default_metric(self, example: dspy.Example, prediction, trace=None) -> float:
        """
        Default metric function that checks field presence and basic matching.
        
        Override with custom metric for better evaluation.
        """
        score = 0.0
        total_fields = 0
        
        # Get expected output fields
        for key in example.keys():
            if key not in [f for f in example.inputs()]:
                total_fields += 1
                expected = getattr(example, key, None)
                predicted = getattr(prediction, key, None)
                
                if predicted is not None:
                    score += 0.5  # Partial credit for having the field
                    
                    # Check for partial match
                    if expected and predicted:
                        expected_str = str(expected).lower()
                        predicted_str = str(predicted).lower()
                        
                        # Exact match
                        if expected_str == predicted_str:
                            score += 0.5
                        # Partial containment
                        elif expected_str in predicted_str or predicted_str in expected_str:
                            score += 0.25
        
        return score / max(total_fields, 1)
    
    def evaluate(self, test_examples: Optional[list[dspy.Example]] = None) -> SkillMetrics:
        """
        Run baseline evaluation on test examples.
        
        Args:
            test_examples: Optional separate test set. Uses skill examples if None.
            
        Returns:
            SkillMetrics with baseline scores
        """
        examples = test_examples or self.examples
        
        if not examples:
            return SkillMetrics(
                accuracy=0.0,
                format_score=0.0,
                sample_size=0,
                evaluated_on=self.model
            )
        
        total_score = 0.0
        format_scores = []
        latencies = []
        token_counts = []
        
        for example in examples:
            # Get input fields
            input_data = {k: getattr(example, k) for k in example.inputs()}
            
            # Time the prediction
            start = time.time()
            try:
                prediction = self.module(**input_data)
                latency = (time.time() - start) * 1000
                latencies.append(latency)
                
                # Calculate accuracy using metric function
                score = self.metric_fn(example, prediction)
                total_score += score
                
                # Calculate format score (presence of expected fields)
                output_keys = [k for k in example.keys() if k not in example.inputs()]
                present = sum(1 for k in output_keys if hasattr(prediction, k) and getattr(prediction, k))
                format_scores.append(present / max(len(output_keys), 1))
                
            except Exception as e:
                print(f"Evaluation error: {e}")
                total_score += 0.0
                format_scores.append(0.0)
        
        return SkillMetrics(
            accuracy=total_score / len(examples),
            format_score=sum(format_scores) / len(format_scores) if format_scores else 0.0,
            latency_ms=sum(latencies) / len(latencies) if latencies else None,
            token_usage=None,  # Would need LLM tracking to capture
            evaluated_on=self.model,
            sample_size=len(examples)
        )


def create_accuracy_metric(expected_fields: list[str]) -> Callable:
    """
    Factory to create accuracy metrics for specific output fields.
    
    Args:
        expected_fields: List of output field names to verify
        
    Returns:
        A metric function for DSPy evaluation
    """
    def metric(example: dspy.Example, prediction, trace=None) -> float:
        score = 0.0
        
        for field in expected_fields:
            expected = getattr(example, field, None)
            predicted = getattr(prediction, field, None)
            
            if expected is None or predicted is None:
                continue
                
            expected_str = str(expected).lower()
            predicted_str = str(predicted).lower()
            
            if expected_str == predicted_str:
                score += 1.0
            elif expected_str in predicted_str:
                score += 0.5
        
        return score / len(expected_fields) if expected_fields else 0.0
    
    return metric
