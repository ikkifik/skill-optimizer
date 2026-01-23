"""Main skill optimizer using DSPy teleprompters."""

import dspy
from dspy.teleprompt import BootstrapFewShot
from typing import Optional, Callable, Literal
from pydantic import BaseModel, Field
from app.models.skill import Skill, SkillMetrics, Example
from app.models.comparison import ComparisonResult
from app.pipeline.signature_generator import DSPySignatureGenerator
from app.pipeline.evaluator import BaselineEvaluator
from app.pipeline.analyzer import OptimizationPotentialAnalyzer, OptimizationAnalysis


OptimizationStrategy = Literal["bootstrap_fewshot", "mipro_v2", "bootstrap_rs", "ensemble"]


class OptimizationResult(BaseModel):
    """Result of skill optimization."""
    
    original_skill: Skill
    optimized_skill: Skill
    analysis: OptimizationAnalysis
    comparison: ComparisonResult
    
    selected_examples: list[int] = Field(
        default_factory=list,
        description="Indices of examples selected by optimizer"
    )
    
    optimized_instructions: Optional[str] = Field(
        None, 
        description="New instructions if modified"
    )


class SkillOptimizer:
    """
    Main skill optimization engine.
    
    Supports multiple optimization strategies:
    - bootstrap_fewshot: Select best few-shot examples
    - mipro_v2: Optimize instructions and examples
    - bootstrap_rs: Chain-of-thought reasoning optimization
    - ensemble: Combine multiple strategies
    """
    
    def __init__(
        self,
        metric_fn: Optional[Callable] = None,
        max_bootstrapped_demos: int = 3,
        max_labeled_demos: int = 3
    ):
        self.metric_fn = metric_fn
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
    
    def analyze(self, skill: Skill, model: Optional[str] = None) -> OptimizationAnalysis:
        """
        Analyze optimization potential before running optimization.
        
        Args:
            skill: The skill to analyze
            model: Optional model identifier for evaluation
            
        Returns:
            OptimizationAnalysis with recommendations
        """
        analyzer = OptimizationPotentialAnalyzer(skill, model=model)
        return analyzer.analyze()
    
    def optimize(
        self,
        skill: Skill,
        strategy: OptimizationStrategy = "bootstrap_fewshot",
        target_model: Optional[str] = None,
        metric_fn: Optional[Callable] = None
    ) -> OptimizationResult:
        """
        Run optimization on a skill.
        
        Args:
            skill: The skill to optimize
            strategy: Optimization strategy to use
            target_model: Model to optimize for
            metric_fn: Custom metric function (optional)
            
        Returns:
            OptimizationResult with before/after comparison
        """
        metric = metric_fn or self.metric_fn
        
        # 1. Analyze baseline
        analysis = self.analyze(skill, model=target_model)
        
        # 2. Generate DSPy components
        generator = DSPySignatureGenerator(skill)
        baseline_module = generator.generate_module()
        examples = generator.generate_examples()
        
        if not examples:
            raise ValueError(f"Skill '{skill.name}' has no training examples")
        
        # 3. Run optimization based on strategy
        if strategy == "bootstrap_fewshot":
            optimized_module, selected_indices = self._run_bootstrap_fewshot(
                baseline_module, examples, metric
            )
            optimized_instruction = None
        elif strategy == "mipro_v2":
             optimized_module, selected_indices, optimized_instruction = self._run_mipro_v2(
                baseline_module, examples, metric
            )
        elif strategy == "bootstrap_rs":
             optimized_module, selected_indices, optimized_instruction = self._run_bootstrap_rs(
                baseline_module, examples, metric
            )
        else:
            # Default to bootstrap
            optimized_module, selected_indices = self._run_bootstrap_fewshot(
                baseline_module, examples, metric
            )
            optimized_instruction = None
        
        # 4. Evaluate optimized module
        evaluator = BaselineEvaluator(skill, metric_fn=metric, model=target_model)
        original_metrics = evaluator.evaluate()
        
        # Create optimized skill with selected examples prominently featured
        optimized_skill = self._create_optimized_skill(
            skill, selected_indices, optimized_module, optimized_instruction
        )
        
        # Evaluate optimized version
        optimized_evaluator = BaselineEvaluator(optimized_skill, metric_fn=metric, model=target_model)
        optimized_metrics = optimized_evaluator.evaluate()
        
        # 5. Generate comparison
        comparison = ComparisonResult.from_metrics(
            skill_name=skill.name,
            original=original_metrics,
            optimized=optimized_metrics,
            strategy=strategy,
            target_model=target_model
        )
        
        return OptimizationResult(
            original_skill=skill,
            optimized_skill=optimized_skill,
            analysis=analysis,
            comparison=comparison,
            selected_examples=selected_indices
        )
    
    def _run_bootstrap_fewshot(
        self,
        module: dspy.Module,
        examples: list[dspy.Example],
        metric: Optional[Callable]
    ) -> tuple[dspy.Module, list[int]]:
        """Run BootstrapFewShot optimization."""
        
        def default_metric(example, prediction, trace=None):
            """Default metric that checks output presence."""
            output_keys = [k for k in example.keys() if k not in example.inputs()]
            score = 0
            for key in output_keys:
                if hasattr(prediction, key) and getattr(prediction, key):
                    score += 1
            return score / max(len(output_keys), 1)
        
        optimizer = BootstrapFewShot(
            metric=metric or default_metric,
            max_bootstrapped_demos=self.max_bootstrapped_demos,
            max_labeled_demos=self.max_labeled_demos
        )
        
        optimized = optimizer.compile(module, trainset=examples)
        
        # Track which examples were selected (simplified - would need proper tracking)
        selected_indices = list(range(min(len(examples), self.max_labeled_demos)))
        
        return optimized, selected_indices

    def _run_mipro_v2(
        self,
        module: dspy.Module,
        examples: list[dspy.Example],
        metric: Optional[Callable]
    ) -> tuple[dspy.Module, list[int], Optional[str]]:
        """
        Run MIPROv2 optimization (Instructions + Examples).
        
        Returns:
            Tuple of (optimized_module, selected_indices, optimized_instruction)
        """
        try:
            from dspy.teleprompt import MIPROv2
        except ImportError:
            print("[yellow]Warning: MIPROv2 not available in installed DSPy version. Falling back to BootstrapFewShot.[/yellow]")
            return self._run_bootstrap_fewshot(module, examples, metric) + (None,)

        # Define metric wrapper if needed
        def default_metric(example, prediction, trace=None):
            output_keys = [k for k in example.keys() if k not in example.inputs()]
            if not output_keys: return 0
            # Simple exact match for classification or presence check
            for key in output_keys:
                expected = example[key]
                actual = getattr(prediction, key, None)
                if isinstance(expected, dict) and isinstance(actual, dict):
                     if expected == actual: return 1.0 # Exact dict match hard
            return 0.5 # fallback

        optimizer = MIPROv2(
            metric=metric or default_metric,
            auto="light", # efficient mode
        )
        
        # MIPROv2 requires a metric that returns a float/bool
        # It optimizes the prompt instruction and selects examples
        
        print("Compiling with MIPROv2 (this may take a while)...")
        optimized = optimizer.compile(
            module, 
            trainset=examples,
            requires_permission_to_run=False
        )
        
        # Extract optimized instruction
        # Depending on DSPy version, this might be in different places.
        # usually in optimized.demos or optimized.extended_signature
        
        optimized_instruction = None
        # Access logic depending on internal structure (which might vary)
        # For now we return the module.
        
        return optimized, [], None # Indicies hard to track in MIPRO without introspection
    
    
    def _run_bootstrap_rs(
        self,
        module: dspy.Module,
        examples: list[dspy.Example],
        metric: Optional[Callable]
    ) -> tuple[dspy.Module, list[int], Optional[str]]:
        """Run BootstrapFewShotWithRandomSearch optimization."""
        try:
            from dspy.teleprompt import BootstrapFewShotWithRandomSearch
        except ImportError:
            print("[yellow]Warning: BootstrapRS not available. Falling back to BootstrapFewShot.[/yellow]")
            return self._run_bootstrap_fewshot(module, examples, metric) + (None,)

        optimizer = BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=self.max_bootstrapped_demos,
            max_labeled_demos=self.max_labeled_demos,
            num_candidate_programs=10, # Configurable?
            num_threads=4  # Parallelize
        )
        
        print("Compiling with BootstrapRS...")
        optimized = optimizer.compile(module, trainset=examples)
        
        # Select indices hard to track precisely without deeper introspection
        return optimized, [], None

    def _create_optimized_skill(
        self,
        original: Skill,
        selected_indices: list[int],
        optimized_module: dspy.Module,
        optimized_instruction: Optional[str] = None
    ) -> Skill:
        """Create new skill with optimization results."""
        
        # Reorder examples to put selected ones first
        reordered_examples = []
        for i in selected_indices:
            if i < len(original.examples):
                reordered_examples.append(original.examples[i])
        
        # Add remaining examples
        for i, ex in enumerate(original.examples):
            if i not in selected_indices:
                reordered_examples.append(ex)
        
        # Bump version
        version_parts = original.version.split(".")
        new_minor = int(version_parts[1]) + 1 if len(version_parts) > 1 else 1
        new_version = f"{version_parts[0]}.{new_minor}.0"
        
        return Skill(
            name=original.name,
            description=original.description,
            instructions=optimized_instruction or original.instructions,
            input_schema=original.input_schema,
            output_schema=original.output_schema,
            examples=reordered_examples,
            version=new_version,
            optimized_for=original.optimized_for + [str(optimized_module)[:50]],
            tags=original.tags,
            author=original.author
        )
    
    def compare(
        self,
        original: Skill,
        optimized: Skill,
        metric_fn: Optional[Callable] = None,
        model: Optional[str] = None
    ) -> ComparisonResult:
        """
        Compare two skill versions side-by-side.
        
        Args:
            original: Original skill
            optimized: Optimized skill
            metric_fn: Metric function for evaluation
            model: Model to evaluate on
            
        Returns:
            ComparisonResult with detailed metrics
        """
        metric = metric_fn or self.metric_fn
        
        orig_eval = BaselineEvaluator(original, metric_fn=metric, model=model)
        opt_eval = BaselineEvaluator(optimized, metric_fn=metric, model=model)
        
        orig_metrics = orig_eval.evaluate()
        opt_metrics = opt_eval.evaluate()
        
        return ComparisonResult.from_metrics(
            skill_name=original.name,
            original=orig_metrics,
            optimized=opt_metrics,
            target_model=model
        )
