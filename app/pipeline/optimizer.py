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


OptimizationStrategy = Literal[
    "bootstrap_fewshot",     # Basic few-shot example selection
    "mipro_v2",              # Instruction + example optimization
    "bootstrap_rs",          # Random search with bootstrapping
    "gepa",                  # Grounded Explanation-based Prompt Alignment
    "simba",                 # Signature-Based Multi-step Bootstrapping
    "better_together",       # Combines multiple optimization strategies
    "knn_fewshot",           # K-Nearest Neighbor example selection
    "copro",                 # Contrastive Prompt Optimization
    "ensemble",              # Ensemble of multiple optimized modules
    "auto",                  # Automatically select best strategy
]


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
    
    def _auto_select_strategy(self, skill: Skill, examples_count: int) -> str:
        """
        Automatically select the best optimization strategy based on skill characteristics.
        
        Logic:
        1. Low data (< 5 examples) -> bootstrap_fewshot (robust)
        2. Medium data (5-20) -> mipro_v2 (optimizes instructions + examples)
        3. Complex reasoning (detected in instructions) -> simba or bootstrap_rs
        4. Alignment focus -> gepa
        """
        # Detect complexity
        is_complex = any(term in skill.instructions.lower() 
                        for term in ["reasoning", "think step by step", "complex", "analyze"])
        
        print(f"[dim]Auto-detecting strategy for '{skill.name}' ({examples_count} examples)...[/dim]")
        
        if examples_count < 5:
            # Low data regime: BootstrapFewShot is most robust
            print("[dim]Low data detected (n<5). Selected: bootstrap_fewshot[/dim]")
            return "bootstrap_fewshot"
            
        if is_complex and examples_count >= 5:
            # Complex task with sufficient data: SIMBA or BootstrapRS
            # SIMBA is state-of-the-art for reasoning
            try:
                from dspy.teleprompt import SIMBA
                print("[dim]Complex task detected. Selected: simba[/dim]")
                return "simba"
            except ImportError:
                print("[dim]Complex task detected (SIMBA unavailable). Selected: bootstrap_rs[/dim]")
                return "bootstrap_rs"
                
        if examples_count >= 10:
            # High data regime: MIPROv2 is best for instruction optimization
            try:
                from dspy.teleprompt import MIPROv2
                print("[dim]High data detected (n>=10). Selected: mipro_v2[/dim]")
                return "mipro_v2"
            except ImportError:
                pass
                
        # Default fallback
        print("[dim]Default selection: bootstrap_fewshot[/dim]")
        return "bootstrap_fewshot"

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
        
        # Resolve automatic strategy selection
        if strategy == "auto":
            strategy = self._auto_select_strategy(skill, len(examples))
        
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
        elif strategy == "gepa":
            optimized_module, selected_indices, optimized_instruction = self._run_gepa(
                baseline_module, examples, metric
            )
        elif strategy == "simba":
            optimized_module, selected_indices, optimized_instruction = self._run_simba(
                baseline_module, examples, metric
            )
        elif strategy == "better_together":
            optimized_module, selected_indices, optimized_instruction = self._run_better_together(
                baseline_module, examples, metric
            )
        elif strategy == "knn_fewshot":
            optimized_module, selected_indices = self._run_knn_fewshot(
                baseline_module, examples, metric
            )
            optimized_instruction = None
        elif strategy == "copro":
            optimized_module, selected_indices, optimized_instruction = self._run_copro(
                baseline_module, examples, metric
            )
        elif strategy == "ensemble":
            optimized_module, selected_indices, optimized_instruction = self._run_ensemble(
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

    def _run_gepa(
        self,
        module: dspy.Module,
        examples: list[dspy.Example],
        metric: Optional[Callable]
    ) -> tuple[dspy.Module, list[int], Optional[str]]:
        """Run GEPA (Grounded Explanation-based Prompt Alignment) optimization."""
        try:
            # Note: GEPA might be in a different path depending on DSPy version
            # Assuming dspy.teleprompt.GEPA is available as per dir() check
            from dspy.teleprompt import GEPA
        except ImportError:
            try:
                # Try alternative import path if standard one fails
                from dspy.teleprompt.gepa import GEPA
            except ImportError:
                print("[yellow]Warning: GEPA not available. Falling back to BootstrapFewShot.[/yellow]")
                return self._run_bootstrap_fewshot(module, examples, metric) + (None,)

        optimizer = GEPA(
            metric=metric,
            max_bootstrapped_demos=self.max_bootstrapped_demos,
            max_labeled_demos=self.max_labeled_demos,
            num_candidate_programs=5,
            num_threads=4
        )
        
        print("Compiling with GEPA...")
        optimized = optimizer.compile(module, trainset=examples)
        return optimized, [], None

    def _run_simba(
        self,
        module: dspy.Module,
        examples: list[dspy.Example],
        metric: Optional[Callable]
    ) -> tuple[dspy.Module, list[int], Optional[str]]:
        """Run SIMBA (Signature-Based Multi-step Bootstrapping) optimization."""
        try:
            from dspy.teleprompt import SIMBA
        except ImportError:
            print("[yellow]Warning: SIMBA not available. Falling back to BootstrapFewShot.[/yellow]")
            return self._run_bootstrap_fewshot(module, examples, metric) + (None,)

        optimizer = SIMBA(
            metric=metric,
            max_bootstrapped_demos=self.max_bootstrapped_demos,
            max_labeled_demos=self.max_labeled_demos,
            num_threads=4
        )
        
        print("Compiling with SIMBA...")
        optimized = optimizer.compile(module, trainset=examples)
        return optimized, [], None

    def _run_better_together(
        self,
        module: dspy.Module,
        examples: list[dspy.Example],
        metric: Optional[Callable]
    ) -> tuple[dspy.Module, list[int], Optional[str]]:
        """Run BetterTogether optimization."""
        try:
            from dspy.teleprompt import BetterTogether
        except ImportError:
            print("[yellow]Warning: BetterTogether not available. Falling back to BootstrapFewShot.[/yellow]")
            return self._run_bootstrap_fewshot(module, examples, metric) + (None,)

        optimizer = BetterTogether(
            metric=metric,
            max_bootstrapped_demos=self.max_bootstrapped_demos,
            max_labeled_demos=self.max_labeled_demos,
        )
        
        print("Compiling with BetterTogether...")
        optimized = optimizer.compile(module, trainset=examples)
        return optimized, [], None

    def _run_knn_fewshot(
        self,
        module: dspy.Module,
        examples: list[dspy.Example],
        metric: Optional[Callable]
    ) -> tuple[dspy.Module, list[int]]:
        """Run KNNFewShot optimization."""
        try:
            from dspy.teleprompt import KNNFewShot
        except ImportError:
            print("[yellow]Warning: KNNFewShot not available. Falling back to BootstrapFewShot.[/yellow]")
            return self._run_bootstrap_fewshot(module, examples, metric)

        # KNN requires vectorizer setup usually, simplified here
        optimizer = KNNFewShot(
            k=self.max_labeled_demos,
            trainset=examples
        )
        
        print("Compiling with KNNFewShot...")
        optimized = optimizer.compile(module, trainset=examples)
        # KNN selects dynamically, so no static indices
        return optimized, []

    def _run_copro(
        self,
        module: dspy.Module,
        examples: list[dspy.Example],
        metric: Optional[Callable]
    ) -> tuple[dspy.Module, list[int], Optional[str]]:
        """Run COPRO (Contrastive Prompt Optimization)."""
        try:
            from dspy.teleprompt import COPRO
        except ImportError:
            print("[yellow]Warning: COPRO not available. Falling back to BootstrapFewShot.[/yellow]")
            return self._run_bootstrap_fewshot(module, examples, metric) + (None,)

        optimizer = COPRO(
            metric=metric,
            breadth=5,
            depth=3,
            track_to_use="score"
        )
        
        print("Compiling with COPRO...")
        optimized = optimizer.compile(module, trainset=examples)
        return optimized, [], None

    def _run_ensemble(
        self,
        module: dspy.Module,
        examples: list[dspy.Example],
        metric: Optional[Callable]
    ) -> tuple[dspy.Module, list[int], Optional[str]]:
        """Run Ensemble optimization."""
        try:
            from dspy.teleprompt import Ensemble
        except ImportError:
            print("[yellow]Warning: Ensemble not available. Falling back to BootstrapFewShot.[/yellow]")
            return self._run_bootstrap_fewshot(module, examples, metric) + (None,)

        # Ensemble usually wraps other optimizers or modules
        # A simple usage might be reducing a list of programs
        # Here we'll simulate by running BootstrapRS first then Ensembling (simplified)
        
        print("Ensemble strategy requires multiple programs. Running BootstrapRS to generate candidates...")
        # Note: Proper ensemble usage needs a list of compiled programs
        # Fallback to BootstrapRS for now as simple Ensemble logic is complex
        return self._run_bootstrap_rs(module, examples, metric)
