"""Skill data models with structured schemas for optimization."""

from pydantic import BaseModel, Field
from typing import Optional, Any
from pathlib import Path
import yaml
import json


class Example(BaseModel):
    """A training/few-shot example for skill optimization."""
    
    input: dict[str, Any] = Field(..., description="Input data for this example")
    expected_output: dict[str, Any] = Field(..., description="Expected output")
    reasoning: Optional[str] = Field(None, description="Optional chain-of-thought reasoning")


class SkillMetrics(BaseModel):
    """Quality metrics for a skill variant."""
    
    accuracy: float = Field(0.0, ge=0.0, le=1.0, description="Task accuracy score")
    format_score: float = Field(0.0, ge=0.0, le=1.0, description="Output format compliance")
    latency_ms: Optional[float] = Field(None, description="Average response latency in ms")
    token_usage: Optional[int] = Field(None, description="Average token consumption")
    evaluated_on: Optional[str] = Field(None, description="Model used for evaluation")
    sample_size: int = Field(0, description="Number of examples evaluated")


class Skill(BaseModel):
    """
    A structured skill definition for agent optimization.
    
    Skills are structured prompts that can be programmatically optimized
    using DSPy and benchmarked across different models.
    """
    
    name: str = Field(..., description="Unique skill identifier")
    description: str = Field(..., description="What this skill does")
    instructions: str = Field(..., description="Core instructions/prompt for the skill")
    
    # Schema definitions
    input_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema for skill inputs"
    )
    output_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema for skill outputs"
    )
    
    # Training and examples
    examples: list[Example] = Field(
        default_factory=list,
        description="Training/few-shot examples"
    )
    
    # Metrics and versioning
    metrics: Optional[SkillMetrics] = Field(None, description="Quality metrics")
    version: str = Field("1.0.0", description="Semantic version")
    optimized_for: list[str] = Field(
        default_factory=list,
        description="Model IDs this skill is optimized for"
    )
    
    # Metadata
    tags: list[str] = Field(default_factory=list, description="Categorization tags")
    author: Optional[str] = Field(None, description="Skill author")
    
    def to_yaml(self) -> str:
        """Serialize skill to YAML format."""
        return yaml.dump(self.model_dump(exclude_none=True), sort_keys=False, allow_unicode=True)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Skill":
        """Load skill from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def save(self, path: str | Path) -> None:
        """Save skill to file (supports .yaml and .md)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == ".md":
            from app.pipeline.parsers import MarkdownSkillParser
            MarkdownSkillParser.save(self, path)
        else:
            with open(path, "w") as f:
                f.write(self.to_yaml())
    
    @classmethod
    def load(cls, name: str, skills_dir: str = "skills") -> "Skill":
        """Load skill by name from skills directory (supports .yaml and .md)."""
        skill = None
        skill_dir = None
        
        # Try structured YAML first
        yaml_path = Path(skills_dir) / f"{name}" / "SKILL.yaml"
        if yaml_path.exists():
            skill = cls.from_yaml(yaml_path)
            skill_dir = yaml_path.parent
        else:
            flat_yaml = Path(skills_dir) / f"{name}.yaml"
            if flat_yaml.exists():
                skill = cls.from_yaml(flat_yaml)
                skill_dir = flat_yaml.parent

        if not skill:
            # Try Markdown format (Agent Skills)
            from app.pipeline.parsers import MarkdownSkillParser
            
            md_path = Path(skills_dir) / f"{name}" / "SKILL.md"
            if md_path.exists():
                skill = MarkdownSkillParser.parse(md_path)
                skill_dir = md_path.parent
            else:
                flat_md = Path(skills_dir) / f"{name}.md"
                if flat_md.exists():
                    skill = MarkdownSkillParser.parse(flat_md)
                    skill_dir = flat_md.parent

        if not skill:
            raise FileNotFoundError(f"Skill '{name}' not found (checked .yaml and .md)")

        # Load sidecar training data if available
        # Check standard location: TRAINING.json in skill dir
        training_path = skill_dir / "TRAINING.json"
        if not training_path.exists():
            # Check for name_training.json
            training_path = skill_dir / f"{name}_training.json"
        
        if training_path.exists():
            try:
                with open(training_path, "r") as f:
                    data = json.load(f)
                    # Support list of examples
                    if isinstance(data, list):
                        new_examples = []
                        for item in data:
                            # Normalize inputs/outputs if they are not dicts
                            # (Similar logic to TrainingDataGenerator normalization)
                            inp = item.get("input", {})
                            if not isinstance(inp, dict): inp = {"input": inp}
                            
                            out = item.get("expected_output", {})
                            if not isinstance(out, dict): out = {"output": out}
                            
                            new_examples.append(Example(
                                input=inp,
                                expected_output=out,
                                reasoning=item.get("reasoning")
                            ))
                        skill.examples.extend(new_examples)
            except Exception as e:
                print(f"Warning: Failed to load training data from {training_path}: {e}")

        return skill
