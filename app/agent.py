"""Agno Agent wrapper for Skill Optimizer with standalone tool functions."""

from agno.agent import Agent
from agno.models.mistral import MistralChat
from agno.db.sqlite import SqliteDb
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

from app.models.skill import Skill
from app.services.skill_service import SkillService
from app.pipeline.optimizer import SkillOptimizer
from app.pipeline.generator import TrainingDataGenerator
from app.pipeline.analyzer import OptimizationPotentialAnalyzer


# --- Structured Output Models ---

class SkillAnalysisResult(BaseModel):
    """Structured output for skill analysis."""
    
    skill_name: str = Field(..., description="Name of the analyzed skill")
    baseline_score: float = Field(..., description="Current performance score (0-1)")
    estimated_ceiling: float = Field(..., description="Maximum achievable score")
    improvement_potential: float = Field(..., description="Percentage improvement possible")
    recommended_strategy: str = Field(..., description="Best optimization strategy to use")
    recommendation: str = Field(..., description="Human-readable recommendation")
    weak_areas: list[str] = Field(default_factory=list, description="Areas needing improvement")


class OptimizationReport(BaseModel):
    """Structured output for optimization results."""
    
    skill_name: str
    original_score: float
    optimized_score: float
    improvement_percent: float
    strategy_used: str
    output_path: str
    summary: str = Field(..., description="Brief summary of what was improved")


# --- Standalone Tool Functions ---

def analyze_skill(skill_name: str, skills_dir: str = "skills") -> str:
    """
    Analyze a skill's potential for optimization.
    
    Args:
        skill_name: Name of the skill to analyze
        skills_dir: Directory containing skills
        
    Returns:
        Analysis results in markdown format
    """
    try:
        service = SkillService(skills_dir)
        skill = service.load(skill_name)
        analyzer = OptimizationPotentialAnalyzer(skill)
        analysis = analyzer.analyze()
        
        return (
            f"### Analysis for {skill_name}\n\n"
            f"- **Baseline Score**: {analysis.baseline_score:.2%}\n"
            f"- **Estimated Ceiling**: {analysis.estimated_ceiling:.2%}\n"
            f"- **Potential Improvement**: {analysis.estimated_improvement:.1f}%\n"
            f"- **Recommended Strategy**: {analysis.recommended_strategy}\n\n"
            f"**Recommendation**: {analysis.recommendation}"
        )
    except Exception as e:
        return f"Error analyzing skill: {str(e)}"


def generate_training_data(
    skill_name: str, 
    count: int = 5, 
    skills_dir: str = "skills"
) -> str:
    """
    Generate synthetic training examples for a skill.
    
    Args:
        skill_name: Name of the skill
        count: Number of examples to generate (default: 5)
        skills_dir: Directory containing skills
        
    Returns:
        Status message about generated examples
    """
    try:
        service = SkillService(skills_dir)
        skill = service.load(skill_name)
        generator = TrainingDataGenerator()
        examples = generator.generate(skill, count=count)
        
        if not examples:
            return "Failed to generate training examples."
        
        # Save using service
        path = service.save_training_data(skill_name, examples)
        
        return f"Generated {len(examples)} examples and saved to {path}."
    except Exception as e:
        return f"Error generating data: {str(e)}"


def optimize_skill(
    skill_name: str, 
    strategy: str = "bootstrap_fewshot",
    target_model: Optional[str] = None,
    skills_dir: str = "skills"
) -> str:
    """
    Run optimization on a skill.
    
    Args:
        skill_name: Name of the skill to optimize
        strategy: Optimization strategy (bootstrap_fewshot, mipro_v2, bootstrap_rs)
        target_model: Optional model to optimize for
        skills_dir: Directory containing skills
        
    Returns:
        Optimization results in markdown format
    """
    try:
        service = SkillService(skills_dir)
        skill = service.load(skill_name)
        optimizer = SkillOptimizer()
        result = optimizer.optimize(skill, strategy=strategy, target_model=target_model)
        
        # Save using service
        out_path = service.save_optimized(result.optimized_skill)
        
        return (
            f"### Optimization Complete for {skill_name}\n\n"
            f"Saved to: `{out_path}`\n\n"
            f"{result.comparison.to_markdown_table()}"
        )
    except Exception as e:
        return f"Error optimizing skill: {str(e)}"


def compare_skills(
    skill_a: str, 
    skill_b: str, 
    skills_dir: str = "skills"
) -> str:
    """
    Compare two skills side by side.
    
    Args:
        skill_a: First skill name
        skill_b: Second skill name
        skills_dir: Directory containing skills
        
    Returns:
        Comparison results in markdown table format
    """
    try:
        service = SkillService(skills_dir)
        s1 = service.load(skill_a)
        s2 = service.load(skill_b)
        optimizer = SkillOptimizer()
        res = optimizer.compare(s1, s2)
        return res.to_markdown_table()
    except Exception as e:
        return f"Error comparing skills: {str(e)}"


def export_to_knowledge(
    skill_name: str,
    format: str = "json",
    skills_dir: str = "skills"
) -> str:
    """
    Export skill examples to Agno Knowledge Base format.
    
    Args:
        skill_name: Name of the skill to export
        format: Export format (json or markdown)
        skills_dir: Directory containing skills
        
    Returns:
        Path to exported knowledge files
    """
    try:
        from app.pipeline.knowledge_exporter import KnowledgeExporter
        
        service = SkillService(skills_dir)
        skill = service.load(skill_name)
        exporter = KnowledgeExporter(skill)
        
        if format == "json":
            path = exporter.export_json()
            return f"Exported knowledge to: {path}"
        else:
            paths = exporter.export_documents()
            return f"Exported {len(paths)} documents to: {paths[0].parent}/"
    except Exception as e:
        return f"Error exporting knowledge: {str(e)}"


# --- Agent Factory ---

def create_skill_optimizer_agent(
    skills_dir: str = "skills",
    model_id: str = "mistral-large-latest",
    enable_memory: bool = True,
    **kwargs
) -> Agent:
    """
    Factory function to create a SkillOptimizerAgent.
    
    Args:
        skills_dir: Directory containing skills
        model_id: Model ID for reasoning (default: mistral-large-latest)
        enable_memory: Enable conversation memory (default: True)
        **kwargs: Additional arguments passed to Agent
        
    Returns:
        Configured Agno Agent with optimization tools
    """
    agent_kwargs = {
        "name": "Skill Optimizer",
        "description": (
            "You are an expert AI Engineer and Prompt Optimizer. "
            "Your goal is to help users improve their agent skills using data-driven optimization. "
            "You can analyze skills for potential improvements, generate synthetic training data, "
            "and run powerful optimization algorithms like MIPROv2 and BootstrapFewShot."
        ),
        "model": MistralChat(id=model_id),
        "instructions": [
            "Always start by analyzing a skill before optimizing it.",
            "If a skill lacks training data, offer to generate it.",
            "When comparing skills, present the results in a clear markdown table.",
            "Explain your reasoning for choosing a specific optimization strategy.",
            "After optimization, suggest exporting to Knowledge Base for dynamic few-shot.",
        ],
        "tools": [
            analyze_skill,
            generate_training_data,
            optimize_skill,
            compare_skills,
            export_to_knowledge,
        ],
        "markdown": True,
        "show_tool_calls": True,
        "reasoning": True,  # Enable chain-of-thought reasoning
    }
    
    # Add memory if enabled
    if enable_memory:
        agent_kwargs["db"] = SqliteDb(
            table_name="agent_sessions",
            db_file="metrics/agent_memory.db"
        )
        agent_kwargs["add_history_to_messages"] = True
    
    agent_kwargs.update(kwargs)
    return Agent(**agent_kwargs)


# --- Backwards-compatible class ---

class SkillOptimizerAgent(Agent):
    """
    An autonomous agent that can analyze, optimize, and improve other agents' skills.
    
    This class provides backwards compatibility. For new code, prefer using
    create_skill_optimizer_agent() factory function.
    """
    
    def __init__(self, name: str = "Skill Optimizer", skills_dir: str = "skills", **kwargs):
        description = (
            "You are an expert AI Engineer and Prompt Optimizer. "
            "Your goal is to help users improve their agent skills using data-driven optimization. "
            "You can analyze skills for potential improvements, generate synthetic training data, "
            "and run powerful optimization algorithms like MIPROv2 and BootstrapFewShot."
        )
        
        super().__init__(
            name=name,
            description=description,
            model=MistralChat(id="mistral-large-latest"),
            instructions=[
                "Always start by analyzing a skill before optimizing it.",
                "If a skill lacks training data, offer to generate it.",
                "When comparing skills, present the results in a clear markdown table.",
                "Explain your reasoning for choosing a specific optimization strategy.",
            ],
            tools=[
                analyze_skill,
                generate_training_data,
                optimize_skill,
                compare_skills,
                export_to_knowledge,
            ],
            db=SqliteDb(
                table_name="agent_sessions",
                db_file="metrics/agent_memory.db"
            ),
            add_history_to_messages=True,
            markdown=True,
            show_tool_calls=True,
            reasoning=True,
            **kwargs
        )
        self.skills_dir = skills_dir

