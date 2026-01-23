"""Agno Agent wrapper for Skill Optimizer with standalone tool functions."""

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from pathlib import Path
from typing import Optional

from app.models.skill import Skill
from app.services.skill_service import SkillService
from app.pipeline.optimizer import SkillOptimizer
from app.pipeline.generator import TrainingDataGenerator
from app.pipeline.analyzer import OptimizationPotentialAnalyzer


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


# --- Agent Factory ---

def create_skill_optimizer_agent(
    skills_dir: str = "skills",
    model_id: str = "gpt-4o",
    **kwargs
) -> Agent:
    """
    Factory function to create a SkillOptimizerAgent.
    
    Args:
        skills_dir: Directory containing skills
        model_id: Model ID for reasoning (default: gpt-4o)
        **kwargs: Additional arguments passed to Agent
        
    Returns:
        Configured Agno Agent with optimization tools
    """
    return Agent(
        name="Skill Optimizer",
        description=(
            "You are an expert AI Engineer and Prompt Optimizer. "
            "Your goal is to help users improve their agent skills using data-driven optimization. "
            "You can analyze skills for potential improvements, generate synthetic training data, "
            "and run powerful optimization algorithms like MIPROv2 and BootstrapFewShot."
        ),
        model=OpenAIChat(id=model_id),
        instructions=[
            "Always start by analyzing a skill before optimizing it.",
            "If a skill lacks training data, offer to generate it.",
            "When comparing skills, present the results in a clear markdown table.",
            "Explain your reasoning for choosing a specific optimization strategy."
        ],
        tools=[
            analyze_skill,
            generate_training_data,
            optimize_skill,
            compare_skills
        ],
        markdown=True,
        show_tool_calls=True,
        **kwargs
    )


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
            model=OpenAIChat(id="gpt-4o"),
            instructions=[
                "Always start by analyzing a skill before optimizing it.",
                "If a skill lacks training data, offer to generate it.",
                "When comparing skills, present the results in a clear markdown table.",
                "Explain your reasoning for choosing a specific optimization strategy."
            ],
            tools=[
                analyze_skill,
                generate_training_data,
                optimize_skill,
                compare_skills
            ],
            markdown=True,
            show_tool_calls=True,
            **kwargs
        )
        self.skills_dir = skills_dir
