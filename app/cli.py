"""
Skill Optimizer CLI - Optimize agent skills using DSPy.

Usage:
    skill-optimizer analyze <skill_name>
    skill-optimizer optimize <skill_name> [--strategy=<strategy>] [--model=<model>]
    skill-optimizer compare <skill_a> <skill_b>
"""

import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional
import json
import os
import dspy
from dotenv import load_dotenv

from app.models.skill import Skill
from app.pipeline.optimizer import SkillOptimizer
from app.pipeline.analyzer import OptimizationPotentialAnalyzer

# Load environment variables
load_dotenv()

# Configure DSPy
def configure_dspy():
    """Configure DSPy LM from environment variables."""
    openai_key = os.getenv("OPENAI_API_KEY")
    mistral_key = os.getenv("MISTRAL_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    dspy_model = os.getenv("DSPY_MODEL")
    
    if dspy_model:
        # User specified model
        lm = dspy.LM(dspy_model)
        dspy.configure(lm=lm)
        return

    # Auto-detect
    if mistral_key:
        # User requested Mistral
        lm = dspy.LM("mistral/mistral-large-latest", api_key=mistral_key)
        dspy.configure(lm=lm)
    elif openai_key:
        lm = dspy.LM("openai/gpt-4o-mini", api_key=openai_key)
        dspy.configure(lm=lm)
    elif gemini_key:
        lm = dspy.LM("gemini/gemini-1.5-flash", api_key=gemini_key)
        dspy.configure(lm=lm)
    else:
        print("[yellow]Warning: No API keys found (MISTRAL_API_KEY, OPENAI_API_KEY or GEMINI_API_KEY). DSPy may fail.[/yellow]")

configure_dspy()


app = typer.Typer(
    name="skill-optimizer",
    help="Optimize agent skills using DSPy teleprompters"
)
console = Console()


@app.command()
def analyze(
    skill_name: str = typer.Argument(..., help="Name of the skill to analyze"),
    skills_dir: str = typer.Option("skills", help="Directory containing skills")
):
    """Analyze a skill's optimization potential before optimizing."""
    try:
        skill = Skill.load(skill_name, skills_dir)
    except FileNotFoundError:
        console.print(f"[red]Skill '{skill_name}' not found in {skills_dir}/[/red]")
        raise typer.Exit(1)
    
    console.print(f"\n[bold]Analyzing skill: {skill.name}[/bold]\n")
    
    analyzer = OptimizationPotentialAnalyzer(skill)
    analysis = analyzer.analyze()
    
    # Display results
    table = Table(title="Optimization Potential Analysis")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Baseline Score", f"{analysis.baseline_score:.2%}")
    table.add_row("Estimated Ceiling", f"{analysis.estimated_ceiling:.2%}")
    table.add_row("Potential Improvement", f"{analysis.estimated_improvement:.1f}%")
    table.add_row("Score Variance", f"{analysis.variance:.4f}")
    table.add_row("Training Examples", str(analysis.sample_size))
    table.add_row("Recommended Strategy", analysis.recommended_strategy)
    
    console.print(table)
    
    if analysis.weakest_areas:
        console.print(f"\n[yellow]Weak areas:[/yellow] {', '.join(analysis.weakest_areas)}")
    
    console.print(Panel(analysis.recommendation, title="Recommendation"))


@app.command()
def optimize(
    skill_name: str = typer.Argument(..., help="Name of the skill to optimize"),
    strategy: str = typer.Option("bootstrap_fewshot", help="Optimization strategy"),
    model: Optional[str] = typer.Option(None, help="Target model to optimize for"),
    skills_dir: str = typer.Option("skills", help="Directory containing skills"),
    output: Optional[str] = typer.Option(None, help="Output path for optimized skill")
):
    """Run optimization on a skill and save the result."""
    try:
        skill = Skill.load(skill_name, skills_dir)
    except FileNotFoundError:
        console.print(f"[red]Skill '{skill_name}' not found in {skills_dir}/[/red]")
        raise typer.Exit(1)
    
    console.print(f"\n[bold]Optimizing skill: {skill.name}[/bold]")
    console.print(f"Strategy: {strategy}")
    if model:
        console.print(f"Target model: {model}")
    
    optimizer = SkillOptimizer()
    
    with console.status("Running optimization..."):
        result = optimizer.optimize(skill, strategy=strategy, target_model=model)
    
    # Display comparison
    console.print("\n" + result.comparison.to_markdown_table())
    
    # Save optimized skill
    if output:
        output_path = Path(output)
    else:
        output_path = Path(skills_dir) / f"{skill_name}_optimized.yaml"
    
    result.optimized_skill.save(output_path)
    console.print(f"\n[green]Optimized skill saved to: {output_path}[/green]")


@app.command()
def compare(
    skill_a: str = typer.Argument(..., help="First skill name"),
    skill_b: str = typer.Argument(..., help="Second skill name"),
    skills_dir: str = typer.Option("skills", help="Directory containing skills"),
    output: Optional[str] = typer.Option(None, help="Output path for comparison report")
):
    """Compare two skill versions side-by-side."""
    try:
        skill_1 = Skill.load(skill_a, skills_dir)
        skill_2 = Skill.load(skill_b, skills_dir)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    
    optimizer = SkillOptimizer()
    comparison = optimizer.compare(skill_1, skill_2)
    
    console.print("\n" + comparison.to_markdown_table())
    
    if output:
        Path(output).write_text(comparison.to_markdown_table())
        console.print(f"\n[green]Report saved to: {output}[/green]")


@app.command()
def create(
    name: str = typer.Argument(..., help="Name for the new skill"),
    description: str = typer.Option("", help="Skill description"),
    skills_dir: str = typer.Option("skills", help="Directory to create skill in")
):
    """Create a new skill template."""
    skill = Skill(
        name=name,
        description=description or f"A skill for {name}",
        instructions="Your task instructions here.",
        input_schema={
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "The input to process"}
            }
        },
        output_schema={
            "type": "object",
            "properties": {
                "output": {"type": "string", "description": "The result"}
            }
        },
        examples=[]
    )
    
    output_dir = Path(skills_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    skill.save(output_dir / "SKILL.yaml")
    
    # Create empty training data file
    training_path = output_dir / "TRAINING.json"
    training_path.write_text(json.dumps([], indent=2))
    
    console.print(f"[green]Created skill template at: {output_dir}/[/green]")
    console.print("  - SKILL.yaml: Skill definition")
    console.print("  - TRAINING.json: Add training examples here (or use 'generate' command)")


@app.command()
def generate(
    skill_name: str = typer.Argument(..., help="Name of the skill"),
    count: int = typer.Option(5, help="Number of examples to generate"),
    skills_dir: str = typer.Option("skills", help="Directory containing skills"),
    output: Optional[str] = typer.Option(None, help="Output file for examples")
):
    """Generate synthetic training data for a skill using LLM."""
    try:
        skill = Skill.load(skill_name, skills_dir)
    except FileNotFoundError:
        console.print(f"[red]Skill '{skill_name}' not found in {skills_dir}/[/red]")
        raise typer.Exit(1)
        
    console.print(f"\n[bold]Generating {count} examples for: {skill.name}[/bold]")
    
    from app.pipeline.generator import TrainingDataGenerator
    generator = TrainingDataGenerator()
    
    with console.status("Generating examples..."):
        examples = generator.generate(skill, count=count)
    
    if not examples:
        console.print("[red]Failed to generate examples.[/red]")
        raise typer.Exit(1)
        
    # Convert back to json format for saving
    data = []
    for ex in examples:
        data.append({
            "input": ex.input,
            "expected_output": ex.expected_output,
            "reasoning": ex.reasoning
        })
        
    if output:
        out_path = Path(output)
    else:
        # Default to TRAINING.json in skill dir if structured, else alongside file
        if (Path(skills_dir) / skill_name).is_dir():
             out_path = Path(skills_dir) / skill_name / "TRAINING.json"
        else:
             out_path = Path(skills_dir) / f"{skill_name}_training.json"
             
    # If file exists, append or merge? For now, just save new set or warn
    if out_path.exists():
        console.print(f"[yellow]Warning: {out_path} already exists. Overwriting.[/yellow]")
        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))
    
    console.print(f"[green]Generated {len(examples)} examples saved to: {out_path}[/green]")



def main():
    app()


if __name__ == "__main__":
    main()
