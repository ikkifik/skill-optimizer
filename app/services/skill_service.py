"""Centralized service for skill operations."""

from pathlib import Path
from typing import Optional
import json

from app.models import Skill, Example


class SkillService:
    """
    Centralized service for skill loading, saving, and management.
    
    This eliminates code duplication between CLI and agent components.
    """
    
    def __init__(self, skills_dir: str = "skills"):
        self.skills_dir = Path(skills_dir)
    
    def load(self, name: str) -> Skill:
        """
        Load a skill by name.
        
        Args:
            name: Name of the skill to load
            
        Returns:
            The loaded Skill object
            
        Raises:
            FileNotFoundError: If skill not found
        """
        return Skill.load(name, str(self.skills_dir))
    
    def save(self, skill: Skill, filename: Optional[str] = None) -> Path:
        """
        Save a skill to disk (defaults to SKILL.md Markdown format).
        
        Args:
            skill: The skill to save
            filename: Optional custom filename (default: skill.name)
            
        Returns:
            Path where skill was saved
        """
        from app.pipeline.parsers import MarkdownSkillParser
        
        if filename:
            output_path = self.skills_dir / filename
        else:
            # Check if structured directory exists
            skill_dir = self.skills_dir / skill.name
            if skill_dir.exists() and skill_dir.is_dir():
                output_path = skill_dir / "SKILL.md"
            else:
                # Default to flat markdown file
                output_path = self.skills_dir / f"{skill.name}.md"
        
        # Use Markdown parser to save
        MarkdownSkillParser.save(skill, output_path)
        return output_path
    
    def save_optimized(self, skill: Skill) -> Path:
        """
        Save an optimized skill with the _optimized suffix (Markdown format).
        
        Args:
            skill: The optimized skill to save
            
        Returns:
            Path where skill was saved
        """
        from app.pipeline.parsers import MarkdownSkillParser
        
        output_path = self.skills_dir / f"{skill.name}_optimized.md"
        MarkdownSkillParser.save(skill, output_path)
        return output_path
    
    def save_training_data(
        self, 
        skill_name: str, 
        examples: list[Example],
        overwrite: bool = True
    ) -> Path:
        """
        Save training data for a skill.
        
        Args:
            skill_name: Name of the skill
            examples: List of Example objects to save
            overwrite: Whether to overwrite existing file
            
        Returns:
            Path where training data was saved
        """
        # Determine output path
        skill_dir = self.skills_dir / skill_name
        if skill_dir.exists() and skill_dir.is_dir():
            out_path = skill_dir / "TRAINING.json"
        else:
            out_path = self.skills_dir / f"{skill_name}_training.json"
        
        # Convert examples to JSON format
        data = [
            {
                "input": ex.input,
                "expected_output": ex.expected_output,
                "reasoning": ex.reasoning
            }
            for ex in examples
        ]
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, indent=2))
        return out_path
    
    def create_skill_template(
        self, 
        name: str, 
        description: str = ""
    ) -> tuple[Skill, Path]:
        """
        Create a new skill template.
        
        Args:
            name: Name for the new skill
            description: Optional description
            
        Returns:
            Tuple of (created Skill, path where saved)
        """
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
        
        # Create structured directory
        output_dir = self.skills_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        skill_path = output_dir / "SKILL.md"
        
        from app.pipeline.parsers import MarkdownSkillParser
        MarkdownSkillParser.save(skill, skill_path)
        
        # Create empty training data file
        training_path = output_dir / "TRAINING.json"
        training_path.write_text(json.dumps([], indent=2))
        
        return skill, skill_path
    
    def list_skills(self) -> list[str]:
        """
        List all available skill names.
        
        Returns:
            List of skill names
        """
        skills = []
        
        if not self.skills_dir.exists():
            return skills
            
        for item in self.skills_dir.iterdir():
            if item.is_dir():
                # Check for SKILL.md or SKILL.yaml
                if (item / "SKILL.md").exists() or (item / "SKILL.yaml").exists():
                    skills.append(item.name)
            elif item.suffix in (".md", ".yaml") and not item.name.startswith("_"):
                # Flat skill file
                skills.append(item.stem)
        
        return sorted(set(skills))
