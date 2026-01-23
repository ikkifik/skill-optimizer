
from app.services.skill_service import SkillService
from app.models.skill import Skill, Example
from pathlib import Path

# Setup dummy skill
skill = Skill(
    name="test_knowledge",
    description="Test skill",
    instructions="Test instructions",
    examples=[
        Example(input={"q": "hi"}, expected_output={"a": "hello"})
    ]
)
service = SkillService("temp_skills")
service.save(skill)

# Test CLI export
import subprocess
try:
    subprocess.run(["skill-optimizer", "export-knowledge", "test_knowledge", "--skills-dir", "temp_skills", "--format", "json"], check=True)
    subprocess.run(["skill-optimizer", "export-knowledge", "test_knowledge", "--skills-dir", "temp_skills", "--format", "markdown"], check=True)
    print("CLI commands succeeded")
except subprocess.CalledProcessError as e:
    print(f"CLI command failed: {e}")
finally:
    import shutil
    if Path("temp_skills").exists():
        shutil.rmtree("temp_skills")
