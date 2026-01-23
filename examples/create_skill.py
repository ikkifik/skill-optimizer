"""
Example: Creating a Custom Skill

This example demonstrates how to programmatically create
a new skill and prepare it for optimization.
"""

from app import Skill, SkillService, Example


def main():
    # Initialize the service
    service = SkillService(skills_dir="skills")
    
    # Define a custom skill
    skill = Skill(
        name="code_reviewer",
        description="Reviews code snippets and provides constructive feedback",
        instructions="""
You are an expert code reviewer. Given a code snippet, provide:

1. **Issues**: List any bugs, potential errors, or anti-patterns
2. **Suggestions**: Recommend improvements for readability and performance
3. **Score**: Rate the code quality from 1-10

Be constructive and specific. Focus on actionable feedback.
        """.strip(),
        input_schema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code snippet to review"
                },
                "language": {
                    "type": "string",
                    "description": "Programming language (e.g., python, javascript)"
                }
            },
            "required": ["code", "language"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "issues": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of issues found"
                },
                "suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Improvement suggestions"
                },
                "score": {
                    "type": "integer",
                    "description": "Quality score 1-10"
                }
            }
        },
        examples=[
            Example(
                input={
                    "code": "def add(a,b): return a+b",
                    "language": "python"
                },
                expected_output={
                    "issues": ["Missing type hints", "No docstring"],
                    "suggestions": [
                        "Add type annotations: def add(a: int, b: int) -> int",
                        "Add a docstring explaining the function"
                    ],
                    "score": 6
                }
            ),
            Example(
                input={
                    "code": "x = eval(input())",
                    "language": "python"
                },
                expected_output={
                    "issues": [
                        "SECURITY: eval() on user input is dangerous",
                        "Variable name 'x' is not descriptive"
                    ],
                    "suggestions": [
                        "Never use eval() on untrusted input",
                        "Use ast.literal_eval() for simple cases",
                        "Use meaningful variable names"
                    ],
                    "score": 2
                }
            )
        ],
        tags=["development", "code-quality", "review"],
        author="skill-optimizer"
    )
    
    # Save the skill
    output_path = service.save(skill)
    print(f"✅ Skill saved to: {output_path}")
    
    # Verify it can be loaded
    loaded = service.load("code_reviewer")
    print(f"📦 Verified: Loaded skill '{loaded.name}' with {len(loaded.examples)} examples")
    
    # Show available skills
    print("\n📋 Available skills:")
    for name in service.list_skills():
        print(f"   - {name}")


if __name__ == "__main__":
    main()
