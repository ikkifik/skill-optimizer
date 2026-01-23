"""
Example: Using the Skill Optimizer Agent

This example shows how to use the AI agent to interactively
analyze and optimize skills through natural language.
"""

from app import create_skill_optimizer_agent


def main():
    # Create the optimizer agent
    agent = create_skill_optimizer_agent(
        skills_dir="skills",
        model_id="gpt-4o"  # or "gpt-4o-mini" for faster/cheaper
    )
    
    print("🤖 Skill Optimizer Agent Ready!")
    print("=" * 50)
    
    # Example 1: Analyze a skill
    print("\n📊 Asking agent to analyze a skill...\n")
    agent.print_response(
        "Analyze the email_summarizer skill and tell me if it's worth optimizing",
        stream=True
    )
    
    print("\n" + "=" * 50)
    
    # Example 2: Generate training data
    print("\n📝 Asking agent to generate training data...\n")
    agent.print_response(
        "Generate 3 training examples for the text_classifier skill",
        stream=True
    )
    
    print("\n" + "=" * 50)
    
    # Example 3: Run optimization
    print("\n⚡ Asking agent to optimize a skill...\n")
    agent.print_response(
        "Optimize the email_summarizer skill using the bootstrap_fewshot strategy",
        stream=True
    )


if __name__ == "__main__":
    main()
