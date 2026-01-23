"""
Skill Optimizer - Main Entry Point

Optimize agent skills using DSPy teleprompters.
"""

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.cli import main

if __name__ == "__main__":
    main()
