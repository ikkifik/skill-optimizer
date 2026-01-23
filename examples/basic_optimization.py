"""
Example: Basic Skill Optimization Flow

This example demonstrates the complete workflow for optimizing a skill:
1. Load a skill
2. Analyze its optimization potential
3. Generate training data (if needed)
4. Run optimization
5. Compare results
"""

from app import SkillService
from app.pipeline import (
    OptimizationPotentialAnalyzer,
    TrainingDataGenerator,
    SkillOptimizer
)


def main():
    # Initialize the service
    service = SkillService(skills_dir="skills")
    
    # Step 1: Load a skill
    print("📦 Loading skill...")
    skill = service.load("email_summarizer")
    print(f"   Loaded: {skill.name}")
    print(f"   Description: {skill.description}")
    print(f"   Examples: {len(skill.examples)}")
    
    # Step 2: Analyze optimization potential
    print("\n🔍 Analyzing optimization potential...")
    analyzer = OptimizationPotentialAnalyzer(skill)
    analysis = analyzer.analyze()
    
    print(f"   Baseline Score: {analysis.baseline_score:.2%}")
    print(f"   Estimated Ceiling: {analysis.estimated_ceiling:.2%}")
    print(f"   Potential Improvement: {analysis.estimated_improvement:.1f}%")
    print(f"   Recommended Strategy: {analysis.recommended_strategy}")
    print(f"   💡 {analysis.recommendation}")
    
    # Step 3: Generate more training data if needed
    if len(skill.examples) < 5:
        print("\n📝 Generating additional training data...")
        generator = TrainingDataGenerator()
        new_examples = generator.generate(skill, count=5)
        service.save_training_data(skill.name, new_examples)
        print(f"   Generated {len(new_examples)} new examples")
        
        # Reload skill to include new examples
        skill = service.load(skill.name)
    
    # Step 4: Run optimization
    print("\n⚡ Running optimization...")
    optimizer = SkillOptimizer()
    result = optimizer.optimize(
        skill, 
        strategy=analysis.recommended_strategy
    )
    
    # Step 5: Save and compare
    output_path = service.save_optimized(result.optimized_skill)
    print(f"\n✅ Optimized skill saved to: {output_path}")
    
    # Display comparison
    print("\n📊 Results:")
    print(result.comparison.to_markdown_table())


if __name__ == "__main__":
    main()
