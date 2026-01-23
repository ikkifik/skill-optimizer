"""Generate DSPy signatures from Skill definitions."""

import dspy
from typing import Type
from app.models.skill import Skill


class DSPySignatureGenerator:
    """
    Converts structured Skill definitions into DSPy Signatures.
    
    This enables automatic optimization of skills using DSPy's
    teleprompters (BootstrapFewShot, MIPROv2, etc.)
    """
    
    def __init__(self, skill: Skill):
        self.skill = skill
    
    def generate_signature(self) -> Type[dspy.Signature]:
        """
        Generate a DSPy Signature class from the skill definition.
        
        Returns a dynamically created Signature class with input/output
        fields derived from the skill's schemas.
        """
        # Build field definitions from schemas
        input_fields = {}
        output_fields = {}
        
        # Parse input schema
        if self.skill.input_schema:
            for field_name, field_def in self.skill.input_schema.get("properties", {}).items():
                desc = field_def.get("description", f"Input: {field_name}")
                input_fields[field_name] = dspy.InputField(desc=desc)
        else:
            # Default input field
            input_fields["input"] = dspy.InputField(desc="The input to process")
        
        # Parse output schema
        if self.skill.output_schema:
            for field_name, field_def in self.skill.output_schema.get("properties", {}).items():
                desc = field_def.get("description", f"Output: {field_name}")
                output_fields[field_name] = dspy.OutputField(desc=desc)
        else:
            # Default output field
            output_fields["output"] = dspy.OutputField(desc="The processed result")
        
        # Create the signature class dynamically
        signature_attrs = {
            "__doc__": self.skill.instructions,
            **input_fields,
            **output_fields
        }
        
        signature_class = type(
            f"{self.skill.name.replace('-', '_').title()}Signature",
            (dspy.Signature,),
            signature_attrs
        )
        
        return signature_class
    
    def generate_module(self, use_cot: bool = False) -> dspy.Module:
        """
        Generate a DSPy Module from the skill.
        
        Args:
            use_cot: If True, use ChainOfThought for complex reasoning tasks
            
        Returns:
            A DSPy Module ready for optimization
        """
        signature = self.generate_signature()
        
        if use_cot:
            return dspy.ChainOfThought(signature)
        return dspy.Predict(signature)
    
    def generate_examples(self) -> list[dspy.Example]:
        """
        Convert skill examples to DSPy Example format.
        
        Returns:
            List of dspy.Example objects for training
        """
        dspy_examples = []
        
        for example in self.skill.examples:
            # Merge input and expected_output into a single dict
            example_data = {**example.input, **example.expected_output}
            
            # Determine input field names
            input_keys = list(example.input.keys())
            
            dspy_example = dspy.Example(**example_data).with_inputs(*input_keys)
            dspy_examples.append(dspy_example)
        
        return dspy_examples
