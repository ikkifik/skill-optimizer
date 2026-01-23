"""Knowledge exporter for Agno Knowledge Base integration.

Exports skill training examples to formats compatible with Agno's 
Knowledge feature for dynamic few-shot learning.
"""

import json
from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field

from app.models.skill import Skill, Example


class KnowledgeChunk(BaseModel):
    """A single knowledge chunk for Agno Knowledge Base."""
    
    content: str = Field(..., description="The text content of the chunk")
    metadata: dict = Field(default_factory=dict, description="Metadata for filtering")


class KnowledgeExporter:
    """
    Exports skill examples to Agno Knowledge Base format.
    
    This enables dynamic few-shot learning where relevant examples
    are retrieved at runtime based on the user's query, rather than
    using a fixed set of examples in the prompt.
    """
    
    def __init__(self, skill: Skill):
        self.skill = skill
    
    def to_chunks(self) -> list[KnowledgeChunk]:
        """
        Convert skill examples to knowledge chunks.
        
        Each example becomes a chunk with:
        - Content: Formatted input/output pair
        - Metadata: Skill name, version, example index
        
        Returns:
            List of KnowledgeChunk objects
        """
        chunks = []
        
        for idx, example in enumerate(self.skill.examples):
            # Format the example as a clear input/output demonstration
            content = self._format_example(example, idx)
            
            metadata = {
                "skill_name": self.skill.name,
                "skill_version": self.skill.version,
                "example_index": idx,
                "has_reasoning": example.reasoning is not None,
            }
            
            # Add input field names for filtering
            if example.input:
                metadata["input_fields"] = list(example.input.keys())
            
            chunks.append(KnowledgeChunk(content=content, metadata=metadata))
        
        return chunks
    
    def _format_example(self, example: Example, idx: int) -> str:
        """Format a single example as readable text for knowledge retrieval."""
        lines = [
            f"## Example {idx + 1} for {self.skill.name}",
            "",
            "### Input:",
            json.dumps(example.input, indent=2),
            "",
            "### Expected Output:",
            json.dumps(example.expected_output, indent=2),
        ]
        
        if example.reasoning:
            lines.extend([
                "",
                "### Reasoning:",
                example.reasoning
            ])
        
        return "\n".join(lines)
    
    def export_json(self, output_path: Optional[Path] = None) -> Path:
        """
        Export examples to JSON format for Agno Knowledge.
        
        Args:
            output_path: Optional output path. Defaults to skill directory.
            
        Returns:
            Path to the exported file
        """
        chunks = self.to_chunks()
        
        # Convert to JSON-serializable format
        data = {
            "skill_name": self.skill.name,
            "skill_version": self.skill.version,
            "skill_description": self.skill.description,
            "total_examples": len(chunks),
            "chunks": [chunk.model_dump() for chunk in chunks]
        }
        
        if output_path is None:
            output_path = Path("knowledge") / f"{self.skill.name}_knowledge.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(data, indent=2))
        
        return output_path
    
    def export_documents(self, output_dir: Optional[Path] = None) -> list[Path]:
        """
        Export examples as individual markdown documents.
        
        Each example becomes a separate .md file that can be
        loaded into an Agno Knowledge Base.
        
        Args:
            output_dir: Output directory. Defaults to knowledge/{skill_name}/
            
        Returns:
            List of paths to exported files
        """
        chunks = self.to_chunks()
        
        if output_dir is None:
            output_dir = Path("knowledge") / self.skill.name
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = []
        for idx, chunk in enumerate(chunks):
            file_path = output_dir / f"example_{idx + 1}.md"
            
            # Add YAML frontmatter for metadata
            content = "---\n"
            for key, value in chunk.metadata.items():
                if isinstance(value, list):
                    content += f"{key}: {json.dumps(value)}\n"
                else:
                    content += f"{key}: {value}\n"
            content += "---\n\n"
            content += chunk.content
            
            file_path.write_text(content)
            paths.append(file_path)
        
        return paths


def create_agno_knowledge_config(
    skill: Skill,
    vector_db: Literal["pgvector", "lancedb", "qdrant"] = "lancedb"
) -> str:
    """
    Generate Python code snippet for using exported knowledge with Agno.
    
    Args:
        skill: The skill to generate config for
        vector_db: Which vector database to use
        
    Returns:
        Python code snippet as a string
    """
    skill_snake = skill.name.replace("-", "_")
    
    if vector_db == "lancedb":
        code = f'''from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from agno.models.mistral import MistralChat
from agno.vectordb.lancedb import LanceDb

# Create knowledge base from exported examples
knowledge_base = Knowledge(
    vector_db=LanceDb(
        table_name="{skill_snake}_examples",
        uri="./lancedb"
    ),
)

# Load the exported knowledge
knowledge_base.insert(json_path="knowledge/{skill.name}_knowledge.json")

# Create agent with dynamic few-shot retrieval
agent = Agent(
    model=MistralChat(id="mistral-large-latest"),
    knowledge=knowledge_base,
    search_knowledge=True,  # Enable automatic knowledge search
    instructions="""
{skill.instructions}

Use the retrieved examples to guide your response format and approach.
""",
)
'''
    elif vector_db == "pgvector":
        code = f'''from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from agno.models.mistral import MistralChat
from agno.vectordb.pgvector import PgVector

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

knowledge_base = Knowledge(
    vector_db=PgVector(table_name="{skill_snake}_examples", db_url=db_url),
)

# Load the exported knowledge
knowledge_base.insert(json_path="knowledge/{skill.name}_knowledge.json")

agent = Agent(
    model=MistralChat(id="mistral-large-latest"),
    knowledge=knowledge_base,
    search_knowledge=True,
)
'''
    else:  # qdrant
        code = f'''from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from agno.models.mistral import MistralChat
from agno.vectordb.qdrant import Qdrant

knowledge_base = Knowledge(
    vector_db=Qdrant(collection="{skill_snake}_examples", url="http://localhost:6333"),
)

knowledge_base.insert(json_path="knowledge/{skill.name}_knowledge.json")

agent = Agent(
    model=MistralChat(id="mistral-large-latest"),
    knowledge=knowledge_base,
    search_knowledge=True,
)
'''
    
    return code
