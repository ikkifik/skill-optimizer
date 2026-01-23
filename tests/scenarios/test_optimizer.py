import scenario
import pytest
import os
from pathlib import Path
from app.models import Skill
from app.agent import SkillOptimizerAgent
from agno.utils.log import logger

# Adapter for Agno Agent
class AgnoAgentAdapter(scenario.AgentAdapter):
    def __init__(self, agent_wrapper):
        self.agent_wrapper = agent_wrapper

    async def call(self, input: scenario.AgentInput) -> scenario.AgentReturnTypes:
        # Run the Agno agent
        # input.last_new_user_message_str() gives the latest message
        response = self.agent_wrapper.run(
            input.last_new_user_message_str(),
            session_id=input.thread_id
        )
        # Return the string content
        return str(response.content)

@pytest.fixture
def setup_skill():
    skill = Skill(
        name="test_skill_for_optimization",
        description="A test skill that needs optimization",
        instructions="Just say hello with a professional greeting.",
    )
    skill_path = Path("skills/test_skill_for_optimization.yaml")
    skill.save(skill_path)
    yield
    try:
        skill_path.unlink(missing_ok=True)
    except Exception:
        pass

@pytest.mark.asyncio
async def test_optimizer_flow(setup_skill):
    optimizer_agent = SkillOptimizerAgent()
    adapter = AgnoAgentAdapter(optimizer_agent)

    result = await scenario.run(
        name="Optimize Skill Flow",
        description="The user asks to optimize a specific skill. The agent should load it, optimize it, and save it.",
        agents=[
            adapter,
            scenario.UserSimulatorAgent(),
            scenario.JudgeAgent(criteria=[
                "Agent should acknowledge the optimization request.",
                "Agent should indicate that the skill has been optimized/saved."
            ])
        ]
    )

    assert result.success

