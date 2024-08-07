from typing import List, Optional
from xagent.agent.tool_agent.agent import ToolAgent  # Ensure this path is correct
from xagent.message_history import Message
from xagent.config import CONFIG

class DialogueAgentWithTools:
    def __init__(self, name: str, agent_configs, system_message: Message, tools: List[ToolAgent], session_id: str, sender_name: str, is_memory: bool = False):
        self.name = name
        self.agent_configs = agent_configs
        self.system_message = system_message
        self.tools = tools
        self.session_id = session_id
        self.sender_name = sender_name
        self.is_memory = is_memory
        self.run_logs_manager = run_logs_manager
        self.memory = None  # Update memory handling according to XAgent
        self.agent = ToolAgent(config=self.agent_configs, prompt_messages=[self.system_message], **tool_kwargs)

    def send(self) -> str:
        prompt = "\n".join(self.message_history + [self.prefix])
        response, tokens = self.agent.parse(messages=[{'role': 'user', 'content': prompt}])
        return response['content']

# Example usage
if __name__ == "__main__":
    system_message = Message("system", "This is a system message.")
    agent_configs = CONFIG
    tools = [ToolAgent(config=CONFIG)]
    agent = DialogueAgentWithTools(name="Agent", agent_configs=agent_configs, system_message=system_message, tools=tools, session_id="session123", sender_name="sender")
    print(agent.send())
