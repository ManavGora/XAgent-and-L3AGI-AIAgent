from xagent.agent.tool_agent.agent import ToolAgent  # Import ToolAgent from XAgent
from xagent.message_history import Message
from xagent.config import CONFIG

def agent_factory():
    system_message = Message(role='system', content='Your system message here')
    agent_configs = CONFIG  # Assume CONFIG is set up appropriately
    tools = []  # Define any tools required for the agent
    return ToolAgent(config=agent_configs, prompt_messages=[system_message], tools=tools)

agent = agent_factory()

client = Client()

eval_config = RunEvalConfig(
    evaluators=[
        "qa",
        RunEvalConfig.Criteria("helpfulness"),
        RunEvalConfig.Criteria("conciseness"),
    ],
    input_key="input",
    eval_llm=ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo"),
)

chain_results = run_on_dataset(
    client,
    dataset_name="test-dataset",
    llm_or_chain_factory=agent_factory,
    evaluation=eval_config,
    concurrency_level=1,
    verbose=True,
)
