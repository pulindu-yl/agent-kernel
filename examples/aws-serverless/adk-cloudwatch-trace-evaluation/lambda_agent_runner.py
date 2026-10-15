from agentkernel.adk import GoogleADKModule
from agentkernel.aws import ServerlessAgentRunner
from google.adk.agents import Agent, LlmAgent
from google.adk.models.lite_llm import LiteLlm

ORDERS = {"A-1001": "Shipped, arriving Friday"}


def get_order_status(order_id: str) -> str:
    """Look up the delivery status of an order by its id, e.g. A-1001."""
    return ORDERS.get(order_id.strip().upper(), f"Order {order_id} not found")


support_agent = Agent(
    name="support",
    model=LiteLlm(model="openai/gpt-4.1-mini"),
    description="Order and delivery questions",
    instruction="You answer questions about orders and deliveries. Always look the order up with "
    "get_order_status and report exactly what it returns. If the order is not found, say so; never guess. "
    "Politely decline anything unrelated to the shop's orders.",
    tools=[get_order_status],
)

triage_agent = LlmAgent(
    name="triage",
    model=LiteLlm(model="openai/gpt-4.1-mini"),
    description="Front desk of an online shop",
    instruction='You are the front desk of an online shop. Transfer order and delivery questions to the agent named "support". '
    "Politely decline anything unrelated to the shop's orders.",
    sub_agents=[support_agent],
)

GoogleADKModule([triage_agent, support_agent])

# Runs each queued request; the CloudWatch runner traces it and flushes the spans before the Lambda freezes
handler = ServerlessAgentRunner.handle
