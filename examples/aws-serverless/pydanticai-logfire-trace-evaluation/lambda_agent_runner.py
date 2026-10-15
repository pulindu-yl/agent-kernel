import logfire
from agentkernel.aws import ServerlessAgentRunner
from agentkernel.pydanticai import PydanticAIModule, PydanticAIToolBuilder
from pydantic_ai import Agent

MODEL = "openai:gpt-4.1-mini"
ORDERS = {"A-1001": "Shipped, arriving Friday"}


def get_order_status(order_id: str) -> str:
    """Look up the delivery status of an order by its id, e.g. A-1001."""
    return ORDERS.get(order_id.strip().upper(), f"Order {order_id} not found")


support_agent = Agent(
    model=MODEL,
    name="support",
    description="Order and delivery questions",
    instructions="You answer questions about orders and deliveries. Always look the order up with "
    "get_order_status and report exactly what it returns. If the order is not found, say so; never guess.",
    tools=PydanticAIToolBuilder.bind([get_order_status]),
)


# Pydantic AI has no handoffs; the triage agent delegates to the support agent through a tool
async def ask_support(question: str) -> str:
    """Hand an order or delivery question to the support agent and return its answer."""
    return str((await support_agent.run(question)).output)


triage_agent = Agent(
    model=MODEL,
    name="triage",
    description="Front desk of an online shop",
    instructions="You are the front desk of an online shop. Pass order and delivery questions to ask_support and "
    "return its answer. Politely decline anything unrelated to the shop's orders.",
    tools=PydanticAIToolBuilder.bind([ask_support]),
)

PydanticAIModule([triage_agent, support_agent])


def handler(event, context):
    """Runs each queued request; the Logfire runner traces it."""
    try:
        return ServerlessAgentRunner.handle(event, context)
    finally:
        # A frozen Lambda sandbox never runs Logfire's background export, so send the spans before returning
        logfire.force_flush()
