from agentkernel.aws import Lambda
from agentkernel.langgraph import LangGraphModule
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from fetch_api import TraceFetchAPI

ORDERS = {"A-1001": "Shipped, arriving Friday"}

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)


# LangChain builds a tool from a function, so this is the one module-level function
@tool
def get_order_status(order_id: str) -> str:
    """Look up the delivery status of an order by its id, e.g. A-1001."""
    return ORDERS.get(order_id.strip().upper(), f"Order {order_id} not found")


support_agent = create_react_agent(
    name="support",
    model=model,
    tools=[get_order_status],
    prompt="You answer questions about orders and deliveries. Always look the order up with get_order_status and "
    "report exactly what it returns. If the order is not found, say so; never guess.",
)

# The supervisor hands order questions to the support agent, like the other examples' triage agent
triage_agent = create_supervisor(
    model=model,
    agents=[support_agent],
    prompt="You are the front desk of an online shop. Hand order and delivery questions to the support agent and "
    "respond with its answer. Politely decline anything unrelated to the shop's orders.",
).compile(name="triage")

LangGraphModule([triage_agent, support_agent])

api = TraceFetchAPI()


@Lambda.register("/fetch", method="GET")
def fetch_spans(event, context):
    """Returns the session's spans from Traceloop."""
    return api.fetch_spans(event, context)


def handler(event, context):
    """Serves /chat and /fetch; flushes the spans before returning, since the OpenLLMetry tracer doesn't on Lambda."""
    try:
        return Lambda.handler(event, context)
    finally:
        TraceFetchAPI.flush()
