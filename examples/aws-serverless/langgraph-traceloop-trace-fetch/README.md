# Agent Kernel LangGraph with Traceloop Trace Fetch in AWS Serverless (Lambda)

A check that `Trace.get().fetch_spans(...)` reads an agent's Traceloop (OpenLLMetry) spans back on AWS Lambda. It
doesn't evaluate anything: the Lambda runs the agent on `/chat` and returns a session's spans on
`GET /fetch`, and the live test passes once they come back. It isn't part of CI.

Why only a fetch: the Traceloop fetcher is best effort (the warehouse API is barely documented, and kinds are
filtered only after the spans arrive), and the OpenLLMetry runners add no Agent Kernel root span, so the trace evaluation examples
([`openai-langfuse-trace-evaluation`](../openai-langfuse-trace-evaluation) and its siblings) can't grade these
traces as they are. This shows what the fetch actually returns.

## Architecture

One Lambda, not in queue mode (`lambda.handler`):

| Route | What it does |
|---|---|
| `POST /api/v1/chat` | Runs the agent: a `triage` LangGraph supervisor that hands order questions to a `support` ReAct agent with a `get_order_status` tool, both on `gpt-4.1-mini`. Each request is one agent run. |
| `GET /api/v1/fetch` | Query `session_id` (required), and optionally `trace_id`, `kind` (`agent`, `llm`, `tool`, `span`) and `minutes` (look-back, default 60, at most 1440). Returns `span_count`, `trace_count`, `truncated` and up to 200 `spans` (each a `FetchedSpan` without `raw`). |

The OpenLLMetry tracer doesn't flush on Lambda, so the handler flushes the tracer provider after every
request, before the sandbox can freeze. Sessions use the Redis cluster of the base `openai` deployment
(`config.yaml`).

## Prerequisites

- AWS CLI configured with appropriate credentials, Docker running, and Terraform `1.9.5` or higher.
- A Traceloop API key. The same key sends the spans and fetches them back.
- The base VPC, security groups and Redis cluster from the `openai` example (`cd ../openai/deploy && ./deploy.sh`).

## Deploy

1. Set the variables (the VPC, subnets and security group come from the openai deployment's outputs):
    ```bash
    export TF_VAR_openai_api_key=<OPENAI_API_KEY>
    export TF_VAR_traceloop_api_key=<TRACELOOP_API_KEY>
    # export TF_VAR_traceloop_base_url=<URL>  # only for a non-default Traceloop endpoint
    export TF_VAR_vpc_id=<VPC_ID_FROM_OPENAI>
    export TF_VAR_private_subnet_ids='["<SUBNET_ID_1>", "<SUBNET_ID_2>"]'
    export TF_VAR_request_handler_security_group_id=<RQH_SG_FROM_OPENAI>
    ```

2. Deploy. The Lambda is a container image, so Docker must be running. `./deploy.sh local` installs
   `agentkernel` from `../../../ak-py/dist`; use it until a release with `Trace.fetch_spans` is on PyPI:
    ```bash
    cd deploy && ./deploy.sh local
    ```

## Try it

```bash
BASE=$(terraform -chdir=deploy output -raw agent_invoke_url | sed 's#/chat$##')
curl -X POST "$BASE/chat" -H "Content-Type: application/json" \
  -d '{"prompt": "Where is my order A-1001?", "session_id": "my-session", "agent": "triage"}'
# a minute or two later
curl "$BASE/fetch?session_id=my-session"
curl "$BASE/fetch?session_id=my-session&kind=tool"
```

## Test

The live test sends two questions under a fresh `traceloop-fetch-<uuid>` session (an order lookup with a
hand-off and a tool call, and an off-topic request), then polls `/fetch` every 15 s until it returns spans and
their count stops growing (up to 5 minutes), and prints them as one tree per trace:

```bash
AK_TEST_ENDPOINT=$(terraform -chdir=deploy output -raw agent_invoke_url) uv run --no-sync pytest lambda_test.py
```

| Test | Passes when |
|---|---|
| `test_session_spans_are_fetched` | `/fetch` returns spans, all with the session id and provider `openllmetry` |
| `test_fetch_by_trace_id` | `/fetch?trace_id=...` returns spans, all of that trace |
| `test_fetch_tool_spans` | `/fetch?kind=tool` returns at least one span, all of kind `tool` |

If the first test finds no spans, check the Traceloop dashboard: spans there but none fetched points at the
session filter (`traceloop.association.properties.session_id`); no spans there points at sending.
