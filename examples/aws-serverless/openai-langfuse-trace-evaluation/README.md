# Agent Kernel OpenAI Agents with Langfuse Trace Evaluation in AWS Serverless (Lambda)

This example evaluates an agent on its real traffic. An OpenAI Agents SDK agent runs on AWS Lambda in queue
mode with Langfuse tracing. On request, the request-handler Lambda reads a session's traces back with
`Trace.get().fetch(...)`, grades each request with four Opik LLM judges, and stores one row per trace in DynamoDB.

It uses only public Agent Kernel API: `Trace.fetch`, `FetchedSpan`, `SpanUsage`, the `AKEvaluator` ABC,
`Lambda`, `ServerlessAgentRunner` and `ResponseHandler`.

## Architecture

```
              POST /chat                                                spans (flushed before the Lambda freezes)
client ──▶ request handler ──▶ SQS input ──▶ agent runner ─────────────────────────────────────▶ Langfuse
                 ▲                               │ reply                                            │
                 └── response store ◀── response handler ◀── SQS output                             │
                                                                                                    │
              POST /eval  {"session_id"}                                                            │ Trace.fetch(session_id=...)
client ──▶ request handler: EvaluationAPI.start ── prepare ◀────────────────────────────────────────┤
                              │                      │ one row per trace, with its progress         │
                              │                      ▼                                              │
                              │                 DynamoDB  ◀── save (one PutItem) ──┐                │
                              │ async self-invoke    ▲                             │                │
                              └──────────────▶ request handler: grade ── 4 Opik judges ◀────────────┘
              GET /check?session_id=...&expected=N   │
client ──▶ request handler: EvaluationAPI.check ─────┘ counts the session's rows by progress
```

The deployment runs in queue mode (`queue_mode = true`, `execution_mode = "rest_sync"`), with three Lambdas:

| Lambda | Entry point | What it does |
|---|---|---|
| Request handler | `lambda_request_handler.handler` (`Lambda.handler`) | Serves the three routes below. `/chat` requests are enqueued for the agent runner and the reply is read back from the response store. Also runs the grading, in an asynchronous invocation of itself, so it is the only tier that carries Opik. |
| Agent runner | `lambda_agent_runner.handler` | Runs the agent (`ServerlessAgentRunner`): a `triage` agent that hands order questions to a `support` agent with a `get_order_status` tool. Its Langfuse runner sends the trace and flushes it before the Lambda freezes. |
| Response handler | `lambda_response_handler.handler` | Stores each reply in the response store (`ResponseHandler`). |

The session and response stores use the Redis cluster of the base `openai` deployment (`config.yaml`).

| Route | What it does |
|---|---|
| `POST /api/v1/chat` | The agent under evaluation. Each request is one trace. |
| `POST /api/v1/eval` | Body `{"session_id": "..."}`. Fetches the session's spans, gives each trace a row and a `progress`, and starts grading the ready ones in an asynchronous self-invocation. Returns `202` with the counts. |
| `GET /api/v1/check` | Query `session_id` and `expected` (the number of requests you sent). Counts the session's rows by `progress`; `complete` is `true` once `graded >= expected`. |

Nothing runs on a schedule: grading happens only when `/eval` is called.

## The evaluation run

`/eval` runs `TraceEvaluationJob.prepare_traces`, inside the request:

1. Fetches every span of the session from the last 24 hours (`Trace.get().fetch(session_id=..., limit=5000)`).
   A full result is an error, so a truncated session is never graded.
2. Groups the spans by trace. A trace with exactly one parentless root span is one request (`TraceCase`).
3. Sets each trace's `progress`:

   | Current row | Trace | New `progress` |
   |---|---|---|
   | `graded` or `unfinished` | any | unchanged |
   | none, `pending`, `failed` or `grading` | the root span has no output | `unfinished` |
   | none, `pending`, `failed` or `grading` | finished less than 2 minutes ago, or a span's parent isn't readable yet | `pending` |
   | none, `pending`, `failed` or `grading` | ready | `grading` |

4. When any trace is `grading`, invokes the request handler itself asynchronously with a `POST /grade`
   request whose body is `{"session_id": ..., "trace_ids": [...]}`.

`/grade` is registered like the other routes (`lambda_request_handler.py`), so the stock `Lambda.handler`
routes the invocation to it. It is not in the API Gateway's endpoints, so it can't be called from outside.
It runs `TraceEvaluationJob.grade_traces`. It re-fetches the session, judges each trace that is still
`grading`, and writes its row in one `PutItem`. That is bound by the request handler's timeout (10 minutes), not by
API Gateway's 29 seconds.

Fault handling:

- A judge call is tried twice. A second failure is stored on that metric (no `score`, the error in
  `reason`) and counted in `metric_errors`; the trace is still `graded`.
- Any other error while grading one trace marks only that row `failed` with the error. A fetch error marks
  every row of the call `failed`. The next `/eval` retries `failed` rows.
- If the grading invocation is killed (timeout or out of memory), its rows stay `grading`. AWS's automatic retry is off,
  so `/check` shows them as `grading` until you call `/eval` again, which queues them again.
- Calling `/eval` while a grading is still running grades those traces twice. The rows stay correct (same
  key, whole-row writes); only judge calls are wasted.

## Metrics

All four are Opik LLM judges on `gpt-4.1-mini` (`AK_TEST__LLM__PROVIDER` / `AK_TEST__LLM__MODEL`), with
threshold `0.5`. None needs an expected answer.

| Metric | Opik metric | Reads | Runs when | Passes when |
|---|---|---|---|---|
| `task_completion` | `AgentTaskCompletionJudge` | the transcript (request, tool calls, answer) | always | score ≥ 0.5 |
| `tool_correctness` | `AgentToolCorrectnessJudge` | the transcript | the trace has a tool call | score ≥ 0.5 |
| `answer_relevance` | `AnswerRelevance` | input, output | always | score ≥ 0.5 |
| `hallucination` | `Hallucination` | input, output, tool results as context | always | score **<** 0.5 |

Tool-call count, failed tool calls, latency, tokens and cost come from the spans and are stored, not judged.

## The results table

`<prefix>-trace-evaluations`, partition key `trace_id`, with a `session_id-index` and a 90-day TTL on
`expires_at`. One row per trace:

| Attribute | On | Meaning |
|---|---|---|
| `progress`, `updated_at`, `error` | every row (`error` on `failed`) | `pending`, `grading`, `graded`, `failed` or `unfinished` |
| `session_id`, `root_name`, `started_at`, `latency_ms`, `input`, `output`, `agents`, `span_count`, `provider` | every row | the trace summary |
| `tool_calls`, `tool_call_count`, `tool_error_count`, `tool_calls_truncated` | every row | up to 25 calls stored; the counts cover all |
| `input_tokens`, `output_tokens`, `total_tokens`, `cost` | every row | `usage` summed over the trace's spans |
| `task_completion`, `tool_correctness`, `answer_relevance`, `hallucination` | `graded` | a map: `score`, `threshold`, `passed`, `reason`, `higher_is_better`, `evaluator`, `attempts` |
| `passed`, `metric_errors`, `evaluated_at` | `graded` | `passed` is true when every scored metric passed |
| `spans_gz` or `spans_truncated` | `graded` | the trace's full spans, gzipped JSON, when they fit |

Text fields are cut to 4,096 characters. The whole item is then kept under a 380 KB budget (DynamoDB's
limit is 400 KB) by dropping tool calls and shortening `input`/`output`, so a large trace never fails the run.

## Changing the metrics or the store

- **Metrics:** add an entry to `OpikTraceEvaluator.METRICS` and `_JUDGE_CLASSES`
  (`evaluator/opik_evaluator.py`). For another judge library, write another `AKEvaluator` subclass with
  `applies_to` and `evaluate_by_llm`, and return it from the factory in `TraceEvaluationJob.from_env`.
- **Store:** `TraceEvaluationJob` and `EvaluationAPI` only call `read_progress`, `write_progress`, `save_results`,
  `mark_failed` and `count_session_progress` on `EvaluationStore` (`evaluator/store.py`); replace it with a
  class offering the same methods.
- **Tracer:** `Trace.fetch` works for Langfuse, Logfire, OpenLLMetry and CloudWatch. This example is built
  and tested on Langfuse, which sets the session id on every span.

## Prerequisites

- AWS CLI configured with appropriate credentials, Docker running, and Terraform `1.9.5` or higher.
- A Langfuse project (Langfuse Cloud or self-hosted) and its public and secret keys.
- The base VPC, security groups and Redis cluster from the `openai` example (`cd ../openai/deploy && ./deploy.sh`).

## Deploy

1. Get the VPC ID, private subnet IDs and the three security group IDs from the openai deployment:
    ```bash
    cd ../openai/deploy && terraform output vpc_id
    cd ../openai/deploy && terraform output private_subnet_ids
    cd ../openai/deploy && terraform output request_handler_security_group_id
    cd ../openai/deploy && terraform output agent_runner_security_group_id
    cd ../openai/deploy && terraform output response_handler_security_group_id
    ```

2. Set the variables:
    ```bash
    export TF_VAR_openai_api_key=<OPENAI_API_KEY>
    export TF_VAR_langfuse_public_key=<LANGFUSE_PUBLIC_KEY>
    export TF_VAR_langfuse_secret_key=<LANGFUSE_SECRET_KEY>
    export TF_VAR_langfuse_base_url=https://cloud.langfuse.com  # the default; set it for another region or self-hosting
    export TF_VAR_vpc_id=<VPC_ID_FROM_OPENAI>
    export TF_VAR_private_subnet_ids='["<SUBNET_ID_1>", "<SUBNET_ID_2>"]'
    export TF_VAR_request_handler_security_group_id=<RQH_SG_FROM_OPENAI>  # optional, each of the three
    export TF_VAR_agent_runner_security_group_id=<AR_SG_FROM_OPENAI>
    export TF_VAR_response_handler_security_group_id=<RSH_SG_FROM_OPENAI>
    ```

3. Deploy. `deploy.sh` builds each Lambda's package from its `pyproject.toml` extra. The request handler
   (Opik) and the agent runner (OpenAI Agents SDK) are over Lambda's 250 MB zip limit, so Terraform builds
   them as container images and Docker must be running; the response handler is a zip. `./deploy.sh local`
   installs `agentkernel` from `../../../ak-py/dist` instead of PyPI:
    ```bash
    cd deploy && ./deploy.sh
    ```

## Try it

1. Send the example's three questions (an order lookup, an unknown order and an off-topic request) under one
   session. The script reads the URL from `AK_TEST_ENDPOINT` or the `agent_invoke_url` Terraform output,
   and prints the session id:
    ```bash
    uv run python simulate_traffic.py
    ```

2. Wait about 2 minutes for Langfuse to ingest the traces, then start grading:
    ```bash
    BASE=$(terraform -chdir=deploy output -raw agent_invoke_url | sed 's#/chat$##')
    curl -X POST "$BASE/eval" -H "Content-Type: application/json" -d '{"session_id": "<SESSION_ID>"}'
    ```
   The response counts the traces found, `grading`, `pending` and `unfinished`. If some are `pending`, call
   `/eval` again a little later.

3. Poll until every request is graded:
    ```bash
    curl "$BASE/check?session_id=<SESSION_ID>&expected=3"
    ```

4. Read the results:
    ```bash
    aws dynamodb query --table-name "$(terraform -chdir=deploy output -raw results_table_name)" \
      --index-name session_id-index --key-condition-expression "session_id = :s" \
      --expression-attribute-values '{":s": {"S": "<SESSION_ID>"}}'
    ```

## Test

Unit tests (no cloud access; the judges, Langfuse and DynamoDB are faked, DynamoDB with `moto`):

```bash
./build.sh local && uv run --no-sync pytest tests/
```

The integration test sends the three questions to a deployment, waits, then runs up to three `/eval`
rounds, and passes once `/check` reports every request graded. It runs in the weekly integration tests;
locally:

```bash
AK_TEST_ENDPOINT=$(terraform -chdir=deploy output -raw agent_invoke_url) uv run --no-sync pytest lambda_test.py
```

It takes about 3 to 5 minutes, most of it waiting for Langfuse ingestion. It checks only that every request
was graded and stored, not the judges' verdicts.
