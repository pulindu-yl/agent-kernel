# Agent Kernel running Claude Agent SDK Agents

This package contains a demo of Agent Kernel running agents built with the
[Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview). Users may interact with agents via the
Agent Kernel CLI.

## Model provider

The Claude Agent SDK runs Claude models only. It needs an Anthropic API key or a cloud provider; Anthropic does
not allow third-party products to use a claude.ai login. Set one of:

    export ANTHROPIC_API_KEY=...          # Claude API
    export CLAUDE_CODE_USE_BEDROCK=1      # Amazon Bedrock (with AWS credentials)
    export CLAUDE_CODE_USE_VERTEX=1       # Google Vertex AI (with GCP credentials)

Each run starts a Claude Code CLI subprocess, bundled in the `claude-agent-sdk` wheel (no Node.js needed).

## Multi-agent routing

Claude's native multi-agent primitive is the **subagent**. The `triage` agent lists the specialist agents in
`subagents=[general_agent, math_agent, weather_agent]`, like `handoffs=` in the OpenAI demo, and delegates to them
through the SDK's `Agent` tool. The weather subagent calls `get_weather`, an Agent Kernel tool bound with
`ClaudeToolBuilder`.

## Built-in tools

By default the Claude Agent SDK gives the model Claude Code's built-in tools (Bash, Read, Write, WebFetch, ...)
acting on this host, and loads `~/.claude` settings. The demo agents turn that off with `tools=[]` and
`setting_sources=[]`, and use `permission_mode="dontAsk"` so any tool call that is not explicitly approved is
denied. Agent Kernel approves the tools bound with `ClaudeToolBuilder`, and the `Agent` tool an agent with subagents
delegates through.

## Running

Install dependencies using:

    ./build.sh

Install local dependencies in development mode using:

    ./build.sh local

Run this demo using the following.

    python demo.py

To run tests:

    uv run pytest -s

> Note: until a release of `agentkernel` that includes the `claude` extra is published to PyPI, build with
> `./build.sh local` against a locally built `ak-py/dist`.
