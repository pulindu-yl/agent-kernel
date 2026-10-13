import asyncio
import os
import time
import uuid

import httpx
import pytest
import pytest_asyncio

from simulate_traffic import TrafficSimulator

pytestmark = pytest.mark.asyncio(loop_scope="session")  # uses a single session for all tests

INITIAL_WAIT = 150  # s: the job's SETTLE_TIME (2 minutes) plus 30 s
ROUNDS = 3  # /eval calls, the first one included
ROUND_INTERVAL = 30  # s between rounds
POLL_INTERVAL = 10  # s between /check calls
POLL_TIMEOUT = 180  # s per round
EXPECTED = len(TrafficSimulator.QUESTIONS)


class EvaluationClient:
    """Calls the deployed /eval and /check routes, next to the agent's /chat."""

    def __init__(self, chat_url: str):
        chat_url = chat_url.rstrip("/")
        if not chat_url.endswith("/chat"):
            raise ValueError(f"expected the agent's /chat URL, got {chat_url}")
        api_url = chat_url.removesuffix("/chat")
        self.eval_url = f"{api_url}/eval"
        self.check_url = f"{api_url}/check"

    async def start_evaluation(self, session_id: str) -> tuple[int, dict]:
        """Calls /eval for the session and returns the status code and body."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.eval_url, json={"session_id": session_id})
        return response.status_code, response.json()

    async def wait_until_graded(
        self, session_id: str, expected: int, timeout: float = POLL_TIMEOUT, poll_interval: float = POLL_INTERVAL
    ) -> dict:
        """Polls /check until complete or nothing is grading, or the timeout passes; returns the last response."""
        deadline = time.monotonic() + timeout
        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                response = await client.get(self.check_url, params={"session_id": session_id, "expected": expected})
                response.raise_for_status()
                progress = response.json()
                if progress["complete"] or progress["grading"] == 0 or time.monotonic() >= deadline:
                    return progress
                await asyncio.sleep(poll_interval)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def chat_url():
    """The deployed agent's /chat URL; skips the test when AK_TEST_ENDPOINT is unset."""
    endpoint = os.getenv("AK_TEST_ENDPOINT")
    if not endpoint:
        pytest.skip("AK_TEST_ENDPOINT is not set; deploy the example first")
    yield endpoint


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def simulator(chat_url):
    """A traffic simulator on a fresh session id."""
    yield TrafficSimulator(chat_url, session_id=str(uuid.uuid4()))


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def client(chat_url):
    """A client for the deployed /eval and /check routes."""
    yield EvaluationClient(chat_url)


@pytest.mark.asyncio
async def test_every_request_is_graded(simulator, client):
    """Sends the questions, runs up to ROUNDS /eval calls, and expects every request graded."""
    replies = await asyncio.to_thread(simulator.send_questions)
    assert len(replies) == EXPECTED and all(replies), f"expected {EXPECTED} non-empty replies, got {replies}"

    await asyncio.sleep(INITIAL_WAIT)

    progress: dict = {}
    for _ in range(ROUNDS):
        code, body = await client.start_evaluation(simulator.session_id)
        assert code == 202, f"/eval failed: {body.get('error')}"
        progress = await client.wait_until_graded(simulator.session_id, expected=EXPECTED)
        if progress["complete"]:
            break
        # not all graded: the next /eval queues pending, failed and leftover grading traces
        await asyncio.sleep(ROUND_INTERVAL)

    assert progress["complete"], (
        f"{progress['graded']} of {EXPECTED} requests graded after {ROUNDS} rounds ({progress['pending']} pending, "
        f"{progress['failed']} failed, {progress['grading']} still grading, {progress['unfinished']} unfinished): "
        f"{progress['errors']}"
    )
