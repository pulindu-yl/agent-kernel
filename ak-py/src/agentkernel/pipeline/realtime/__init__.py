"""Realtime orchestration for live calls.

The realtime connection pool, one model connection per session, and ``RealtimeCall``, application code's handle on
a live call (``pool``); and the background jobs of live calls, which the realtime host runs beside the pool
(``jobs``). The adapter contract and the audio primitives they build on are in ``agentkernel.core.realtime``.
"""

from .jobs import BackgroundJob, BackgroundJobCancelled, BackgroundJobRefused
from .pool import RealtimeCall

__all__ = ["BackgroundJob", "BackgroundJobCancelled", "BackgroundJobRefused", "RealtimeCall"]
