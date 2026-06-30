"""Continuous RSI runner — spends RunPod credits on self-training until they run out.

Rotates through the weak/seed classes, harvesting fresh Bright Data imagery and submitting a
GPU fine-tune cycle each round, then publishing new detector weights. Stops when the RunPod
balance falls below `rsi_burn_min_balance_usd` (or after `max_cycles`, if given).

Run standalone:  python -m app.rsi.runner            (from backend/)
or trigger one-off cycles via POST /api/rsi/cycle.
"""
from __future__ import annotations

import logging
import time

import httpx

from ..config import get_settings
from .loop import RSILoop

log = logging.getLogger("pathfinder.rsi.runner")

_BALANCE_URL = "https://api.runpod.io/v2/billing/balance"


def _balance_usd(api_key: str) -> float | None:
    """Best-effort RunPod balance via the GraphQL API; None if unavailable."""
    try:
        resp = httpx.post(
            "https://api.runpod.io/graphql",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"query": "query { myself { clientBalance } }"},
            timeout=20,
        )
        resp.raise_for_status()
        return float(resp.json()["data"]["myself"]["clientBalance"])
    except Exception:  # noqa: BLE001
        return None


def run_forever(max_cycles: int | None = None) -> None:
    settings = get_settings()
    loop = RSILoop()
    classes = [c.strip() for c in settings.rsi_burn_classes.split(",") if c.strip()]
    batch = max(1, settings.rsi_burn_batch)

    if not (settings.rsi_trainer_url and settings.runpod_api_key):
        log.error("RunPod not configured (rsi_trainer_url / runpod_api_key) — nothing to run.")
        return

    cycle = 0
    i = 0
    while max_cycles is None or cycle < max_cycles:
        bal = _balance_usd(settings.runpod_api_key)
        if bal is not None and bal < settings.rsi_burn_min_balance_usd:
            log.info("RunPod balance $%.2f below floor $%.2f — stopping.", bal, settings.rsi_burn_min_balance_usd)
            break

        targets = [classes[(i + k) % len(classes)] for k in range(batch)]
        i = (i + batch) % len(classes)
        cycle += 1
        log.info("RSI burn cycle %d (balance=%s) targets=%s", cycle, bal, targets)
        try:
            result = loop.run_cycle(labels=targets)
            log.info("cycle %d job=%s", cycle, (result.get("job") or {}).get("status"))
        except Exception as exc:  # noqa: BLE001 - one bad cycle shouldn't kill the burner
            log.warning("cycle %d failed: %s", cycle, exc)
            time.sleep(10)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_forever()
