# data/

Runtime artifacts from the RSI loop (git-ignored):

- `rsi_ledger.jsonl` — one line per self-improvement cycle: weak classes found, data
  harvested, and the retrain job submitted to RunPod.
- `harvest/<label>.jsonl` — reference imagery + descriptions gathered via Bright Data for
  classes the detector is weak on, used as training data for the next fine-tune.

Trigger a cycle manually: `POST /api/rsi/cycle`.
