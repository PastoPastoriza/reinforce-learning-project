# AGENTS.md

## Setup
- Python: 3.9
- Install: `pip install -r requirements.txt`

## Verify (run these on every change)
- Tests: `PYTHONWARNINGS=error pytest -q`
- Lint: `ruff check .`

## Smoke (quick sanity run)
- `python linear_rl_trader.py --episodes 1 --seed 42`

## Rules (safety rails)
- Open a **Pull Request** for any code change; **do not merge automatically**.
- Show a **unified diff** and **test logs** in the PR description.
- **Only edit files explicitly listed in the task prompt.**
- If uncertain about scope/side effects, **stop and ask**.
