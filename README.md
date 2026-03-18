# EPANET Net3 Pump Scheduling RL (Project Skeleton)

This repository is initialized as a `src`-layout Python project for reproducing an EPANET Net3 pump scheduling reinforcement learning paper.

Current scope:
- Project skeleton and package structure only
- Existing EPANET network inputs kept under `networks/`
- No PPO training implementation yet

## Project Structure

```text
EPANET_RL/
├─ environment.yml
├─ .gitignore
├─ README.md
├─ requirements.txt
├─ networks/
│  ├─ Net3.inp
│  └─ Anytown.inp
├─ scripts/
│  └─ .gitkeep
├─ src/
│  └─ epanet_rl/
│     ├─ __init__.py
│     └─ constants.py
└─ tests/
   ├─ __init__.py
   └─ test_import.py
```

## Quick Start

1. Create and activate environment (recommended):

```bash
conda env create -f environment.yml
conda activate epanet_rl
```

2. Or install from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

3. Run tests:

```bash
pytest -q
```
