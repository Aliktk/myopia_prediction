"""
run_all_agents.py — Sequential Multi-Agent Orchestrator
=========================================================
Runs all specialist agents in order, passing shared state between them.
Each agent is self-contained and can also be run independently.

Usage
-----
    python agents/run_all_agents.py
    python agents/run_all_agents.py --agents 1 2 3   # run only agents 1–3
"""
from __future__ import annotations
import argparse
import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


AGENTS = {
    1: ("agent_01_data_analyzer",  "Data Analyzer"),
    2: ("agent_02_preprocessor",   "Preprocessor & Feature Engineering"),
    3: ("agent_03_eda",            "EDA Visualizer"),
    4: ("agent_04_model_trainer",  "Model Trainer"),
    5: ("agent_05_evaluator",      "Model Evaluator"),
    6: ("agent_06_inference",      "Inference & Diagrams"),
}


def run_agent(module_name: str, display_name: str):
    print(f"\n{'#' * 68}")
    print(f"#  {display_name.upper()}")
    print(f"{'#' * 68}")
    t0 = time.time()
    try:
        mod = importlib.import_module(f"agents.{module_name}")
        mod.run()
        elapsed = time.time() - t0
        print(f"\n  OK {display_name} completed in {elapsed:.1f}s")
    except ImportError:
        print(f"  ! Agent module '{module_name}' not implemented yet — skipping")
    except Exception as e:
        print(f"  FAIL {display_name} failed: {e}")
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agents", nargs="*", type=int, default=list(AGENTS.keys()),
                   help="Which agent numbers to run (default: all)")
    args = p.parse_args()

    print("\n" + "=" * 68)
    print("  MYOPIA PREDICTION — MULTI-AGENT PIPELINE ORCHESTRATOR")
    print("=" * 68)

    t_total = time.time()
    for num in sorted(args.agents):
        if num in AGENTS:
            module_name, display_name = AGENTS[num]
            run_agent(module_name, display_name)
        else:
            print(f"  Unknown agent number: {num}")

    total = time.time() - t_total
    print(f"\n{'=' * 68}")
    print(f"  ALL AGENTS COMPLETE — Total time: {total:.1f}s")
    print(f"{'=' * 68}")


if __name__ == "__main__":
    main()
