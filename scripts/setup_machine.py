#!/usr/bin/env python3
"""One-time machine setup for IMBench data collection.

Writes the machine ID to ``~/.imbench/machine.json`` and creates the global
run counter ``~/.imbench/run_counter.json`` at 0 if it does not exist yet.
An existing counter is always kept, so re-running this script (e.g. to fix
the machine ID) never re-issues run IDs; ``--set-counter`` overwrites it
explicitly.

Usage:
    python scripts/setup_machine.py                    # interactive
    python scripts/setup_machine.py --machine-id 03    # non-interactive
    python scripts/setup_machine.py --set-counter 1200 # repair / move the counter
"""

import argparse
import json
import os
import sys

from imbench.utils.run_id import (
    MAX_COUNTER,
    RunCounterError,
    counter_path,
    init_counter,
    machine_config_path,
    validate_machine_id,
)


def _prompt_machine_id() -> str:
    while True:
        mid = input("Enter machine ID (00–99): ").strip()
        try:
            return validate_machine_id(mid)
        except ValueError:
            print("  Invalid — must be exactly 2 digits (00–99).")


def main():
    parser = argparse.ArgumentParser(description="Configure this machine for IMBench data collection.")
    parser.add_argument("--machine-id", default=None,
                        help="Two-digit machine ID (00-99); prompts when omitted")
    parser.add_argument("--set-counter", type=int, default=None, metavar="N",
                        help="Overwrite the run counter with N (the next run ID uses N+1). "
                             "Without it an existing counter is kept.")
    args = parser.parse_args()

    if args.set_counter is not None and not 0 <= args.set_counter <= MAX_COUNTER:
        parser.error(f"--set-counter must be in [0, {MAX_COUNTER}]")

    print("═══ IMBench Machine Setup ═══\n")

    machine_path = machine_config_path()
    existing = None
    if machine_path.exists():
        try:
            with open(machine_path) as f:
                existing = json.load(f).get("machine_id")
        except (OSError, json.JSONDecodeError):
            existing = None
        print(f"Existing config found: machine_id = {existing}")
        try:
            validate_machine_id(existing)
        except ValueError:
            print("  Existing machine_id is invalid and will be replaced.")
            existing = None

    if args.machine_id is not None:
        try:
            mid = validate_machine_id(args.machine_id)
        except ValueError as exc:
            parser.error(str(exc))
    elif existing is not None and args.set_counter is not None:
        mid = existing
    else:
        if existing is not None:
            resp = input("Overwrite? [y/N]: ").strip().lower()
            if resp != "y":
                mid = existing
            else:
                mid = _prompt_machine_id()
        else:
            mid = _prompt_machine_id()

    os.makedirs(machine_path.parent, exist_ok=True)
    if mid != existing:
        with open(machine_path, "w") as f:
            json.dump({"machine_id": mid}, f, indent=2)
        print(f"  Wrote {machine_path}")

    try:
        if args.set_counter is not None:
            counter = init_counter(args.set_counter, overwrite=True)
            print(f"  Set {counter_path()}  (counter = {counter})")
        else:
            counter = init_counter(0)
            print(f"  {counter_path()}  (counter = {counter}, kept if it already existed)")
    except RunCounterError as exc:
        sys.exit(f"ERROR: {exc}")

    print(f"\nDone. This machine is M{mid}.")
    print(f"Output directories will be:  demos/<task_id>/M{mid}/")


if __name__ == "__main__":
    main()
