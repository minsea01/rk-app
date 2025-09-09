#!/usr/bin/env python3
import os, runpy, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tune'))
TARGET = os.path.join(ROOT, 'auto_tune_pid.py')
if not os.path.exists(TARGET):
    print(f"Missing target script: {TARGET}", file=sys.stderr)
    sys.exit(2)

# Delegate execution to the relocated script
runpy.run_path(TARGET, run_name='__main__')
