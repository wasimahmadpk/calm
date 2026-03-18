#!/usr/bin/env python3
"""
Run the CALM causal discovery agent from the command line.

Usage:
  python run_agent.py "Load the sachs dataset and run PC causal discovery"
  python run_agent.py "What is the causal effect of pip2 on pkc?"   # will load sachs, run discovery, then estimate

API key: put OPENAI_API_KEY in a .env file in this directory (see .env.example).
Never commit .env or share your key.
"""

import os
import sys

# Load .env from the same folder as this script (project root)
try:
    from dotenv import load_dotenv
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(_script_dir, ".env"))
except ImportError:
    pass

from calm.agent import run_agent


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_agent.py \"Your question or request\"")
        print("Example: python run_agent.py \"Load sachs and run LiNGAM, then estimate effect of pip2 on pkc\"")
        sys.exit(1)

    user_message = " ".join(sys.argv[1:])
    print("You:", user_message)
    print("\nAgent:")
    try:
        final, _ = run_agent(user_message)
        print(final)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
