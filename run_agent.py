#!/usr/bin/env python3
"""
Run the CALM causal discovery agent from the command line.

Usage:
  python run_agent.py --chat                    # interactive: keep talking to the agent
  python run_agent.py "Load sachs and run PC"   # single question then exit

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


def chat_loop() -> None:
    """Interactive chat: keep talking to the agent until you type quit or exit."""
    print("CALM causal discovery agent. Type your question, or 'quit' / 'exit' to stop.\n")
    messages = None
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        print("\nAgent:")
        try:
            final, messages = run_agent(user_input, previous_messages=messages)
            print(final)
        except Exception as e:
            print(f"Error: {e}")
        print()


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] in ("--chat", "-c", "chat"):
        chat_loop()
        return

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run_agent.py --chat              # interactive chat")
        print('  python run_agent.py "Your question"     # single question')
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
