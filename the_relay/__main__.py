"""CLI entrypoint for The Relay.

Usage:
    python -m the_relay "What is the Mimi codec in Moshi?"
    python -m the_relay --real "What is Multi-Head Attention?"
    python -m the_relay  # interactive mode
"""

import argparse
import sys
from the_relay.config import config
from the_relay.client import RelayModelClient
from the_relay.models import SourceType
from the_relay.pipeline.core import RelayPipeline
from the_relay.resilience.dispatcher import ResilientDispatcher


def main():
    parser = argparse.ArgumentParser(description="The Relay: Resilient RAG Assistant")
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Question to ask (if omitted, starts interactive mode)",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Use real Groq API (requires GROQ_API_KEY in .env)",
    )

    args = parser.parse_args()

    source = SourceType.REAL if args.real else SourceType.STAND_IN

    if source == SourceType.REAL and not config.groq_api_key:
        print("[!] Error: GROQ_API_KEY not set in .env. Using --standin instead.")
        source = SourceType.STAND_IN

    client = RelayModelClient(source=source)
    pipeline = RelayPipeline(client=client)
    dispatcher = ResilientDispatcher(pipeline=pipeline)

    print("=" * 60)
    print(f"  THE RELAY - Resilient Research Assistant")
    print(f"  Mode: {'REAL (Groq)' if source == SourceType.REAL else 'STAND-IN (Calibrated)'}")
    print("=" * 60)

    if args.query:
        run_query(dispatcher, args.query)
        return

    print("\nAsk questions about Attention Is All You Need, Moshi, or Dynamic Chunking.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            query = input("Ask > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            run_query(dispatcher, query)
            print("-" * 60)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def run_query(dispatcher: ResilientDispatcher, query: str):
    print(f"\nProcessing: \"{query}\" ...")
    resp = dispatcher.dispatch(query)

    print("\n--- Answer ---")
    print(resp.answer)
    print("\n--- Execution Telemetry ---")
    print(f"Status:           {resp.status.value}")
    print(f"Time to 1st Token: {resp.latency.first_token_ms:.1f} ms")
    print(f"Total Time:       {resp.latency.total_ms:.1f} ms")
    print(f"  - Waiting:      {resp.latency.waiting_ms:.1f} ms")
    print(f"  - Working:      {resp.latency.working_ms:.1f} ms")
    print(f"  - Retrying:     {resp.latency.retrying_ms:.1f} ms")
    print(f"Retries:          {resp.retries}")
    print(f"Tokens:           {resp.tokens_in} in / {resp.tokens_out} out")
    if resp.citations:
        print(f"Citations:        {', '.join(resp.citations)}")


if __name__ == "__main__":
    main()
