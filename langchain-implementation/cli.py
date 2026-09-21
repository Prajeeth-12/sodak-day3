"""CLI for running the LangChain Placement Assistant."""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path so imports work
ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = Path(__file__).resolve().parent
for p in (str(ROOT), str(MODULE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent import run_query


def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain Placement Assistant CLI")
    parser.add_argument("query", help="The question or command for the agent")
    parser.add_argument("--student", default="22CS045", help="Roll number of the student (default: 22CS045)")
    parser.add_argument("--quiet", action="store_true", help="Suppress intermediate tool call logs")
    args = parser.parse_args()

    print(f"\n[Student: {args.student}] Query: {args.query}\n" + "=" * 60)
    result = run_query(args.query, student_id=args.student, verbose=not args.quiet)

    print("\n" + "=" * 60)
    print(f"Assistant Reply:\n{result['output']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
