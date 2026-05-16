"""Interactive CLI smoke test for the GrabWise supervisor + driver agent.

Usage (from project root, with .venv active):

    # As driver #1, single shot:
    python -m scripts.run_chat --role driver --user-id 1 \\
        --message "It's Friday evening — where should I drive to maximize earnings?"

    # Interactive REPL:
    python -m scripts.run_chat --role driver --user-id 1
"""
from __future__ import annotations
import argparse
import sys
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.json import JSON
from rich.rule import Rule
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from backend.agents.supervisor import GRAPH
from backend.llm.bedrock import llm_provider_name

console = Console()


def _print_trace(trace: list[dict[str, Any]]) -> None:
    if not trace:
        return
    console.print(Rule("[bold cyan]Agent Activity[/]"))
    for i, step in enumerate(trace, 1):
        header = f"[bold]{i}. {step['agent']} → {step['tool']}[/]  [dim]{step.get('ts','')}[/]"
        body = JSON.from_data({"input": step.get("input"), "output": step.get("output")})
        console.print(Panel(body, title=header, border_style="cyan", expand=True))


def _print_final(messages: list) -> None:
    # Find the last AI message that isn't a [supervisor] note and isn't a tool-call shell.
    final = None
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            text = m.content if isinstance(m.content, str) else ""
            if text and not text.startswith("[supervisor]") and not getattr(m, "tool_calls", None):
                final = text
                break
    if final:
        console.print(Rule("[bold green]Assistant[/]"))
        console.print(Panel(final, border_style="green", expand=True))
    else:
        console.print("[yellow]⚠ No final assistant message found.[/]")


def run_once(role: str, user_id: str, message: str) -> None:
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "user_role": role,
        "user_id": user_id,
        "agent_trace": [],
        "next_agent": None,
    }
    console.print(Rule(f"[bold]🛺 GrabWise[/] · provider=[magenta]{llm_provider_name()}[/] · role=[cyan]{role}[/] · user_id=[cyan]{user_id}[/]"))
    console.print(Panel(message, title="[bold]You[/]", border_style="white"))

    final_state = GRAPH.invoke(initial_state, config={"recursion_limit": 25})

    _print_trace(final_state.get("agent_trace", []))
    _print_final(final_state.get("messages", []))


def repl(role: str, user_id: str) -> None:
    console.print(Rule("[bold]GrabWise REPL[/]  ([dim]Ctrl-C or 'exit' to quit[/])"))
    while True:
        try:
            msg = console.input("\n[bold cyan]you ▸[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye.[/]")
            return
        if not msg or msg.lower() in {"exit", "quit"}:
            return
        run_once(role, user_id, msg)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--role", default="driver", choices=["driver", "customer", "merchant", "admin"])
    p.add_argument("--user-id", default="1")
    p.add_argument("--message", default=None, help="Single-shot message; omit for REPL.")
    args = p.parse_args()

    if args.message:
        run_once(args.role, args.user_id, args.message)
    else:
        repl(args.role, args.user_id)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
