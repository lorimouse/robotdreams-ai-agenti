"""Domácí úkol 1 – LLM API + tool calling.

Použití:
    uv run main.py                         # ukázkový dotaz (víc nástrojů naráz)
    uv run main.py "Kolik je 17 % z 2 450 EUR v korunách?"
    uv run main.py --chat                  # interaktivní chat s historií
    uv run main.py --compare               # LLM bez nástroje vs. s nástrojem
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from agent import ReactAgent, Usage, make_client

DEMO_QUESTION = (
    "Kupuju byt za 285 000 EUR. Kolik to je dnes v korunách? Mám naspořeno "
    "1,2 milionu Kč, zbytek chci na hypotéku na 30 let s úrokem 4,79 %. "
    "Jaká bude měsíční splátka a kolik bance přeplatím?"
)
COMPARE_QUESTION = "Kolik je 987654321 × 123456789? A kolik je dnes 1 337 EUR v korunách?"

# --- hezký výstup do terminálu -------------------------------------------------
COLOR = sys.stdout.isatty() and not os.getenv("NO_COLOR")


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if COLOR else text


def print_event(kind: str, data: dict) -> None:
    if kind == "iteration":
        print(c(f"\n── krok {data['n']} · volám LLM", "2"))
    elif kind == "tool_call":
        args = json.loads(data["args"] or "{}")
        pretty = ", ".join(f"{k}={v!r}" for k, v in args.items())
        print(c(f"   🔧 {data['name']}({pretty})", "33"))
    elif kind == "tool_result":
        print(c(f"   ↳ {data['result']}", "36"))


def print_usage(usage: Usage) -> None:
    cost = usage.cost_usd()
    cost_txt = f" · cena ≈ ${cost:.6f}" if cost is not None else ""
    print(
        c(
            f"\n📊 volání LLM: {usage.llm_calls} · nástrojů: {usage.tool_calls} · "
            f"tokeny: {usage.prompt_tokens} vstup + {usage.completion_tokens} výstup "
            f"= {usage.total_tokens}{cost_txt}",
            "2",
        )
    )


# --- režimy --------------------------------------------------------------------
def ask(question: str, quiet: bool = False) -> None:
    client, model = make_client()
    agent = ReactAgent(client, model, on_event=(lambda *_: None) if quiet else print_event)
    print(c(f"🤖 model: {model}", "2"))
    print(c(f"\n❓ {question}", "1"))
    answer = agent.run(question)
    print(c("\n✅ Odpověď:", "1;32"))
    print(answer)
    print_usage(agent.usage)


def chat() -> None:
    """Chat s pamětí: jeden agent = jedna historie zpráv, která roste."""
    client, model = make_client()
    agent = ReactAgent(client, model, on_event=print_event)
    print(c(f"🤖 model: {model} · napiš 'konec' pro ukončení", "2"))
    while True:
        try:
            question = input(c("\nTy: ", "1"))
        except (EOFError, KeyboardInterrupt):
            break
        if question.strip().lower() in {"konec", "exit", "quit", ""}:
            break
        print(c("\nAgent: ", "1;32") + agent.run(question))
        print_usage(agent.usage)  # sleduj, jak tokeny s historií rostou


def compare() -> None:
    """Stejný dotaz bez nástrojů a s nástroji – proč agenti potřebují tools."""
    client, model = make_client()
    exact = 987654321 * 123456789
    print(c(f"❓ {COMPARE_QUESTION}", "1"))
    print(c(f"   (správně: 987654321 × 123456789 = {exact:,})".replace(",", " "), "2"))

    for use_tools, title in [(False, "❌ BEZ nástrojů"), (True, "✅ S nástroji")]:
        agent = ReactAgent(
            client, model, use_tools=use_tools, on_event=print_event if use_tools else (lambda *_: None)
        )
        print(c(f"\n═════ {title} ═════", "1"))
        answer = agent.run(COMPARE_QUESTION)
        print(answer)
        digits = "".join(ch for ch in answer if ch.isdigit())
        verdict = "✔ součin sedí" if str(exact) in digits else "✘ součin NESEDÍ"
        print(c(f"   → kontrola: {verdict}", "32" if "✔" in verdict else "31"))
        print_usage(agent.usage)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM agent s výpočetními nástroji")
    parser.add_argument("question", nargs="*", help="dotaz (jinak ukázkový)")
    parser.add_argument("--chat", action="store_true", help="interaktivní chat s historií")
    parser.add_argument("--compare", action="store_true", help="porovnání bez/s nástroji")
    parser.add_argument("--quiet", action="store_true", help="nevypisovat jednotlivé kroky")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Chybí OPENAI_API_KEY – zkopíruj .env.example do .env a doplň klíč.")

    if args.chat:
        chat()
    elif args.compare:
        compare()
    else:
        ask(" ".join(args.question) or DEMO_QUESTION, quiet=args.quiet)


if __name__ == "__main__":
    main()
