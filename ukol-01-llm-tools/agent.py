"""ReAct agent: smyčka nad tool callingem.

    while iterace < max_iterations:
        odpověď = LLM(zprávy, nástroje)
        pokud odpověď obsahuje tool_calls:
            spusť nástroje → výsledky přidej do zpráv jako role "tool" → znovu
        jinak:
            vrať text = finální odpověď

LLM API je bezstavové – celou historii (self.messages) posíláme při každém volání.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from openai import OpenAI

from tools import TOOLS, execute_tool

SYSTEM_PROMPT = """Jsi přesný finanční a matematický asistent. Odpovídáš česky.
Dnes je {today}.
Pravidla:
- NIKDY nepočítej z hlavy. Každý výpočet proveď nástrojem `calculate`.
- Kurzy měn neznáš – vždy použij `convert_currency` (oficiální kurz ČNB).
- Splátky úvěrů počítej nástrojem `loan_payment`.
- Nezávislé výpočty si klidně vyžádej najednou (víc tool calls v jedné odpovědi).
- Ve finální odpovědi uveď výsledek srozumitelně a u kurzu i datum a zdroj."""


@dataclass
class Usage:
    """Počítadlo tokenů – s každým kolem posíláme celou historii, takže rostou."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0

    def add(self, usage: Any) -> None:
        self.llm_calls += 1
        if usage is not None:
            self.prompt_tokens += usage.prompt_tokens or 0
            self.completion_tokens += usage.completion_tokens or 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cost_usd(self) -> float | None:
        """Cena podle PRICE_INPUT_PER_M / PRICE_OUTPUT_PER_M v .env (USD za 1M tokenů)."""
        price_in = os.getenv("PRICE_INPUT_PER_M")
        price_out = os.getenv("PRICE_OUTPUT_PER_M")
        if not (price_in and price_out):
            return None
        return (
            self.prompt_tokens * float(price_in) + self.completion_tokens * float(price_out)
        ) / 1_000_000


@dataclass
class ReactAgent:
    client: OpenAI
    model: str
    max_iterations: int = 8
    use_tools: bool = True
    # Callback pro výpis průběhu (main.py ho používá na hezký výstup do terminálu).
    on_event: Callable[[str, dict], None] = lambda kind, data: None
    messages: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        if not self.messages:
            prompt = SYSTEM_PROMPT.format(today=date.today().isoformat())
            if not self.use_tools:
                prompt = f"Jsi asistent. Odpovídáš česky. Dnes je {date.today().isoformat()}."
            self.messages.append({"role": "system", "content": prompt})

    def _call_llm(self):
        kwargs: dict[str, Any] = {"model": self.model, "messages": self.messages}
        if self.use_tools:
            kwargs["tools"] = TOOLS
            kwargs["tool_choice"] = "auto"
        response = self.client.chat.completions.create(**kwargs)
        self.usage.add(response.usage)
        return response.choices[0].message

    def run(self, user_message: str) -> str:
        """Jeden dotaz uživatele → ReAct smyčka → finální odpověď."""
        self.messages.append({"role": "user", "content": user_message})

        for iteration in range(1, self.max_iterations + 1):
            self.on_event("iteration", {"n": iteration})
            message = self._call_llm()

            # 1) Model nechce nástroj → máme finální odpověď.
            if not message.tool_calls:
                answer = message.content or ""
                self.messages.append({"role": "assistant", "content": answer})
                return answer

            # 2) Model chce nástroj(e). Jeho požadavek MUSÍ jít do historie,
            #    jinak by API odmítlo následující zprávy s rolí "tool".
            self.messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
            )

            # 3) Spustíme každý nástroj a výsledek vrátíme modelu (role "tool").
            for tc in message.tool_calls:
                self.on_event(
                    "tool_call", {"name": tc.function.name, "args": tc.function.arguments}
                )
                result = execute_tool(tc.function.name, tc.function.arguments)
                self.usage.tool_calls += 1
                self.on_event("tool_result", {"name": tc.function.name, "result": result})
                self.messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )

        # Pojistka: smyčka nesmí běžet donekonečna (a vypálit kredit).
        return f"Nepodařilo se dokončit ani po {self.max_iterations} krocích."


def make_client() -> tuple[OpenAI, str]:
    """Jeden OpenAI klient pro všechno.

    - přímo OpenAI:      OPENAI_API_KEY=sk-...
    - LiteLLM proxy:     OPENAI_BASE_URL=http://localhost:4000 + OPENAI_API_KEY=<master key>
    - lokální model:     OPENAI_BASE_URL=http://localhost:1234/v1 (LM Studio) apod.
    OpenAI SDK si OPENAI_API_KEY i OPENAI_BASE_URL načte z prostředí sám.
    """
    model = os.getenv("MODEL", "gpt-5.4-nano")
    return OpenAI(), model
