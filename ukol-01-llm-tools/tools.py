"""Nástroje (tools), které může LLM zavolat.

Každý nástroj má dvě části:
  1. obyčejnou Python funkci, která dělá skutečnou práci,
  2. JSON schéma, podle kterého LLM pozná, KDY a JAK nástroj použít.
     Nejdůležitější je `description` – podle ní se model rozhoduje.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import urllib.request
from collections.abc import Callable
from typing import Any

# ---------------------------------------------------------------------------
# 1) calculate – bezpečná kalkulačka (AST místo eval)
# ---------------------------------------------------------------------------
# Proč ne eval()? Model generuje text a ten může obsahovat cokoliv, třeba
# "__import__('os').system('rm -rf ~')". Proto výraz rozparsujeme na strom
# (AST) a povolíme jen čísla, operátory a pár matematických funkcí.

_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}

MAX_EXPONENT = 10_000  # ochrana proti "9**9**9**9", které by zamrazilo počítač


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"Exponent je příliš velký (max {MAX_EXPONENT}).")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*(_eval_node(a) for a in node.args))
    raise ValueError(f"Nepovolený prvek ve výrazu: {ast.dump(node)[:60]}")


def calculate(expression: str) -> dict:
    """Spočítá matematický výraz, např. '(1250 * 1.21) / 12'."""
    # Modely občas pošlou "^" pro mocninu nebo symboly × a ÷.
    cleaned = expression.replace("^", "**").replace("×", "*").replace("÷", "/")
    tree = ast.parse(cleaned, mode="eval")
    result = _eval_node(tree)
    if isinstance(result, float):
        result = round(result, 10)
    return {"expression": expression, "result": result}


# ---------------------------------------------------------------------------
# 2) convert_currency – převod měn podle dnešního kurzu ČNB
# ---------------------------------------------------------------------------
# Živá data, která model z tréninku znát NEMŮŽE (jako cena akcie v lekci).

CNB_URL = (
    "https://www.cnb.cz/cs/financni-trhy/devizovy-trh/"
    "kurzy-devizoveho-trhu/kurzy-devizoveho-trhu/denni_kurz.txt"
)
_rates_cache: dict[str, Any] = {}


def parse_cnb_rates(text: str) -> tuple[str, dict[str, float]]:
    """Rozparsuje textový kurzovní lístek ČNB → (datum, {kód: CZK za 1 jednotku})."""
    lines = text.strip().splitlines()
    rate_date = lines[0].split(" ")[0]
    rates = {"CZK": 1.0}
    for line in lines[2:]:
        _country, _name, amount, code, rate = line.split("|")
        rates[code] = float(rate.replace(",", ".")) / int(amount)
    return rate_date, rates


def get_cnb_rates() -> tuple[str, dict[str, float]]:
    """Stáhne kurzy ČNB (jednou za běh programu – cache)."""
    if "data" not in _rates_cache:
        with urllib.request.urlopen(CNB_URL, timeout=10) as resp:
            _rates_cache["data"] = parse_cnb_rates(resp.read().decode("utf-8"))
    return _rates_cache["data"]


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Převede částku mezi měnami podle aktuálního kurzu ČNB."""
    rate_date, rates = get_cnb_rates()
    src, dst = from_currency.upper(), to_currency.upper()
    for code in (src, dst):
        if code not in rates:
            raise ValueError(
                f"Měnu '{code}' ČNB nevede. Dostupné: {', '.join(sorted(rates))}"
            )
    result = amount * rates[src] / rates[dst]
    return {
        "amount": amount,
        "from": src,
        "to": dst,
        "result": round(result, 2),
        "rate": round(rates[src] / rates[dst], 6),
        "rate_date": rate_date,
        "source": "Česká národní banka",
    }


# ---------------------------------------------------------------------------
# 3) loan_payment – měsíční splátka úvěru / hypotéky (anuita)
# ---------------------------------------------------------------------------


def loan_payment(principal: float, annual_rate_percent: float, years: float) -> dict:
    """Měsíční anuitní splátka, celková zaplacená částka a přeplatek."""
    if principal <= 0 or years <= 0:
        raise ValueError("Výše úvěru i doba splácení musí být kladné.")
    n = round(years * 12)
    r = annual_rate_percent / 100 / 12
    monthly = principal / n if r == 0 else principal * r / (1 - (1 + r) ** -n)
    total = monthly * n
    return {
        "monthly_payment": round(monthly, 2),
        "number_of_payments": n,
        "total_paid": round(total, 2),
        "total_interest": round(total - principal, 2),
    }


# ---------------------------------------------------------------------------
# JSON schémata pro LLM (OpenAI formát – LiteLLM ho přeloží pro ostatní)
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Přesně spočítá matematický výraz. Použij VŽDY, když je potřeba "
                "cokoliv počítat – nikdy nepočítej z hlavy. Podporuje + - * / // % **, "
                "závorky, sqrt, log, log10, exp, sin, cos, tan, round, abs, floor, "
                "ceil, factorial a konstanty pi, e. Desetinná tečka, ne čárka."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Výraz v Python syntaxi, např. '(1250 * 1.21) / 12'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": (
                "Převede částku mezi měnami podle DNEŠNÍHO kurzu České národní banky. "
                "Použij pro jakýkoliv dotaz na kurz nebo převod měn (včetně CZK)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Částka k převodu"},
                    "from_currency": {
                        "type": "string",
                        "description": "ISO kód zdrojové měny, např. EUR",
                    },
                    "to_currency": {
                        "type": "string",
                        "description": "ISO kód cílové měny, např. CZK",
                    },
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "loan_payment",
            "description": (
                "Spočítá měsíční splátku úvěru nebo hypotéky (anuita), celkem "
                "zaplacenou částku a přeplatek na úrocích."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "principal": {"type": "number", "description": "Výše úvěru"},
                    "annual_rate_percent": {
                        "type": "number",
                        "description": "Roční úroková sazba v procentech, např. 4.9",
                    },
                    "years": {"type": "number", "description": "Doba splácení v letech"},
                },
                "required": ["principal", "annual_rate_percent", "years"],
            },
        },
    },
]

TOOL_FUNCTIONS: dict[str, Callable[..., dict]] = {
    "calculate": calculate,
    "convert_currency": convert_currency,
    "loan_payment": loan_payment,
}


def execute_tool(name: str, arguments_json: str) -> str:
    """Spustí nástroj podle jména a vrátí výsledek jako JSON string.

    Chyby NEvyhazujeme – vracíme je modelu jako {"error": ...}. Model si chybu
    přečte a může to zkusit znovu (např. opravit výraz). Agent tak nespadne.
    """
    try:
        func = TOOL_FUNCTIONS[name]
    except KeyError:
        return json.dumps({"error": f"Neznámý nástroj '{name}'."}, ensure_ascii=False)
    try:
        args = json.loads(arguments_json or "{}")
        result = func(**args)
    except Exception as exc:  # noqa: BLE001 – chceme modelu vrátit jakoukoliv chybu
        result = {"error": f"{type(exc).__name__}: {exc}"}
    return json.dumps(result, ensure_ascii=False)
