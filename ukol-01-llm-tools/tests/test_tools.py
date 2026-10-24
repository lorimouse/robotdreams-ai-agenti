import json

import pytest

import tools
from tools import (
    calculate,
    convert_currency,
    execute_tool,
    loan_payment,
    parse_cnb_rates,
)

CNB_SAMPLE = """23.10.2026 #205
země|měna|množství|kód|kurz
EMU|euro|1|EUR|24,500
Japonsko|jen|100|JPY|15,000
USA|dolar|1|USD|22,000
"""


# --- calculate -------------------------------------------------------------
@pytest.mark.parametrize(
    "expr, expected",
    [
        ("2 + 3 * 4", 14),
        ("(1250 * 1.21) / 12", 126.0416666667),
        ("2 ^ 10", 1024),
        ("987654321 * 123456789", 121932631112635269),
        ("sqrt(144) + abs(-3)", 15.0),
        ("round(pi, 2)", 3.14),
        ("-5 % 3", 1),
    ],
)
def test_calculate(expr, expected):
    assert calculate(expr)["result"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "evil",
    [
        "__import__('os').system('echo hacked')",
        "open('/etc/passwd').read()",
        "(1).__class__",
        "x + 1",
    ],
)
def test_calculate_rejects_code(evil):
    with pytest.raises((ValueError, SyntaxError)):
        calculate(evil)


def test_calculate_blocks_huge_exponent():
    with pytest.raises(ValueError, match="Exponent"):
        calculate("9 ** 99999")


# --- loan_payment ------------------------------------------------------------
def test_loan_payment_known_value():
    # 1 000 000 na 30 let při 5 % → 5 368,22 měsíčně (ověřitelné v kalkulačce banky)
    r = loan_payment(1_000_000, 5, 30)
    assert r["monthly_payment"] == pytest.approx(5368.22, abs=0.01)
    assert r["number_of_payments"] == 360


def test_loan_payment_zero_interest():
    assert loan_payment(120_000, 0, 10)["monthly_payment"] == 1000


def test_loan_payment_invalid():
    with pytest.raises(ValueError):
        loan_payment(-1, 5, 30)


# --- convert_currency (bez internetu – kurzy podstrčíme) ---------------------
@pytest.fixture
def fake_rates(monkeypatch):
    monkeypatch.setitem(tools._rates_cache, "data", parse_cnb_rates(CNB_SAMPLE))


def test_parse_cnb_rates_handles_amount():
    rate_date, rates = parse_cnb_rates(CNB_SAMPLE)
    assert rate_date == "23.10.2026"
    assert rates["JPY"] == pytest.approx(0.15)  # 100 JPY = 15 Kč
    assert rates["CZK"] == 1.0


def test_convert_eur_to_czk(fake_rates):
    assert convert_currency(100, "eur", "CZK")["result"] == 2450.0


def test_convert_cross_rate(fake_rates):
    # EUR → USD přes korunu: 24,5 / 22
    assert convert_currency(22, "EUR", "USD")["result"] == pytest.approx(24.5)


def test_convert_unknown_currency(fake_rates):
    with pytest.raises(ValueError, match="XYZ"):
        convert_currency(1, "XYZ", "CZK")


# --- execute_tool: chyby se vrací modelu, ne vyhazují --------------------------
def test_execute_tool_ok():
    assert json.loads(execute_tool("calculate", '{"expression": "6*7"}'))["result"] == 42


def test_execute_tool_returns_error_instead_of_crashing():
    out = json.loads(execute_tool("calculate", '{"expression": "1/0"}'))
    assert "ZeroDivisionError" in out["error"]


def test_execute_tool_unknown():
    assert "error" in json.loads(execute_tool("hack_the_planet", "{}"))


def test_all_schemas_have_implementation():
    names = {t["function"]["name"] for t in tools.TOOLS}
    assert names == set(tools.TOOL_FUNCTIONS)
