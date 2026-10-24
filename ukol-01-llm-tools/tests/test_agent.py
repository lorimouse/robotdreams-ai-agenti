"""Test ReAct smyčky BEZ API klíče – LLM nahradíme falešným klientem,
který vrací předem připravené odpovědi (jako by je poslal model)."""

import json
from types import SimpleNamespace as NS

from agent import ReactAgent


def tool_call(id_, name, args):
    return NS(id=id_, function=NS(name=name, arguments=json.dumps(args)))


def llm_response(content=None, tool_calls=None):
    message = NS(content=content, tool_calls=tool_calls)
    usage = NS(prompt_tokens=100, completion_tokens=20)
    return NS(choices=[NS(message=message)], usage=usage)


class FakeClient:
    """Napodobí client.chat.completions.create(...) a zapamatuje si, co dostal."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        # kopie zpráv – agent je po volání dál upravuje
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def test_full_loop_with_parallel_tool_calls():
    client = FakeClient(
        [
            # 1. kolo: model chce DVA nástroje najednou
            llm_response(
                tool_calls=[
                    tool_call("c1", "calculate", {"expression": "6 * 7"}),
                    tool_call("c2", "loan_payment",
                              {"principal": 120000, "annual_rate_percent": 0, "years": 10}),
                ]
            ),
            # 2. kolo: model má výsledky a odpoví textem
            llm_response(content="Výsledek je 42 a splátka 1000 Kč."),
        ]
    )
    agent = ReactAgent(client, "fake-model")
    answer = agent.run("Spočítej to.")

    assert answer == "Výsledek je 42 a splátka 1000 Kč."
    roles = [m["role"] for m in agent.messages]
    assert roles == ["system", "user", "assistant", "tool", "tool", "assistant"]

    # výsledky nástrojů se opravdu vrátily modelu
    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    assert json.loads(tool_msgs[0]["content"])["result"] == 42
    assert json.loads(tool_msgs[1]["content"])["monthly_payment"] == 1000
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]

    # druhé volání LLM už obsahovalo výsledky nástrojů (API je bezstavové)
    assert [m["role"] for m in client.requests[1]["messages"]][-2:] == ["tool", "tool"]
    assert client.requests[0]["tools"]  # nástroje se posílají s každým voláním

    assert agent.usage.llm_calls == 2
    assert agent.usage.tool_calls == 2
    assert agent.usage.total_tokens == 240


def test_tool_error_goes_back_to_model():
    client = FakeClient(
        [
            llm_response(tool_calls=[tool_call("c1", "calculate", {"expression": "1/0"})]),
            llm_response(content="Nulou dělit nelze."),
        ]
    )
    agent = ReactAgent(client, "fake-model")
    assert agent.run("1/0?") == "Nulou dělit nelze."
    assert "ZeroDivisionError" in agent.messages[3]["content"]


def test_max_iterations_guard():
    endless = [
        llm_response(tool_calls=[tool_call(f"c{i}", "calculate", {"expression": "1+1"})])
        for i in range(10)
    ]
    agent = ReactAgent(FakeClient(endless), "fake-model", max_iterations=3)
    assert "3 krocích" in agent.run("smyčka")
    assert agent.usage.llm_calls == 3


def test_history_is_kept_between_questions():
    client = FakeClient([llm_response(content="Ahoj Martine."), llm_response(content="Martin.")])
    agent = ReactAgent(client, "fake-model")
    agent.run("Jmenuju se Martin.")
    agent.run("Jak se jmenuju?")
    # druhý request nese i první otázku a odpověď → model si „pamatuje“
    contents = [m["content"] for m in client.requests[1]["messages"]]
    assert "Jmenuju se Martin." in contents and "Ahoj Martine." in contents


def test_without_tools_sends_no_tools():
    client = FakeClient([llm_response(content="Asi 1,2 × 10^17.")])
    ReactAgent(client, "fake-model", use_tools=False).run("987654321 × 123456789?")
    assert "tools" not in client.requests[0]
