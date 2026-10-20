"""Run one episode in LangChain, SmolAgents or the OpenAI Agents SDK.

Every framework gets the same tools (built from the task family's ToolSpecs) and every tool
body is `gateway.call(name, kwargs)` -- so replay and the guardrail sit at the tool boundary,
below the framework, and the framework's own message handling is exercised unchanged.
Models are reached through each framework's stock OpenAI-compatible client, so any
OpenAI / OpenRouter / vLLM / Ollama endpoint works.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from .replay import Gateway
from .world import tool_specs

FRAMEWORKS = ("langchain", "smolagents", "openai_agents")
logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class ModelCfg:
    name: str            # label used in results
    model_id: str        # id sent to the endpoint
    base_url: str
    api_key: str
    temperature: float = 0.0


_PY = {"string": str, "number": float}


def _langchain(g: Gateway, m: ModelCfg, system: str, user: str, max_steps: int) -> str:
    from langchain.agents import create_agent
    from langchain_core.tools import StructuredTool
    from langchain_openai import ChatOpenAI
    from pydantic import Field, create_model

    tools = []
    for s in tool_specs(g.inst.family):
        schema = create_model(f"{s.name}_args", **{p: (_PY[t], Field(description=d)) for p, (t, d) in s.params.items()})
        tools.append(StructuredTool.from_function(
            func=(lambda _n: lambda **kw: g.call(_n, kw))(s.name), name=s.name, description=s.description,
            args_schema=schema))
    llm = ChatOpenAI(model=m.model_id, base_url=m.base_url, api_key=m.api_key, temperature=m.temperature, max_retries=3)
    agent = create_agent(llm, tools, system_prompt=system)
    out = agent.invoke({"messages": [{"role": "user", "content": user}]},
                       config={"recursion_limit": 2 * max_steps + 1})
    return str(out["messages"][-1].content)


def _smolagents(g: Gateway, m: ModelCfg, system: str, user: str, max_steps: int) -> str:
    from smolagents import OpenAIServerModel, Tool, ToolCallingAgent
    from smolagents.monitoring import LogLevel

    tools = []
    for s in tool_specs(g.inst.family):
        attrs = {
            "name": s.name, "description": s.description, "output_type": "string",
            "inputs": {p: {"type": t, "description": d} for p, (t, d) in s.params.items()},
            "skip_forward_signature_validation": True,
            "forward": (lambda _n: lambda self, **kw: g.call(_n, kw))(s.name),
        }
        tools.append(type(f"T_{s.name}", (Tool,), attrs)())
    model = OpenAIServerModel(model_id=m.model_id, api_base=m.base_url, api_key=m.api_key, temperature=m.temperature)
    agent = ToolCallingAgent(tools=tools, model=model, max_steps=max_steps, verbosity_level=LogLevel.OFF,
                             instructions=system)
    return str(agent.run(user))


def _openai_agents(g: Gateway, m: ModelCfg, system: str, user: str, max_steps: int) -> str:
    from agents import Agent, FunctionTool, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
    from agents.model_settings import ModelSettings
    from openai import AsyncOpenAI

    set_tracing_disabled(True)
    tools = []
    for s in tool_specs(g.inst.family):
        async def invoke(ctx, args_json, _n=s.name):
            return g.call(_n, json.loads(args_json or "{}"))
        tools.append(FunctionTool(
            name=s.name, description=s.description, on_invoke_tool=invoke,
            params_json_schema={"type": "object", "additionalProperties": False, "required": list(s.params),
                                "properties": {p: {"type": t, "description": d} for p, (t, d) in s.params.items()}}))
    client = AsyncOpenAI(base_url=m.base_url, api_key=m.api_key, max_retries=3)
    agent = Agent(name="ops", instructions=system, tools=tools,
                  model=OpenAIChatCompletionsModel(model=m.model_id, openai_client=client),
                  model_settings=ModelSettings(temperature=m.temperature, parallel_tool_calls=False))
    return str(Runner.run_sync(agent, user, max_turns=max_steps).final_output)


RUNNERS = {"langchain": _langchain, "smolagents": _smolagents, "openai_agents": _openai_agents}


def run_episode(framework: str, g: Gateway, m: ModelCfg, system: str, user: str, max_steps: int = 10) -> tuple[str, str | None]:
    try:
        return RUNNERS[framework](g, m, system, user, max_steps), None
    except Exception as e:  # max-steps, API errors, parse failures: record, don't crash the sweep
        return "", f"{type(e).__name__}: {str(e)[:300]}"
