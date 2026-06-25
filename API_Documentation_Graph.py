# =============================================================================
# API Documentation Generator -- A LangGraph Learning Project
# =============================================================================
#
# This project builds on the Mental Wellness pattern to generate documentation
# from an API endpoint spec or raw endpoint code.
#
# WHAT THIS DOES:
# A user pastes an API spec or endpoint code (e.g. FastAPI routes, OpenAPI YAML,
# Express handlers). The system runs 3 specialist nodes in PARALLEL
# (describe_endpoints, document_params_and_responses, generate_usage_examples),
# then a decision node picks the right output format and routes to either a
# QUICK REFERENCE (concise cheat-sheet) or FULL DOCUMENTATION (comprehensive
# docs with all detail) based on complexity.
#
# LANGGRAPH CONCEPTS COVERED:
# 1. State Management (Pydantic) -- API spec flows through the graph
# 2. Nodes -- each function does one job (describe, document params, etc.)
# 3. Parallel Execution -- 3 specialist nodes run at the same time
# 4. Fan-in -- waiting for all 3 before the decision node runs
# 5. Conditional Edges -- routing to quick vs full based on complexity
# 6. Graph Compilation -- turning the graph definition into a runnable app
#
# GRAPH STRUCTURE:
#
#   START
#     |
#   parse_api_input
#     |
#     +---> describe_endpoints ----------------+
#     |                                        |
#     +---> document_params_and_responses -----+---> pick_doc_style
#     |                                        |         |
#     +---> generate_usage_examples -----------+    (conditional)
#                                                  /           \
#                                              quick?          full?
#                                                |               |
#                                       quick_api_reference  full_api_documentation
#                                                |               |
#                                               END             END
#
# HOW TO RUN:
#   python api_doc_generator.py
#
# DEPENDENCIES (same as requirements.txt):
#   langgraph, langchain-openai, python-dotenv, pydantic
#
# =============================================================================

import sys
import os
import operator
import json
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY not found. "
        "Add it to your .env file as: OPENAI_API_KEY=sk-..."
    )


class ApiDocState(BaseModel):
    api_input: str = ""
    endpoint_descriptions: str = ""
    params_and_responses: str = ""
    usage_examples: str = ""
    needs_full_docs: bool = False
    doc_style_reason: str = ""
    final_documentation: str = ""
    messages: Annotated[list, operator.add] = []


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=OPENAI_API_KEY)


def parse_api_input(state: ApiDocState) -> dict:
    response = llm.invoke(
        f"You are a senior API analyst. "
        f"A user has provided the following API spec or endpoint code:\n\n"
        f"```\n{state.api_input}\n```\n\n"
        f"Briefly acknowledge what kind of API this appears to be (REST, GraphQL, gRPC, etc.), "
        f"the tech stack if identifiable, and roughly how many endpoints are present. "
        f"Keep it to 2-3 sentences."
    )
    return {
        "messages": [f"[parse_api_input] {response.content}"]
    }


def describe_endpoints(state: ApiDocState) -> dict:
    response = llm.invoke(
        f"You are an API documentation specialist focused on endpoint descriptions. "
        f"Given this API spec or code:\n\n"
        f"```\n{state.api_input}\n```\n\n"
        f"For each endpoint you can identify, describe:\n"
        f"- The HTTP method and path (or operation name for non-REST)\n"
        f"- A clear, one-sentence description of what it does\n"
        f"- Who or what would call it (use-case context)\n\n"
        f"Format as a clean list. Be precise and developer-friendly."
    )
    return {
        "endpoint_descriptions": response.content,
        "messages": [f"[describe_endpoints] Done"]
    }


def document_params_and_responses(state: ApiDocState) -> dict:
    response = llm.invoke(
        f"You are an API documentation specialist focused on parameters and schemas. "
        f"Given this API spec or code:\n\n"
        f"```\n{state.api_input}\n```\n\n"
        f"For each endpoint, document:\n"
        f"- Path/query/header parameters (name, type, required, description)\n"
        f"- Request body schema (fields, types, required vs optional)\n"
        f"- Response schema for success and common error codes\n\n"
        f"Format as a clean, structured list per endpoint. "
        f"If a field is inferred rather than explicit, note it."
    )
    return {
        "params_and_responses": response.content,
        "messages": [f"[document_params_and_responses] Done"]
    }


def generate_usage_examples(state: ApiDocState) -> dict:
    response = llm.invoke(
        f"You are an API documentation specialist focused on practical examples. "
        f"Given this API spec or code:\n\n"
        f"```\n{state.api_input}\n```\n\n"
        f"For each endpoint, provide:\n"
        f"- A realistic example request (curl command or JSON body)\n"
        f"- An example success response (JSON)\n"
        f"- One example error response where relevant\n\n"
        f"Use realistic placeholder values. Keep examples concise but complete."
    )
    return {
        "usage_examples": response.content,
        "messages": [f"[generate_usage_examples] Done"]
    }


def pick_doc_style(state: ApiDocState) -> dict:
    response = llm.invoke(
        f"You are a documentation decision system. "
        f"The user provided this API:\n\n"
        f"```\n{state.api_input}\n```\n\n"
        f"Here is what three specialists produced:\n\n"
        f"ENDPOINT DESCRIPTIONS:\n{state.endpoint_descriptions}\n\n"
        f"PARAMS AND RESPONSES:\n{state.params_and_responses}\n\n"
        f"USAGE EXAMPLES:\n{state.usage_examples}\n\n"
        f"Decide: should this produce QUICK REFERENCE docs (concise cheat-sheet, "
        f"for simple APIs with few endpoints or straightforward patterns) "
        f"or FULL DOCUMENTATION (comprehensive, structured docs for complex APIs, "
        f"many endpoints, or non-obvious behavior)?\n\n"
        f"Reply STRICTLY in this JSON format (no other text):\n"
        f'{{"needs_full_docs": true/false, "reason": "one sentence explanation"}}'
    )
    try:
        # Strip markdown fences if the model wraps its JSON in them
        cleaned = response.content.strip()
        for fence in ("```json", "```"):
            if cleaned.startswith(fence):
                cleaned = cleaned[len(fence):]
        cleaned = cleaned.rstrip("`").strip()
        result = json.loads(cleaned)
        needs_full = result["needs_full_docs"]
        reason = result["reason"]
    except (json.JSONDecodeError, KeyError):
        needs_full = False
        reason = "Could not parse decision, defaulting to quick reference."

    return {
        "needs_full_docs": needs_full,
        "doc_style_reason": reason,
        "messages": [f"[pick_doc_style] full_docs={needs_full}"]
    }


def quick_api_reference(state: ApiDocState) -> dict:
    response = llm.invoke(
        f"You are a technical writer producing a quick-reference card. "
        f"The user provided this API:\n\n"
        f"```\n{state.api_input}\n```\n\n"
        f"Using these specialist inputs, produce a CONCISE quick-reference:\n\n"
        f"DESCRIPTIONS: {state.endpoint_descriptions}\n"
        f"PARAMS: {state.params_and_responses}\n"
        f"EXAMPLES: {state.usage_examples}\n\n"
        f"Format:\n"
        f"- A one-line summary of the API at the top\n"
        f"- Each endpoint as: METHOD /path — description — key params — quick example\n"
        f"- Keep the whole thing scannable in under 2 minutes\n"
        f"Use markdown formatting. Aim for brevity over completeness."
    )
    return {
        "final_documentation": (
            f"QUICK API REFERENCE\n{'=' * 45}\n{response.content}"
        ),
        "messages": [f"[quick_api_reference] Generated quick reference"]
    }


def full_api_documentation(state: ApiDocState) -> dict:
    response = llm.invoke(
        f"You are a technical writer producing comprehensive API documentation. "
        f"The user provided this API:\n\n"
        f"```\n{state.api_input}\n```\n\n"
        f"Using these specialist inputs, produce FULL documentation:\n\n"
        f"DESCRIPTIONS: {state.endpoint_descriptions}\n"
        f"PARAMS: {state.params_and_responses}\n"
        f"EXAMPLES: {state.usage_examples}\n\n"
        f"Structure it with these sections:\n"
        f"1. Overview — what this API does and who it's for\n"
        f"2. Base URL & Authentication — inferred if not explicit\n"
        f"3. Endpoints — one subsection per endpoint with full detail:\n"
        f"   - Description, method, path\n"
        f"   - Parameters table (name, type, required, description)\n"
        f"   - Request body schema\n"
        f"   - Response schema (success + errors)\n"
        f"   - Example request and response\n"
        f"4. Common Error Codes — consolidated error reference\n\n"
        f"Use markdown formatting with clear headers. Be thorough and developer-friendly."
    )
    return {
        "final_documentation": (
            f"FULL API DOCUMENTATION\n{'=' * 45}\n{response.content}"
        ),
        "messages": [f"[full_api_documentation] Generated full documentation"]
    }


def route_after_decision(state: ApiDocState) -> str:
    if state.needs_full_docs:
        return "full"
    else:
        return "quick"


graph = StateGraph(ApiDocState)

graph.add_node("parse_api_input", parse_api_input)
graph.add_node("describe_endpoints", describe_endpoints)
graph.add_node("document_params_and_responses", document_params_and_responses)
graph.add_node("generate_usage_examples", generate_usage_examples)
graph.add_node("pick_doc_style", pick_doc_style)
graph.add_node("quick_api_reference", quick_api_reference)
graph.add_node("full_api_documentation", full_api_documentation)

graph.add_edge(START, "parse_api_input")

graph.add_edge("parse_api_input", "describe_endpoints")
graph.add_edge("parse_api_input", "document_params_and_responses")
graph.add_edge("parse_api_input", "generate_usage_examples")

graph.add_edge("describe_endpoints", "pick_doc_style")
graph.add_edge("document_params_and_responses", "pick_doc_style")
graph.add_edge("generate_usage_examples", "pick_doc_style")

graph.add_conditional_edges(
    "pick_doc_style",
    route_after_decision,
    {
        "quick": "quick_api_reference",
        "full": "full_api_documentation",
    }
)

graph.add_edge("quick_api_reference", END)
graph.add_edge("full_api_documentation", END)

app = graph.compile()


def run_api_doc_generator(api_input: str):
    print("=" * 55)
    print("  API DOCUMENTATION GENERATOR")
    print("=" * 55)

    result = app.invoke({
        "api_input": api_input,
        "messages": [],
    })

    print("\n" + "=" * 55)
    print("  YOUR GENERATED DOCUMENTATION")
    print("=" * 55)
    print(f"\n{result['final_documentation']}")

    print("\n" + "-" * 55)
    print(f"  Decision reason: {result['doc_style_reason']}")
    print("-" * 55)
    print("  MESSAGE LOG")
    print("-" * 55)
    for msg in result["messages"]:
        print(f"  {msg}")

    return result


EXAMPLE_SPEC = """
POST /auth/login
  Body: { "email": string, "password": string }
  Returns: { "token": string, "expires_in": number }

GET /users/{id}
  Headers: Authorization: Bearer <token>
  Returns: { "id": string, "email": string, "name": string, "created_at": string }

PUT /users/{id}
  Headers: Authorization: Bearer <token>
  Body: { "name"?: string, "email"?: string }
  Returns: updated user object

DELETE /users/{id}
  Headers: Authorization: Bearer <token>
  Returns: 204 No Content

GET /users/{id}/posts
  Headers: Authorization: Bearer <token>
  Query: ?page=1&limit=20
  Returns: { "data": Post[], "total": number, "page": number }
"""


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  API DOCUMENTATION GENERATOR")
    print("=" * 55)
    print("\n  Paste your API spec or endpoint code below.")
    print("  Type 'DONE' on a new line when finished.")
    print("  Type 'example' to use a built-in example spec.")
    print("  Type 'quit' to exit.\n")

    while True:
        first_line = input("  > ").strip()

        if first_line.lower() in ("quit", "exit", "q"):
            print("\n  Happy documenting! Goodbye!\n")
            break

        if first_line.lower() == "example":
            print("\n  Using built-in example spec...\n")
            run_api_doc_generator(EXAMPLE_SPEC)
            print("\n")
            continue

        if not first_line:
            continue

        # Multi-line input collection
        lines = [first_line]
        print("  (continue pasting, then type DONE on a new line)")
        while True:
            line = input()
            if line.strip().upper() == "DONE":
                break
            lines.append(line)

        api_input = "\n".join(lines)
        if api_input.strip():
            run_api_doc_generator(api_input)
            print("\n")