"""Prefix normalizer for Claude Code SWE-bench context.

This normalizer is loaded by gateway via FQN and runs in the gateway prefix
comparison path. It must be symmetric: both sides of the comparison are
normalized independently, and the function must not depend on which side is
processed first.

Contract:
- Input: a base-canonicalized message (already through
  codec.canonicalize_message_for_prefix_comparison). The gateway passes a deep
  copy, so this function may mutate the input in place -- doing so does not
  affect the real request/session payload or token truth.
  Tool call arguments shape is ("json", dict) for JSON-parsed args.
- Output: normalized message (returning the mutated input is fine).
- Exceptions: caught by the gateway and treated as mismatch.

Other recipes can follow this pattern: keep recipe-specific constants local to
the recipe normalizer, then expose one FQN in recipe config.
"""

from typing import Any


def normalize_for_claude_code_swe(message: dict[str, Any]) -> dict[str, Any]:
    """Normalize assistant tool calls for the Claude Code SWE allowlist.

    Four equivalence rules:
    1. Edit.replace_all: absent is equivalent to false.
    2. Bash.command: strip the "cd /testbed && " prefix.
    3. Edit.new_string: rstrip each line.
    4. Write.content: rstrip each line.

    "/testbed" is hardcoded here because it is Claude Code's known shell cwd
    for this recipe. The gateway does not receive or interpret that parameter.
    Input is a base-canonicalized message and is mutated in place (the gateway
    passes a deep copy). Must remain symmetric: applying it to history/request
    in either order produces the same normalized representation for equivalent
    messages.
    """
    if message.get("role") != "assistant":
        return message

    tool_calls = message.get("tool_calls") or []
    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        name = function.get("name")
        args_tuple = function.get("arguments")

        if not (isinstance(args_tuple, tuple) and len(args_tuple) == 2 and args_tuple[0] == "json"):
            continue

        args = dict(args_tuple[1])

        if name == "Edit":
            if "replace_all" not in args or args.get("replace_all") is False:
                args.pop("replace_all", None)
            if isinstance(args.get("new_string"), str):
                args["new_string"] = "\n".join(line.rstrip() for line in args["new_string"].split("\n"))
        elif name == "Write":
            if isinstance(args.get("content"), str):
                args["content"] = "\n".join(line.rstrip() for line in args["content"].split("\n"))
        elif name == "Bash":
            if isinstance(args.get("command"), str):
                args["command"] = args["command"].removeprefix("cd /testbed && ")

        function["arguments"] = ("json", args)

    return message
