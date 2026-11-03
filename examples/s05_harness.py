"""Offline tool loop with a SCRIPTED model, not DeepSeek or dsh.

Demonstrates a tool whitelist, argument validation, observation messages,
and bounded termination. No shell, file or network execution.
"""
import json


def dispatch(call):
    if not isinstance(call, dict) or call.get("name") != "add":
        raise ValueError("Unknown tool")
    args = call.get("arguments")
    if not isinstance(args, dict) or set(args) != {"a", "b"}:
        raise ValueError("add requires exactly a and b")
    if any(type(v) is not int or abs(v) > 10**6 for v in args.values()):
        raise ValueError("Arguments must be bounded integers")
    return args["a"] + args["b"]


def scripted_model(messages):
    # A fixture: this function does not reason or call a language model.
    if messages[-1]["role"] == "tool":
        return {"type": "final", "content": str(messages[-1]["content"])}
    return {"type": "tool_call", "name": "add",
            "arguments": {"a": 19, "b": 23}}


def run(model=scripted_model, max_steps=4):
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be positive")
    messages = [{"role": "user", "content": "Compute 19 + 23 using add."}]
    for _ in range(max_steps):
        response = model(messages)
        if not isinstance(response, dict):
            raise ValueError("Model response must be an object")
        if response.get("type") == "final":
            content = response.get("content")
            if not isinstance(content, str):
                raise ValueError("Final content must be text")
            messages.append({"role": "assistant", "content": content})
            return content, messages
        if response.get("type") != "tool_call":
            raise ValueError("Unsupported response type")
        # This teaching demo fails closed on invalid calls.
        result = dispatch(response)
        messages.append({"role": "assistant", "tool_call": response})
        messages.append({"role": "tool", "name": response["name"], "content": result})
    raise RuntimeError("Step budget exhausted")


if __name__ == "__main__":
    answer, messages = run()
    print(json.dumps(messages, indent=2, ensure_ascii=False))
    print("final =", answer)
