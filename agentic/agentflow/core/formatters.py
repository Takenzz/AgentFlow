import re


def extract_context_subgoal_and_tool(raw_response: str) -> tuple[str, str, str]:
    """
    Extract Context / Sub-Goal / Tool Name from the raw LLM response.
    Compatible with Markdown bold markers and multi-line content; expected format:
      **Justification:** ...
      **Context:** ...
      **Sub-Goal:** ...
      **Tool Name:** ...
    """

    text = raw_response

    # Keep only the last assistant segment (guard against leftover system/user template text)
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]

    # Strip <think> ... </think> blocks, keeping only the formal answer
    if "</think>" in text:
        text = text.split("</think>", 1)[1]

    # Remove Markdown bold markers to simplify pattern matching
    text = text.replace("**", "")

    # Match:
    #   Context:   <any content>
    #   Sub-Goal: <any content>
    #   Tool Name:<any content>
    pattern = r"Context:\s*(.*?)Sub-Goal:\s*(.*?)Tool Name:\s*(.*?)\s*(?:```)?\s*(?=\n\n|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return "", "", ""

    # Use the last match (typically the result of the current step)
    context, sub_goal, tool_name = matches[-1]
    context = context.strip()
    sub_goal = sub_goal.strip()
    # Strip surrounding whitespace and trim trailing special tokens (e.g. <|im_end|>)
    tool_name = tool_name.strip()
    if "<|im_end|>" in tool_name:
        tool_name = tool_name.split("<|im_end|>", 1)[0].strip()
    return context, sub_goal, tool_name
