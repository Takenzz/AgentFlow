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

    # Normalize numbered headers such as "1. Context:" or inline
    # "... 4. Tool Name:" before section extraction.
    header_names = r"(Justification|Context|Sub-Goal|Tool Name)"
    text = re.sub(rf"(?im)(?:^|\n)\s*\d+\.\s*{header_names}\s*:", r"\n\1:", text)
    text = re.sub(rf"(?im)\s+\d+\.\s*{header_names}\s*:", r"\n\1:", text)

    # Match:
    #   Context:   <any content>
    #   Sub-Goal: <any content>
    #   Tool Name:<any content>
    pattern = (
        r"Context:\s*(.*?)\s*"
        r"Sub-Goal:\s*(.*?)\s*"
        r"Tool Name:\s*(.*?)\s*(?:```)?\s*"
        r"(?=\n\s*(?:Justification|Context|Sub-Goal|Tool Name):|\Z)"
    )
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return "", "", ""

    # Use the last match (typically the result of the current step)
    context, sub_goal, tool_name = matches[-1]
    context = context.strip()
    sub_goal = re.sub(r"\s+\d+\.\s*$", "", sub_goal.strip()).strip()
    # Strip surrounding whitespace and trim trailing special tokens (e.g. <|im_end|>)
    tool_name = tool_name.strip()
    if "<|im_end|>" in tool_name:
        tool_name = tool_name.split("<|im_end|>", 1)[0].strip()
    tool_name = tool_name.splitlines()[0].strip()
    tool_name = tool_name.strip("`'\" ")
    tool_name = re.sub(r"[.,;:]+$", "", tool_name).strip()
    return context, sub_goal, tool_name
