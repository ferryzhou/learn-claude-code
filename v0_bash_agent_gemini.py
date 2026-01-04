#!/usr/bin/env python
"""
v0_bash_agent_gemini.py - A new bash agent using the google-genai library.

Based on https://github.com/shareAI-lab/learn-claude-code/blob/main/v0_bash_agent.py.

Generated with antigravity with prompt: "create a new bash agent python file that use google genai library and bash tool."


Why Bash is Enough:
------------------
Unix philosophy says everything is a file, everything can be piped.
Bash is the gateway to this world:

    | You need      | Bash command                           |
    |---------------|----------------------------------------|
    | Read files    | cat, head, tail, grep                  |
    | Write files   | echo '...' > file, cat << 'EOF' > file |
    | Search        | find, grep, rg, ls                     |
    | Execute       | python, npm, make, any command         |
    | **Subagent**  | python v0_bash_agent.py "task"         |

The last line is the KEY INSIGHT: calling itself via bash implements subagents!
No Task tool, no Agent Registry - just recursion through process spawning.

How Subagents Work:
------------------
    Main Agent
      |-- bash: python v0_bash_agent.py "analyze architecture"
           |-- Subagent (isolated process, fresh history)
                |-- bash: find . -name "*.py"
                |-- bash: cat src/main.py
                |-- Returns summary via stdout

Process isolation = Context isolation:
- Child process has its own history=[]
- Parent captures stdout as tool result
- Recursive calls enable unlimited nesting

Usage:
    # Interactive mode
    python v0_bash_agent.py

    # Subagent mode (called by parent agent or directly)
    python v0_bash_agent.py "explore src/ and summarize"
"""

import os
import sys
import subprocess
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Configuration
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("MODEL_NAME", "gemini-2.5-flash")

if not API_KEY:
    sys.exit("Please set GEMINI_API_KEY in .env")

client = genai.Client(api_key=API_KEY)

TOOLS = [{
    "name": "bash",
    "description": """Execute shell command. Common patterns:
- Read: cat/head/tail, grep/find/rg/ls, wc -l
- Write: echo 'content' > file, sed -i 's/old/new/g' file
- Subagent: python v0_bash_agent_gemini.py 'task description' (spawns isolated agent, returns summary)""",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"]
    }
}]

# System prompt teaches the model HOW to use bash effectively
# Notice the subagent guidance - this is how we get hierarchical task decomposition
SYSTEM = f"""You are a CLI agent at {os.getcwd()}. Solve problems using bash commands.

Rules:
- Prefer tools over prose. Act first, explain briefly after.
- Read files: cat, grep, find, rg, ls, head, tail
- Write files: echo '...' > file, sed -i, or cat << 'EOF' > file
- Subagent: For complex subtasks, spawn a subagent to keep context clean:
  python v0_bash_agent.py "explore src/ and summarize the architecture"

When to use subagent:
- Task requires reading many files (isolate the exploration)
- Task is independent and self-contained
- You want to avoid polluting current conversation with intermediate details

The subagent runs in isolation and returns only its final summary."""

def execute_bash(command: str) -> str:
    """Executes a shell command and returns output."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout + result.stderr
    except Exception as e:
        return f"Execution error: {e}"

def chat(prompt, history=None):
    """
    The complete agent loop in ONE function.

    This is the core pattern that ALL coding agents share:
        while not done:
            response = model(messages, tools)
            if no tool calls: return
            execute tools, append results

    Args:
        prompt: User's request
        history: Conversation history (mutable, shared across calls in interactive mode)

    Returns:
        Final text response from the model
    """
    if history is None:
        history = []
    
    # Add user message
    history.append({"role": "user", "parts": [{"text": prompt}]})
    
    while True:
        # 1. Call the model with tools
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM,
                    tools=[{"function_declarations": TOOLS}]
                )
            )
        except Exception as e:
            return f"Error communicating with API: {e}"

        if not response.candidates:
            return "No response from model."

        # 2. Build assistant message content (preserve both text and tool_use blocks)
        candidate = response.candidates[0]
        content = candidate.content
        history.append(content)
        
        # Check for function calls
        function_calls = []
        if content.parts:
            for part in content.parts:
                if part.function_call:
                    function_calls.append(part.function_call)
        
        # 3. If model didn't call tools, we're done
        if not function_calls:
            # Return text response
            if not content.parts:
                 return ""
            text_parts = [p.text for p in content.parts if p.text]
            return "".join(text_parts)
        
        # 4. Execute each tool call and collect results
        tool_outputs = []
        for fc in function_calls:
            if fc.name == "bash":
                cmd = fc.args["command"]
                # print(f"\033[33m[Executing: {cmd}]\033[0m", file=sys.stderr)
                result = execute_bash(cmd)
                tool_outputs.append({
                    "function_response": {
                        "name": "bash",
                        "response": {"result": result}
                    }
                })
        
        # 5. Append results and continue the loop
        history.append({"role": "tool", "parts": tool_outputs})

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Subagent mode: execute task and print result
        # This is how parent agents spawn children via bash
        print(chat(sys.argv[1]))
    else:
        # Interactive REPL mode
        print("Bash Agent. Type 'exit' to quit.")
        history = []
        while True:
            try:
                user_input = input("\033[36m>> \033[0m")
                if user_input.lower() in ["exit", "quit"]:
                    break
                if not user_input.strip():
                    continue
                response = chat(user_input, history)
                print(response)
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break
