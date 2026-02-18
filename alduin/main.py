"""Alduin - A minimal CLI coding agent."""

import os
from typing import Any

import anthropic
import dotenv
from rich.console import Console
import rich.pretty
from alduin import llm, theme, ui, system_prompt, schema_converter, tool


def execute_tool(
    name_of_the_tool_to_execute: str,
    tools_lookup_table: dict[str, Any],
    args: Any,
    console: Console,
) -> str:
    # get the tool function to execute
    tool_fn = tools_lookup_table.get(name_of_the_tool_to_execute)
    if not tool_fn:
        error_msg = f"Error: Requested tool do not exists {name_of_the_tool_to_execute}"
        ui.print_tool_error(
            console=console, name=name_of_the_tool_to_execute, error=error_msg
        )
    ui.print_tool_request(console=console, name=name_of_the_tool_to_execute, args=args)
    try:
        result = tool_fn(**args)
        ui.print_tool_result(
            console=console, name=name_of_the_tool_to_execute, result=result
        )
        return result
    except Exception as e:
        error_msg = f'Error: Calling tool {name_of_the_tool_to_execute}\n{e}'
        ui.print_tool_error(
            console=console, name=name_of_the_tool_to_execute, error=error_msg
        )

        return error_msg


def agent_loop(client: anthropic.Anthropic, console: Console) -> None:
    """Run the main agent loop: read input, call LLM, execute tools, repeat.

    Args:
        client: The initialized Anthropic client.
        console: The Rich Console for logging and UI.
    """

    conversation: list[dict[str, Any]] = []

    active_tools = [tool.read_file, tool.list_files]

    tools_lookup = {t.__name__: t for t in active_tools}

    while True:
        try:
            user_input = input("🧑‍💻 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            ui.clear_previous_line()
            ui.print_goodbye(console)
            return

        if not user_input:
            continue

        conversation.append({"role": "user", "content": user_input})

        ui.clear_previous_line()
        ui.print_user_message(console, user_input)

        # Start sub agent loop break when there is no call for tools
        while True:
            # import the call method from llm module
            llm_response = llm.call(
                console=console,
                client=client,
                system_prompt=system_prompt.get(),
                messages=conversation,
                tool_schemas=schema_converter.generate_tool_schema(active_tools),
            )
            # Append previous conversation to the list
            conversation.append({'role': 'assistant', 'content': llm_response.content})

            tool_results = []

            # display llm response
            rich.pretty.pprint(llm_response)

            # assistant_reply = (
            #     "Krosis. That knowledge cannot be known to me. "
            #     "Even the Firstborn of Akatosh has limits. Very few. But they exist."
            # )
            for block in llm_response.content:
                # check if response block wants to use tool
                if block.type == 'text':
                    ui.print_assistant_reply(
                        console=console,
                        text=block.text,
                        input_tokens=llm_response.usage.input_tokens,
                        output_tokens=llm_response.usage.output_tokens,
                    )
                elif block.type == 'tool_use':
                    result = execute_tool(
                        name_of_the_tool_to_execute=block.name,
                        tools_lookup_table=tools_lookup,
                        args=block.input,
                        console=console,
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )
                    # print(
                    #     f'tool use requested for tool {block.name} with args {block.input}'
                    # )
            # break when there are no more tool to call
            if not tool_results:
                break
            conversation.append(
                {
                    "role": "user",
                    "content": tool_results,
                }
            )


def main() -> None:
    """Entry point for the Alduin CLI agent.

    Initializes console, checks API key, and starts the agent loop.
    """

    console = Console(theme=theme.ALDUIN_THEME)
    ui.print_banner(console)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        ui.print_error(console, "ANTHROPIC_API_KEY environment variable is not set.")
        return

    client = anthropic.Anthropic(api_key=api_key)
    agent_loop(client=client, console=console)


if __name__ == "__main__":
    dotenv.load_dotenv()
    main()
