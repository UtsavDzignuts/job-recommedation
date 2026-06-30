"""AI Agent Executor for multi-step task reasoning.

Uses LangChain's AgentExecutor with ReAct-style reasoning to autonomously
solve complex tasks by invoking tools (API query, vector search, LLM reasoning)
and combining their results.
"""

import logging
from typing import List, Optional

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from app.ai.config import AIConfig
from app.ai.llm_factory import create_chat_llm
from app.ai.models import AgentResponse, ToolInvocation

logger = logging.getLogger(__name__)

# ReAct-style prompt template for the agent.
# This follows LangChain's expected format with tools, tool_names,
# agent_scratchpad placeholders.
_REACT_PROMPT_TEMPLATE = """You are an intelligent AI agent that solves tasks step-by-step using available tools.
Your goal is to complete the given task accurately and efficiently.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Instructions:
1. Think step-by-step about how to accomplish the task.
2. Choose the most appropriate tool for each step.
3. After each tool use, evaluate the result and decide the next action.
4. Stop when the task is fully completed or when you have gathered enough information to provide a final answer.
5. If a tool fails, try an alternative approach using a different tool.
6. Be concise in your reasoning but thorough in your execution.

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


class AIAgentExecutor:
    """Executes complex multi-step tasks using a ReAct-style LangChain agent.

    The executor creates an agent with three tools (api_query, vector_search,
    llm_reasoning) and runs it with a configurable maximum number of iterations.
    Each tool invocation is logged with inputs, outputs, and reasoning.

    Args:
        config: Optional AIConfig instance. Loaded from environment if not provided.
        tools: Optional list of LangChain tools. Uses default tools if not provided.
    """

    def __init__(
        self,
        config: Optional[AIConfig] = None,
        tools: Optional[list] = None,
    ) -> None:
        self._config = config or AIConfig()

        # Initialize LLM
        self._llm = create_chat_llm(self._config, temperature=0)

        # Initialize tools
        if tools is not None:
            self._tools = tools
        else:
            from app.ai.agent.tools import (
                api_query_tool,
                llm_reasoning_tool,
                vector_search_tool,
            )

            self._tools = [api_query_tool, vector_search_tool, llm_reasoning_tool]

        # Create the ReAct-style agent
        prompt = PromptTemplate.from_template(_REACT_PROMPT_TEMPLATE)
        agent = create_react_agent(
            llm=self._llm,
            tools=self._tools,
            prompt=prompt,
        )

        # Create the AgentExecutor with max iterations from config
        self._executor = AgentExecutor(
            agent=agent,
            tools=self._tools,
            max_iterations=self._config.AGENT_MAX_STEPS,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            verbose=False,
        )

    async def execute_task(self, task: str) -> AgentResponse:
        """Execute a complex task using the ReAct agent.

        The agent autonomously decides which tools to call, in what order,
        and when to stop. Each intermediate step is captured and returned
        in the response.

        Args:
            task: Natural language task description.

        Returns:
            AgentResponse with the answer, tool invocation steps,
            completion status, and optional message.
        """
        logger.info("Agent starting task: %s", task[:200])

        steps: List[ToolInvocation] = []
        completed = True
        message: Optional[str] = None
        answer = ""

        try:
            result = await self._executor.ainvoke({"input": task})

            # Extract final answer
            answer = result.get("output", "")

            # Extract intermediate steps
            intermediate_steps = result.get("intermediate_steps", [])
            steps = self._parse_intermediate_steps(intermediate_steps)

            # Check if max iterations was reached
            # When AgentExecutor hits max_iterations, it sets the output to
            # an "Agent stopped" message. We detect this and mark as incomplete.
            if self._is_max_iterations_exceeded(result, intermediate_steps):
                completed = False
                message = (
                    f"The agent reached the maximum step limit of "
                    f"{self._config.AGENT_MAX_STEPS} without completing the task. "
                    f"A partial result has been provided."
                )
                logger.warning(
                    "Agent exceeded max iterations (%d) for task: %s",
                    self._config.AGENT_MAX_STEPS,
                    task[:200],
                )

        except Exception as e:
            logger.error("Agent execution error: %s", str(e), exc_info=True)
            completed = False
            answer = f"An error occurred during task execution: {str(e)}"
            message = "The agent encountered an unexpected error during execution."

        logger.info(
            "Agent completed task (completed=%s, steps=%d): %s",
            completed,
            len(steps),
            task[:200],
        )

        return AgentResponse(
            answer=answer,
            steps=steps,
            completed=completed,
            message=message,
        )

    def _parse_intermediate_steps(
        self, intermediate_steps: list
    ) -> List[ToolInvocation]:
        """Parse LangChain intermediate steps into ToolInvocation objects.

        Each intermediate step is a tuple of (AgentAction, observation).
        The AgentAction contains the tool name, tool input, and the agent's
        reasoning (log).

        Args:
            intermediate_steps: List of (AgentAction, observation) tuples.

        Returns:
            List of ToolInvocation objects.
        """
        invocations: List[ToolInvocation] = []

        for step in intermediate_steps:
            try:
                action, observation = step

                tool_name = getattr(action, "tool", "unknown")
                tool_input = getattr(action, "tool_input", "")
                reasoning = getattr(action, "log", "")

                # Normalize tool input to dict
                if isinstance(tool_input, str):
                    input_dict = {"query": tool_input}
                elif isinstance(tool_input, dict):
                    input_dict = tool_input
                else:
                    input_dict = {"input": str(tool_input)}

                # Normalize observation output to string
                if isinstance(observation, str):
                    output_str = observation
                else:
                    output_str = str(observation)

                invocation = ToolInvocation(
                    tool_name=tool_name,
                    input=input_dict,
                    output=output_str,
                    reasoning=reasoning.strip(),
                )
                invocations.append(invocation)

                # Log each tool invocation
                logger.info(
                    "Agent tool invocation: tool=%s, input=%s, output_length=%d",
                    tool_name,
                    str(input_dict)[:200],
                    len(output_str),
                )

            except Exception as e:
                logger.warning(
                    "Failed to parse intermediate step: %s", str(e)
                )
                # Include the failure information in the steps
                invocations.append(
                    ToolInvocation(
                        tool_name="parse_error",
                        input={"raw_step": str(step)[:500]},
                        output=f"Failed to parse step: {str(e)}",
                        reasoning="Error occurred while parsing agent step",
                    )
                )

        return invocations

    def _is_max_iterations_exceeded(
        self, result: dict, intermediate_steps: list
    ) -> bool:
        """Determine if the agent stopped due to max iterations being exceeded.

        LangChain's AgentExecutor sets a specific output message when max
        iterations are reached. We also check if the number of steps equals
        or exceeds the configured maximum.

        Args:
            result: The full result dict from AgentExecutor.
            intermediate_steps: List of intermediate steps taken.

        Returns:
            True if the agent hit the iteration limit.
        """
        # LangChain uses "Agent stopped due to iteration limit" or similar
        output = result.get("output", "")
        if "iteration limit" in output.lower() or "max iterations" in output.lower():
            return True

        # Also check if steps count equals max_iterations
        if len(intermediate_steps) >= self._config.AGENT_MAX_STEPS:
            return True

        return False
