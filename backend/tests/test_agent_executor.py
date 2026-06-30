"""Unit tests for AIAgentExecutor."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.ai.agent.executor import AIAgentExecutor
from app.ai.config import AIConfig
from app.ai.models import AgentResponse, ToolInvocation


@pytest.fixture
def config():
    """Create an AIConfig with test values."""
    return AIConfig(
        OPENAI_API_KEY="test-api-key",
        OPENAI_CHAT_MODEL="gpt-4o-mini",
        AGENT_MAX_STEPS=10,
    )


@pytest.fixture
def mock_tools():
    """Create mock LangChain tools for testing."""
    from langchain_core.tools import tool

    @tool
    async def mock_tool_a(query: str) -> str:
        """A mock tool that returns data."""
        return f"Result for: {query}"

    @tool
    async def mock_tool_b(query: str) -> str:
        """Another mock tool for testing."""
        return f"Analysis of: {query}"

    return [mock_tool_a, mock_tool_b]


def _create_executor_with_mocked_internals(config, tools):
    """Helper to create an AIAgentExecutor with mocked LangChain internals."""
    with patch("app.ai.agent.executor.ChatOpenAI"), \
         patch("app.ai.agent.executor.create_react_agent") as mock_create, \
         patch("app.ai.agent.executor.AgentExecutor") as mock_agent_exec_cls:
        mock_create.return_value = MagicMock()
        mock_executor_instance = AsyncMock()
        mock_agent_exec_cls.return_value = mock_executor_instance
        executor = AIAgentExecutor(config=config, tools=tools)
    return executor


class TestAIAgentExecutorInit:
    """Tests for AIAgentExecutor initialization."""

    def test_creates_executor_with_config(self, config, mock_tools):
        """Executor initializes with provided config and tools."""
        executor = _create_executor_with_mocked_internals(config, mock_tools)
        assert executor._config.AGENT_MAX_STEPS == 10
        assert len(executor._tools) == 2

    def test_creates_executor_with_default_tools(self, config):
        """Executor uses default tools when none provided."""
        with patch("app.ai.agent.executor.ChatOpenAI"), \
             patch("app.ai.agent.executor.create_react_agent") as mock_create, \
             patch("app.ai.agent.executor.AgentExecutor") as mock_agent_exec_cls:
            mock_create.return_value = MagicMock()
            mock_agent_exec_cls.return_value = AsyncMock()
            executor = AIAgentExecutor(config=config)

        assert executor._tools is not None
        assert len(executor._tools) == 3

    def test_max_iterations_passed_to_executor(self, config, mock_tools):
        """The max_iterations parameter is passed to AgentExecutor."""
        with patch("app.ai.agent.executor.ChatOpenAI"), \
             patch("app.ai.agent.executor.create_react_agent") as mock_create, \
             patch("app.ai.agent.executor.AgentExecutor") as mock_agent_exec_cls:
            mock_create.return_value = MagicMock()
            mock_agent_exec_cls.return_value = AsyncMock()
            executor = AIAgentExecutor(config=config, tools=mock_tools)

        # Verify AgentExecutor was called with correct max_iterations
        mock_agent_exec_cls.assert_called_once()
        call_kwargs = mock_agent_exec_cls.call_args[1]
        assert call_kwargs["max_iterations"] == 10
        assert call_kwargs["return_intermediate_steps"] is True
        assert call_kwargs["handle_parsing_errors"] is True


class TestAIAgentExecutorExecuteTask:
    """Tests for execute_task method."""

    @pytest.mark.asyncio
    async def test_successful_task_execution(self, config, mock_tools):
        """Agent returns completed response with steps on successful execution."""
        executor = _create_executor_with_mocked_internals(config, mock_tools)

        # Mock the AgentExecutor.ainvoke
        mock_action = MagicMock()
        mock_action.tool = "mock_tool_a"
        mock_action.tool_input = "find python jobs"
        mock_action.log = "I need to search for Python jobs"

        mock_result = {
            "output": "Found 3 Python developer positions.",
            "intermediate_steps": [
                (mock_action, "Result for: find python jobs"),
            ],
        }

        executor._executor = AsyncMock()
        executor._executor.ainvoke = AsyncMock(return_value=mock_result)

        response = await executor.execute_task("Find Python jobs")

        assert isinstance(response, AgentResponse)
        assert response.completed is True
        assert response.answer == "Found 3 Python developer positions."
        assert len(response.steps) == 1
        assert response.steps[0].tool_name == "mock_tool_a"
        assert response.steps[0].input == {"query": "find python jobs"}
        assert response.steps[0].output == "Result for: find python jobs"
        assert "Python jobs" in response.steps[0].reasoning
        assert response.message is None

    @pytest.mark.asyncio
    async def test_max_iterations_exceeded_by_output_message(self, config, mock_tools):
        """Agent returns completed=false when output indicates iteration limit."""
        config_limited = AIConfig(
            OPENAI_API_KEY="test-api-key",
            OPENAI_CHAT_MODEL="gpt-4o-mini",
            AGENT_MAX_STEPS=3,
        )
        executor = _create_executor_with_mocked_internals(config_limited, mock_tools)

        mock_action = MagicMock()
        mock_action.tool = "mock_tool_a"
        mock_action.tool_input = "query"
        mock_action.log = "Trying again"

        mock_result = {
            "output": "Agent stopped due to iteration limit or time limit.",
            "intermediate_steps": [
                (mock_action, "output1"),
                (mock_action, "output2"),
                (mock_action, "output3"),
            ],
        }

        executor._executor = AsyncMock()
        executor._executor.ainvoke = AsyncMock(return_value=mock_result)

        response = await executor.execute_task("Complex task")

        assert response.completed is False
        assert response.message is not None
        assert "maximum step limit" in response.message
        assert "3" in response.message

    @pytest.mark.asyncio
    async def test_max_iterations_exceeded_by_step_count(self, config, mock_tools):
        """Agent returns completed=false when step count >= max iterations."""
        config_limited = AIConfig(
            OPENAI_API_KEY="test-api-key",
            OPENAI_CHAT_MODEL="gpt-4o-mini",
            AGENT_MAX_STEPS=2,
        )
        executor = _create_executor_with_mocked_internals(config_limited, mock_tools)

        mock_action = MagicMock()
        mock_action.tool = "mock_tool_a"
        mock_action.tool_input = "query"
        mock_action.log = "Trying"

        mock_result = {
            "output": "Some partial result.",
            "intermediate_steps": [
                (mock_action, "output1"),
                (mock_action, "output2"),
            ],
        }

        executor._executor = AsyncMock()
        executor._executor.ainvoke = AsyncMock(return_value=mock_result)

        response = await executor.execute_task("Complex task")

        assert response.completed is False
        assert "maximum step limit" in response.message

    @pytest.mark.asyncio
    async def test_handles_execution_error(self, config, mock_tools):
        """Agent handles unexpected errors gracefully."""
        executor = _create_executor_with_mocked_internals(config, mock_tools)

        executor._executor = AsyncMock()
        executor._executor.ainvoke = AsyncMock(
            side_effect=Exception("LLM service unavailable")
        )

        response = await executor.execute_task("Do something")

        assert response.completed is False
        assert "error occurred" in response.answer.lower()
        assert response.message is not None
        assert "unexpected error" in response.message.lower()
        assert response.steps == []

    @pytest.mark.asyncio
    async def test_tool_failure_included_in_steps(self, config, mock_tools):
        """Tool failures are logged and included in the response steps."""
        executor = _create_executor_with_mocked_internals(config, mock_tools)

        # Simulate a tool failure in intermediate steps
        mock_action1 = MagicMock()
        mock_action1.tool = "mock_tool_a"
        mock_action1.tool_input = "search for jobs"
        mock_action1.log = "Let me search for jobs"

        mock_action2 = MagicMock()
        mock_action2.tool = "mock_tool_b"
        mock_action2.tool_input = "analyze results"
        mock_action2.log = "Tool A failed, trying alternative with Tool B"

        mock_result = {
            "output": "Based on the analysis, here are the results.",
            "intermediate_steps": [
                (mock_action1, "Error querying the platform API: Connection refused"),
                (mock_action2, "Analysis of: analyze results"),
            ],
        }

        executor._executor = AsyncMock()
        executor._executor.ainvoke = AsyncMock(return_value=mock_result)

        response = await executor.execute_task("Find and analyze jobs")

        assert response.completed is True
        assert len(response.steps) == 2
        # First step has error output
        assert "Error" in response.steps[0].output
        # Second step used alternative tool
        assert response.steps[1].tool_name == "mock_tool_b"

    @pytest.mark.asyncio
    async def test_dict_tool_input_preserved(self, config, mock_tools):
        """Dict tool inputs are preserved as-is in the response."""
        executor = _create_executor_with_mocked_internals(config, mock_tools)

        mock_action = MagicMock()
        mock_action.tool = "mock_tool_a"
        mock_action.tool_input = {"query": "jobs", "filter": "python"}
        mock_action.log = "Searching with filters"

        mock_result = {
            "output": "Done.",
            "intermediate_steps": [
                (mock_action, "Some results"),
            ],
        }

        executor._executor = AsyncMock()
        executor._executor.ainvoke = AsyncMock(return_value=mock_result)

        response = await executor.execute_task("Search")

        assert response.steps[0].input == {"query": "jobs", "filter": "python"}

    @pytest.mark.asyncio
    async def test_empty_intermediate_steps(self, config, mock_tools):
        """Agent can return a response with no intermediate steps."""
        executor = _create_executor_with_mocked_internals(config, mock_tools)

        mock_result = {
            "output": "I can answer this directly without tools.",
            "intermediate_steps": [],
        }

        executor._executor = AsyncMock()
        executor._executor.ainvoke = AsyncMock(return_value=mock_result)

        response = await executor.execute_task("What is 2+2?")

        assert response.completed is True
        assert response.answer == "I can answer this directly without tools."
        assert response.steps == []

    @pytest.mark.asyncio
    async def test_response_includes_all_required_fields(self, config, mock_tools):
        """AgentResponse always has answer, steps, completed fields."""
        executor = _create_executor_with_mocked_internals(config, mock_tools)

        mock_action = MagicMock()
        mock_action.tool = "api_query"
        mock_action.tool_input = "get jobs"
        mock_action.log = "Querying API"

        mock_result = {
            "output": "Here are the jobs.",
            "intermediate_steps": [(mock_action, "Job list")],
        }

        executor._executor = AsyncMock()
        executor._executor.ainvoke = AsyncMock(return_value=mock_result)

        response = await executor.execute_task("List jobs")

        # Verify all required fields exist
        assert hasattr(response, "answer")
        assert hasattr(response, "steps")
        assert hasattr(response, "completed")
        assert hasattr(response, "message")
        assert isinstance(response.answer, str)
        assert isinstance(response.steps, list)
        assert isinstance(response.completed, bool)

    @pytest.mark.asyncio
    async def test_tool_invocation_has_all_fields(self, config, mock_tools):
        """Each ToolInvocation has tool_name, input, output, reasoning."""
        executor = _create_executor_with_mocked_internals(config, mock_tools)

        mock_action = MagicMock()
        mock_action.tool = "vector_search"
        mock_action.tool_input = "python developer"
        mock_action.log = "Need to find similar candidates"

        mock_result = {
            "output": "Found matches.",
            "intermediate_steps": [(mock_action, "3 results found")],
        }

        executor._executor = AsyncMock()
        executor._executor.ainvoke = AsyncMock(return_value=mock_result)

        response = await executor.execute_task("Find candidates")

        step = response.steps[0]
        assert isinstance(step, ToolInvocation)
        assert step.tool_name == "vector_search"
        assert isinstance(step.input, dict)
        assert isinstance(step.output, str)
        assert isinstance(step.reasoning, str)
        assert len(step.reasoning) > 0


class TestParseIntermediateSteps:
    """Tests for _parse_intermediate_steps helper."""

    def test_handles_malformed_steps_gracefully(self, config, mock_tools):
        """Malformed steps produce parse_error entries instead of crashing."""
        executor = _create_executor_with_mocked_internals(config, mock_tools)

        # A malformed step that can't be unpacked as a 2-tuple
        malformed_steps = [("only_one_element",)]

        result = executor._parse_intermediate_steps(malformed_steps)

        assert len(result) == 1
        assert result[0].tool_name == "parse_error"
        assert "Failed to parse step" in result[0].output

    def test_handles_non_string_observation(self, config, mock_tools):
        """Non-string observations are converted to strings."""
        executor = _create_executor_with_mocked_internals(config, mock_tools)

        mock_action = MagicMock()
        mock_action.tool = "api_query"
        mock_action.tool_input = "get jobs"
        mock_action.log = "Querying"

        steps = [(mock_action, {"data": [1, 2, 3]})]

        result = executor._parse_intermediate_steps(steps)

        assert len(result) == 1
        assert isinstance(result[0].output, str)
        assert "data" in result[0].output


class TestIsMaxIterationsExceeded:
    """Tests for _is_max_iterations_exceeded helper."""

    def test_detects_iteration_limit_in_output(self, config, mock_tools):
        """Detects iteration limit from the output message."""
        executor = _create_executor_with_mocked_internals(config, mock_tools)

        result = {"output": "Agent stopped due to iteration limit or time limit."}
        assert executor._is_max_iterations_exceeded(result, []) is True

    def test_detects_max_iterations_keyword(self, config, mock_tools):
        """Detects max iterations keyword in output."""
        executor = _create_executor_with_mocked_internals(config, mock_tools)

        result = {"output": "Exceeded max iterations, returning partial."}
        assert executor._is_max_iterations_exceeded(result, []) is True

    def test_detects_by_step_count(self, config, mock_tools):
        """Detects limit exceeded when steps >= max_iterations."""
        config_limited = AIConfig(
            OPENAI_API_KEY="test-api-key",
            OPENAI_CHAT_MODEL="gpt-4o-mini",
            AGENT_MAX_STEPS=2,
        )
        executor = _create_executor_with_mocked_internals(config_limited, mock_tools)

        result = {"output": "Some answer"}
        steps = [MagicMock(), MagicMock()]
        assert executor._is_max_iterations_exceeded(result, steps) is True

    def test_returns_false_for_normal_completion(self, config, mock_tools):
        """Returns False when task completed normally."""
        executor = _create_executor_with_mocked_internals(config, mock_tools)

        result = {"output": "Here is your answer."}
        steps = [MagicMock()]  # Only 1 step, max is 10
        assert executor._is_max_iterations_exceeded(result, steps) is False
