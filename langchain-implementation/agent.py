"""LangChain agent setup for the Placement Assistant."""
import os
from pathlib import Path
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

try:
    from .tools import ALL_TOOLS
except (ImportError, ValueError):
    from tools import ALL_TOOLS


def _load_env() -> None:
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            if k and k not in os.environ:
                os.environ[k] = v


_load_env()

DEFAULT_SYSTEM_TEMPLATE = """You are the Placement Assistant for an engineering college's placement cell.
You are talking to the student with roll number {student_id}. Act only for this student.
Use the tools for every fact about drives, eligibility, applications and slots; never guess.
Eligibility is decided by check_eligibility, not by you. Keep replies short and concrete.

Rules for tool usage:
- Answer questions directly once you have the facts. Do not call redundant tools.
- Never call apply_to_drive, book_interview_slot, or notify_student unless the student explicitly asks to apply, book a slot, or receive a notification.
- If a drive or company is not open or not found, inform the student directly without calling further tools."""


def get_llm(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
) -> ChatOpenAI:
    _load_env()
    key = api_key or os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise ValueError("NVIDIA_API_KEY is not set in environment or .env file.")

    model = model_name or os.environ.get("NVIDIA_MODEL", "openai/gpt-oss-20b")
    url = base_url or os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

    return ChatOpenAI(
        model=model,
        api_key=key,
        base_url=url,
        temperature=temperature,
    )


def create_placement_agent(
    student_id: str = "22CS045",
    tools: Optional[list] = None,
    llm: Optional[ChatOpenAI] = None,
    verbose: bool = True,
) -> AgentExecutor:
    """Create a LangChain Tool-Calling AgentExecutor for the Placement Assistant."""
    model = llm or get_llm()
    agent_tools = tools or ALL_TOOLS

    system_prompt = DEFAULT_SYSTEM_TEMPLATE.format(student_id=student_id)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(model, agent_tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=agent_tools,
        verbose=verbose,
        return_intermediate_steps=True,
        max_iterations=10,
    )


def run_query(query: str, student_id: str = "22CS045", verbose: bool = True) -> dict[str, Any]:
    """Execute a query through the LangChain Placement Agent and return the result."""
    executor = create_placement_agent(student_id=student_id, verbose=verbose)
    return executor.invoke({"input": query})
