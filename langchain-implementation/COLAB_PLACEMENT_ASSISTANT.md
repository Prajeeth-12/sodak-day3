# Placement Assistant Agent in Google Colab (Pure LangChain + NVIDIA NIM)

This notebook is focused purely on **LangChain tool creation**, **Agent setup**, and **calling the LLM via NVIDIA NIM API**. It eliminates all database / SQLite boilerplate so you can immediately experiment with LangChain agent tool-calling in [Google Colab](https://colab.research.google.com/).

---

## Cell 1: Install Dependencies

```bash
!pip install -q langchain langchain-core langchain-openai
```

---

## Cell 2: Configure NVIDIA API Key & Model

```python
import os

# Set your NVIDIA API credentials
os.environ["NVIDIA_API_KEY"] = "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"
os.environ["NVIDIA_BASE_URL"] = "https://integrate.api.nvidia.com/v1"
os.environ["NVIDIA_MODEL"] = "openai/gpt-oss-20b"

print("✓ NVIDIA NIM API credentials configured!")
```

---

## Cell 3: Placement Data & LangChain Tools (`@tool`)

Here, we define clean in-memory placement data and wrap each action into a LangChain `@tool`.

```python
import json
from typing import Optional
from langchain_core.tools import tool

# -------------------------------------------------------------
# In-Memory Placement Cell Data (No Database Needed)
# -------------------------------------------------------------
STUDENTS = {
    "22CS045": {"name": "Priya Raman", "branch": "CSE", "cgpa": 8.4, "backlogs": 0, "grad_year": 2026},
    "22IT017": {"name": "Arjun Kumar", "branch": "IT", "cgpa": 6.8, "backlogs": 1, "grad_year": 2026},
    "22EC031": {"name": "Divya Sekar", "branch": "ECE", "cgpa": 7.2, "backlogs": 2, "grad_year": 2026},
}

DRIVES = [
    {
        "drive_id": 1,
        "company": "Zoho",
        "role": "Member Technical Staff",
        "ctc_lpa": 8.4,
        "status": "open",
        "min_cgpa": 7.0,
        "max_backlogs": 0,
        "branches": ["CSE", "IT", "ECE"],
        "grad_year": 2026
    },
    {
        "drive_id": 2,
        "company": "TCS",
        "role": "Ninja",
        "ctc_lpa": 3.6,
        "status": "open",
        "min_cgpa": 6.0,
        "max_backlogs": 1,
        "branches": ["CSE", "IT", "ECE", "MECH"],
        "grad_year": 2026
    },
    {
        "drive_id": 3,
        "company": "Freshworks",
        "role": "Software Engineer I",
        "ctc_lpa": 12.0,
        "status": "closed",
        "min_cgpa": 8.0,
        "max_backlogs": 0,
        "branches": ["CSE", "IT"],
        "grad_year": 2026
    }
]

APPLICATIONS = []
SLOTS = [
    {"slot_id": 1, "drive_id": 1, "time": "2026-10-22 10:00 AM", "student_id": None},
    {"slot_id": 2, "drive_id": 1, "time": "2026-10-22 11:30 AM", "student_id": None},
    {"slot_id": 3, "drive_id": 2, "time": "2026-10-23 02:00 PM", "student_id": None},
]
NOTIFICATIONS = []

# -------------------------------------------------------------
# LangChain Tools
# -------------------------------------------------------------

@tool
def list_open_drives(branch: Optional[str] = None) -> str:
    """List placement drives accepting applications right now, with company name, role, CTC, and drive_id.
    Use when user asks which companies are visiting or what drives are open.
    """
    open_drives = [
        {"drive_id": d["drive_id"], "company": d["company"], "role": d["role"], "ctc_lpa": d["ctc_lpa"]}
        for d in DRIVES
        if d["status"] == "open" and (not branch or branch in d["branches"])
    ]
    return json.dumps({"open_drives": open_drives})


@tool
def get_student_profile(student_id: str) -> str:
    """Fetch profile details for a student: name, branch, CGPA, backlogs, and graduation year.
    Use when user asks 'what is my CGPA', 'how many backlogs do I have', or checks profile.
    """
    profile = STUDENTS.get(student_id)
    if not profile:
        return json.dumps({"error": f"Student '{student_id}' not found."})
    return json.dumps({"student_id": student_id, **profile})


@tool
def check_eligibility(student_id: str, drive_id: int) -> str:
    """Evaluate whether a student meets the criteria for a specific placement drive.
    Returns whether they are eligible and lists reasons if they do not meet the criteria.
    """
    student = STUDENTS.get(student_id)
    if not student:
        return json.dumps({"error": f"Student '{student_id}' not found."})

    drive = next((d for d in DRIVES if d["drive_id"] == int(drive_id)), None)
    if not drive:
        return json.dumps({"error": f"Drive id {drive_id} not found."})

    reasons = []
    if student["cgpa"] < drive["min_cgpa"]:
        reasons.append(f"CGPA {student['cgpa']} is below required {drive['min_cgpa']}")
    if student["backlogs"] > drive["max_backlogs"]:
        reasons.append(f"Backlogs ({student['backlogs']}) exceed maximum allowed ({drive['max_backlogs']})")
    if student["branch"] not in drive["branches"]:
        reasons.append(f"Branch {student['branch']} is not eligible (allowed: {drive['branches']})")
    if student["grad_year"] != drive["grad_year"]:
        reasons.append(f"Grad year {student['grad_year']} does not match {drive['grad_year']}")

    return json.dumps({
        "student_id": student_id,
        "company": drive["company"],
        "eligible": len(reasons) == 0,
        "reasons": reasons
    })


@tool
def apply_to_drive(student_id: str, drive_id: int) -> str:
    """Submit an application for a student to a specific drive.
    Call this ONLY when the user explicitly asks to apply or register.
    """
    drive = next((d for d in DRIVES if d["drive_id"] == int(drive_id)), None)
    if not drive or drive["status"] != "open":
        return json.dumps({"error": f"Drive {drive_id} is not open for application."})

    # Check if already applied
    for app in APPLICATIONS:
        if app["student_id"] == student_id and app["drive_id"] == int(drive_id):
            return json.dumps({"status": "already_applied", "company": drive["company"]})

    APPLICATIONS.append({"student_id": student_id, "drive_id": int(drive_id), "status": "applied"})

    # Return available slots
    available_slots = [
        {"slot_id": s["slot_id"], "time": s["time"]}
        for s in SLOTS if s["drive_id"] == int(drive_id) and s["student_id"] is None
    ]
    return json.dumps({
        "status": "applied_successfully",
        "company": drive["company"],
        "available_slots": available_slots
    })


@tool
def book_interview_slot(student_id: str, slot_id: int) -> str:
    """Book an interview slot for a student who has already applied.
    Call this ONLY after the user explicitly selects an interview slot.
    """
    slot = next((s for s in SLOTS if s["slot_id"] == int(slot_id)), None)
    if not slot:
        return json.dumps({"error": f"Slot {slot_id} not found."})
    if slot["student_id"] is not None:
        return json.dumps({"error": f"Slot {slot_id} is already taken."})

    slot["student_id"] = student_id
    return json.dumps({"status": "booked", "slot_id": slot_id, "time": slot["time"]})


@tool
def notify_student(student_id: str, message: str) -> str:
    """Send SMS/email notification to student. Call ONLY when explicitly requested to notify/remind."""
    NOTIFICATIONS.append({"student_id": student_id, "message": message})
    return json.dumps({"status": "sent", "student_id": student_id, "message": message})


# Tools list
TOOLS = [
    list_open_drives,
    get_student_profile,
    check_eligibility,
    apply_to_drive,
    book_interview_slot,
    notify_student,
]

print(f"✓ Registered {len(TOOLS)} LangChain tools!")
```

---

## Cell 4: Build the LangChain Tool-Calling Agent

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Initialize LLM with NVIDIA NIM
llm = ChatOpenAI(
    model=os.environ["NVIDIA_MODEL"],
    api_key=os.environ["NVIDIA_API_KEY"],
    base_url=os.environ["NVIDIA_BASE_URL"],
    temperature=0.0
)

# System Prompt with clear tool-calling guidelines
SYSTEM_PROMPT = """You are the Placement Assistant for an engineering college's placement cell.
You are assisting the student with roll number: {student_id}. Act only for this student.

Rules for tool execution:
1. Always check facts using the tools (drives, eligibility, profile). Never guess or hallucinate.
2. Answer the user directly once you have the facts. Do not call redundant tools.
3. NEVER call apply_to_drive, book_interview_slot, or notify_student unless the user explicitly asks to apply, book, or be notified.
4. If a company is not in the open list or is closed, inform the user directly."""

def create_agent(student_id: str = "22CS045", verbose: bool = True):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT.format(student_id=student_id)),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    try:
        from langchain.agents import create_tool_calling_agent, AgentExecutor
    except ImportError:
        from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=TOOLS, verbose=verbose, max_iterations=6)

print("✓ LangChain Agent Builder create_agent() ready!")
```

---

## Cell 5: Run Sample Queries (See Tool-Calling in Action)

```python
# Create agent for student Priya Raman (22CS045 - CSE, CGPA 8.4, 0 backlogs)
agent = create_agent(student_id="22CS045", verbose=True)

def test_query(prompt: str):
    print("\n" + "="*60)
    print(f"USER: {prompt}")
    print("="*60)
    result = agent.invoke({"input": prompt})
    print(f"\nASSISTANT: {result['output']}\n")

# Query 1: Information query (Agent calls list_open_drives and replies)
test_query("list companies names?")

# Query 2: Eligibility check (Agent checks rules and answers)
test_query("Am I eligible for Zoho?")

# Query 3: Closed drive inquiry (Agent checks drives, sees Freshworks is closed/not open, and answers)
test_query("Can I apply to freshworks?")

# Query 4: Profile check
test_query("What is my CGPA and backlogs record?")
```

---

## Cell 6: Action Execution (Applying & Booking)

```python
# Query 5: Explicit action to apply
test_query("Apply me to Zoho")

# Query 6: Explicit action to book a slot
test_query("Book slot 1 for me")
```

---

## Cell 7: Interactive Chat Loop

```python
# Chat with the agent in real time
chat_agent = create_agent(student_id="22CS045", verbose=False)

print("🎓 Placement Assistant Chatbot is ready! (Type 'exit' to quit)\n")
while True:
    user_msg = input("You: ").strip()
    if not user_msg or user_msg.lower() in ("exit", "quit"):
        print("Goodbye!")
        break
    response = chat_agent.invoke({"input": user_msg})
    print(f"\nAssistant: {response['output']}\n")
```
