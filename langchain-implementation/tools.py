"""LangChain tools wrapping the placement cell database tools."""
import json
from typing import Optional
from langchain_core.tools import tool

from app.placement_db import PlacementDb
from app.tools.placement_tools import PlacementTools


_placement_db: Optional[PlacementDb] = None
_placement_tools: Optional[PlacementTools] = None


def get_tools_instance() -> PlacementTools:
    global _placement_db, _placement_tools
    if _placement_tools is None:
        _placement_db = PlacementDb("placement.db")
        _placement_db.migrate(seed=True)
        _placement_tools = PlacementTools(_placement_db)
    return _placement_tools


@tool
def list_open_drives(branch: Optional[str] = None, grad_year: Optional[int] = None) -> str:
    """List placement drives accepting applications right now, soonest deadline first.
    Use when the user asks which companies are coming, what drives are open, or what they could apply to.
    Do NOT use to decide whether a specific student is eligible; use check_eligibility for that.
    """
    tools = get_tools_instance()
    res = tools.list_open_drives(branch=branch, grad_year=grad_year)
    return json.dumps(res)


@tool
def get_student(student_id: str) -> str:
    """Fetch the placement record for one student: name, branch, CGPA, backlogs, graduation year.
    Use when the user asks what is on record for them (e.g. 'what is my CGPA', 'how many backlogs do I have').
    """
    tools = get_tools_instance()
    res = tools.get_student(student_id=student_id)
    return json.dumps(res)


@tool
def check_eligibility(student_id: str, drive_id: int) -> str:
    """Decide whether ONE student may apply to ONE drive, using the drive's eligibility rules.
    Use before apply_to_drive, or when the user asks 'can I apply', 'am I eligible for <company>', or 'why can't I apply'.
    """
    tools = get_tools_instance()
    res = tools.check_eligibility(student_id=student_id, drive_id=int(drive_id))
    return json.dumps(res)


@tool
def list_my_applications(student_id: str) -> str:
    """List the placement applications one student has already submitted, oldest first,
    with the interview slot they booked for each, if any.
    Use when the user asks 'where have I applied', 'did my application go through', or 'when is my interview'.
    """
    tools = get_tools_instance()
    res = tools.list_my_applications(student_id=student_id)
    return json.dumps(res)


@tool
def apply_to_drive(student_id: str, drive_id: int) -> str:
    """Submit a placement application for ONE student to ONE drive.
    Side effect: creates an application record. Call it ONLY when the user explicitly asks to apply or register.
    """
    tools = get_tools_instance()
    res = tools.apply_to_drive(student_id=student_id, drive_id=int(drive_id))
    return json.dumps(res)


@tool
def book_interview_slot(student_id: str, slot_id: int) -> str:
    """Book one interview slot for a student who has already applied to that slot's drive.
    Side effect: reserves the slot. Call it only after the user has picked a specific slot.
    """
    tools = get_tools_instance()
    res = tools.book_interview_slot(student_id=student_id, slot_id=int(slot_id))
    return json.dumps(res)


@tool
def notify_student(student_id: str, message: str) -> str:
    """Send a short reminder or notice to a student by SMS and email.
    Side effect: sends a message to the student. Use ONLY when the user explicitly asks to be reminded or notified.
    Do NOT use to answer questions in chat.
    """
    tools = get_tools_instance()
    res = tools.notify_student(student_id=student_id, message=message)
    return json.dumps(res)


ALL_TOOLS = [
    list_open_drives,
    get_student,
    check_eligibility,
    list_my_applications,
    apply_to_drive,
    book_interview_slot,
    notify_student,
]
