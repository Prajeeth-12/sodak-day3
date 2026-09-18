"""Lab 3 — two runs booking the same slot; one wins cleanly."""
import threading

from app.memory import RunStore
from app.placement_db import PlacementDb
from app.providers import ModelTurn, PositionalMock, ToolCall
from app.tools.placement_tools import PlacementTools
from app.worker import Worker


def _mock_for(student, drive, slot):
    return PositionalMock([
        ModelTurn(text=None, tool_calls=[ToolCall("check_eligibility", {"student_id": student, "drive_id": drive})],
                  tokens_in=100, tokens_out=10),
        ModelTurn(text=None, tool_calls=[ToolCall("apply_to_drive", {"student_id": student, "drive_id": drive})],
                  tokens_in=120, tokens_out=10),
        ModelTurn(text=None, tool_calls=[ToolCall("book_interview_slot", {"student_id": student, "slot_id": slot})],
                  tokens_in=140, tokens_out=10),
        ModelTurn(text="Done.", tokens_in=160, tokens_out=10),
    ])


def test_two_workers_two_runs_one_slot(db_files, clock):
    agent_path, place_path = db_files

    store_a = RunStore(agent_path, clock)
    store_b = RunStore(agent_path, clock)
    place_a = PlacementDb(place_path)
    place_b = PlacementDb(place_path)

    run_a = store_a.enqueue(store_a.create_thread("22IT017"), "Apply me to TCS, book slot 3", "mock")
    run_b = store_b.enqueue(store_b.create_thread("22CS045"), "Apply me to TCS, book slot 3", "mock")

    worker_a = Worker(store_a, place_a, _mock_for("22IT017", 2, 3), worker_id="wa")
    worker_b = Worker(store_b, place_b, _mock_for("22CS045", 2, 3), worker_id="wb")

    result_a = worker_a.run_once()
    result_b = worker_b.run_once()

    assert result_a is not None and result_b is not None
    assert sorted([result_a[1], result_b[1]]) == ["succeeded", "succeeded"]

    check = PlacementDb(place_path)
    booked = check.conn.execute(
        "SELECT count(*) FROM interview_slot WHERE id = 3 AND student_id IS NOT NULL"
    ).fetchone()[0]
    assert booked == 1

    steps_a = store_a.get_run(run_a)["steps"]
    steps_b = store_b.get_run(run_b)["steps"]
    book_results = []
    for steps in [steps_a, steps_b]:
        for s in steps:
            if s["kind"] == "tool" and s["tool_name"] == "book_interview_slot":
                book_results.append(s["result"])
    statuses = sorted(r.get("status", r.get("error")) for r in book_results)
    assert statuses == ["booked", "slot_taken"]


def test_truly_concurrent_claims_have_one_winner(db_files):
    _, place_path = db_files
    main_db = PlacementDb(place_path)
    main_db.create_application(1, 2)
    main_db.create_application(2, 2)
    version = main_db.slot_version(3)

    barrier = threading.Barrier(8)
    results = []
    lock = threading.Lock()

    def try_claim(student_id):
        db = PlacementDb(place_path)
        barrier.wait()
        won = db.claim_slot(3, student_id, version)
        with lock:
            results.append(won)

    threads = []
    for i in range(8):
        student_id = 1 if i % 2 == 0 else 2
        t = threading.Thread(target=try_claim, args=(student_id,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert results.count(True) == 1
    assert results.count(False) == 7

    check = PlacementDb(place_path)
    assert check.slot_version(3) == version + 1
