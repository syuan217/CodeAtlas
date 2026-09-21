"""cost.py:费用流水写入与汇总。"""

from codeatlas.cost import cost_summary, log_usage, model_summary, total_cost


def test_log_and_summary(db):
    log_usage(db, stage="doctor", model="m-a", prompt_tokens=100, completion_tokens=50,
              cost=0.001)
    log_usage(db, stage="index", model="m-b", prompt_tokens=200, completion_tokens=0,
              cost=0.002)
    log_usage(db, stage="doctor", model="m-a", prompt_tokens=10, completion_tokens=5,
              cost=0.0001)

    by_stage = {r["stage"]: r for r in cost_summary(db)}
    assert by_stage["doctor"]["calls"] == 2
    assert by_stage["doctor"]["prompt_tokens"] == 110
    assert by_stage["doctor"]["cost"] == 0.0011
    assert by_stage["index"]["calls"] == 1

    by_model = {r["model"]: r for r in model_summary(db)}
    assert by_model["m-a"]["calls"] == 2
    assert by_model["m-b"]["prompt_tokens"] == 200

    assert total_cost(db) == 0.0031


def test_empty_summary(db):
    assert cost_summary(db) == []
    assert total_cost(db) == 0.0
