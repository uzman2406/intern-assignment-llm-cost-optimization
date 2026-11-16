"""Cost budgets from the assignment specs, plus holdout correctness."""
from __future__ import annotations

import pytest

from eval.reference import TICKET_EXPECTATIONS
from question1.agent import SupportCopilot
from question2.agent import SupportPipeline

PUBLIC = ["T-1001", "T-1002", "T-1003", "T-1004"]


@pytest.mark.parametrize("tid", PUBLIC)
def test_q1_points_per_ticket_under_budget(tid):
    copilot = SupportCopilot()
    copilot.handle_ticket(tid)
    assert copilot.session.total_points() <= 3000


@pytest.mark.parametrize("tid", PUBLIC)
def test_q2_points_per_ticket_under_budget(tid):
    pipe = SupportPipeline()
    pipe.handle_ticket(tid)
    assert pipe.session.total_points() <= 2500


@pytest.mark.parametrize("agent_cls", [SupportCopilot, SupportPipeline])
def test_holdout_subscription_still_correct(agent_cls):
    got = agent_cls().handle_ticket("T-1091")
    exp = TICKET_EXPECTATIONS["T-1091"]
    assert (got.action, got.kb_article) == (exp.action, exp.kb_article)


def test_q2_uses_a_single_llm_call_on_happy_path():
    pipe = SupportPipeline()
    pipe.handle_ticket("T-1001")
    assert len(pipe.session.calls) == 1