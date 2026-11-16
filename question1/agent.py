from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache

from shared.billing import CostSession
from shared.kb import article_ids, load_article
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket


@dataclass
class TicketResolution:
    action: str
    kb_article: str
    customer_reply: str
    raw_completion: str


def parse_resolution(text: str) -> TicketResolution:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"No JSON in model output:\n{text[:400]}")
    data = json.loads(m.group())
    return TicketResolution(
        action=str(data["action"]).upper(),
        kb_article=str(data["kb_article"]),
        customer_reply=str(data.get("customer_reply", "")),
        raw_completion=text,
    )


"""optimised prompt

Where the baseline's ~8.9k points/ticket went (T-1001):
  * 485 output tokens x 15 = 7,275 pts: "Think step by step ... quote any article"
    makes the model write a long reasoning trace the product never uses.
  * 524 input tokens x 3   = 1,572 pts: random run-id, few-shot block, full
    markdown headings, pretty-printed JSON snapshot.

Fix: ask for the JSON object only (no reasoning/quoting), keep the prompt
static + compact, and send the order snapshot as minified JSON."""

_SYSTEM_HEADER = (
    "ShopAssist Tier-1 support copilot. Pick one action per ticket, using the order "
    "facts and the help-center article that applies.\n"
    "Actions: REFUND, REPLACE, SEND_ARTICLE, ESCALATE_TIER2.\n"
    'Reply with only this JSON: {"action":...,"kb_article":<article id>,'
    '"customer_reply":<one short sentence>}'
)


@lru_cache
def _compact_help_center() -> str:
    """One line per article: `id: policy text` (id heading + markdown bold removed)."""
    lines = []
    for aid in article_ids():
        body = load_article(aid).replace(f"# {aid}", "").replace("**", "")
        lines.append(f"{aid}: {' '.join(body.split())}")
    return "\n".join(lines)


class SupportCopilot:
    """
    ShopAssist Tier-1 copilot (baseline).

    Given a support ticket + order snapshot, choose an action, cite the help-center
    article, and draft the customer reply.
    """

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)

    def build_messages(self, ticket_id: str) -> list[ChatMessage]:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)
        snapshot = json.dumps(order, separators=(",", ":"))

        # Static content first (stable prefix -> cacheable once it grows past the 1024-token minimum), per-ticket content last.
        system = f"{_SYSTEM_HEADER}\nHELP CENTER:\n{_compact_help_center()}"
        user = f"{thread}\nORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT"
        return [ChatMessage("system", system), ChatMessage("user", user)]

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        messages = self.build_messages(ticket_id)
        completion = self.llm.chat("handle_ticket", messages)
        return parse_resolution(completion)