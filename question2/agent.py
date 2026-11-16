from __future__ import annotations

import json
from dataclasses import dataclass

from shared.billing import CostSession
from shared.kb import article_ids, load_article
from shared.llm import ChatMessage, LlmClient
from shared.tools import format_thread, get_order, get_ticket

from question1.agent import parse_resolution
import math
import re
from collections import Counter
from functools import lru_cache
"""
Design (numbers in SUBMISSION.md)

Baseline: 4 LLM calls, ~75k points/ticket. Triage shipped up to 120k tokens of
help center to the model, retrieve asked it to *print* eight articles
(retrieval paid for in output tokens), compose asked for "full reasoning", and
audit re-sent 80k tokens just to "double-check".

Same four stages, but only one of them needs a language model:
  triage   -> local: turn structured order facts + customer text into a query
  retrieve -> local: BM25 over the article store, top-K articles only
  compose  -> LLM:   one call, small prompt, JSON-only output
  audit    -> local: validate the JSON against the article store / action enum;
              only if validation fails, retry once with all articles in the
              prompt, then fall back to a human hand-off instead of guessing.

"""
ACTIONS = {"REFUND", "REPLACE", "SEND_ARTICLE", "ESCALATE_TIER2"}
TOP_K = 3

_SYSTEM = (
    "ShopAssist Tier-1 support pipeline. Pick one action per ticket using the order "
    "facts and the help-center articles provided.\n"
    "Actions: REFUND, REPLACE, SEND_ARTICLE, ESCALATE_TIER2.\n"
    'Reply with only this JSON: {"action":...,"kb_article":<article id>,'
    '"customer_reply":<one short sentence>}'
)


@dataclass
class TicketResolution:
    action: str
    kb_article: str

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "the a an and or of to in on for is are was were be been it its this that with "
    "as at by from i my me we you your our not no can may if after within than "
    "have has had please hi thanks".split()
)

def _terms(text: str) -> list[str]:
    # Crude stemming (5-char prefix) so refund/refunded/refundable, damage/damaged,etc. all meet.
    return [w[:5] for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2]


class _Index:
    """BM25 over the real article store; built once, reused for every ticket."""

    K1, B = 1.5, 0.75

    def __init__(self) -> None:
        self.ids = article_ids()
        self.docs = {
            aid: Counter(_terms(aid.replace("-", " ") + " " + load_article(aid)))
            for aid in self.ids
        }
        self.lens = {aid: sum(c.values()) for aid, c in self.docs.items()}
        self.avg = sum(self.lens.values()) / max(len(self.lens), 1)
        df: Counter[str] = Counter()
        for c in self.docs.values():
            df.update(c.keys())
        n = len(self.ids)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def top(self, query: str, k: int) -> list[str]:
        q = set(_terms(query))
        scored = []
        for aid in self.ids:
            tf, dl = self.docs[aid], self.lens[aid]
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    norm = f + self.K1 * (1 - self.B + self.B * dl / self.avg)
                    s += self.idf[t] * f * (self.K1 + 1) / norm
            scored.append((s, aid))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [aid for _, aid in scored[:k]]


@lru_cache
def _index() -> _Index:
    return _Index()


def _article_line(aid: str) -> str:
    body = load_article(aid).replace(f"# {aid}", "").replace("**", "")
    return f"{aid}: {' '.join(body.split())}"

class SupportPipeline:
    """Same product as Question 1 — triage, retrieve, compose, audit."""

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        thread = format_thread(ticket)
        snapshot = json.dumps(order, separators=(",", ":"))

        query = self._triage(thread, order)
        retrieved = self._retrieve(query, TOP_K)
        draft = self._compose(thread, snapshot, retrieved)
        return self._audit(thread, snapshot, retrieved, draft)

    # -- triage (local): structured facts + customer text -> retrieval query --
    @staticmethod
    def _triage(thread: str, order: dict) -> str:
        facts = [str(order.get("product_type", "")), str(order.get("title", ""))]
        for key, val in order.items():
            if val is True:  # customized, damage_reported, license_revealed, opened ...
                facts.append(key.replace("_", " "))
        if "subscription_age_days" in order:
            facts.append("annual subscription cancelled")
        return f"{thread}\n{' '.join(facts)}"

    # -- retrieve (local): article ids, no LLM, nothing printed ---------------
    @staticmethod
    def _retrieve(query: str, k: int | None) -> list[str]:
        return _index().top(query, k) if k else list(_index().ids)

    # -- compose (LLM): the only model call on the happy path -----------------
    def _compose(
        self, thread: str, snapshot: str, article_ids_: list[str], label: str = "compose"
    ) -> str:
        articles = "\n".join(_article_line(a) for a in article_ids_)
        system = f"{_SYSTEM}\nARTICLES:\n{articles}"
        user = f"{thread}\nORDER_SNAPSHOT:\n{snapshot}\nEND_SNAPSHOT"
        return self.llm.chat(label, [ChatMessage("system", system), ChatMessage("user", user)])

    # -- audit (local): validate; widen context once; else hand off to a human -
    def _audit(
        self, thread: str, snapshot: str, retrieved: list[str], draft: str
    ) -> TicketResolution:
        ok = self._valid(draft)
        if ok is None:
            retry = self._compose(thread, snapshot, self._retrieve("", None), "compose_retry")
            ok = self._valid(retry)
        if ok is None:
            return TicketResolution("ESCALATE_TIER2", retrieved[0])
        return ok

    @staticmethod
    def _valid(text: str) -> TicketResolution | None:
        try:
            parsed = parse_resolution(text)
        except (ValueError, KeyError):
            return None
        if parsed.action in ACTIONS and parsed.kb_article in _index().ids:
            return TicketResolution(parsed.action, parsed.kb_article)
        return None