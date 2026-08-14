"""Query layer over the loaded dataset.

Everything here is a pure function of the dataset plus an explicit `as_of`
timestamp — no wall-clock reads, no hidden state — so the same inputs always
produce the same ticket set, in the same order.

## Why tickets join to accounts on `company`

`tickets.json` carries an `account_id`, but in this corpus it is not a usable
foreign key: only 4 of 500 tickets match an account by id, and in all four the
ticket's `company` disagrees with the account's. All 50 company names, by
contrast, overlap exactly and yield 4-17 tickets per account. We therefore join
on the company name and additionally union in any id match, so a future dataset
with real ids keeps working. `scripts/profile_dataset.py` prints this evidence.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Iterable

from app.data.loader import Dataset, load_dataset, parse_timestamp

_URGENCY_WEIGHT = {"P1": 4, "P2": 3, "P3": 2, "P4": 1}
_UNRESOLVED_STATUSES = frozenset({"Open", "In Progress", "Pending Customer"})


class SupportRepository:
    """Read-only access to tickets and accounts."""

    def __init__(self, dataset: Dataset | None = None) -> None:
        self._dataset = dataset or load_dataset()
        self._accounts_by_id = {
            str(a["account_id"]).strip().upper(): a for a in self._dataset.accounts
        }

    # --- accounts ---------------------------------------------------------- #

    @property
    def as_of(self) -> datetime:
        return self._dataset.as_of

    @property
    def accounts(self) -> list[dict[str, Any]]:
        return sorted(self._dataset.accounts, key=lambda a: a["account_id"])

    def account_ids(self) -> list[str]:
        return [a["account_id"] for a in self.accounts]

    def get_account(self, account_id: str) -> dict[str, Any] | None:
        if not account_id or not account_id.strip():
            return None
        return self._accounts_by_id.get(account_id.strip().upper())

    # --- tickets ----------------------------------------------------------- #

    def tickets_for_account(
        self,
        account: dict[str, Any],
        *,
        window_days: int | None = None,
        as_of: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Tickets belonging to an account, optionally limited to a window.

        Sorted newest-first with `ticket_id` as the tie-break so repeated calls
        produce a byte-identical ordering.
        """
        reference = as_of or self._dataset.as_of
        company = str(account.get("company", "")).strip().lower()
        account_id = str(account.get("account_id", "")).strip().upper()

        matched: dict[str, dict[str, Any]] = {}
        for ticket in self._dataset.tickets:
            same_company = str(ticket.get("company", "")).strip().lower() == company
            same_id = str(ticket.get("account_id", "")).strip().upper() == account_id
            if same_company or same_id:
                matched[ticket["ticket_id"]] = ticket

        selected: Iterable[dict[str, Any]] = matched.values()
        if window_days is not None:
            cutoff = reference - timedelta(days=window_days)
            selected = [
                t for t in selected if parse_timestamp(t["created_at"]) >= cutoff
            ]

        return sorted(
            selected,
            key=lambda t: (-parse_timestamp(t["created_at"]).timestamp(), t["ticket_id"]),
        )

    def get_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        wanted = ticket_id.strip().upper()
        for ticket in self._dataset.tickets:
            if str(ticket["ticket_id"]).strip().upper() == wanted:
                return ticket
        return None


def rank_tickets_for_brief(
    tickets: list[dict[str, Any]], *, limit: int, as_of: datetime
) -> list[dict[str, Any]]:
    """Pick the highest-signal tickets for the extraction stage.

    Deterministic and cheap: severity dominates, unresolved work and poor CSAT
    add weight, recency breaks near-ties. This is the cost control that keeps a
    brief at two LLM calls regardless of how noisy an account is.
    """
    if limit <= 0:
        return []

    def score(ticket: dict[str, Any]) -> tuple[float, str]:
        weight = float(_URGENCY_WEIGHT.get(ticket.get("urgency", ""), 1)) * 10.0
        if ticket.get("status") in _UNRESOLVED_STATUSES:
            weight += 6.0
        csat = ticket.get("satisfaction_score")
        if isinstance(csat, int) and csat <= 2:
            weight += 5.0
        age_days = (as_of - parse_timestamp(ticket["created_at"])).days
        weight += max(0.0, 5.0 - age_days / 20.0)
        # Negate for descending sort; ticket_id keeps ordering total.
        return (-weight, ticket["ticket_id"])

    return sorted(tickets, key=score)[:limit]


def summarise_recurring_themes(tickets: list[dict[str, Any]], *, top_n: int = 3) -> list[str]:
    """Product-area pairs that appear more than once, most frequent first."""
    counts: dict[str, int] = {}
    for ticket in tickets:
        key = f"{ticket.get('product', 'Unknown')} / {ticket.get('product_area', 'Unknown')}"
        counts[key] = counts.get(key, 0) + 1
    repeated = [(name, n) for name, n in counts.items() if n > 1]
    repeated.sort(key=lambda item: (-item[1], item[0]))
    return [f"{name} ({n} tickets)" for name, n in repeated[:top_n]]


def compute_metrics(
    account: dict[str, Any],
    tickets: list[dict[str, Any]],
    *,
    as_of: datetime,
) -> dict[str, Any]:
    """Facts the model is never asked to derive."""
    seats_licensed = int(account.get("seats_licensed") or 0)
    seats_active = int(account.get("seats_active") or 0)
    utilisation = round(seats_active / seats_licensed, 3) if seats_licensed else 0.0

    days_to_renewal: int | None = None
    renewal = account.get("renewal_date")
    if renewal:
        try:
            renewal_dt = parse_timestamp(f"{renewal}T00:00:00Z")
            days_to_renewal = (renewal_dt - as_of).days
        except Exception:  # noqa: BLE001 - a bad date must not break the brief
            days_to_renewal = None

    scores = [
        t["satisfaction_score"]
        for t in tickets
        if isinstance(t.get("satisfaction_score"), int)
    ]
    product_counts: dict[str, int] = {}
    for ticket in tickets:
        product = ticket.get("product", "Unknown")
        product_counts[product] = product_counts.get(product, 0) + 1
    top_products = [
        name
        for name, _ in sorted(product_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    ]

    return {
        "health_status": account.get("health_status", "Unknown"),
        "usage_trend": account.get("usage_trend", "Unknown"),
        "plan_tier": account.get("plan_tier", "Unknown"),
        "arr_usd": int(account.get("arr_usd") or 0),
        "seats_licensed": seats_licensed,
        "seats_active": seats_active,
        "seat_utilisation": utilisation,
        "renewal_date": str(account.get("renewal_date") or "unknown"),
        "days_to_renewal": days_to_renewal,
        "nps_score": account.get("nps_score"),
        "open_tickets": int(account.get("open_tickets") or 0),
        "tickets_in_window": len(tickets),
        "p1_in_window": sum(1 for t in tickets if t.get("urgency") == "P1"),
        "p2_in_window": sum(1 for t in tickets if t.get("urgency") == "P2"),
        "unresolved_in_window": sum(
            1 for t in tickets if t.get("status") in _UNRESOLVED_STATUSES
        ),
        "avg_satisfaction": round(sum(scores) / len(scores), 2) if scores else None,
        "top_products_by_volume": top_products,
        "recurring_themes": summarise_recurring_themes(tickets),
    }
