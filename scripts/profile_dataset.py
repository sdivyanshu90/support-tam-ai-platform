"""Print the dataset evidence behind three design decisions.

Run with `python -m app profile` (or `python scripts/profile_dataset.py`).

The three decisions this justifies:

1. The dataset's own `category` and `urgency` fields are not ground truth.
2. Tickets join to accounts on `company`, not on `account_id`.
3. The 90-day window is anchored to the newest ticket, not to wall-clock now.

Every claim in the README about the data is reproducible from this script.
"""

from __future__ import annotations

import collections
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.loader import load_dataset, parse_timestamp  # noqa: E402


def _rule(title: str) -> None:
    print(f"\n{title}\n{'=' * len(title)}")


def _lexical_signal(ticket: dict) -> str:
    """A crude, content-only guess at what a ticket is about."""
    text = f"{ticket['subject']} {ticket['body']}".lower()
    if "error:" in text or "error " in text:
        return "mentions an error"
    if "invoice" in text or "billing" in text or "charged" in text:
        return "mentions billing"
    if "best practice" in text or "how do" in text or "could you point us" in text:
        return "asks how-to"
    if text.strip().startswith("request:") or "expected behaviour" in text:
        return "requests a feature"
    return "other"


def main() -> int:
    dataset = load_dataset()
    tickets, accounts = dataset.tickets, dataset.accounts

    _rule("1. The dataset's own labels are uncorrelated with ticket content")
    print(
        "If `category` were ground truth, tickets that literally contain an error\n"
        "message would concentrate in Bug/Integration. They do not — the label is\n"
        "close to uniform within every content group:\n"
    )
    table: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for ticket in tickets:
        table[_lexical_signal(ticket)][ticket["category"]] += 1
    for signal in sorted(table):
        counts = table[signal]
        spread = ", ".join(f"{name} {n}" for name, n in counts.most_common())
        print(f"  tickets that {signal:<20} -> {spread}")

    urgency_by_tone = collections.Counter(
        (("urgent" in f"{t['subject']}{t['body']}".lower()), t["urgency"]) for t in tickets
    )
    print("\nAnd urgency does not track the word 'urgent' either:")
    for tone in (True, False):
        row = ", ".join(
            f"{tier} {urgency_by_tone[(tone, tier)]}" for tier in ("P1", "P2", "P3", "P4")
        )
        label = "contains 'urgent'" if tone else "does not"
        print(f"  {label:<20} -> {row}")
    print(
        "\n=> These fields are used only as a source of controlled vocabulary.\n"
        "   They are never used as supervision, and never as eval ground truth."
    )

    _rule("2. Tickets join to accounts on `company`, not `account_id`")
    account_ids = {a["account_id"] for a in accounts}
    by_id = collections.Counter(t["account_id"] for t in tickets)
    id_matches = {aid: by_id[aid] for aid in account_ids if by_id.get(aid)}
    company_overlap = {t["company"] for t in tickets} & {a["company"] for a in accounts}
    by_company = collections.Counter(t["company"] for t in tickets)

    print(f"  distinct account_id values in tickets.json : {len(by_id)}")
    print(f"  account_id values that match an account    : {len(id_matches)} of {len(account_ids)}")
    print(f"  company names that match an account        : {len(company_overlap)} of {len(accounts)}")
    mismatched = 0
    for ticket in tickets:
        for account in accounts:
            if ticket["account_id"] == account["account_id"] and ticket["company"] != account["company"]:
                mismatched += 1
    print(f"  id-matched tickets whose company disagrees : {mismatched} of {sum(id_matches.values())}")
    counts = sorted(by_company.get(a["company"], 0) for a in accounts)
    print(f"  tickets per account via company join       : min {counts[0]}, max {counts[-1]}")
    print(
        "\n=> `account_id` on a ticket is not a usable foreign key in this corpus.\n"
        "   The repository joins on company and unions in any id match, so a\n"
        "   future dataset with real ids keeps working."
    )

    _rule("3. The 90-day window is anchored to the data, not to the wall clock")
    stamps = [parse_timestamp(t["created_at"]) for t in tickets]
    now = dt.datetime.now(dt.timezone.utc)
    from_data = sum(1 for s in stamps if s >= max(stamps) - dt.timedelta(days=90))
    from_clock = sum(1 for s in stamps if s >= now - dt.timedelta(days=90))
    print(f"  ticket dates span            : {min(stamps).date()} to {max(stamps).date()}")
    print(f"  wall-clock today             : {now.date()}")
    print(f"  tickets in last 90d (as-of max created_at) : {from_data} of {len(tickets)}")
    print(f"  tickets in last 90d (wall clock)           : {from_clock} of {len(tickets)}")
    print(f"  configured as-of             : {dataset.as_of.isoformat()}")
    print(
        "\n=> The snapshot is historical. Anchoring to wall-clock time would empty\n"
        "   the window and silently produce empty briefs. Override with\n"
        "   APP_AS_OF_DATE if you need a different reference point."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
