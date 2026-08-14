"""Controlled vocabulary and routing policy.

The taxonomy is *derived from the supplied corpus*, not invented: products and
product areas are the distinct values present in `data/tickets.json`, and issue
categories are the eight documented in `DATA_SCHEMA.md`. Constraining the model
to this vocabulary is what stops it minting a new category per request.

Note on the dataset's own labels: `scripts/profile_dataset.py` shows `category`
and `urgency` in tickets.json are uncorrelated with ticket content, so they are
used *only* as a vocabulary source — never as supervision or eval ground truth.
"""

from __future__ import annotations

from functools import lru_cache

from app.data.loader import load_dataset

UNKNOWN = "Unknown"

# The eight categories documented in DATA_SCHEMA.md.
ISSUE_CATEGORIES: tuple[str, ...] = (
    "Bug",
    "Feature Request",
    "How-To",
    "Performance",
    "Billing",
    "Integration",
    "Onboarding",
    "Data Loss",
)

# Responder teams. Routing is a business policy, so it is computed in Python
# from the model's classification rather than asked of the model.
TEAMS: tuple[str, ...] = (
    "Data Platform Engineering",
    "Sync & Storage Engineering",
    "Analytics Engineering",
    "Security & Identity Engineering",
    "Automation Engineering",
    "Billing Operations",
    "Customer Onboarding",
    "Product Management",
    "Tier-1 Support",
)

_PRODUCT_TEAM: dict[str, str] = {
    "DataBridge Pro": "Data Platform Engineering",
    "CloudSync": "Sync & Storage Engineering",
    "AnalyticsHub": "Analytics Engineering",
    "SecureVault": "Security & Identity Engineering",
    "WorkflowEngine": "Automation Engineering",
}

# Category routing wins over product routing: a billing question about
# AnalyticsHub belongs to Billing Ops, not the analytics engineers.
_CATEGORY_TEAM: dict[str, str] = {
    "Billing": "Billing Operations",
    "Onboarding": "Customer Onboarding",
    "Feature Request": "Product Management",
    "How-To": "Tier-1 Support",
}

# Categories that stay with the owning product team no matter what.
_ENGINEERING_CATEGORIES = frozenset({"Bug", "Performance", "Integration", "Data Loss"})


class Taxonomy:
    """Allowed classification values, derived from the corpus."""

    def __init__(
        self,
        products: tuple[str, ...],
        areas_by_product: dict[str, tuple[str, ...]],
    ) -> None:
        self.products = products
        self.areas_by_product = areas_by_product
        self.all_areas: tuple[str, ...] = tuple(
            sorted({area for areas in areas_by_product.values() for area in areas})
        )
        self.categories = ISSUE_CATEGORIES
        self.teams = TEAMS

    def is_valid_product(self, product: str) -> bool:
        return product in self.products

    def is_valid_area(self, product: str, area: str) -> bool:
        return area in self.areas_by_product.get(product, ())

    def coerce(self, product: str, area: str, category: str) -> tuple[str, str, str]:
        """Snap model output onto the controlled vocabulary.

        Anything unrecognised becomes `Unknown` rather than being passed
        through, so downstream consumers never see an invented label.
        """
        product = product if self.is_valid_product(product) else UNKNOWN
        if product == UNKNOWN or not self.is_valid_area(product, area):
            area = area if area in self.all_areas and product == UNKNOWN else UNKNOWN
        category = category if category in self.categories else UNKNOWN
        return product, area, category

    def prompt_block(self) -> str:
        """Render the vocabulary for inclusion in a system prompt."""
        lines = ["Products and their valid product areas:"]
        for product in self.products:
            areas = ", ".join(self.areas_by_product[product])
            lines.append(f"- {product}: {areas}")
        lines.append("")
        lines.append("Issue categories: " + ", ".join(self.categories))
        return "\n".join(lines)


@lru_cache(maxsize=1)
def get_taxonomy() -> Taxonomy:
    dataset = load_dataset()
    areas: dict[str, set[str]] = {}
    for ticket in dataset.tickets:
        areas.setdefault(ticket["product"], set()).add(ticket["product_area"])
    areas_by_product = {p: tuple(sorted(a)) for p, a in sorted(areas.items())}
    return Taxonomy(tuple(sorted(areas_by_product)), areas_by_product)


def route(product: str, issue_category: str, urgency: str) -> tuple[str, str]:
    """Deterministic routing policy -> (team, rationale).

    Kept out of the LLM on purpose: routing is a stable business rule, and a
    rule is cheaper to audit, change, and test than a prompt.
    """
    if issue_category in _ENGINEERING_CATEGORIES:
        team = _PRODUCT_TEAM.get(product)
        if team:
            reason = f"{issue_category} in {product} is owned by the product engineering team"
            if urgency == "P1":
                return team, f"{reason}; P1 pages the on-call rota for that team"
            return team, reason

    team = _CATEGORY_TEAM.get(issue_category)
    if team:
        return team, f"{issue_category} requests are handled by {team} regardless of product"

    team = _PRODUCT_TEAM.get(product)
    if team:
        return team, f"Routed on product ownership ({product})"

    return "Tier-1 Support", "Product and category could not be identified; Tier-1 to qualify"
