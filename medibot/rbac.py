"""Role-based access matrix and the Qdrant filter that enforces it."""

from qdrant_client import models

ROLES: tuple[str, ...] = ("doctor", "nurse", "billing_executive", "technician", "admin")
COLLECTIONS: tuple[str, ...] = ("general", "clinical", "nursing", "billing", "equipment")

# Who may read each document collection. This is the single source of truth:
# ingestion stamps it onto every chunk as `access_roles`, and retrieval filters on it.
COLLECTION_ROLES: dict[str, tuple[str, ...]] = {
    "general": ROLES,
    "clinical": ("doctor", "admin"),
    "nursing": ("nurse", "doctor", "admin"),
    "billing": ("billing_executive", "admin"),
    "equipment": ("technician", "admin"),
}

SQL_ROLES: tuple[str, ...] = ("billing_executive", "admin")


class UnknownRoleError(ValueError):
    pass


class UnknownCollectionError(ValueError):
    pass


def _check_role(role: str) -> None:
    if role not in ROLES:
        raise UnknownRoleError(f"Unknown role '{role}'. Expected one of {', '.join(ROLES)}.")


def collections_for(role: str) -> list[str]:
    _check_role(role)
    return [c for c in COLLECTIONS if role in COLLECTION_ROLES[c]]


def roles_for(collection: str) -> list[str]:
    if collection not in COLLECTION_ROLES:
        raise UnknownCollectionError(
            f"Unknown collection '{collection}'. Expected one of {', '.join(COLLECTIONS)}."
        )
    return list(COLLECTION_ROLES[collection])


def access_filter(role: str) -> models.Filter:
    """Qdrant filter passed into every retrieval query.

    `access_roles` is a list on each chunk's payload; MatchValue on a list field
    matches when any element equals the role, so this reads as "role in access_roles".
    """
    _check_role(role)
    return models.Filter(
        must=[models.FieldCondition(key="access_roles", match=models.MatchValue(value=role))]
    )


def can_use_sql(role: str) -> bool:
    _check_role(role)
    return role in SQL_ROLES


def refusal_message(role: str, collection: str) -> str:
    allowed = collections_for(role)
    return (
        f"As a {role.replace('_', ' ')}, you don't have access to {collection} documents. "
        f"I can only answer questions from the {_join(allowed)} collection"
        f"{'s' if len(allowed) > 1 else ''}."
    )


def sql_refusal_message(role: str) -> str:
    _check_role(role)
    return (
        f"As a {role.replace('_', ' ')}, you don't have access to claims and maintenance "
        f"analytics. Those reports are available to {_join(SQL_ROLES)} roles only."
    )


def _join(items: tuple[str, ...] | list[str]) -> str:
    names = [i.replace("_", " ") for i in items]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]
