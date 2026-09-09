import pytest
from qdrant_client import models

from medibot import rbac


ALL_ROLES = ["doctor", "nurse", "billing_executive", "technician", "admin"]


@pytest.mark.parametrize("role", ALL_ROLES)
def test_every_role_can_read_general(role: str) -> None:
    assert "general" in rbac.collections_for(role)


def test_admin_sees_every_collection() -> None:
    assert set(rbac.collections_for("admin")) == {
        "general", "clinical", "nursing", "billing", "equipment"
    }


@pytest.mark.parametrize(
    ("role", "allowed"),
    [
        ("doctor", {"general", "clinical", "nursing"}),
        ("nurse", {"general", "nursing"}),
        ("billing_executive", {"general", "billing"}),
        ("technician", {"general", "equipment"}),
    ],
)
def test_access_matrix_matches_spec(role: str, allowed: set[str]) -> None:
    assert set(rbac.collections_for(role)) == allowed


def test_roles_for_collection_is_the_inverse_of_collections_for() -> None:
    for collection in rbac.COLLECTIONS:
        for role in ALL_ROLES:
            in_forward = collection in rbac.collections_for(role)
            in_inverse = role in rbac.roles_for(collection)
            assert in_forward == in_inverse, (role, collection)


def test_unknown_role_raises() -> None:
    with pytest.raises(rbac.UnknownRoleError):
        rbac.collections_for("ceo")


def test_unknown_collection_raises() -> None:
    with pytest.raises(rbac.UnknownCollectionError):
        rbac.roles_for("finance")


def test_access_filter_matches_role_inside_access_roles_payload() -> None:
    flt = rbac.access_filter("nurse")
    assert isinstance(flt, models.Filter)
    assert len(flt.must) == 1
    condition = flt.must[0]
    assert isinstance(condition, models.FieldCondition)
    assert condition.key == "access_roles"
    assert condition.match == models.MatchValue(value="nurse")


def test_access_filter_rejects_unknown_role() -> None:
    with pytest.raises(rbac.UnknownRoleError):
        rbac.access_filter("hacker")


def test_only_billing_and_admin_may_use_sql() -> None:
    assert rbac.can_use_sql("billing_executive")
    assert rbac.can_use_sql("admin")
    for role in ("doctor", "nurse", "technician"):
        assert not rbac.can_use_sql(role)


def test_refusal_message_names_role_blocked_collection_and_allowed_ones() -> None:
    msg = rbac.refusal_message("nurse", "billing")
    assert "nurse" in msg
    assert "billing" in msg
    assert "nursing" in msg and "general" in msg
    assert "clinical" not in msg


def test_sql_refusal_message_names_role_and_permitted_roles() -> None:
    msg = rbac.sql_refusal_message("technician")
    assert "technician" in msg
    assert "billing" in msg and "admin" in msg


def test_join_reads_naturally_for_one_two_and_three_items() -> None:
    assert rbac._join(["general"]) == "general"
    assert rbac._join(["general", "nursing"]) == "general and nursing"
    assert rbac._join(["a", "b", "c"]) == "a, b and c"
