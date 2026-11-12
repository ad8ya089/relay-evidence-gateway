import pytest
from pydantic import ValidationError

from standin.profile import Profile, load_profile, profiles


def test_development_is_runnable_with_provenance() -> None:
    profile = load_profile()
    assert profile.name == "development"
    assert profile.models[profile.model].numbers()["rpm"] == 600
    assert profile.capacity.source == "assumed"
    assert profile.cache.ttl_s.value == 7200
    assert all(value.source != "fitted" for value in profile.faults.__dict__.values())


@pytest.mark.parametrize("name", ["free_exact", "scaled"])
def test_account_profiles_refuse_missing_evidence(name: str) -> None:
    with pytest.raises(ValueError, match="rpm, tpm, rpd, tpd"):
        load_profile(name)


def test_scaled_multiplies_all_base_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = profiles().copy()
    data = catalog["free_exact"].model_dump()
    for item in data["models"][data["model"]].values():
        item.update(value=10, source="observed", observed_at="2026-11-12")
    catalog["free_exact"] = Profile.model_validate(data)
    monkeypatch.setattr("standin.profile.profiles", lambda: catalog)
    assert set(load_profile("scaled").models[data["model"]].numbers().values()) == {300}
    assert load_profile("scaled").capacity.value == 8


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_fault_probability_validation(value: float) -> None:
    data = load_profile().model_dump()
    data["faults"]["timeout_prob"]["value"] = value
    with pytest.raises(ValidationError):
        Profile.model_validate(data)
