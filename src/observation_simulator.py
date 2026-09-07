"""Causal, calendar-aware observation simulation shared by validation runs.

The simulator deliberately receives the state/feature panel and labels as separate
objects.  It walks each location in time order: a hidden TWS value never enters the
visible state, even if a later source row is visible.  This is the single source of
truth for availability-faithful validation ledgers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    family: str
    first_source_month: str
    last_source_month: str
    training_target_cutoff: str
    visibility_rule: str
    notes: str = ""

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SimulatedFold:
    spec: ScenarioSpec
    ledger: pd.DataFrame
    labels: pd.DataFrame
    exclusions: dict[str, int]


def _period(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values).dt.to_period("M")


def conservative_training_cutoff(first_source_month: str | pd.Period) -> pd.Period:
    """Latest allowed target month, strictly before a simulated source month."""
    return pd.Period(first_source_month, freq="M") - 1


def simulate_observations(
    panel: pd.DataFrame,
    *,
    score_ids: pd.Series | None = None,
    labels: pd.DataFrame | None = None,
    scenario: ScenarioSpec,
) -> SimulatedFold:
    """Stream visible TWS state through a predeclared panel.

    ``panel`` must contain every prefix row needed to establish state and one row
    for each scored source row.  ``tws_visible`` defines the information set.  The
    returned ledger contains no target; labels are aligned separately by sample ID.
    Calendar gaps are retained rather than compressed.
    """
    required = {"sample_id", "time", "lat", "lon", "TWS_t", "tws_visible"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"panel missing required columns: {sorted(missing)}")
    if panel.duplicated(["sample_id"]).any():
        raise AssertionError("panel sample_id must be unique")
    if labels is not None:
        if {"sample_id", "target"}.difference(labels.columns):
            raise ValueError("labels must contain sample_id and target")
        if labels["sample_id"].duplicated().any():
            raise AssertionError("labels sample_id must be unique")

    x = panel.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t", "tws_visible"]].copy()
    x["source_date"] = pd.to_datetime(x["time"])
    x["source_period"] = x["source_date"].dt.to_period("M")
    x["_input_order"] = np.arange(len(x), dtype=np.int64)
    x = x.sort_values(["lat", "lon", "source_date", "_input_order"], kind="mergesort")

    x["tws_visible"] = x["tws_visible"].astype(bool)
    if x.loc[x["tws_visible"], "TWS_t"].isna().any():
        raise AssertionError("visible observations must have a TWS value")
    x["_candidate_date"] = x["source_date"].where(x["tws_visible"])
    x["_candidate_tws"] = x["TWS_t"].where(x["tws_visible"])
    groups = x.groupby(["lat", "lon"], sort=False)
    x["last_observed_date"] = groups["_candidate_date"].ffill()
    x["last_observed_TWS"] = groups["_candidate_tws"].ffill()

    score_mask = pd.Series(True, index=x.index)
    if score_ids is not None:
        wanted = set(score_ids.astype(str))
        score_mask = x["sample_id"].astype(str).isin(wanted)
        absent = wanted.difference(set(x["sample_id"].astype(str)))
        if absent:
            raise AssertionError(f"{len(absent)} score IDs are absent from the panel")
    scored = x.loc[score_mask].copy()
    exclusions = {"no_legal_anchor": int(scored["last_observed_date"].isna().sum())}
    scored = scored.loc[scored["last_observed_date"].notna()].copy()
    if scored.empty:
        raise AssertionError("Scenario has no scored rows with legal anchors")

    scored["target_date"] = scored["source_date"] + pd.offsets.MonthBegin(1)
    scored["h"] = (
        (scored["target_date"].dt.year * 12 + scored["target_date"].dt.month)
        - (scored["last_observed_date"].dt.year * 12 + scored["last_observed_date"].dt.month)
    ).astype(np.int16)
    if (scored["h"] < 1).any():
        raise AssertionError("A simulated horizon must be positive")
    if ((scored["h"] > 1) & (scored["last_observed_date"] == scored["source_date"])).any():
        raise AssertionError("A hidden source row used its own TWS")
    scored["location_id"] = scored.groupby(["lat", "lon"], sort=True).ngroup()
    scored["scenario_id"] = scenario.scenario_id
    ledger = scored.loc[:, [
        "sample_id", "source_date", "target_date", "source_period", "lat", "lon",
        "location_id", "tws_visible", "last_observed_date", "last_observed_TWS", "h",
        "scenario_id",
    ]].sort_values(["source_date", "lat", "lon"], kind="mergesort").reset_index(drop=True)

    if labels is None:
        aligned = pd.DataFrame({"sample_id": ledger["sample_id"]})
    else:
        aligned = labels.loc[:, ["sample_id", "target"]].set_index("sample_id").loc[
            ledger["sample_id"]
        ].reset_index()
        if aligned["target"].isna().any():
            raise AssertionError("Scored row has no supplied target")
    return SimulatedFold(spec=scenario, ledger=ledger, labels=aligned, exclusions=exclusions)


def build_mask_block_fold(
    train: pd.DataFrame,
    *,
    anchor_month: str | pd.Period,
    end_month: str | pd.Period,
    scenario_id: str,
    family: str = "recent_mask_block",
) -> SimulatedFold:
    """Visible anchor then hidden source-month block, with a full legal prefix."""
    required = {"sample_id", "time", "lat", "lon", "TWS_t", "target"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"train missing required columns: {sorted(missing)}")
    anchor = pd.Period(anchor_month, freq="M")
    end = pd.Period(end_month, freq="M")
    if end < anchor:
        raise ValueError("end_month precedes anchor_month")
    x = train.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t", "target"]].copy()
    x["source_period"] = _period(x["time"])
    prefix = x.loc[x["source_period"] < anchor].copy()
    window = x.loc[(x["source_period"] >= anchor) & (x["source_period"] <= end)].copy()
    if not (window["source_period"] == anchor).any():
        raise AssertionError(f"No source rows at requested anchor month {anchor}")
    prefix["tws_visible"] = True
    window["tws_visible"] = window["source_period"] == anchor
    panel = pd.concat([prefix, window], ignore_index=True, sort=False)
    spec = ScenarioSpec(
        scenario_id=scenario_id,
        family=family,
        first_source_month=str(anchor),
        last_source_month=str(end),
        training_target_cutoff=str(conservative_training_cutoff(anchor)),
        visibility_rule="full prefix; anchor visible; later supplied source rows TWS-hidden",
    )
    return simulate_observations(
        panel.drop(columns=["target"]),
        score_ids=window["sample_id"],
        labels=x.loc[:, ["sample_id", "target"]],
        scenario=spec,
    )


def build_template_replay_fold(
    train: pd.DataFrame,
    template: pd.DataFrame,
    *,
    start_month: str | pd.Period,
    scenario_id: str,
    family: str,
) -> SimulatedFold:
    """Replay the real Test's source-row presence and visibility schedule.

    Unlike the legacy direct anchor lookup, the visible state is generated by the
    same streaming simulator used for every other scenario.  The full historical
    prefix is visible; inside the window only template-visible TWS rows update
    state.  Template horizon equality is asserted rather than assumed.
    """
    required_train = {"sample_id", "time", "lat", "lon", "TWS_t", "target"}
    required_template = {"lat", "lon", "offset_months", "template_visible", "template_h"}
    if missing := required_train.difference(train.columns):
        raise ValueError(f"train missing required columns: {sorted(missing)}")
    if missing := required_template.difference(template.columns):
        raise ValueError(f"template missing required columns: {sorted(missing)}")
    start = pd.Period(start_month, freq="M")
    x = train.loc[:, ["sample_id", "time", "lat", "lon", "TWS_t", "target"]].copy()
    x["source_period"] = _period(x["time"])
    mapping = template.loc[:, ["lat", "lon", "offset_months", "template_visible", "template_h"]].copy()
    mapping["source_period"] = mapping["offset_months"].map(lambda offset: start + int(offset))
    candidate = x.merge(
        mapping, on=["lat", "lon", "source_period"], how="inner", validate="one_to_one"
    )
    # A global source-month match is insufficient: an individual historical grid
    # point may be absent for one of the template's source months.  Retain only
    # locations with the complete row schedule so stream-derived horizons remain
    # comparable to the real Test template.  This coverage rule is fixed before
    # any candidate predictions are constructed.
    expected = mapping.groupby(["lat", "lon"], sort=False).size().rename("expected_rows")
    found = candidate.groupby(["lat", "lon"], sort=False).size().rename("found_rows")
    coverage = expected.to_frame().join(found, how="left").fillna({"found_rows": 0})
    complete_locations = coverage.loc[
        coverage["found_rows"] == coverage["expected_rows"]
    ].reset_index()[["lat", "lon"]]
    window = candidate.merge(complete_locations, on=["lat", "lon"], how="inner", validate="many_to_one")
    if window.empty:
        raise AssertionError("Template replay has no historical source rows")
    prefix = x.loc[x["source_period"] < start].copy()
    prefix["tws_visible"] = True
    window["tws_visible"] = window["template_visible"].astype(bool)
    panel = pd.concat([prefix, window[x.columns.tolist() + ["tws_visible"]]], ignore_index=True, sort=False)
    spec = ScenarioSpec(
        scenario_id=scenario_id,
        family=family,
        first_source_month=str(start),
        last_source_month=str(window["source_period"].max()),
        training_target_cutoff=str(conservative_training_cutoff(start)),
        visibility_rule="full prefix plus transplanted Test source-row and TWS-visibility schedule",
    )
    fold = simulate_observations(
        panel.drop(columns=["target"]), score_ids=window["sample_id"],
        labels=x.loc[:, ["sample_id", "target"]], scenario=spec,
    )
    expected = window.set_index("sample_id").loc[fold.ledger["sample_id"], "template_h"].to_numpy(dtype=np.int16)
    actual = fold.ledger["h"].to_numpy(dtype=np.int16)
    if not np.array_equal(actual, expected):
        mismatch = int((actual != expected).sum())
        raise AssertionError(
            f"Streaming replay horizon differs from template for {mismatch}/{len(actual)} fully-covered rows"
        )
    exclusions = dict(fold.exclusions)
    exclusions["incomplete_template_locations"] = int((coverage["found_rows"] != coverage["expected_rows"]).sum())
    exclusions["incomplete_template_rows"] = int(len(mapping) - len(window))
    return SimulatedFold(spec=fold.spec, ledger=fold.ledger, labels=fold.labels, exclusions=exclusions)
