"""Campaign-based intervention scheduling for cohort-based simulation.

Provides ``Campaign``, a component that loads a schedule from a dict, list,
JSON file, or CSV file and dispatches named interventions at the right ticks,
nodes, and compartment states.

Each schedule entry specifies:

- **who**: ``"*"`` (all states) or a list of state names, e.g. ``["S", "R"]``
- **what**: name of the registered intervention class
- **when**: ``"*"`` (every tick), an integer tick, a list of integer ticks, or a ``"YYYY-MM-DD"`` date
- **where**: ``"*"`` (all nodes), a single node ID, or a list of node IDs
- **parameters**: arbitrary ``{key: value}`` pairs forwarded to the intervention
- **notes**: free-text string forwarded to the intervention

Date-based ``when`` values require a ``start_date`` argument on the Campaign.
Integer ticks and date strings cannot be mixed in the same schedule.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from laser.cohorts.model import Model

from laser.cohorts.utils import PropertyType

logger = logging.getLogger(__name__)

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _is_date_string(value: str) -> bool:
    return bool(_DATE_RE.match(value))


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _normalize_who(raw) -> list[str] | None:
    if raw == "*":
        return None
    if isinstance(raw, list):
        return raw
    return json.loads(raw)


def _normalize_where(raw) -> list[int] | None:
    if raw == "*":
        return None
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, list):
        return [int(v) for v in raw]
    stripped = str(raw).strip()
    if stripped.startswith("["):
        return [int(v) for v in json.loads(stripped)]
    return [int(stripped)]


class Intervention:
    """Base class for all campaign interventions.

    Subclass this, implement ``execute``, and register with
    ``Campaign.register("MyName", MyClass)``.

    Example:
        >>> class Vaccination(Intervention):
        ...     def execute(self, tick, who, where, params, notes):
        ...         coverage = params.get("coverage", 0.5)
        ...         # move fraction of S to R in the target nodes
        ...         pass
        >>> Campaign.register("Vaccination", Vaccination)
    """

    def __init__(self, model: Model) -> None:
        """Initialize the Intervention.

        Args:
            model (Model): The parent model instance.
        """
        self.model = model

    def execute(
        self,
        tick: int,
        who: list[str] | None,
        where: list[int] | None,
        params: dict[str, Any],
        notes: str,
    ) -> None:
        """Execute the intervention.

        Args:
            tick (int): Current simulation tick (0-indexed).
            who (list[str] | None): Target state names; ``None`` means all states.
            where (list[int] | None): Target node IDs; ``None`` means all nodes.
            params (dict[str, Any]): Arbitrary parameters from the schedule entry.
            notes (str): Free-text annotation from the schedule entry.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(f"{type(self).__name__}.execute is not implemented")


class Campaign:
    """Intervention scheduling component.

    Loads a campaign schedule and fires named interventions at the specified
    ticks, nodes, and compartment states.

    Intervention classes are registered on the class-level registry with
    ``Campaign.register`` and looked up by name at each tick.

    Example:
        >>> Campaign.register(Vaccination)
        >>> schedule = [
        ...     {"who": "*", "what": "Vaccination", "when": 30,
        ...      "where": [0, 1], "parameters": {"coverage": 0.8}, "notes": ""},
        ...     {"who": ["S"], "what": "Vaccination", "when": [60, 90, 120],
        ...      "where": "*", "parameters": {"coverage": 0.6}, "notes": "boosters"},
        ... ]
        >>> campaign = Campaign(model, schedule)
        >>> model.components = [..., campaign]
    """

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, intervention_cls: type) -> None:
        """Register an intervention class using its ``__name__`` as the schedule key.

        Args:
            intervention_cls (type): Subclass of ``Intervention`` to register.
                The class name becomes the value expected in ``what`` fields.
        """
        name = intervention_cls.__name__
        logger.info("Campaign: registering intervention '%s'", name)
        cls._registry[name] = intervention_cls

        return

    def __init__(
        self,
        model: Model,
        source: dict | list | str | Path,
        start_date: str | date | None = None,
    ) -> None:
        """Initialize the Campaign component.

        Args:
            model (Model): The parent model instance.
            source (dict | list | str | Path): Campaign schedule.  One of:

                - a single entry ``dict``
                - a ``list`` of entry dicts
                - a file path (``str`` or ``Path``) to a ``.json`` or ``.csv``
                  file containing the schedule

            start_date (str | date | None): Simulation start date in
                ``"YYYY-MM-DD"`` format or as a ``datetime.date`` object.
                Required when any ``when`` value is a date string.

        Raises:
            ValueError: If the source path has an unsupported suffix.
            ValueError: If ``when`` values mix integer ticks and date strings.
            ValueError: If a list ``when`` contains date strings (unsupported).
            ValueError: If date-valued ``when`` entries are present but
                ``start_date`` is not provided.
        """
        self.model = model
        self._start_date = _parse_date(start_date) if start_date is not None else None

        raw = self._load(source)
        self._parsed = self._parse_entries(raw)

        self._every_tick: "list[dict]" = []
        self._at_tick: "dict[int, list[dict]]" = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, source: "dict | list | str | Path") -> "list[dict]":
        if isinstance(source, dict):
            return [source]
        if isinstance(source, list):
            return source
        path = Path(source)
        if path.suffix == ".json":
            data = json.loads(path.read_text())
            return [data] if isinstance(data, dict) else data
        if path.suffix == ".csv":
            return self._load_csv(path)
        raise ValueError(f"Unsupported source: expected dict, list, .json, or .csv; got {path.suffix!r}")

    def _load_csv(self, path: Path) -> "list[dict]":
        entries = []
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                entry: dict[str, Any] = dict(row)

                raw_params = entry.get("parameters", "").strip()
                entry["parameters"] = json.loads(raw_params) if raw_params else {}

                raw_who = entry.get("who", "*").strip()
                if raw_who != "*":
                    entry["who"] = json.loads(raw_who)

                raw_where = entry.get("where", "*").strip()
                if raw_where != "*" and not raw_where.lstrip("-").isdigit() and not raw_where.startswith("["):
                    pass  # leave as-is; _normalize_where handles string ints
                elif raw_where.startswith("["):
                    entry["where"] = json.loads(raw_where)

                raw_when = entry.get("when", "*").strip()
                if raw_when != "*" and not _is_date_string(raw_when):
                    if raw_when.startswith("["):
                        entry["when"] = json.loads(raw_when)
                    else:
                        entry["when"] = int(raw_when)

                entries.append(entry)
        return entries

    # ------------------------------------------------------------------
    # Parsing and validation
    # ------------------------------------------------------------------

    def _parse_entries(self, raw: "list[dict]") -> "list[dict]":
        has_dates = any(isinstance(e.get("when"), str) and e["when"] != "*" and _is_date_string(e["when"]) for e in raw)
        has_int_ticks = any(
            isinstance(e.get("when"), int)
            or isinstance(e.get("when"), list)
            or (isinstance(e.get("when"), str) and e["when"] != "*" and not _is_date_string(e["when"]))
            for e in raw
        )

        if has_dates and has_int_ticks:
            raise ValueError("Campaign schedule mixes date strings and integer ticks in 'when'. Use either dates or integers consistently.")
        if has_dates and self._start_date is None:
            raise ValueError("Campaign has date-valued 'when' entries but start_date was not provided.")

        parsed = []
        for entry in raw:
            when = entry.get("when", "*")
            base = {
                "what": entry["what"],
                "who": _normalize_who(entry.get("who", "*")),
                "where": _normalize_where(entry.get("where", "*")),
                "params": entry.get("parameters", {}),
                "notes": str(entry.get("notes", "")),
            }

            if when == "*":
                parsed.append({**base, "tick": None})
            elif isinstance(when, list):
                for t in when:
                    if isinstance(t, str) and _is_date_string(t):
                        raise ValueError("List of date strings in 'when' is not supported. Use a list of integer ticks instead.")
                    parsed.append({**base, "tick": int(t)})
            elif isinstance(when, int):
                parsed.append({**base, "tick": when})
            elif _is_date_string(str(when)):
                d = _parse_date(when)
                parsed.append({**base, "tick": (d - self._start_date).days})
            else:
                parsed.append({**base, "tick": int(when)})

        return parsed

    # ------------------------------------------------------------------
    # Component protocol
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Build the tick-indexed schedule from parsed entries."""
        for entry in self._parsed:
            if entry["tick"] is None:
                self._every_tick.append(entry)
            else:
                self._at_tick.setdefault(entry["tick"], []).append(entry)
        logger.info(
            "Campaign: %d every-tick and %d tick-specific interventions scheduled",
            len(self._every_tick),
            sum(len(v) for v in self._at_tick.values()),
        )

    def start_step(self, tick: int) -> None:
        """No-op start-of-step hook.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        pass

    def step(self, tick: int) -> None:
        """Dispatch all interventions scheduled for this tick.

        Fires every-tick interventions first, then tick-specific ones, in
        the order they appear in the schedule.

        Args:
            tick (int): Current simulation tick (0-indexed).

        Raises:
            KeyError: If a ``what`` name in the schedule is not registered.
        """
        entries = self._every_tick + self._at_tick.get(tick, [])
        for entry in entries:
            name = entry["what"]
            if name not in self._registry:
                raise KeyError(
                    f"Intervention '{name}' is not registered. Call Campaign.register(<class>) where <class>.__name__ == '{name}'."
                )
            cls = self._registry[name]
            logger.info(
                "Campaign tick %d: dispatching '%s' who=%s where=%s",
                tick,
                name,
                entry["who"],
                entry["where"],
            )
            intervention = cls(self.model)
            intervention.execute(
                tick=tick,
                who=entry["who"],
                where=entry["where"],
                params=entry["params"],
                notes=entry["notes"],
            )

    def end_step(self, tick: int) -> None:
        """No-op end-of-step hook.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        pass

    @property
    def properties(self) -> "list[PropertyType]":
        """Return node properties required by this component.

        Returns:
            list[PropertyType]: Empty list; the Campaign declares no node
                properties itself — individual interventions access the model
                directly.
        """
        return []

    @property
    def states(self) -> "list[str]":
        """Return compartment state names required by this component.

        Returns:
            list[str]: Empty list; the Campaign declares no new states.
        """
        return []
