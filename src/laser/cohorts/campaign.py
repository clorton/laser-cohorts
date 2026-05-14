"""Campaign-based intervention scheduling for cohort-based simulation.

Provides ``Campaign``, a component that loads a schedule from a dict, list,
JSON file, or CSV file and dispatches named interventions at the right ticks,
nodes, and compartment states.

Each schedule entry specifies:

- **who** *(required)*: ``"*"`` (all states) or a list of state names, e.g. ``["S", "R"]``
- **what** *(required)*: name of the registered intervention class
- **when** *(optional, defaults to* ``"*"`` *)*: ``"*"`` (every tick), an integer tick, a list of integer ticks, a ``"YYYY-MM-DD"`` date, or a list of date strings
- **where** *(required)*: ``"*"`` (all nodes), a single node ID, or a list of node IDs
- **parameters** *(optional)*: arbitrary ``{key: value}`` pairs forwarded to the intervention
- **notes** *(optional)*: free-text string forwarded to the intervention

``who`` and ``where`` are required for every entry; omitting them raises
``ValueError``.  Use ``"*"`` explicitly to target all states or all nodes.

Date-based ``when`` values require a ``start_date`` argument on the Campaign.
Integer ticks and date strings cannot be mixed in the same schedule.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
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
    stripped = str(raw).strip()
    if stripped.startswith("["):
        return json.loads(stripped)
    return [stripped]


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


def _is_int_like(value) -> bool:
    """Return True when ``value`` is an int or a string of digits.

    Booleans are explicitly rejected; Python treats ``bool`` as a subtype of
    ``int`` but a ``True``/``False`` in a schedule almost certainly indicates
    a mistake.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, str):
        return value.strip().lstrip("-").isdigit()
    return False


def _is_date_like(value) -> bool:
    """Return True when ``value`` is a ``"YYYY-MM-DD"`` date string."""
    return isinstance(value, str) and _is_date_string(value)


def _validate_who_value(value) -> None:
    """Validate a ``who`` value: ``"*"``, a single string, or a list of strings.

    Raises:
        ValueError: If ``value`` is not one of the allowed forms.
    """
    if value == "*":
        return
    if isinstance(value, str):
        return
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return
    raise ValueError(
        f"Invalid 'who' value {value!r}: use '*', a single string, or a list of strings."
    )


def _validate_where_value(value) -> None:
    """Validate a ``where`` value: ``"*"``, a single int, or a list of ints.

    Raises:
        ValueError: If ``value`` is not one of the allowed forms.
    """
    if value == "*":
        return
    if _is_int_like(value):
        return
    if isinstance(value, list):
        for v in value:
            if not _is_int_like(v):
                raise ValueError(
                    f"Invalid 'where' list element {v!r}: list elements must be ints."
                )
        return
    raise ValueError(
        f"Invalid 'where' value {value!r}: use '*', an int, or a list of ints."
    )


def _validate_when_value(value) -> None:
    """Validate a ``when`` value: ``"*"``, a single int or date string, or a list.

    Lists must be homogeneous — all ints or all date strings, never mixed.

    Raises:
        ValueError: If ``value`` is not one of the allowed forms, or if a
            list mixes date strings and ints.
    """
    if value == "*":
        return
    if _is_int_like(value) or _is_date_like(value):
        return
    if isinstance(value, list):
        has_date = False
        has_int = False
        for v in value:
            if _is_date_like(v):
                has_date = True
            elif _is_int_like(v):
                has_int = True
            else:
                raise ValueError(
                    f"Invalid 'when' list element {v!r}: list elements must be ints or "
                    "'YYYY-MM-DD' date strings."
                )
        if has_date and has_int:
            raise ValueError(
                "Campaign 'when' list mixes date strings and integer ticks. "
                "Use either dates or integers consistently."
            )
        return
    raise ValueError(
        f"Invalid 'when' value {value!r}: use '*', an int, a 'YYYY-MM-DD' date string, "
        "or a list of ints / date strings."
    )


def _normalize_when(raw, start_date: date | None = None) -> list[int] | None:
    """Convert a pre-validated ``when`` value into a list of integer ticks.

    Pure conversion — assumes `Campaign._validate` has already approved the
    value, including the presence of ``start_date`` whenever date strings
    are used.

    Args:
        raw: Validated ``when`` value (``"*"``, an int, a date string, or a
            homogeneous list of those).
        start_date (date | None): Reference date for date-to-tick conversion.

    Returns:
        list[int] | None: Integer ticks in the order given, or ``None`` when
            ``raw == "*"``.
    """
    if raw == "*":
        return None

    if TYPE_CHECKING:
        assert start_date is not None

    items = raw if isinstance(raw, list) else [raw]

    result: list[int] = []
    for item in items:
        if isinstance(item, str) and _is_date_string(item):
            d = _parse_date(item)
            result.append((d - start_date).days)
        else:
            result.append(int(item))

    return result


@dataclass
class _ScheduledEntry:
    """A single fully-parsed schedule entry — one intervention dispatch.

    Each instance corresponds to one ``(tick, intervention)`` pair after the
    raw schedule has been validated and expanded — ``when`` lists are flattened
    into individual entries, ``who``/``where`` are normalised to lists or
    ``None``, and date strings are converted to integer tick offsets.

    Attributes:
        what (str): Name of the registered intervention class.
        who (list[str] | None): Target compartment states, or ``None`` (all).
        where (list[int] | None): Target node IDs, or ``None`` (all).
        params (dict[str, Any]): Arbitrary parameters forwarded to ``execute``.
        notes (str): Free-text annotation forwarded to ``execute``.
        tick (int | None): Tick on which to fire, or ``None`` to fire on every tick.
    """

    what: str
    who: list[str] | None
    where: list[int] | None
    params: dict[str, Any]
    notes: str
    tick: int | None


class Intervention:
    """Base class for all campaign interventions.

    Subclass this, implement ``execute``, and register with
    ``Campaign.register(MyClass)``.

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

    @property
    def states(self) -> list[str]:
        """Return compartment states required by this intervention.

        Override in subclasses to declare any new states the intervention
        needs (e.g. ``["V"]`` for a vaccination intervention).  These states
        are surfaced through ``Campaign.states`` so the model allocates them
        before the simulation runs.

        Returns:
            list[str]: Empty list; subclasses override as needed.
        """
        return []

    @property
    def properties(self) -> list[PropertyType]:
        """Return node properties required by this intervention.

        Override in subclasses to declare per-tick, per-node arrays for
        recording intervention outputs.  These properties are surfaced through
        ``Campaign.properties`` so the model allocates them before the
        simulation runs.

        Returns:
            list[PropertyType]: Empty list; subclasses override as needed.
        """
        return []

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
            ValueError: If any schedule entry omits the required ``who`` or
                ``where`` field.  Use ``"*"`` explicitly to target all states
                or all nodes.
            ValueError: If any ``what`` value names an intervention class that
                has not been registered with ``Campaign.register``.
            ValueError: If ``when`` values mix integer ticks and date strings
                (across entries or within a single list).
            ValueError: If date-valued ``when`` entries are present but
                ``start_date`` is not provided.
            ValueError: If a ``when`` date is earlier than ``start_date``.
        """
        self.model = model
        self._start_date = _parse_date(start_date) if start_date is not None else None

        raw = self._load(source)
        self._validate(raw)
        self._parsed = self._parse_entries(raw)

        self._every_tick: list[_ScheduledEntry] = []
        self._at_tick: dict[int, list[_ScheduledEntry]] = {}

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
        # `who`, `where`, and `when` all share the same cell grammar: an empty
        # or missing cell, a literal "*", a bracketed JSON list, or a single
        # bare scalar.  This helper normalises the shape uniformly; the
        # downstream `_normalize_*` functions are responsible for any field-
        # specific type coercion (e.g. int parsing for `where`/`when`).
        def _normalize_cell(value: str | None) -> Any:
            if value is None:
                return None
            stripped = value.strip()
            if stripped == "":
                return None
            if stripped == "*":
                return "*"
            if stripped.startswith("["):
                return json.loads(stripped)
            return stripped

        entries = []
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                entry: dict[str, Any] = dict(row)

                raw_params = entry.get("parameters", "").strip()
                entry["parameters"] = json.loads(raw_params) if raw_params else {}

                # Apply the same shape-normalisation to all three fields.
                # An absent cell drops the key — `_parse_entries` then either
                # raises (who/where, required) or defaults to "*" (when).
                for field in ("who", "where", "when"):
                    value = _normalize_cell(entry.get(field))
                    if value is None:
                        entry.pop(field, None)
                    else:
                        entry[field] = value

                entries.append(entry)
        return entries

    # ------------------------------------------------------------------
    # Parsing and validation
    # ------------------------------------------------------------------

    def _validate(self, raw: "list[dict]") -> None:
        """Validate a loaded schedule before parsing.

        Runs after `_load` and before `_parse_entries`, raising eagerly at
        construction time for any problem so downstream code can assume
        well-formed input.  Checks performed:

        - Every entry has the required ``who``, ``what``, and ``where`` keys.
        - Each entry's ``who``, ``where``, and ``when`` value matches the
          documented per-field grammar (see `_validate_who_value`,
          `_validate_where_value`, `_validate_when_value`).
        - Every entry's ``what`` value names an intervention class that has
          already been registered via `Campaign.register`.
        - A ``when`` list is homogeneous — all ints or all date strings.
        - The schedule as a whole does not mix date-valued and integer-valued
          ``when`` entries across different entries.
        - When any ``when`` is date-valued, ``start_date`` was provided and
          every date is on or after ``start_date``.

        Args:
            raw (list[dict]): The schedule entries returned by `_load`.

        Raises:
            ValueError: For any required-field omission, value-grammar
                violation, unregistered ``what`` name, mixed-dates-and-ints
                schedule, missing ``start_date`` when dates are used, or date
                earlier than ``start_date``.
        """
        for entry in raw:
            if "who" not in entry:
                raise ValueError(
                    "Campaign schedule entry is missing the required 'who' field. "
                    "Use '*' to target all states explicitly."
                )
            if "what" not in entry:
                raise ValueError(
                    "Campaign schedule entry is missing the required 'what' field."
                )
            if "where" not in entry:
                raise ValueError(
                    "Campaign schedule entry is missing the required 'where' field. "
                    "Use '*' to target all nodes explicitly."
                )
            _validate_who_value(entry["who"])
            _validate_where_value(entry["where"])
            _validate_when_value(entry.get("when", "*"))
            if entry["what"] not in self._registry:
                raise ValueError(
                    f"Intervention '{entry['what']}' is not registered. "
                    f"Call Campaign.register(<class>) where <class>.__name__ == '{entry['what']}' "
                    "before constructing the Campaign."
                )

        # Cross-entry consistency of `when`: date schedule or tick schedule,
        # not both.
        whens = [e.get("when", "*") for e in raw]

        def _has_date(when) -> bool:
            if _is_date_like(when):
                return True
            if isinstance(when, list):
                return any(_is_date_like(v) for v in when)
            return False

        def _has_int(when) -> bool:
            if when == "*":
                return False
            if _is_int_like(when):
                return True
            if isinstance(when, list):
                return any(_is_int_like(v) for v in when)
            return False

        any_date = any(_has_date(w) for w in whens)
        any_int = any(_has_int(w) for w in whens)

        if any_date:
            if any_int:
                raise ValueError(
                    "Campaign schedule mixes date strings and integer ticks in 'when'. "
                    "Use either dates or integers consistently."
                )
            if self._start_date is None:
                raise ValueError(
                    "Campaign has date-valued 'when' entries but start_date was not provided."
                )

            for w in whens:
                items = w if isinstance(w, list) else [w]
                for item in items:
                    if _is_date_like(item):
                        d = _parse_date(item)
                        if d < self._start_date:
                            raise ValueError(
                                f"Intervention date ({d:%Y-%m-%d}) is before campaign "
                                f"start date {self._start_date:%Y-%m-%d}."
                            )

    def _parse_entries(self, raw: "list[dict]") -> "list[_ScheduledEntry]":
        """Convert pre-validated schedule entries into the tick-indexed form.

        Assumes `_validate` has already approved every entry.  Each entry is
        expanded into one or more `_ScheduledEntry` instances — one per
        resolved tick — and every-tick entries get ``tick=None``.

        Args:
            raw (list[dict]): Validated schedule entries from `_load`.

        Returns:
            list[_ScheduledEntry]: Flattened list of per-tick dispatch entries.
        """
        parsed: list[_ScheduledEntry] = []
        for entry in raw:
            ticks = _normalize_when(entry.get("when", "*"), self._start_date)
            what = entry["what"]
            who = _normalize_who(entry["who"])
            where = _normalize_where(entry["where"])
            params = entry.get("parameters", {})
            notes = str(entry.get("notes", ""))

            if ticks is None:
                parsed.append(_ScheduledEntry(what=what, who=who, where=where, params=params, notes=notes, tick=None))
            else:
                for t in ticks:
                    parsed.append(_ScheduledEntry(what=what, who=who, where=where, params=params, notes=notes, tick=t))

        return parsed

    # ------------------------------------------------------------------
    # Component protocol
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Build the tick-indexed schedule from parsed entries."""
        for entry in self._parsed:
            if entry.tick is None:
                self._every_tick.append(entry)
            else:
                self._at_tick.setdefault(entry.tick, []).append(entry)
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
        the order they appear in the schedule.  All ``what`` names are
        validated against the registry at construction time, so the lookup
        here is unconditional.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        entries = self._every_tick + self._at_tick.get(tick, [])
        for entry in entries:
            cls = self._registry[entry.what]
            logger.info(
                "Campaign tick %d: dispatching '%s' who=%s where=%s",
                tick,
                entry.what,
                entry.who,
                entry.where,
            )
            intervention = cls(self.model)
            intervention.execute(
                tick=tick,
                who=entry.who,
                where=entry.where,
                params=entry.params,
                notes=entry.notes,
            )

    def end_step(self, tick: int) -> None:
        """No-op end-of-step hook.

        Args:
            tick (int): Current simulation tick (0-indexed).
        """
        pass

    @property
    def properties(self) -> "list[PropertyType]":
        """Return node properties required by this component and its interventions.

        Iterates over the unique intervention class names referenced in the
        schedule, instantiates each registered class with the model, and
        accumulates their ``properties`` declarations into a set so duplicates
        are dropped automatically.

        Returns:
            list[PropertyType]: Union of all property declarations from
                interventions used in this campaign's schedule.  Order is
                unspecified.
        """
        seen_classes: set[str] = set()
        result: set[PropertyType] = set()
        for entry in self._parsed:
            if entry.what in seen_classes:
                continue
            seen_classes.add(entry.what)
            intervention = self._registry[entry.what](self.model)
            result.update(intervention.properties)
        return list(result)

    @property
    def states(self) -> "list[str]":
        """Return compartment states required by this component and its interventions.

        Iterates over the unique intervention class names referenced in the
        schedule, instantiates each registered class with the model, and
        accumulates their ``states`` declarations into a set so duplicates are
        dropped automatically.

        Returns:
            list[str]: Union of all state declarations from interventions used
                in this campaign's schedule.  Order is unspecified.
        """
        seen_classes: set[str] = set()
        result: set[str] = set()
        for entry in self._parsed:
            if entry.what in seen_classes:
                continue
            seen_classes.add(entry.what)
            intervention = self._registry[entry.what](self.model)
            result.update(intervention.states)
        return list(result)
