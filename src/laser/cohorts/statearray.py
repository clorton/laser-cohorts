"""StateArray: a NumPy ndarray subclass with named compartment access."""

from __future__ import annotations

import numpy as np
from typing import Any, Type


class StateArray(np.ndarray):
    """
    A numpy array wrapper that provides attribute access to state compartments.

    This class allows accessing state compartments by name (e.g., states.S, states.I, states.R)
    while maintaining full numpy array functionality and backward compatibility with
    numeric indexing (e.g., states[0], states[1]).

    Example:
        >>> states = StateArray(source_array=np.zeros((3, 100)), state_names=["S", "I", "R"], state_axis=0)
        >>> states.S[0] = 1000  # Set susceptible population in patch 0
        >>> prevalence = states.I / states.sum(axis=0)  # Calculate prevalence
        >>> states[0] += births  # Numeric indexing still works
        >>> N = states.sum(axis=states.state_axis)  # Sum over state axis to get total population per patch
    """

    def __new__(
        cls,
        state_names: list[str] | tuple[str, ...],
        state_axis: int,
        source_array: np.ndarray | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: Type[int] | Type[float] | type[np.generic] = np.uint32,
        default_value: int | float = 0,
    ) -> "StateArray":
        """Create a new StateArray instance.

        The StateArray can be created either by providing a source_array or by specifying the shape and default_value.
        In either case the state_names and state_axis parameters are required. state_names defines the names of the state compartments, and state_axis specifies the axis along which these states are stored.

        Args:
            state_names (list[str] | tuple[str, ...]): List or tuple of state compartment names (e.g., ["S", "E", "I", "R"])
            state_axis (int): The axis along which the state compartments are stored
            source_array (np.ndarray | None): The numpy array to wrap
            shape (tuple[int, ...] | None): The shape of the array if source_array is not provided
            dtype (np.dtype): The data type of the array
            default_value (number): The default value to fill the array with if source_array is not provided

        Returns:
            StateArray: An instance of StateArray with the specified properties.
        """

        if (source_array is not None) and (shape is not None):
            raise ValueError("specify either source_array or shape, but not both")

        if (source_array is None) and (shape is None):
            raise ValueError("must specify either source_array or shape")

        if state_axis < 0:
            raise ValueError(f"state_axis must be >= 0, got {state_axis}")

        if shape is None:
            shape = source_array.shape

        if state_axis >= len(shape):
            raise ValueError(f"state_axis must be between >= 0 and < {len(shape)}, got {state_axis}")

        if len(state_names) != shape[state_axis]:
            raise ValueError(f"Number of states {len(state_names)} does not match array states dimension {shape[state_axis]}.")

        if source_array is None:
            source_array = np.full(shape, default_value, dtype=dtype)

        arr = np.asarray(source_array)

        for state in state_names:
            if not state.isidentifier():
                raise ValueError(f"Invalid state name: {state!r}")
            if hasattr(np.ndarray, state):
                raise ValueError(f"State name collides with ndarray attribute: {state!r}")

        obj = arr.view(cls)
        obj._state_axis = state_axis
        obj._state_names = tuple(state_names)

        StateArray._cache_state_views(obj)

        return obj

    @staticmethod
    def _cache_state_views(obj: "StateArray") -> None:
        """Build and cache a plain-ndarray view for each named state compartment.

        Pre-computes an ndarray slice selecting each state's index along
        `state_axis` so that named attribute access (e.g. ``arr.S``) returns
        the view without recomputing the index each time.

        Args:
            obj (StateArray): The StateArray instance whose views are to be cached.
        """
        state_axis = obj._state_axis
        state_names = obj._state_names
        shape = obj.shape

        def get_indexing(value):
            """Get tuple representing an axis slice."""
            # Example np.ndarray[:,n,:,:] actually calls np.ndarray.__getitem__(tuple)
            # where tuple is (Slice(None), n, Slice(None), Slice(None))
            # So, we build the required tuple here based on state_axis, the value of n, and shape
            return tuple([slice(None) for i in range(state_axis)] + [value] + [slice(None) for i in range(state_axis + 1, len(shape))])

        # Instantiate and cache a np.ndarray view for each state.
        obj._state_to_view = {state: obj.view(np.ndarray)[get_indexing(i)] for i, state in enumerate(state_names)}

        return

    def __array_finalize__(self, obj: np.ndarray | None) -> None:
        """Propagate StateArray metadata to views and new instances.

        Called by NumPy whenever a new StateArray is created via slicing, view
        casting, or ufunc output.  Copies `_state_axis` and `_state_names` from
        the template object and re-caches state views when the shape along the
        state axis is still consistent.

        Args:
            obj (np.ndarray | None): The array from which this instance was derived,
                or ``None`` during direct construction.
        """
        if obj is None:
            return
        self._state_axis = getattr(obj, "_state_axis", None)
        self._state_names = getattr(obj, "_state_names", None)
        self._state_to_view = None
        if self._state_axis is not None and self._state_names is not None:
            # This does not guarantee that the state views will be valid, but it
            # is a best effort to cache them for any view that has the same shape
            # along the state axis.
            # If the shape along the state axis changes, the views will be invalid
            if self._state_axis < self.ndim and self.shape[self._state_axis] == len(self._state_names):
                StateArray._cache_state_views(self)

        return

    def __getattr__(self, name: str) -> Any:
        """Return the cached ndarray view for a registered state name.

        Only invoked when normal attribute lookup has already failed.  Checks
        the ``_state_to_view`` mapping and returns the pre-computed slice if
        `name` is a registered state name, otherwise delegates to
        ``ndarray.__getattribute__``.

        Args:
            name (str): Attribute name being looked up.

        Returns:
            np.ndarray: The cached state-compartment view.

        Raises:
            AttributeError: If `name` is not a registered state name.
        """
        # only called if regular attribute lookup fails

        mapping = getattr(self, "_state_to_view", None)
        if mapping is not None and name in mapping:
            return mapping[name]

        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Assign a value to a registered state compartment or an internal attribute.

        Private attributes (prefixed with ``_``) bypass the state-name check and
        are stored normally.  For public names, if the name matches a registered
        state compartment the value is broadcast-assigned into the cached view;
        otherwise an ``AttributeError`` is raised to prevent silent typo-based
        compartment creation.

        Args:
            name (str): Attribute name to assign.
            value: Value to assign.  For state names this is broadcast into the
                existing ndarray view via ``view[...] = value``.

        Raises:
            AttributeError: If `name` is not a registered state name, ``"dtype"``,
                or ``"shape"``.
        """
        # Intercept field assignment like x.E = ...
        if name.startswith("_"):
            super().__setattr__(name, value)
            return

        mapping = getattr(self, "_state_to_view", None)
        if mapping is not None:
            if name in mapping:
                view = mapping[name]
                view[...] = value
                return
            else:
                if name in ["dtype", "shape"]:
                    raise AttributeError(f"attribute '{name}' of 'laser.measles.utils.StateArray' objects is not writable")
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        super().__setattr__(name, value)

        return

    def __getitem__(self, key: Any) -> np.ndarray:
        """Index the underlying ndarray, always returning a plain ndarray.

        Delegates to the underlying ``np.ndarray`` view so that all indexing
        operations (integer, slice, tuple, fancy) return base ndarrays rather
        than StateArray subclass instances.

        Args:
            key: Any valid NumPy index (int, slice, tuple, array, etc.).

        Returns:
            np.ndarray: The indexed data as a plain ndarray.
        """
        return self.view(np.ndarray)[key]

    @property
    def state_names(self) -> tuple[str, ...] | None:
        """Return the tuple of registered state compartment names.

        Returns:
            tuple[str, ...] | None: Compartment names in axis order, or ``None``
                if the instance was created via view casting without metadata.
        """
        # We can just return the tuple, it's immutable
        return self._state_names

    @property
    def state_axis(self) -> int:
        """Return the axis index along which state compartments are stored.

        Returns:
            int: Zero-based axis index for the state dimension.

        Raises:
            RuntimeError: If ``_state_axis`` is ``None``, which occurs when the
                instance was created via view casting without metadata.
        """
        if self._state_axis is None:
            raise RuntimeError("state_axis is None")
        return self._state_axis

    def get_state_index(self, name: str) -> int | None:
        """Return the numeric axis index for a named state compartment.

        Args:
            name (str): State compartment name to look up.

        Returns:
            int | None: Zero-based index of `name` along the state axis, or
                ``None`` if `name` is not registered or state metadata is absent.
        """
        return self._state_names.index(name) if (self._state_names is not None) and (name in self._state_names) else None

    def get_state_mask(self, states: str | list[str]) -> np.ndarray:
        """Return a boolean mask selecting the specified state compartments.

        The returned array has length equal to the number of registered states
        and is ``True`` at each position corresponding to a named state in
        ``states``.  Useful for vectorised operations that apply to a subset
        of compartments (e.g. mortality restricted to ``["S", "I"]``).

        Args:
            states (str | list[str]): A single state name or a list of state
                names to include in the mask.

        Returns:
            np.ndarray: Boolean array of length ``n_states`` (the size of the
                state axis) with ``True`` at each index corresponding to a
                state in ``states`` and ``False`` elsewhere.

        Raises:
            ValueError: If ``states`` is neither a string nor a list.
            ValueError: If any name in ``states`` is not a registered state.

        Example:
            >>> sa = StateArray(["S", "I", "R"], 0, shape=(3, 10))
            >>> sa.get_state_mask("S")
            array([ True, False, False])
            >>> sa.get_state_mask(["S", "R"])
            array([ True, False,  True])
        """
        if isinstance(states, str):
            states = [states]

        if not isinstance(states, list):
            raise ValueError(f"'states' must be a string or list of strings, got {type(states)}")

        mask = np.zeros(self.shape[self.state_axis], dtype=bool)

        for state in states:
            idx = self.get_state_index(state)
            if idx is not None:
                mask[idx] = True
            else:
                raise ValueError(f"'{state}' is not a valid state for this StateArray, must be one of {self._state_names}")

        return mask
