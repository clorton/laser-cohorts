import numpy as np


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
        dtype=np.uint32,
        default_value=0,
    ):
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

            (StateArray): An instance of StateArray with the specified properties
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
    def _cache_state_views(obj):

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

    def __array_finalize__(self, obj):
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

    def __getattr__(self, name):
        # only called if regular attribute lookup fails

        mapping = getattr(self, "_state_to_view", None)
        if mapping is not None and name in mapping:
            return mapping[name]

        return super().__getattribute__(name)

    def __setattr__(self, name, value):
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

    def __getitem__(self, key):
        return self.view(np.ndarray)[key]

    @property
    def state_names(self):
        """Return the list of state compartment names."""
        # We can just return the tuple, it's immutable
        return self._state_names

    @property
    def state_axis(self) -> int:
        """Get the axis index for the state compartments."""
        if self._state_axis is None:
            raise RuntimeError("state_axis is None")
        return self._state_axis

    def get_state_index(self, name):
        """Get the numeric index for a state compartment name."""
        return self._state_names.index(name) if (self._state_names is not None) and (name in self._state_names) else None
