"""Scale-down stabilization shared by the controller and the simulator.

Both the live controller (``services/forescale-controller``) and the offline
simulator (``experiments/simulator.py``) must apply the **same** max-hold rule so
that the headline comparison reflects one algorithm, not two slightly different
ones. Holding the peak desired replica count over a trailing window prevents a
brief forecast dip from stripping pre-warmed pods just before an imminent burst
peaks inside the lookahead window.

Isolating it here (with no Kubernetes, NumPy or I/O dependencies) keeps it
directly unit-testable and impossible to let drift between the two callers.
"""

from __future__ import annotations

from collections import deque


class MaxHold:
    """Track the maximum desired replica count over a trailing time window.

    Each scaling decision is recorded with its timestamp; :meth:`update` evicts
    entries older than ``window_s`` and returns the peak of what remains. This is
    the scale-down stabilization used identically by the controller and the
    simulator.

    Timestamps are caller-supplied (wall-clock seconds for the controller,
    simulated seconds for the simulator), so the class carries no clock of its
    own and stays deterministic under test.
    """

    def __init__(self, window_s: float) -> None:
        if window_s < 0:
            raise ValueError(f"window_s must be >= 0, got {window_s!r}")
        self.window_s = window_s
        self._entries: deque[tuple[float, int]] = deque()

    def update(self, now: float, desired: int) -> int:
        """Record ``desired`` at time ``now`` and return the windowed peak.

        Args:
            now: Current time (seconds); must be non-decreasing across calls.
            desired: The freshly computed desired replica count.

        Returns:
            The maximum desired count observed within the trailing
            ``window_s`` (always ``>= desired``).
        """
        self._entries.append((now, desired))
        cutoff = now - self.window_s
        while self._entries and self._entries[0][0] < cutoff:
            self._entries.popleft()
        return max(d for _, d in self._entries)
