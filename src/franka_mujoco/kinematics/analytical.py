from __future__ import annotations

from collections.abc import Sequence


class AnalyticalPandaKinematics:
    """Starter placeholder for your own FK/Jacobian implementation.

    Use this class when you are ready to implement Denavit-Hartenberg or
    product-of-exponentials kinematics yourself. Until then, use
    MuJoCoKinematics as simulator ground truth for validation.
    """

    def forward(self, q: Sequence[float]):
        raise NotImplementedError("Implement analytical 7-DOF forward kinematics here.")

    def jacobian(self, q: Sequence[float]):
        raise NotImplementedError("Implement analytical 6x7 geometric Jacobian here.")
