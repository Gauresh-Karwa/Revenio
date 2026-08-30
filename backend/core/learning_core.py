"""
Learning core: bandit policies over a domain module's discrete action space.

Per architecture doc 6.2, the learning core must be drift-aware from the
start, not a stationary bandit retrofitted later. Three policies with
genuinely different behavior, plus a LearningCore manager that owns one
policy per domain and enforces single-writer updates (architecture doc
9.5) so concurrent case processing can't race on the same policy's state.

WHAT AN "ARM" IS HERE: domain-agnostic — an integer index into whatever
discrete action set a module already has (subscription's RETRY_BACKOFF_HOURS,
4 arms; checkout-abandonment's channel escalation, 3 arms).

WHAT "REWARD" MEANS: a float in [0, 1] — amount_recovered normalized by the
case amount (1.0 = fully recovered, 0.0 = lost), not a raw dollar amount.

WHAT "POOLING TWO DOMAINS" MEANS HERE: subscription's action space (retry
timing) and checkout-abandonment's (channel choice) are NOT the same set of
arms, so sharing one policy's Beta counts across both would be dimensionally
meaningless. Pooling means: each domain gets its OWN policy instance, but
both live under one LearningCore, updated through the same single-writer
observer (see backend.core.bandit_observer.BanditUpdateObserver). "Pooling
improves the system" is measured at the LearningCore level (aggregate money
recovered), not by merging incompatible action spaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Any

import numpy as np


def _validate_reward(reward: float) -> None:
    if not (0.0 <= reward <= 1.0):
        raise ValueError(f"reward must be in [0, 1], got {reward!r}")


class BanditPolicy(ABC):
    n_arms: int

    @abstractmethod
    def select_arm(self, context: Any = None) -> int: ...

    @abstractmethod
    def update(self, arm: int, reward: float) -> None: ...

    @abstractmethod
    def snapshot(self) -> dict[str, Any]: ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "BanditPolicy": ...


class StaticHeuristicPolicy(BanditPolicy):
    """The baseline: a fixed rule that never learns."""

    def __init__(self, n_arms: int, fixed_arm: int = 0) -> None:
        if n_arms < 1:
            raise ValueError(f"n_arms must be >= 1, got {n_arms}")
        if not (0 <= fixed_arm < n_arms):
            raise ValueError(f"fixed_arm must be in [0, {n_arms}), got {fixed_arm}")

        self.n_arms = n_arms
        self.fixed_arm = fixed_arm
        self._pull_counts = np.zeros(n_arms, dtype=int)
        self._reward_sums = np.zeros(n_arms, dtype=float)

    def select_arm(self, context: Any = None) -> int:
        return self.fixed_arm

    def update(self, arm: int, reward: float) -> None:
        _validate_reward(reward)
        if not (0 <= arm < self.n_arms):
            raise ValueError(f"arm must be in [0, {self.n_arms}), got {arm}")
        self._pull_counts[arm] += 1
        self._reward_sums[arm] += reward

    def snapshot(self) -> dict[str, Any]:
        return {
            "policy_type": "StaticHeuristicPolicy",
            "fixed_arm": self.fixed_arm,
            "arms": [
                {
                    "arm": i,
                    "pull_count": int(self._pull_counts[i]),
                    "mean_reward": (
                        float(self._reward_sums[i] / self._pull_counts[i])
                        if self._pull_counts[i] > 0 else None
                    ),
                }
                for i in range(self.n_arms)
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_type": "StaticHeuristicPolicy",
            "n_arms": self.n_arms,
            "fixed_arm": self.fixed_arm,
            "pull_counts": self._pull_counts.tolist(),
            "reward_sums": self._reward_sums.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StaticHeuristicPolicy":
        obj = cls(n_arms=data["n_arms"], fixed_arm=data["fixed_arm"])
        obj._pull_counts = np.array(data["pull_counts"], dtype=int)
        obj._reward_sums = np.array(data["reward_sums"], dtype=float)
        return obj


class ThompsonSamplingBandit(BanditPolicy):
    """
    Beta-Bernoulli Thompson Sampling, with two OPTIONAL drift-adaptation
    mechanisms: discount_factor (gamma in (0,1]) and/or window_size. Do not
    instantiate directly for "the stationary candidate" or "the drift-aware
    candidate" — use the two subclasses below.
    """

    def __init__(
        self,
        n_arms: int,
        discount_factor: float = 1.0,
        window_size: int | None = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if n_arms < 1:
            raise ValueError(f"n_arms must be >= 1, got {n_arms}")
        if not (0.0 < discount_factor <= 1.0):
            raise ValueError(f"discount_factor must be in (0, 1], got {discount_factor}")
        if window_size is not None and window_size < 1:
            raise ValueError(f"window_size must be >= 1 if set, got {window_size}")
        if window_size is not None and discount_factor < 1.0:
            raise ValueError(
                "discount_factor < 1.0 and window_size are mutually exclusive — "
                "pick one drift mechanism, not both."
            )
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError(
                f"prior_alpha and prior_beta must be > 0, got {prior_alpha}, {prior_beta}"
            )

        self.n_arms = n_arms
        self.discount_factor = discount_factor
        self.window_size = window_size
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

        self.alpha = np.full(n_arms, prior_alpha, dtype=float)
        self.beta = np.full(n_arms, prior_beta, dtype=float)
        self._pull_counts = np.zeros(n_arms, dtype=int)
        self._windows: list[deque] | None = (
            [deque(maxlen=window_size) for _ in range(n_arms)] if window_size else None
        )
        self._rng = np.random.default_rng(seed)

    def select_arm(self, context: Any = None) -> int:
        samples = self._rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        _validate_reward(reward)
        if not (0 <= arm < self.n_arms):
            raise ValueError(f"arm must be in [0, {self.n_arms}), got {arm}")

        self._pull_counts[arm] += 1

        if self._windows is not None:
            self._windows[arm].append(reward)
            rewards = self._windows[arm]
            self.alpha[arm] = self.prior_alpha + sum(rewards)
            self.beta[arm] = self.prior_beta + sum(1.0 - r for r in rewards)
        else:
            self.alpha[arm] = (
                self.prior_alpha + self.discount_factor * (self.alpha[arm] - self.prior_alpha) + reward
            )
            self.beta[arm] = (
                self.prior_beta + self.discount_factor * (self.beta[arm] - self.prior_beta) + (1.0 - reward)
            )

    def snapshot(self) -> dict[str, Any]:
        return {
            "policy_type": type(self).__name__,
            "discount_factor": self.discount_factor,
            "window_size": self.window_size,
            "arms": [
                {
                    "arm": i,
                    "pull_count": int(self._pull_counts[i]),
                    "alpha": float(self.alpha[i]),
                    "beta": float(self.beta[i]),
                    "mean_estimate": float(self.alpha[i] / (self.alpha[i] + self.beta[i])),
                }
                for i in range(self.n_arms)
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_type": type(self).__name__,
            "n_arms": self.n_arms,
            "discount_factor": self.discount_factor,
            "window_size": self.window_size,
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "alpha": self.alpha.tolist(),
            "beta": self.beta.tolist(),
            "pull_counts": self._pull_counts.tolist(),
            "windows": [list(w) for w in self._windows] if self._windows is not None else None,
            "rng_state": self._rng.bit_generator.state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThompsonSamplingBandit":
        # Deliberately bypasses cls(...) — StationaryThompsonSampling and
        # DriftAwareThompsonSampling narrow __init__'s accepted kwargs on
        # purpose (that narrowing enforces "stationary can't accidentally
        # be configured as drift-aware" at construction time).
        obj = object.__new__(cls)
        obj.n_arms = data["n_arms"]
        obj.discount_factor = data["discount_factor"]
        obj.window_size = data["window_size"]
        obj.prior_alpha = data["prior_alpha"]
        obj.prior_beta = data["prior_beta"]
        obj.alpha = np.array(data["alpha"], dtype=float)
        obj.beta = np.array(data["beta"], dtype=float)
        obj._pull_counts = np.array(data["pull_counts"], dtype=int)
        obj._windows = (
            [deque(w, maxlen=data["window_size"]) for w in data["windows"]]
            if data["windows"] is not None else None
        )
        obj._rng = np.random.default_rng()
        obj._rng.bit_generator.state = data["rng_state"]
        return obj


class StationaryThompsonSampling(ThompsonSamplingBandit):
    """Candidate 1: standard Beta-Bernoulli TS. Accumulates history uniformly forever."""

    def __init__(
        self, n_arms: int, prior_alpha: float = 1.0, prior_beta: float = 1.0, seed: int | None = None,
    ) -> None:
        super().__init__(
            n_arms=n_arms, discount_factor=1.0, window_size=None,
            prior_alpha=prior_alpha, prior_beta=prior_beta, seed=seed,
        )


class DriftAwareThompsonSampling(ThompsonSamplingBandit):
    """Candidate 2: discounted OR sliding-window TS. Requires a real drift mechanism."""

    def __init__(
        self,
        n_arms: int,
        discount_factor: float = 1.0,
        window_size: int | None = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if discount_factor >= 1.0 and window_size is None:
            raise ValueError(
                "DriftAwareThompsonSampling requires discount_factor < 1.0 and/or "
                "window_size to be set — otherwise use StationaryThompsonSampling."
            )
        super().__init__(
            n_arms=n_arms, discount_factor=discount_factor, window_size=window_size,
            prior_alpha=prior_alpha, prior_beta=prior_beta, seed=seed,
        )


class LearningCore:
    """
    Owns one bandit policy per domain_type. Update semantics match
    architecture doc 9.5: assumes a single-writer consumer calls update()
    sequentially. One domain's policy being absent must never block another's.
    """

    def __init__(self) -> None:
        self._policies: dict[str, BanditPolicy] = {}

    def register_policy(self, domain_type: str, policy: BanditPolicy) -> None:
        self._policies[domain_type] = policy

    def has_policy(self, domain_type: str) -> bool:
        return domain_type in self._policies

    def select_arm(self, domain_type: str, context: Any = None) -> int:
        if domain_type not in self._policies:
            raise KeyError(f"No policy registered for domain '{domain_type}'.")
        return self._policies[domain_type].select_arm(context)

    def update(self, domain_type: str, arm: int, reward: float) -> None:
        if domain_type not in self._policies:
            raise KeyError(f"No policy registered for domain '{domain_type}'.")
        self._policies[domain_type].update(arm, reward)

    def snapshot(self) -> dict[str, Any]:
        return {domain: policy.snapshot() for domain, policy in self._policies.items()}
