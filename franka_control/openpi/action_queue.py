"""Synchronous action chunk queue for remote policy servers."""

from __future__ import annotations

from typing import Any

import numpy as np

ActionKey = str | list[str] | tuple[str, ...] | None


class ActionChunkQueue:
    """Fetch an action chunk when empty and pop one action per control step."""

    def __init__(
        self,
        action_key: ActionKey = "actions",
        return_horizon: int | None = None,
    ) -> None:
        self._action_key = action_key
        self._return_horizon = None if return_horizon is None else int(return_horizon)
        if self._return_horizon is not None and self._return_horizon <= 0:
            raise ValueError("return_horizon must be positive")

        self._pending: list[np.ndarray] = []
        self._reset_pending = True
        self._global_step = 0
        self._closed = False

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def queue_length(self) -> int:
        return len(self._pending)

    def next_action(self, policy: Any, observation: dict[str, Any]) -> np.ndarray:
        if self._closed:
            raise RuntimeError("Action chunk queue is closed")
        if not self._pending:
            request_obs = dict(observation)
            if self._reset_pending:
                request_obs["_reset"] = True
                self._reset_pending = False

            actions = actions_from_result(policy.infer(request_obs), self._action_key)
            _validate_return_horizon(actions, self._return_horizon)
            self._pending = [
                np.asarray(action, dtype=np.float32).reshape(-1)
                for action in actions
            ]
            if not self._pending:
                raise RuntimeError("Policy did not return any usable action")

        return np.asarray(self._pending.pop(0), dtype=np.float32)

    def mark_action_executed(self) -> int:
        self._global_step += 1
        return self._global_step

    def reset(self) -> None:
        self._pending.clear()
        self._reset_pending = True
        self._global_step = 0

    def close(self) -> None:
        self._closed = True
        self._pending.clear()


def actions_from_result(result: Any, action_key: ActionKey) -> np.ndarray:
    """Extract a contiguous ``(T, D)`` action chunk from a policy response."""
    if (
        isinstance(result, (list, tuple))
        and len(result) == 2
        and isinstance(result[0], dict)
    ):
        result = result[0]

    if isinstance(action_key, (list, tuple)):
        if not isinstance(result, dict):
            raise ValueError("action_key list requires a dict action result")
        action_chunks = [
            _coerce_action_chunk(_dict_value(result, str(key)), key=str(key))
            for key in action_key
        ]
        horizon = action_chunks[0].shape[0]
        for key, chunk in zip(action_key, action_chunks, strict=False):
            if chunk.shape[0] != horizon:
                raise ValueError(
                    f"Action key {key!r} has horizon {chunk.shape[0]}, "
                    f"expected {horizon}"
                )
        return np.ascontiguousarray(
            np.concatenate(action_chunks, axis=-1),
            dtype=np.float32,
        )

    if action_key:
        if not isinstance(result, dict):
            raise ValueError(f"action_key={action_key!r} requires a dict result")
        actions = _dict_value(result, action_key)
    else:
        actions = result
    return _coerce_action_chunk(actions, key=str(action_key or "actions"))


def _validate_return_horizon(
    actions: np.ndarray,
    return_horizon: int | None,
) -> None:
    if return_horizon is not None and actions.shape[0] != return_horizon:
        raise ValueError(
            f"Policy returned horizon {actions.shape[0]}, expected "
            f"return_horizon {return_horizon}"
        )


def _dict_value(mapping: dict[str, Any], key: str) -> Any:
    if key in mapping:
        return mapping[key]
    prefixed = f"action.{key}"
    if prefixed in mapping:
        return mapping[prefixed]
    if key.startswith("action.") and key.removeprefix("action.") in mapping:
        return mapping[key.removeprefix("action.")]
    raise KeyError(
        f"Action key {key!r} not found; available keys: {sorted(mapping)}"
    )


def _coerce_action_chunk(value: Any, *, key: str) -> np.ndarray:
    actions = np.asarray(value, dtype=np.float32)
    if actions.ndim == 3:
        if actions.shape[0] != 1:
            raise ValueError(
                f"Action key {key!r} must have batch size 1 when batched, "
                f"got {actions.shape}"
            )
        actions = actions[0]
    elif actions.ndim == 1:
        actions = actions[None, :]
    elif actions.ndim != 2:
        raise ValueError(
            f"Expected action key {key!r} to have shape (T,D), got {actions.shape}"
        )
    return np.ascontiguousarray(actions, dtype=np.float32)
