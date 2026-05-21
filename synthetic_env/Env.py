# finite_mdp_jax.py
from __future__ import annotations

from typing import Any, Tuple, Optional

import jax
import jax.numpy as jnp
from flax import struct


# =============================================================================
# Helpers: sample simplex vectors (Dirichlet(1))
# =============================================================================
def random_simplex(key: jax.Array, shape: Tuple[int, ...], d: int) -> jax.Array:
    """Returns array of shape (*shape, d), last axis sums to 1."""
    x = jax.random.exponential(key, shape=(*shape, d))
    return x / jnp.sum(x, axis=-1, keepdims=True)


# =============================================================================
# Minimal gymnax-style Env API container
# =============================================================================
@struct.dataclass
class EnvAPI:
    reset: Any
    step: Any


# =============================================================================
# Params / State (PyTrees)
# =============================================================================
@struct.dataclass
class FiniteMDPParams:
    P: jax.Array      # (S, A, S)
    R: jax.Array      # (S, A)
    mu0: jax.Array    # (S,)

    # IMPORTANT: static metadata (do NOT treat as arrays in PyTree)
    S: int = struct.field(pytree_node=False)
    A: int = struct.field(pytree_node=False)


@struct.dataclass
class FiniteMDPState:
    s: jax.Array      # scalar int32


@struct.dataclass
class TabularModel:
    P: jax.Array
    R: jax.Array
    mu0: jax.Array


# =============================================================================
# Pure functions: make_params / reset / step
# =============================================================================
def make_finite_mdp_params(
    key: jax.Array,
    S: int,
    A: int,
    epsilon_p: float = 0.0,
    epsilon_r: float = 0.0,
    common_transition: Optional[jax.Array] = None,  # (S,A,S)
    common_reward: Optional[jax.Array] = None,      # (S,A)
    mu0: Optional[jax.Array] = None,                # (S,)
) -> Tuple[FiniteMDPParams, TabularModel]:
    keyP, keyR, keyMu = jax.random.split(key, 3)

    local_P = random_simplex(keyP, shape=(S, A), d=S).astype(jnp.float32)

    if common_transition is not None:
        P = (1.0 - epsilon_p) * common_transition + epsilon_p * local_P
        P = P / jnp.sum(P, axis=-1, keepdims=True)
    else:
        P = local_P

    local_R = jax.random.uniform(keyR, shape=(S, A), minval=0.0, maxval=1.0).astype(jnp.float32)

    if common_reward is not None:
        R = (1.0 - epsilon_r) * common_reward + epsilon_r * local_R
    else:
        R = local_R

    if mu0 is None:
        mu0 = jnp.ones((S,), dtype=jnp.float32) / S
    else:
        mu0 = mu0.astype(jnp.float32)
        mu0 = mu0 / jnp.sum(mu0)

    params = FiniteMDPParams(P=P, R=R, mu0=mu0, S=int(S), A=int(A))
    model = TabularModel(P=P, R=R, mu0=mu0)
    return params, model


def finite_mdp_reset(key: jax.Array, params: FiniteMDPParams) -> Tuple[jax.Array, FiniteMDPState]:
    s0 = jax.random.choice(key, a=jnp.arange(params.S), p=params.mu0).astype(jnp.int32)
    return s0, FiniteMDPState(s=s0)


def finite_mdp_step(
    key: jax.Array,
    state: FiniteMDPState,
    action: jax.Array,
    params: FiniteMDPParams,
) -> Tuple[jax.Array, FiniteMDPState, jax.Array, jax.Array, Any]:
    s = state.s.astype(jnp.int32)
    a = action.astype(jnp.int32)

    r = params.R[s, a]
    p = params.P[s, a]  # (S,)

    s_next = jax.random.choice(key, a=jnp.arange(params.S), p=p).astype(jnp.int32)
    next_state = FiniteMDPState(s=s_next)

    done = jnp.array(False)
    info = {}
    return s_next, next_state, r, done, info


def make_finite_mdp_env() -> EnvAPI:
    """Returns an EnvAPI(reset, step) compatible with your training script."""
    return EnvAPI(reset=finite_mdp_reset, step=finite_mdp_step)


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    S, A = 5, 3
    key = jax.random.PRNGKey(0)

    params, model = make_finite_mdp_params(key, S=S, A=A, epsilon_p=0.2, epsilon_r=0.1)
    env = make_finite_mdp_env()

    key, k1, k2 = jax.random.split(key, 3)
    obs, st = env.reset(k1, params)
    a = jnp.array(1, dtype=jnp.int32)
    obs2, st2, r, done, info = env.step(k2, st, a, params)
    print("obs:", obs, "a:", a, "obs2:", obs2, "r:", r)
