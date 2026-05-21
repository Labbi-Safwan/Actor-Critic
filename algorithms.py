# er_actor_critic_softmax_jax.py
# Single-agent, tabular entropy-regularized Actor-Critic under softmax parametrization.
# Structure matches Algorithm 1 in your screenshot:
#   for k=0..K-1:
#     Critic: H inner TD steps on q_hat using samples X~ν_c(θ_k)
#     Compute v_hat, a_hat from q_hat and π_θk
#     Actor: one step on θ using sample Y~ν_a(θ_k)
# No projection operator is used.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Tuple, Optional

import jax
import jax.numpy as jnp


# =============================================================================
# 0) Minimal gymnax-style Env API (pure functions)
# =============================================================================
# reset(key, params) -> obs, env_state
# step(key, env_state, action, params) -> obs_next, env_state_next, reward, done, info

class EnvAPI(NamedTuple):
    reset: Callable[[jax.Array, Any], Tuple[jax.Array, Any]]
    step: Callable[[jax.Array, Any, jax.Array, Any], Tuple[jax.Array, Any, jax.Array, jax.Array, Any]]


# =============================================================================
# 1) Config + Train state + Model (for exact objective)
# =============================================================================
@dataclass(frozen=True)
class ERACConfig:
    S: int
    A: int
    gamma: float
    lam: float                # entropy temperature λ
    eta_c: float              # critic step size η_c
    eta_a: float              # actor step size η_a
    H: int                    # critic inner steps
    K: int                    # outer iterations
    max_geometric_steps: int = 2048
    eps_log: float = 1e-20
    obs_to_state: Callable[[jax.Array], jax.Array] = lambda obs: obs  # obs must map to int state id


class ERACTrainState(NamedTuple):
    theta: jax.Array          # (S, A) logits
    q_hat: jax.Array          # (S, A) critic estimate


@dataclass(frozen=True)
class TabularModel:
    P: jax.Array              # (S, A, S)
    R: jax.Array              # (S, A)
    mu0: jax.Array            # (S,)


# =============================================================================
# 2) Softmax policy + sampling helpers
# =============================================================================
def softmax_policy(theta: jax.Array) -> jax.Array:
    # theta: (S,A) -> pi: (S,A)
    theta = theta - jnp.max(theta, axis=1, keepdims=True)
    exp_logits = jnp.exp(theta)
    return exp_logits / jnp.sum(exp_logits, axis=1, keepdims=True)


def sample_action(key: jax.Array, pi: jax.Array, s: jax.Array) -> jax.Array:
    # categorical over actions at state s
    return jax.random.categorical(key, jnp.log(pi[s]))


# =============================================================================
# 3) Discounted occupancy sampling (ν_a and ν_c) via geometric-time rollout
# =============================================================================
def sample_geometric_time(key: jax.Array, gamma: float, max_steps: int) -> jax.Array:
    # T ~ Geom(stop prob=1-gamma) on {0,1,2,...}, implemented via inversion:
    # T = floor(log(U)/log(gamma))
    u = jax.random.uniform(key, (), minval=1e-12, maxval=1.0)
    T = jnp.floor(jnp.log(u) / jnp.log(gamma)).astype(jnp.int32)
    return jnp.minimum(T, jnp.int32(max_steps - 1))


def rollout_to_time_T(
    key: jax.Array,
    env: EnvAPI,
    env_params: Any,
    cfg: ERACConfig,
    theta: jax.Array,
    T: jax.Array,
) -> Tuple[jax.Array, Any, jax.Array]:
    """
    Roll out from reset for T steps (with resets upon done to continue sampling).
    Returns (s_T, env_state_T, key_out).
    """
    k_reset, k_loop = jax.random.split(key, 2)
    obs0, env_state0 = env.reset(k_reset, env_params)
    s0 = cfg.obs_to_state(obs0).astype(jnp.int32)

    pi = softmax_policy(theta)

    def body(i, carry):
        key_i, env_state, s = carry
        do = i < T

        key_i, k_act, k_step, k_reset2 = jax.random.split(key_i, 4)
        a = sample_action(k_act, pi, s)

        obs_next, env_state_next, r, done, info = env.step(k_step, env_state, a, env_params)
        s_next = cfg.obs_to_state(obs_next).astype(jnp.int32)

        # If done, reset (so occupancy sampling is well-defined)
        obs_r, env_state_r = env.reset(k_reset2, env_params)
        s_r = cfg.obs_to_state(obs_r).astype(jnp.int32)

        s_next2 = jnp.where(done, s_r, s_next)
        env_state_next2 = jax.tree_util.tree_map(
            lambda x, y: jnp.where(done, y, x), env_state_next, env_state_r
        )

        # Only apply transition if i < T
        s_out = jnp.where(do, s_next2, s)
        env_state_out = jax.tree_util.tree_map(
            lambda new, old: jnp.where(do, new, old), env_state_next2, env_state
        )
        return (key_i, env_state_out, s_out)

    key_out, env_state_T, s_T = jax.lax.fori_loop(
        0, cfg.max_geometric_steps, body, (k_loop, env_state0, s0)
    )
    return s_T, env_state_T, key_out


def sample_nu_a(
    key: jax.Array,
    env: EnvAPI,
    env_params: Any,
    cfg: ERACConfig,
    theta: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Sample Y=(S,A) ~ ν_a(θ): pick geometric time T, roll out to s_T, sample a_T ~ π(·|s_T).
    """
    kT, kroll, kact = jax.random.split(key, 3)
    T = sample_geometric_time(kT, cfg.gamma, cfg.max_geometric_steps)
    s, _env_state, key_mid = rollout_to_time_T(kroll, env, env_params, cfg, theta, T)
    pi = softmax_policy(theta)
    a = sample_action(kact, pi, s)
    return s, a, key_mid


def sample_nu_c(
    key: jax.Array,
    env: EnvAPI,
    env_params: Any,
    cfg: ERACConfig,
    theta: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Sample X=(S,A,S',A') ~ ν_c(θ):
      - sample (s,a) at geometric time,
      - take one env step to get s' and reward,
      - sample a' ~ π(·|s').
    Returns (s, a, s2, a2, r, key_out).
    """
    kT, kroll, kact, kstep, kact2, kreset = jax.random.split(key, 6)
    T = sample_geometric_time(kT, cfg.gamma, cfg.max_geometric_steps)
    s, env_state, key_mid = rollout_to_time_T(kroll, env, env_params, cfg, theta, T)

    pi = softmax_policy(theta)
    a = sample_action(kact, pi, s)

    obs2, env_state2, r, done, info = env.step(kstep, env_state, a, env_params)
    s2 = cfg.obs_to_state(obs2).astype(jnp.int32)

    # If done, reset to define (s2,a2)
    obs_r, env_state_r = env.reset(kreset, env_params)
    s_r = cfg.obs_to_state(obs_r).astype(jnp.int32)

    s2 = jnp.where(done, s_r, s2)
    env_state2 = jax.tree_util.tree_map(
        lambda x, y: jnp.where(done, y, x), env_state2, env_state_r
    )

    a2 = sample_action(kact2, pi, s2)
    return s, a, s2, a2, r, key_mid


# =============================================================================
# 4) Exact objective (model-based): J_lambda(pi_theta) = mu0^T V
# =============================================================================
def exact_objective_tabular(theta: jax.Array, model: TabularModel, cfg: ERACConfig) -> jax.Array:
    """
    Exact discounted entropy-regularized objective under π_θ:
      r̃_π(s) = Σ_a π(a|s) [ R(s,a) - λ log π(a|s) ]
      P_π(s,s') = Σ_a π(a|s) P(s,a,s')
      V = (I - γ P_π)^{-1} r̃_π
      J = μ0^T V
    """
    P, R, mu0 = model.P, model.R, model.mu0
    S = cfg.S

    pi = softmax_policy(theta)
    logpi = jnp.log(pi + cfg.eps_log)

    P_pi = jnp.einsum("sa,san->sn", pi, P)                   # (S,S)
    r_tilde_pi = jnp.sum(pi * (R - cfg.lam * logpi), axis=1)  # (S,)

    I = jnp.eye(S, dtype=theta.dtype)
    V = jnp.linalg.solve(I - cfg.gamma * P_pi, r_tilde_pi)    # (S,)
    return jnp.dot(mu0, V)


# =============================================================================
# 5) One iteration of Algorithm 1
# =============================================================================
def one_iteration(
    key: jax.Array,
    env: EnvAPI,
    env_params: Any,
    cfg: ERACConfig,
    st: ERACTrainState,
) -> Tuple[ERACTrainState, dict]:
    """
    Algorithm 1 (no projection):
      Critic:
        q^0_k = q_{k-1}
        for h=0..H-1:
          sample X=(S^h,A^h,S^{h+1},A^{h+1}) ~ ν_c(θ_k)
          TD update on q
      then compute v_hat, a_hat
      Actor:
        sample Y=(S,A) ~ ν_a(θ_k)
        θ_{k+1} = θ_k + η_a * g_a(Y)
    """
    theta, q_prev = st.theta, st.q_hat

    # ---- Critic inner loop
    def critic_body(h, carry):
        key_h, q = carry
        key_h, k_samp = jax.random.split(key_h, 2)

        s, a, s2, a2, r, key_h2 = sample_nu_c(k_samp, env, env_params, cfg, theta)

        pi = softmax_policy(theta)
        logpi_sa = jnp.log(pi[s, a] + cfg.eps_log)

        r_tilde = r - cfg.lam * logpi_sa
        td = r_tilde + cfg.gamma * q[s2, a2] - q[s, a]
        q = q.at[s, a].add(cfg.eta_c * td)
        return (key_h2, q)

    key, k_critic = jax.random.split(key, 2)
    key_out, q_hat = jax.lax.fori_loop(0, cfg.H, critic_body, (k_critic, q_prev))

    # ---- Compute v_hat and a_hat (tabular)
    pi = softmax_policy(theta)
    logpi = jnp.log(pi + cfg.eps_log)

    v_hat = jnp.sum(pi * (q_hat - cfg.lam * logpi), axis=1)       # (S,)
    a_hat = q_hat - cfg.lam * logpi - v_hat[:, None]              # (S,A)

    # ---- Actor step
    key_out, k_actor = jax.random.split(key_out, 2)
    sY, aY, key_out2 = sample_nu_a(k_actor, env, env_params, cfg, theta)

    g_sa = a_hat[sY, aY] / (1.0 - cfg.gamma)                      # scalar
    theta_next = theta.at[sY, aY].add(cfg.eta_a * g_sa)

    st_next = ERACTrainState(theta=theta_next, q_hat=q_hat)
    metrics = {
        "s_actor": sY,
        "a_actor": aY,
        "adv_sa": a_hat[sY, aY],
        "q_sa": q_hat[sY, aY],
        "v_s": v_hat[sY],
    }
    return st_next, metrics


# =============================================================================
# 6) Training loop with exact objective recording
# =============================================================================
def train_er_actor_critic(
    key: jax.Array,
    env: EnvAPI,
    env_params: Any,
    cfg: ERACConfig,
    model: Optional[TabularModel] = None,   # if provided, record exact objective
    theta0: Optional[jax.Array] = None,
    q0: Optional[jax.Array] = None,
) -> Tuple[ERACTrainState, dict]:
    """
    Runs K iterations and returns:
      - final train state
      - history dict with fields:
          metrics_hist: dict of arrays length K (s_actor, a_actor, adv_sa, ...)
          J_exact: array length K (nan if model=None)
    """
    if theta0 is None:
        theta0 = jnp.zeros((cfg.S, cfg.A), dtype=jnp.float32)
    if q0 is None:
        q0 = jnp.zeros((cfg.S, cfg.A), dtype=jnp.float32)

    init_state = ERACTrainState(theta=theta0, q_hat=q0)

    def scan_step(carry, _):
        key_i, st = carry
        key_i, k = jax.random.split(key_i, 2)

        st_next, metrics = one_iteration(k, env, env_params, cfg, st)

        # exact objective (if model is provided)
        J = jnp.nan
        if model is not None:
            J = exact_objective_tabular(st_next.theta, model, cfg)

        metrics = dict(metrics)
        metrics["J_exact"] = J
        return (key_i, st_next), metrics

    (key_final, st_final), metrics_hist = jax.lax.scan(
        scan_step, (key, init_state), xs=None, length=cfg.K
    )
    return st_final, metrics_hist


# =============================================================================
# 7) Example (template)
# =============================================================================
if __name__ == "__main__":
    # You need to plug your environment here (gymnax-style).
    #
    # Example:
    #   import gymnax
    #   env_obj, env_params = gymnax.make("DeepSea-v0")   # (example name)
    #   env = EnvAPI(reset=env_obj.reset, step=env_obj.step)
    #
    # For exact objective, you must provide model arrays P,R,mu0:
    #   model = TabularModel(P=P, R=R, mu0=mu0)

    # ---- Dummy placeholders (replace with your real env)
    raise SystemExit(
        "Template file. Plug in your env (EnvAPI) + env_params. "
        "Optionally provide TabularModel(P,R,mu0) to record exact objective."
    )
