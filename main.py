# er_actor_critic_softmax_jax_cli_fast.py
# ------------------------------------------------------------
# Fast(er) tabular entropy-regularized Actor-Critic (softmax) in JAX.
#
# Key speed fixes vs your version:
#  1) NO "always 2048-step" rollout: rollout now runs exactly T steps (while_loop).
#  2) NO repeated softmax inside vmaps: compute pi/logpi ONCE per outer iteration.
#  3) For the built-in random_mdp (tabular P,R,mu0), we can sample from the
#     exact discounted state distribution d_pi^gamma (no rollout at all).
#  4) Optionally compute J_exact only every J_every iterations (default 10).
#
# Usage examples:
#   python er_actor_critic_softmax_jax_cli_fast.py --debug_print --print_every 50
#   python er_actor_critic_softmax_jax_cli_fast.py --verbose
#
# ------------------------------------------------------------

from __future__ import annotations

import os
import argparse
import pickle
from dataclasses import dataclass, asdict
from typing import Any, Callable, NamedTuple, Tuple, Optional, Dict

import jax
import jax.numpy as jnp
from flax import struct


# =============================================================================
# Utilities: IO
# =============================================================================
def create_folder_if_not_exists(folder_path: str) -> None:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def save_results_to_pickle(file_path: str, data: Any) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


# =============================================================================
# Minimal gymnax-style Env API (pure functions)
# =============================================================================
class EnvAPI(NamedTuple):
    reset: Callable[[jax.Array, Any], Tuple[jax.Array, Any]]
    step: Callable[[jax.Array, Any, jax.Array, Any], Tuple[jax.Array, Any, jax.Array, jax.Array, Any]]


# =============================================================================
# Config + Train state + Tabular model
# =============================================================================
@dataclass(frozen=True)
class ERACConfig:
    S: int
    A: int
    gamma: float
    lam: float
    eta_c: float
    eta_a: float
    H: int
    K: int
    Bc: int = 1
    Ba: int = 1
    max_geometric_steps: int = 2048
    eps_log: float = 1e-20

    # Used inside JAX computation. Do NOT pickle cfg directly.
    obs_to_state: Callable[[jax.Array], jax.Array] = lambda obs: obs

    # Debug printing inside jit/scan
    debug_print: bool = False
    print_every: int = 1

    # Exact objective frequency (0 => never, 1 => every iter, 10 => every 10 iters, ...)
    J_every: int = 10


def config_to_pickle_safe_dict(cfg: ERACConfig) -> Dict[str, Any]:
    d = asdict(cfg)
    d.pop("obs_to_state", None)
    return d


class ERACTrainState(NamedTuple):
    theta: jax.Array  # (S,A)
    q_hat: jax.Array  # (S,A)


@struct.dataclass
class TabularModel:
    P: jax.Array    # (S,A,S)
    R: jax.Array    # (S,A)
    mu0: jax.Array  # (S,)


# =============================================================================
# Softmax policy + helpers
# =============================================================================
def softmax_policy(theta: jax.Array) -> jax.Array:
    theta = theta - jnp.max(theta, axis=1, keepdims=True)
    exp_logits = jnp.exp(theta)
    return exp_logits / jnp.sum(exp_logits, axis=1, keepdims=True)


def sample_action_from_pi(key: jax.Array, pi_row: jax.Array) -> jax.Array:
    # pi_row shape (A,)
    return jax.random.categorical(key, jnp.log(pi_row))


# =============================================================================
# Discounted occupancy sampling: geometric-time (general env)
# =============================================================================
def sample_geometric_time(key: jax.Array, gamma: float, max_steps: int) -> jax.Array:
    # Geometric on {0,1,2,...} with P(T=t)=(1-gamma)*gamma^t
    # Using inverse CDF; clip to max_steps-1.
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
    Roll out from reset for exactly T steps using while_loop (fast).
    If env returns done=True, we reset to keep the chain going.
    Returns (s_T, env_state_T, key_out).
    """
    k_reset, k_loop = jax.random.split(key, 2)
    obs0, env_state0 = env.reset(k_reset, env_params)
    s0 = cfg.obs_to_state(obs0).astype(jnp.int32)

    pi = softmax_policy(theta)

    def cond(carry):
        i, _key_i, _env_state, _s = carry
        return i < T

    def body(carry):
        i, key_i, env_state, s = carry

        key_i, k_act, k_step, k_reset2 = jax.random.split(key_i, 4)
        a = sample_action_from_pi(k_act, pi[s])

        obs_next, env_state_next, r, done, info = env.step(k_step, env_state, a, env_params)
        s_next = cfg.obs_to_state(obs_next).astype(jnp.int32)

        # reset-on-done
        obs_r, env_state_r = env.reset(k_reset2, env_params)
        s_r = cfg.obs_to_state(obs_r).astype(jnp.int32)

        s_next2 = jnp.where(done, s_r, s_next)
        env_state_next2 = jax.tree_util.tree_map(
            lambda x, y: jnp.where(done, y, x), env_state_next, env_state_r
        )

        return (i + 1, key_i, env_state_next2, s_next2)

    init = (jnp.int32(0), k_loop, env_state0, s0)
    _, key_out, env_state_T, s_T = jax.lax.while_loop(cond, body, init)
    return s_T, env_state_T, key_out


def sample_nu_a_general(
    key: jax.Array,
    env: EnvAPI,
    env_params: Any,
    cfg: ERACConfig,
    theta: jax.Array,
    pi: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    kT, kroll, kact = jax.random.split(key, 3)
    T = sample_geometric_time(kT, cfg.gamma, cfg.max_geometric_steps)
    s, _env_state, key_mid = rollout_to_time_T(kroll, env, env_params, cfg, theta, T)
    a = sample_action_from_pi(kact, pi[s])
    return s, a, key_mid


def sample_nu_c_general(
    key: jax.Array,
    env: EnvAPI,
    env_params: Any,
    cfg: ERACConfig,
    theta: jax.Array,
    pi: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    kT, kroll, kact, kstep, kact2, kreset = jax.random.split(key, 6)
    T = sample_geometric_time(kT, cfg.gamma, cfg.max_geometric_steps)
    s, env_state, key_mid = rollout_to_time_T(kroll, env, env_params, cfg, theta, T)

    a = sample_action_from_pi(kact, pi[s])

    obs2, env_state2, r, done, info = env.step(kstep, env_state, a, env_params)
    s2 = cfg.obs_to_state(obs2).astype(jnp.int32)

    # reset-on-done
    obs_r, env_state_r = env.reset(kreset, env_params)
    s_r = cfg.obs_to_state(obs_r).astype(jnp.int32)

    s2 = jnp.where(done, s_r, s2)
    env_state2 = jax.tree_util.tree_map(
        lambda x, y: jnp.where(done, y, x), env_state2, env_state_r
    )

    a2 = sample_action_from_pi(kact2, pi[s2])
    return s, a, s2, a2, r, key_mid


# =============================================================================
# Exact discounted state distribution for tabular model
# =============================================================================
def discounted_state_dist(theta: jax.Array, model: TabularModel, cfg: ERACConfig) -> jax.Array:
    """
    d = (1-gamma) * sum_{t>=0} gamma^t mu0^T P_pi^t
      = (1-gamma) * (I - gamma P_pi)^{-T} mu0
    Returned as a probability vector (S,).
    """
    pi = softmax_policy(theta)
    P_pi = jnp.einsum("sa,san->sn", pi, model.P)  # (S,S)
    I = jnp.eye(cfg.S, dtype=theta.dtype)
    d = (1.0 - cfg.gamma) * jnp.linalg.solve((I - cfg.gamma * P_pi).T, model.mu0)
    d = jnp.clip(d, 0.0)
    d = d / (jnp.sum(d) + 1e-12)
    return d


def sample_nu_a_tabular(
    key: jax.Array,
    cfg: ERACConfig,
    model: TabularModel,
    pi: jax.Array,
    d: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    # s ~ d, a ~ pi(.|s)
    k_s, k_a = jax.random.split(key, 2)
    s = jax.random.choice(k_s, a=jnp.arange(cfg.S), p=d).astype(jnp.int32)
    a = sample_action_from_pi(k_a, pi[s]).astype(jnp.int32)
    return s, a


def sample_nu_c_tabular(
    key: jax.Array,
    cfg: ERACConfig,
    model: TabularModel,
    pi: jax.Array,
    d: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    # s ~ d, a ~ pi(.|s), s2 ~ P[s,a], a2 ~ pi(.|s2), r = R[s,a]
    k_s, k_a, k_s2, k_a2 = jax.random.split(key, 4)
    s = jax.random.choice(k_s, a=jnp.arange(cfg.S), p=d).astype(jnp.int32)
    a = sample_action_from_pi(k_a, pi[s]).astype(jnp.int32)
    s2 = jax.random.choice(k_s2, a=jnp.arange(cfg.S), p=model.P[s, a]).astype(jnp.int32)
    a2 = sample_action_from_pi(k_a2, pi[s2]).astype(jnp.int32)
    r = model.R[s, a].astype(jnp.float32)
    return s, a, s2, a2, r


# =============================================================================
# Exact objective (tabular model)
# =============================================================================
def exact_objective_tabular(theta: jax.Array, model: TabularModel, cfg: ERACConfig) -> jax.Array:
    P, R, mu0 = model.P, model.R, model.mu0
    S = cfg.S

    pi = softmax_policy(theta)
    logpi = jnp.log(pi + cfg.eps_log)

    P_pi = jnp.einsum("sa,san->sn", pi, P)  # (S,S)
    r_tilde_pi = jnp.sum(pi * (R - cfg.lam * logpi), axis=1)  # (S,)

    I = jnp.eye(S, dtype=theta.dtype)
    V = jnp.linalg.solve(I - cfg.gamma * P_pi, r_tilde_pi)
    return jnp.dot(mu0, V)


# =============================================================================
# Algorithm: one iteration (fast)
# =============================================================================
def one_iteration(
    key: jax.Array,
    env: EnvAPI,
    env_params: Any,
    cfg: ERACConfig,
    st: ERACTrainState,
    model: Optional[TabularModel] = None,
) -> Tuple[ERACTrainState, Dict[str, jax.Array]]:
    theta, q_prev = st.theta, st.q_hat

    # Compute policy ONCE
    pi = softmax_policy(theta)
    logpi = jnp.log(pi + cfg.eps_log)

    use_tabular = model is not None
    d = discounted_state_dist(theta, model, cfg) if use_tabular else None

    def critic_body(_h, carry):
        key_h, q = carry
        key_h, k_batch = jax.random.split(key_h, 2)
        keys = jax.random.split(k_batch, cfg.Bc)

        def one_sample(k):
            if use_tabular:
                s, a, s2, a2, r = sample_nu_c_tabular(k, cfg, model, pi, d)
            else:
                s, a, s2, a2, r, _ = sample_nu_c_general(k, env, env_params, cfg, theta, pi)

            r_tilde = r - cfg.lam * logpi[s, a]
            td = r_tilde + cfg.gamma * q[s2, a2] - q[s, a]
            return s, a, td

        s_b, a_b, td_b = jax.vmap(one_sample)(keys)

        updates = jnp.zeros((cfg.S, cfg.A), dtype=q.dtype)
        updates = updates.at[s_b, a_b].add(td_b / cfg.Bc)

        q = q + cfg.eta_c * updates
        return (key_h, q)

    key, k_critic = jax.random.split(key, 2)
    key_out, q_hat = jax.lax.fori_loop(0, cfg.H, critic_body, (k_critic, q_prev))

    # Advantage estimate (regularized)
    v_hat = jnp.sum(pi * (q_hat - cfg.lam * logpi), axis=1)          # (S,)
    a_hat = q_hat - cfg.lam * logpi - v_hat[:, None]                # (S,A)

    # Actor batch
    key_out, k_actor = jax.random.split(key_out, 2)
    keys_a = jax.random.split(k_actor, cfg.Ba)

    def one_actor_sample(k):
        if use_tabular:
            sY, aY = sample_nu_a_tabular(k, cfg, model, pi, d)
        else:
            sY, aY, _ = sample_nu_a_general(k, env, env_params, cfg, theta, pi)
        g = a_hat[sY, aY] / (1.0 - cfg.gamma)
        return sY, aY, g

    sY_b, aY_b, g_b = jax.vmap(one_actor_sample)(keys_a)

    theta_updates = jnp.zeros((cfg.S, cfg.A), dtype=theta.dtype)
    theta_updates = theta_updates.at[sY_b, aY_b].add(g_b / cfg.Ba)

    theta_next = theta + cfg.eta_a * theta_updates
    st_next = ERACTrainState(theta=theta_next, q_hat=q_hat)

    metrics = {
        "adv_mean_batch": jnp.mean(a_hat[sY_b, aY_b]),
        "g_mean_batch": jnp.mean(g_b),
        "q_mean_batch": jnp.mean(q_hat[sY_b, aY_b]),
        "v_mean_batch": jnp.mean(v_hat[sY_b]),
        "theta_norm": jnp.linalg.norm(theta_next),
        "theta_update_norm": jnp.linalg.norm(cfg.eta_a * theta_updates),
    }
    return st_next, metrics


def train_er_actor_critic(
    key: jax.Array,
    env: EnvAPI,
    env_params: Any,
    cfg: ERACConfig,
    model: Optional[TabularModel] = None,
    theta0: Optional[jax.Array] = None,
    q0: Optional[jax.Array] = None,
) -> Tuple[ERACTrainState, Dict[str, jax.Array]]:
    if theta0 is None:
        theta0 = jnp.zeros((cfg.S, cfg.A), dtype=jnp.float32)
    if q0 is None:
        q0 = jnp.zeros((cfg.S, cfg.A), dtype=jnp.float32)

    init_state = ERACTrainState(theta=theta0, q_hat=q0)

    def scan_step(carry, k_idx):
        key_i, st = carry
        key_i, k = jax.random.split(key_i, 2)

        st_next, metrics = one_iteration(k, env, env_params, cfg, st, model=model)

        # Compute J_exact only every cfg.J_every iters (if model is provided)
        if model is not None and cfg.J_every > 0:
            do_J = (k_idx % cfg.J_every) == 0

            def _do(_):
                return exact_objective_tabular(st_next.theta, model, cfg)

            def _no(_):
                return jnp.nan

            J = jax.lax.cond(do_J, _do, _no, operand=0)
        else:
            J = jnp.nan

        metrics = dict(metrics)
        metrics["J_exact"] = J

        if cfg.debug_print:
            pred = (k_idx % cfg.print_every) == 0

            def _do_print(_):
                jax.debug.print(
                    "[k={k}] J_exact={J} | theta_norm={tn:.4f} | dtheta_norm={dtn:.4f} | adv_mean={adv:.4f}",
                    k=k_idx,
                    J=J,
                    tn=metrics["theta_norm"],
                    dtn=metrics["theta_update_norm"],
                    adv=metrics["adv_mean_batch"],
                )
                return 0

            _ = jax.lax.cond(pred, _do_print, lambda _: 0, operand=0)

        return (key_i, st_next), metrics

    ks = jnp.arange(cfg.K, dtype=jnp.int32)
    (key_final, st_final), hist = jax.lax.scan(scan_step, (key, init_state), xs=ks)
    return st_final, hist


# =============================================================================
# Environments (pure JAX, with exact tabular model)
# =============================================================================
@struct.dataclass
class RandomMDPParams:
    P: jax.Array
    R: jax.Array
    mu0: jax.Array


def make_random_mdp(S: int, A: int, seed: int = 0) -> Tuple[EnvAPI, RandomMDPParams, TabularModel]:
    key = jax.random.PRNGKey(seed)
    keyP, keyR = jax.random.split(key, 2)

    logits = jax.random.normal(keyP, (S, A, S))
    P = jax.nn.softmax(logits, axis=-1).astype(jnp.float32)

    R = jax.random.uniform(keyR, (S, A), minval=0.0, maxval=1.0).astype(jnp.float32)

    mu0 = (jnp.ones((S,), dtype=jnp.float32) / S)

    params = RandomMDPParams(P=P, R=R, mu0=mu0)
    model = TabularModel(P=P, R=R, mu0=mu0)

    def reset(key: jax.Array, p: RandomMDPParams):
        s0 = jax.random.choice(key, a=jnp.arange(S), p=p.mu0)
        return s0.astype(jnp.int32), s0.astype(jnp.int32)

    def step(key: jax.Array, env_state: jax.Array, action: jax.Array, p: RandomMDPParams):
        s = env_state.astype(jnp.int32)
        a = action.astype(jnp.int32)
        probs = p.P[s, a]
        s2 = jax.random.choice(key, a=jnp.arange(S), p=probs).astype(jnp.int32)
        r = p.R[s, a].astype(jnp.float32)
        done = jnp.array(False)
        info = {}
        return s2, s2, r, done, info

    env = EnvAPI(reset=reset, step=step)
    return env, params, model


# =============================================================================
# CLI
# =============================================================================
def build_env_from_args(args) -> Tuple[EnvAPI, Any, TabularModel, int, int]:
    if args.env == "random_mdp":
        env, env_params, model = make_random_mdp(S=args.S, A=args.A, seed=args.env_seed)
        return env, env_params, model, args.S, args.A
    raise ValueError(f"Unknown env: {args.env}")


def main():
    parser = argparse.ArgumentParser("Fast Entropy-regularized Actor-Critic (softmax) - JAX (batched)")

    parser.add_argument("--env", type=str, default="random_mdp", choices=["random_mdp"])
    parser.add_argument("--env_seed", type=int, default=0)

    parser.add_argument("--S", type=int, default=10)
    parser.add_argument("--A", type=int, default=4)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.05)
    parser.add_argument("--eta_c", type=float, default=0.1)
    parser.add_argument("--eta_a", type=float, default=0.05)
    parser.add_argument("--H", type=int, default=50)
    parser.add_argument("--K", type=int, default=2000)

    parser.add_argument("--Bc", type=int, default=16)
    parser.add_argument("--Ba", type=int, default=6)

    parser.add_argument("--max_geometric_steps", type=int, default=2048)
    parser.add_argument("--eps_log", type=float, default=1e-20)

    parser.add_argument("--J_every", type=int, default=10,
                        help="Compute exact objective every J_every iterations (0 disables).")

    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--seed_base", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="./experiments/ERAC")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--debug_print", action="store_true",
                        help="Print after actor step (jit-safe).")
    parser.add_argument("--print_every", type=int, default=50,
                        help="Print every N actor steps (k).")

    args = parser.parse_args()

    env, env_params, model, S, A = build_env_from_args(args)

    cfg = ERACConfig(
        S=S,
        A=A,
        gamma=args.gamma,
        lam=args.lam,
        eta_c=args.eta_c,
        eta_a=args.eta_a,
        H=args.H,
        K=args.K,
        Bc=args.Bc,
        Ba=args.Ba,
        max_geometric_steps=args.max_geometric_steps,
        eps_log=args.eps_log,
        obs_to_state=lambda obs: obs,
        debug_print=args.debug_print,
        print_every=args.print_every,
        J_every=args.J_every,
    )

    tag = (
        f"{args.env}_S{S}_A{A}_K{cfg.K}_H{cfg.H}"
        f"_Bc{cfg.Bc}_Ba{cfg.Ba}"
        f"_etaC{cfg.eta_c}_etaA{cfg.eta_a}_lam{cfg.lam}_g{cfg.gamma}"
        f"_Jevery{cfg.J_every}"
    )
    outdir = os.path.join(args.outdir, tag)
    create_folder_if_not_exists(outdir)

    # JIT compile training.
    # env and cfg are static so JAX can trace the function once.
    train_jit = jax.jit(train_er_actor_critic, static_argnames=("env", "cfg"))

    all_runs = []
    for run in range(args.runs):
        seed = args.seed_base + run
        key = jax.random.PRNGKey(seed)

        st_final, hist = train_jit(key, env, env_params, cfg, model)

        # Force device sync so debug prints flush + results are ready
        _ = jax.block_until_ready(hist["theta_norm"])

        hist_np = {k: jax.device_get(v) for k, v in hist.items()}
        theta_np = jax.device_get(st_final.theta)
        qhat_np = jax.device_get(st_final.q_hat)

        run_data = {
            "seed": seed,
            "config": config_to_pickle_safe_dict(cfg),
            "history": hist_np,
            "theta_final": theta_np,
            "qhat_final": qhat_np,
        }
        all_runs.append(run_data)

        if args.verbose:
            # J_exact may be nan for most entries if J_every>1
            J_vals = hist_np["J_exact"]
            last_finite = jnp.nan
            # find last non-nan
            for v in reversed(J_vals.tolist()):
                if v == v:  # not nan
                    last_finite = v
                    break
            print(f"[run {run+1}/{args.runs}] seed={seed}  last_logged_J_exact={last_finite}")

        save_results_to_pickle(os.path.join(outdir, f"run_{run}_results.pkl"), run_data)

    save_results_to_pickle(os.path.join(outdir, "all_runs.pkl"), all_runs)

    if args.verbose:
        print(f"Saved results in: {outdir}")


if __name__ == "__main__":
    main()
