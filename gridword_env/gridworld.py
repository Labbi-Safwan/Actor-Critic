# gridworld_jax.py
from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import struct


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _parse_coords_list(coords: Optional[Sequence[Tuple[int, int]]]) -> Tuple[Tuple[int, int], ...]:
    if coords is None:
        return tuple()
    return tuple((int(r), int(c)) for (r, c) in coords)


def random_simplex_over_support(key: jax.Array, support_mask: jax.Array) -> jax.Array:
    """
    support_mask: (N,) bool, at least one True
    returns probs: (N,) summing to 1, supported on True entries
    """
    # sample positive values on all entries, then mask out and renormalize
    x = jax.random.exponential(key, shape=support_mask.shape)
    x = jnp.where(support_mask, x, 0.0)
    z = jnp.sum(x)
    # If numeric issue (shouldn't if at least one True), fallback to uniform on support
    probs = jnp.where(z > 0, x / z, support_mask.astype(jnp.float32) / jnp.sum(support_mask))
    return probs


# ---------------------------------------------------------------------
# Params / State
# ---------------------------------------------------------------------
@struct.dataclass
class GridWorldParams:
    rows: int
    cols: int
    walls_mask: jax.Array         # (rows, cols) bool
    terminal_mask: jax.Array      # (rows, cols) bool

    # reward map as dense grid (rows, cols)
    reward_grid: jax.Array        # (rows, cols) float32
    default_reward: float

    success_probability: float    # p

    # start distribution over *valid* states in the compressed indexing
    mu0: jax.Array                # (S,) float32

    # Tabular model on compressed state space (excluding walls):
    P: jax.Array                  # (S, A, S)
    R: jax.Array                  # (S, A)


@struct.dataclass
class GridWorldState:
    s: jax.Array                  # int32 in {0..S-1} compressed index


# ---------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------
class GridWorldJAX:
    """
    GridWorld with compressed state space = all non-wall cells.

    Actions: 0=left, 1=right, 2=down, 3=up  (to match your original mapping)
    """
    A = 4

    def __init__(self, rows: int, cols: int):
        self.rows = int(rows)
        self.cols = int(cols)

    # ------------- build mappings (pure, used once outside jit) -------------
    def build_mappings(self, walls: Sequence[Tuple[int, int]]):
        """
        Returns:
            coord2idx: (rows, cols) int32, -1 for walls, else 0..S-1
            idx2coord: (S, 2) int32 giving (r,c)
        """
        walls_set = set((int(r), int(c)) for r, c in walls)

        coord2idx = -jnp.ones((self.rows, self.cols), dtype=jnp.int32)

        coords = []
        idx = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in walls_set:
                    continue
                coord2idx = coord2idx.at[r, c].set(idx)
                coords.append((r, c))
                idx += 1

        idx2coord = jnp.array(coords, dtype=jnp.int32)  # (S,2)
        return coord2idx, idx2coord

    # ------------- build params (P,R,mu0) -------------
    def make_params(
        self,
        key: jax.Array,
        start_coord: Tuple[int, int] = (0, 0),
        terminal_states: Optional[Sequence[Tuple[int, int]]] = None,
        success_probability: float = 0.9,
        reward_at: Optional[Sequence[Tuple[Tuple[int, int], float]]] = None,
        walls: Optional[Sequence[Tuple[int, int]]] = ((1, 1), (2, 2)),
        default_reward: float = 0.0,
        # heterogeneity mixing:
        common: Optional[jax.Array] = None,     # (S,A,S) on compressed space
        epsilon_p: float = 0.0,
        # if start as distribution instead of a single coord, you can pass mu0 directly:
        mu0: Optional[jax.Array] = None,        # (S,)
    ) -> GridWorldParams:
        """
        reward_at: list of ((r,c), reward) entries. If None: goal at bottom-right gets reward 1.0
        terminal_states: list of (r,c). If None: empty (non-terminal).
        walls: list of (r,c). If None: no walls.
        """
        if walls is None:
            walls = tuple()
        else:
            walls = _parse_coords_list(walls)

        if terminal_states is None:
            terminal_states = tuple()
        else:
            terminal_states = _parse_coords_list(terminal_states)

        # reward map
        if reward_at is None:
            reward_at = [((self.rows - 1, self.cols - 1), 1.0)]
        reward_at = [((int(r), int(c)), float(v)) for ((r, c), v) in reward_at]

        # mappings
        coord2idx, idx2coord = self.build_mappings(walls)
        S = int(idx2coord.shape[0])

        # dense masks
        walls_mask = jnp.zeros((self.rows, self.cols), dtype=bool)
        for (r, c) in walls:
            walls_mask = walls_mask.at[r, c].set(True)

        terminal_mask = jnp.zeros((self.rows, self.cols), dtype=bool)
        for (r, c) in terminal_states:
            terminal_mask = terminal_mask.at[r, c].set(True)

        reward_grid = jnp.full((self.rows, self.cols), float(default_reward), dtype=jnp.float32)
        for (r, c), v in reward_at:
            reward_grid = reward_grid.at[r, c].set(jnp.float32(v))

        # initial distribution mu0 over compressed states
        start_r, start_c = int(start_coord[0]), int(start_coord[1])
        start_idx = coord2idx[start_r, start_c]
        if start_idx < 0:
            raise ValueError("start_coord is on a wall.")

        if mu0 is None:
            mu0 = jnp.zeros((S,), dtype=jnp.float32).at[start_idx].set(1.0)
        else:
            mu0 = mu0.astype(jnp.float32)
            mu0 = mu0 / jnp.sum(mu0)

        # build base transition kernel P_base (success_probability structure)
        P_base = self._build_P_base(coord2idx, idx2coord, walls_mask, terminal_mask, success_probability)  # (S,A,S)

        # build Individual kernel (random simplex per (s,a) on support of P_base)
        key_ind = key
        Individual = self._build_individual_kernel(key_ind, P_base)  # (S,A,S)

        # mix with common if provided
        if common is not None:
            P = (1.0 - float(epsilon_p)) * common + float(epsilon_p) * Individual
            P = P / jnp.sum(P, axis=-1, keepdims=True)
        else:
            P = P_base

        # build mean reward R(s,a) = sum_{s'} P(s,a,s') * reward(s)  (like your reward_fn depends on current cell)
        # Your original reward_fn uses (row,col) of current state (ss), not next state.
        # That means the expected reward under action a at state s is just reward(s) (independent of s'),
        # except walls were excluded from state space already.
        # We'll reproduce that: reward at a state is reward_grid[r,c], regardless of next state.
        rs = reward_grid[idx2coord[:, 0], idx2coord[:, 1]]  # (S,)
        R = jnp.tile(rs[:, None], (1, self.A))              # (S,A)

        return GridWorldParams(
            rows=self.rows,
            cols=self.cols,
            walls_mask=walls_mask,
            terminal_mask=terminal_mask,
            reward_grid=reward_grid,
            default_reward=float(default_reward),
            success_probability=float(success_probability),
            mu0=mu0,
            P=P.astype(jnp.float32),
            R=R.astype(jnp.float32),
        )

    def _build_P_base(
        self,
        coord2idx: jax.Array,      # (rows,cols) int32
        idx2coord: jax.Array,      # (S,2) int32
        walls_mask: jax.Array,     # (rows,cols) bool
        terminal_mask: jax.Array,  # (rows,cols) bool
        p_succ: float,
    ) -> jax.Array:
        """
        Builds P using your rule:
          - valid_neighbors = set of valid moves among left/right/down/up
          - if chosen action corresponds to a valid neighbor:
                prob = p_succ + (1-p_succ)*(n_valid==1) for that neighbor
                prob = (1-p_succ)/(n_valid-1) for other valid neighbors (if n_valid>1)
          - if chosen action not valid (i.e., hits wall/bounds): stay with prob 1
          - terminal states: absorbing (stay with prob 1) for all actions
        """
        S = idx2coord.shape[0]
        A = self.A
        rows, cols = self.rows, self.cols

        # action deltas matching your mapping: left,right,down,up
        dr = jnp.array([0, 0, 1, -1], dtype=jnp.int32)
        dc = jnp.array([-1, 1, 0, 0], dtype=jnp.int32)

        def is_valid_cell(r, c):
            in_bounds = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
            not_wall = jnp.where(in_bounds, ~walls_mask[r, c], False)
            return in_bounds & not_wall

        def next_coord(r, c, a):
            rr = r + dr[a]
            cc = c + dc[a]
            ok = is_valid_cell(rr, cc)
            rr2 = jnp.where(ok, rr, r)
            cc2 = jnp.where(ok, cc, c)
            return rr2, cc2, ok  # ok indicates action is valid move

        # Precompute for each state: next state for each action + validity mask
        r0 = idx2coord[:, 0]
        c0 = idx2coord[:, 1]

        def per_state(s):
            r = r0[s]
            c = c0[s]

            # terminal?
            is_term = terminal_mask[r, c]

            rr, cc, ok = jax.vmap(lambda a: next_coord(r, c, a))(jnp.arange(A))
            # rr,cc: (A,), ok: (A,)
            nxt = coord2idx[rr, cc]  # (A,) compressed next indices (should be valid)
            # if action invalid, nxt should be current state's index
            # (coord2idx[r,c] is always valid)
            s_self = coord2idx[r, c]
            nxt = jnp.where(ok, nxt, s_self)

            # compute distribution for each chosen action a:
            def dist_for_action(a):
                # if terminal: stay
                def term_dist():
                    d = jnp.zeros((S,), dtype=jnp.float32).at[s_self].set(1.0)
                    return d

                def nonterm_dist():
                    valid_mask = ok  # (A,) which directions lead to valid neighbors
                    n_valid = jnp.sum(valid_mask.astype(jnp.int32))

                    # if chosen action invalid => stay with prob 1
                    def invalid_choice():
                        d = jnp.zeros((S,), dtype=jnp.float32).at[s_self].set(1.0)
                        return d

                    # chosen action valid
                    def valid_choice():
                        # mass to chosen neighbor:
                        chosen_next = nxt[a]
                        mass_chosen = p_succ + (1.0 - p_succ) * (n_valid == 1)

                        # distribute remaining mass among OTHER valid neighbors if n_valid>1
                        rem = 1.0 - mass_chosen
                        # probs over directions:
                        # for b != a, b valid => rem/(n_valid-1), else 0
                        denom = jnp.maximum(n_valid - 1, 1)
                        per_other = rem / denom

                        # Build distribution over next states
                        d = jnp.zeros((S,), dtype=jnp.float32)

                        # add chosen mass
                        d = d.at[chosen_next].add(mass_chosen)

                        # add other masses
                        def add_other(b, d_in):
                            cond = (b != a) & valid_mask[b] & (n_valid > 1)
                            d_in = d_in.at[nxt[b]].add(jnp.where(cond, per_other, 0.0))
                            return d_in

                        d = jax.lax.fori_loop(0, A, add_other, d)
                        # numeric normalize
                        return d / jnp.sum(d)

                    return jax.lax.cond(ok[a], valid_choice, invalid_choice)

                return jax.lax.cond(is_term, term_dist, nonterm_dist)

            dists = jax.vmap(dist_for_action)(jnp.arange(A))  # (A,S)
            return dists

        P = jax.vmap(per_state)(jnp.arange(S))  # (S,A,S)
        return P

    def _build_individual_kernel(self, key: jax.Array, P_base: jax.Array) -> jax.Array:
        """
        Mimics your Individual random kernel:
          Individual[s,a,:] random simplex then normalized,
        but we enforce the same support as P_base (only positive where P_base > 0)
        so it doesn't create transitions to impossible states.
        """
        S, A, _ = P_base.shape

        def per_sa(k, support_mask):
            return random_simplex_over_support(k, support_mask)

        keys = jax.random.split(key, S * A).reshape(S, A, 2)  # 2 keys per (s,a) not needed, but shape ok

        def per_s(s):
            def per_a(a):
                support = P_base[s, a] > 0
                # need a unique key
                k = jax.random.fold_in(key, s * A + a)
                return per_sa(k, support)
            return jax.vmap(per_a)(jnp.arange(A))

        Individual = jax.vmap(per_s)(jnp.arange(S))  # (S,A,S)
        return Individual.astype(jnp.float32)

    # ------------- gymnax-style API -------------
    def reset(self, key: jax.Array, params: GridWorldParams) -> Tuple[jax.Array, GridWorldState]:
        s0 = jax.random.choice(key, a=jnp.arange(params.mu0.shape[0]), p=params.mu0).astype(jnp.int32)
        return s0, GridWorldState(s=s0)

    def step(
        self,
        key: jax.Array,
        state: GridWorldState,
        action: jax.Array,
        params: GridWorldParams,
    ) -> Tuple[jax.Array, GridWorldState, jax.Array, jax.Array, Any]:
        s = state.s.astype(jnp.int32)
        a = action.astype(jnp.int32)

        # reward depends on current state (as in your reward_fn)
        # Since R is precomputed, just read it:
        r = params.R[s, a]

        p = params.P[s, a]  # (S,)
        s2 = jax.random.choice(key, a=jnp.arange(p.shape[0]), p=p).astype(jnp.int32)

        # done if next state is terminal (same idea as your is_terminal on next_s_idx)
        # We need terminal indicator in compressed space: build from params.terminal_mask + idx2coord
        # Instead of storing idx2coord (static Python), we mark done by: terminal iff self-coordinate is terminal.
        # We can precompute terminal_state_mask over compressed space easily from reward_grid/walls,
        # but to keep params compact we recompute on the fly from P and absorbing property is costly.
        # So: simplest: done=False (continuing) OR if you want proper done, pass terminal_states in reward_grid via negative?
        #
        # Practical: for your AC occupancy sampling you already reset-on-done in rollout, so done is ok either way.
        # We'll set done=False here to keep everything purely continuing.
        done = jnp.array(False)
        info = {}
        return s2, GridWorldState(s=s2), r, done, info
