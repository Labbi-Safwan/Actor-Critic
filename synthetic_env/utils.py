# utils_jax.py
from __future__ import annotations

from typing import Sequence, Tuple, Union
import jax
import jax.numpy as jnp


ShapeLike = Union[Sequence[int], Tuple[int, ...]]


def random_simplex_vector(key: jax.Array, d: int = 5, size: ShapeLike = (1,)) -> jax.Array:
    """
    JAX replacement for:
        vec = np.random.exponential(size=size+[d])
        vec = vec / np.sum(vec, axis=-1).reshape(size+[1])

    Args:
        key: PRNGKey
        d: simplex dimension
        size: leading shape (like your 'size' list). Example:
              size=(S,A) -> returns shape (S,A,d)

    Returns:
        Array of shape (*size, d) with last axis summing to 1.
    """
    size = tuple(int(x) for x in size)
    x = jax.random.exponential(key, shape=size + (int(d),))
    return x / jnp.sum(x, axis=-1, keepdims=True)
