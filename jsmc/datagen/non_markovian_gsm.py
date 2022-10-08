import jax
import chex
import jax.numpy as jnp
from functools import partial

@chex.dataclass
class ModelParameters:
    """
    Non-Markovian Gaussian Sequence Model
    parameters of the form

    f(xt | xt-1) = N(xt | phi * xt-1, q)
    g(yt | x{1:t}) = N(yt | sum_{k=1}^t beta^{t-k} * xt, r)

    Parameters
    ----------
    phi: mean of latent conditional dependency
    beta: strength of dependence on previous latent variables
    q: variance of latent conditional dependency
    r: variance of observation conditional dependency
    """
    phi: float
    beta: float
    q: float
    r: float


@chex.dataclass
class State:
    """
    State of the NM-GSM

    x: latent variable
    mu: sum of previous latent variables
    """
    x: float
    mu: float


@chex.dataclass
class DataHist:
    """
    Data history
    """
    x: jnp.ndarray
    y: jnp.ndarray


def latent_step(key, x_prev, params):
    """
    Latent step of the Non-Markovian Gaussian Sequence Model
    """
    x = jax.random.normal(key) * jnp.sqrt(params.q) + params.phi * x_prev
    return x


def observation_step(key, x, mu, params):
    """
    Observation step of the Non-Markovian Gaussian Sequence Model
    """
    mu_next =  params.beta * mu + x
    y = jax.random.normal(key) * jnp.sqrt(params.r) + mu_next
    return y, mu_next
    

def step(state, key, params):
    """
    Step of the Non-Markovian Gaussian Sequence Model
    """
    key_latent, key_observation = jax.random.split(key)
    x = latent_step(key_latent, state.x, params)
    y, mu = observation_step(key_observation, x, state.mu, params)

    state_next = state.replace(x=x, mu=mu)

    memory = DataHist(x=x, y=y)

    return state_next, memory


def non_markovian_gsm(key, params, num_steps):
    """
    Non-Markovian Gaussian Sequence Model
    """
    key_init, keys_steps = jax.random.split(key)
    keys_steps = jax.random.split(keys_steps, num_steps)
    x_init = jax.random.normal(key_init) * jnp.sqrt(params.q)
    state = State(x=x_init, mu=0)

    partial_step = partial(step, params=params)
    _, steps = jax.lax.scan(partial_step, state, keys_steps)

    return steps