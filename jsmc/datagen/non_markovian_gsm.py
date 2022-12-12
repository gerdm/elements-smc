"""
Non-Markvoian Gaussian Sequence Model (NM-GSM)
implementation.

We assume that the system is described by 
    x_t = \phi x_{t-1} + \epsilon_t 
    y_t = \sum_{i=0}^t \beta^{t-i} x_i + \eta_t

with x0 = 0.0
"""
import jax
import chex
import distrax
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


def simulate(key, params, num_steps):
    """
    Non-Markovian Gaussian Sequence Model
    """
    key_init, keys_steps = jax.random.split(key)
    keys_steps = jax.random.split(keys_steps, num_steps)
    # x_init = jax.random.normal(key_init)
    x_init = 0.0
    state = State(x=x_init, mu=0)

    partial_step = partial(step, params=params)
    _, steps = jax.lax.scan(partial_step, state, keys_steps)

    return steps


def step_target_mean(carry_row, row, params, x_latent):
    """
    Helper function to build means of non-Markovian
    GSM one row at a time.
    """
    carry_row = carry_row * params.beta + row
    mean_val = jnp.einsum("i...,i->...", x_latent, carry_row)
    return carry_row, mean_val


def eval_observation_mean(params, x_latent):
    """
    Evaluate the means of the observation distribution
    given the latent estimates.
    """
    num_steps = x_latent.shape[0]
    init_row = jnp.zeros(num_steps)
    eval_rows = jnp.eye(num_steps)

    partial_target = partial(step_target_mean, params=params, x_latent=x_latent)
    _, means = jax.lax.scan(partial_target, init_row, eval_rows)

    return means


def step_latent_logpdf(x_prev, params):
    mean =  params.phi * x_prev
    logprob = distrax.Normal(loc=mean, scale=jnp.sqrt(params.q)).log_prob(x_prev)
    return logprob


def log_observation(x_latent, y_obs, params):
    """
    Log target density of the Non-Markovian Gaussian Sequence Model
    """
    mean_est = eval_observation_mean(params, x_latent)
    scale_est = jnp.sqrt(params.r)
    log_probs = distrax.Normal(loc=mean_est, scale=scale_est).log_prob(y_obs)
    return log_probs


def log_transition(x_latent, params):
    """
    Log target density of the Non-Markovian Gaussian Sequence Model
    """
    x_cond = jnp.roll(x_latent, 1).at[0].set(0.0) * params.phi
    log_probs = distrax.Normal(loc=x_cond, scale=jnp.sqrt(params.q)).log_prob(x_latent)
    return log_probs


partial(jax.jit, static_argnums=(2,))
def log_joint(x_latent, y_obs, params):
    """
    Log target density of the Non-Markovian Gaussian Sequence Model
    """
    log_probs = log_transition(x_latent, params) + log_observation(x_latent, y_obs, params)
    return log_probs
