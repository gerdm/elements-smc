import jax
import chex
import jax.numpy as jnp
from functools import partial

@chex.dataclass
class StateSIS:
    """
    State of the Sequential Importance Sampler

    Parameters
    ----------
    particle_count:
        history of sampled proposals
    observations:
        history of observations
    vcounter:
        zeros -> ones array of timesteps
    log_weights:
        log weights of the particles
    """
    particles: jnp.ndarray
    observations: jnp.ndarray
    vcounter: jnp.ndarray
    log_weights: jnp.ndarray


def step_accumulate_count(row_carry, row_target):
    """
    Accumulate a single observation
    """
    row_carry = row_carry + row_target 
    return row_carry


def step_accumulate_obs(row_carry, row_target, y_obs):
    """
    Accumulate a single observation
    """
    row_carry = row_carry + y_obs * row_target 
    return row_carry


def step_accumulate_sample(key, state_sis, row, proposal):
    """
    Accumulate the sample
    """
    carry_particles = state_sis.particles
    sample_new = proposal.sample(key, state_sis)
    carry_particles = carry_particles + sample_new * row
    return carry_particles, sample_new


def compound_elements(state: StateSIS, xs, proposal):
    key, row, y_obs = xs
    observations_new = step_accumulate_obs(state.observations, row, y_obs)
    vcounter_new = step_accumulate_count(state.vcounter, row)
    res_new = step_accumulate_sample(key, state, row, proposal)
    particles_new, sample_new = res_new

    state = state.replace(
        particles=particles_new,
        observations=observations_new,
        vcounter=vcounter_new,
    )
    return state, sample_new


def step_sis(state: StateSIS, xs, target, proposal):
    """
    Update the weights
    """
    state_new, sample_new = compound_elements(state, xs, proposal)

    log_weigth_new = (state.log_weights
                    + target.logpdf(state_new) # x{1:t}
                    - target.logpdf(state) # x{1:t-1}
                    - proposal.logpdf(sample_new, state)) # x{t} | x{1:t-1}
    
    state_new = state_new.replace(log_weights=log_weigth_new)
    return state_new, log_weigth_new


def init_sis(num_steps):
    """
    Initialize the Sequential Importance Sampler
    """
    state = StateSIS(
        particles=jnp.zeros(num_steps),
        observations=jnp.zeros(num_steps),
        vcounter=jnp.zeros(num_steps),
        log_weights=0.0
    )
    return state


def eval(key, observations, target, proposal):
    num_steps = len(observations)
    state_init = init_sis(num_steps)
    rows = jnp.eye(num_steps)
    keys = jax.random.split(key, num_steps)

    xs = (keys, rows, observations)
    partial_step = partial(
        step_sis,
        proposal=proposal,
        target=target,
    )
    state, log_weights = jax.lax.scan(partial_step, state_init, xs)
    state = state.replace(
        log_weights=log_weights
    )
    return state
