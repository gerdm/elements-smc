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
    observations_compound:
        history of observations
    selection_compound:
        zeros -> ones array of timesteps
    log_weights:
        log weights of the particles
    """
    particles_compound: jnp.ndarray
    observations_compound: jnp.ndarray
    selection_compound: jnp.ndarray
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


def step_accumulate_sample(key, carry_particles, row, proposal, carry_transform):
    """
    Accumulate the sample
    """
    sample_new = proposal.sample(key, carry_particles)
    carry_particles = carry_transform(carry_particles) + sample_new * row
    return carry_particles, sample_new


def compound_elements(state, xs, proposal, carry_transform):
    key, row, y_obs = xs
    observations_compound_new = step_accumulate_obs(state.observations_compound, row, y_obs)
    selection_compound_new = step_accumulate_count(state.selection_compound, row)
    res_new = step_accumulate_sample(key, state.particles_compound, row, proposal, carry_transform)
    particles_compound_new, sample_new = res_new

    state = state.replace(
        particles_compound=particles_compound_new,
        observations_compound=observations_compound_new,
        selection_compound=selection_compound_new,
    )
    return state, sample_new


def step_sis(state: StateSIS, xs, target, proposal, carry_transform):
    """
    Update the weights
    """
    state_new, sample_new = compound_elements(state, xs, proposal, carry_transform)
    observations_compound_new = state_new.observations_compound
    particles_compound_new = state_new.particles_compound

    log_weigth_new = (state.log_weight_prev
                    + target.logpdf(particles_compound_new, observations_compound_new) # x{1:t}
                    - target.logpdf(state.particles_compound, state.observations_compound) # x{1:t-1}
                    - proposal.logpdf(sample_new, state.particles_compound)) # x{t} | x{1:t-1}
    
    state_new = state_new.replace(log_weight_prev=log_weigth_new)
    return state_new, (sample_new, log_weigth_new)


def init_sis(num_steps):
    """
    Initialize the Sequential Importance Sampler
    """
    state = StateSIS(
        particles_compound=jnp.zeros(num_steps),
        observations_compound=jnp.zeros(num_steps),
        selection_compound=jnp.zeros(num_steps),
        log_weights=0.0
    )
    return state


def eval(key, observations, target, proposal, carry_transform=None):
    num_steps = len(observations)
    state_init = init_sis(num_steps)
    rows = jnp.eye(num_steps)
    keys = jax.random.split(key, num_steps)

    xs = (keys, rows, observations)
    partial_step = partial(compound_elements, proposal=proposal, carry_transform=lambda x: x)
    state, _ = jax.lax.scan(partial_step, state_init, xs)
    return state
