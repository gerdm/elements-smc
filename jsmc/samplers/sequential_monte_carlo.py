import jax
import chex
import jax.numpy as jnp
from jaxtyping import Array, Float
from functools import partial
from typing import Callable

@chex.dataclass
class State:
    """
    State of the Sequential Importance Sampler
    """
    particles: Float[Array, "num_steps dim_particle"]
    log_weights: Float[Array, "num_steps"]
    step: int = 0


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


def compound_elements(state: State, xs, proposal):
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


def step_smc(state: State, xs, target, proposal):
    """
    Update the weights
    """
    state_new, sample_new = compound_elements(state, xs, proposal)

    log_weights_new = (state.log_weights
                    + target.logpdf(state_new) # x{1:t}
                    - target.logpdf(state) # x{1:t-1}
                    - proposal.logpdf(sample_new, state)) # x{t} | x{1:t-1}
    
    state_new = state_new.replace(log_weights=log_weights_new)
    return state_new, log_weights_new


def _step_smc(
    state: State,
    xs,
    proposal: Callable,
    y: Float[Array, "num_steps dim_obs"],
):
    """
    SMC step for a single particle
    """
    key, row, y_obs = xs
    y_obs = row * y_obs # Masking observation
    # 1. Resample
    ix_particle = jax.random.categorical(key, state.log_weights)
    resample_particle = state.particles[ix_particle]
    # 2. Propagate
    # TODO: Add additional input to sample proposal
    particle_new = proposal.sample(key, resample_particle, y, state.step)
    # 3. Concatenate and update state
    particles_new = resample_particle.at[state.step].set(particle_new)
    state_new = state.replace(particles=particles_new, step=state.step + 1)
    return state_new


def _init_state_single(_, num_steps, dim_particle):
    """
    Initialise single state of the particle SMC
    """
    state = State(
        step=0,
        particles=jnp.zeros((num_steps, dim_particle)),
        log_weights=jnp.zeros(num_steps),
    )
    return state

def _init_state(num_particles: int, num_steps: int, dim_particle: int):
    """
    Initialise the state of the particle SMC
    """
    dummyp = jnp.zeros(num_particles)
    state = jax.vmap(_init_state_single, in_axes=(0, None, None))
    state = state(dummyp, num_steps, dim_particle)
    return state


def eval(key, observations, target, proposal):
    num_steps = len(observations)
    state_init = init_smc(num_steps)
    rows = jnp.eye(num_steps)
    keys = jax.random.split(key, num_steps)

    xs = (keys, rows, observations)
    partial_step = partial(
        step_smc,
        proposal=proposal,
        target=target,
    )
    state, log_weights = jax.lax.scan(partial_step, state_init, xs)
    state = state.replace(
        log_weights=log_weights
    )
    return state
