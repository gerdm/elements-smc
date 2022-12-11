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
    particles: Float[Array, "num_particles num_steps dim_particles"]
    log_weights: Float[Array, "num_particles"]
    step: int = 0


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
    key,
    state: State,
    proposal: Callable,
    y: Float[Array, "num_steps dim_obs"],
):
    """
    SMC step for a single particle.

    In this case we assume that the proposal distribution
    can consider samples from all previous and future steps.
    It's up to the user to make sure that the proposal distribution
    handles the inputs correctly.

    TODO: We need a better way to handle the different
          kinds of inputs to the proposal distribution
          in size and number of arguments.
    """
    # 1. Resample
    ix_particle = jax.random.categorical(key, state.log_weights)
    resample_particle = state.particles[ix_particle]
    # 2. Propagate
    # TODO: Add additional input to sample proposal.
    particle_new = proposal.sample(key, resample_particle, state.step, y)
    # 3. Concatenate and update state
    particles_new = resample_particle.at[state.step].set(particle_new)
    return particles_new


def _init_state(num_particles: int, num_steps: int, dim_particle: int):
    """
    Initialise the state of the particle SMC. We assume that the
    proposal distribution can consider samples from all previous
    steps.

    TODO: We wight want to be more smart on the size and handling
          of the particles.
    """
    particles_init = jnp.zeros((num_particles, num_steps, dim_particle))
    log_weights_init = jnp.zeros(num_particles)

    state = State(
        step=0,
        particles=particles_init,
        log_weights=log_weights_init,
    )
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
