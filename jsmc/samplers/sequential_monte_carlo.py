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
    key_resample, key_propagate = jax.random.split(key)
    # 1. Resample
    ix_particle = jax.random.categorical(key_resample, state.log_weights)
    resample_particle = state.particles[ix_particle]
    # 2. Propagate
    # TODO: Add additional input to sample proposal.
    particle_new = proposal.sample(key_propagate, resample_particle, state.step, y)
    # 3. Concatenate and update state
    particles_new = resample_particle.at[state.step + 1].set(particle_new)
    return particles_new


def estimate_log_weights(particles, state_old, target, proposal, y):
    """
    Compute the unnormalised log-weights of the new (resampled) particles.
    """
    step_prev = state_old.step
    step_next = state_old.step + 1
    particle_next  = particles[:, step_next]

    target_logpdf = jax.vmap(target.logpdf, in_axes=(0, None, None))
    proposal_logpdf = jax.vmap(proposal.logpdf, in_axes=(0, 0, None))
    log_weights = (
                    + target_logpdf(particles, step_next, y) # x{1:t}
                    - target_logpdf(particles, step_prev, y) # x{1:t-1}
                    - proposal_logpdf(particle_next,  particles, step_prev).squeeze() # x{t} | x{1:t-1}
                    ) 
    
    return log_weights


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


def step_and_update(
    key,
    state: State,
    y: Float[Array, "num_steps dim_obs"],
    proposal: Callable,
    target: Callable,
):
    num_particles = state.particles.shape[0]
    keys = jax.random.split(key, num_particles)
    # Resample and propagate
    particles_new = jax.vmap(_step_smc, in_axes=(0, None, None, None))
    particles_new = particles_new(keys, state, proposal, y)

    log_weights_new = estimate_log_weights(particles_new, state, target, proposal, y)

    state_new = state.replace(
        particles=particles_new,
        step=state.step + 1,
        log_weights=log_weights_new,
    )
    return state_new
