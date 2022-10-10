import jax
import chex
import jax.numpy as jnp

@chex.dataclass
class StateSIS:
    """
    State of the Sequential Importance Sampler
    """
    particles_compound: jnp.ndarray
    log_weights: jnp.ndarray


def step_accumulate_obs(row_carry, row_target, y_obs):
    """
    Accumulate a single observation
    """
    row_carry = row_carry + y_obs * row_target 
    return row_carry


def step_accumulate_sample(row_carry, xs, proposal, params, carry_transform):
    """
    Accumulate the sample
    """
    key, row, _ = xs
    sample_new = proposal.sample(key, row_carry, params)
    row_carry = carry_transform(row_carry) + sample_new * row

    return row_carry, sample_new


def step_update_weights(state, xs, target, proposal, params, carry_transform):
    """
    Update the weights
    """
    key, row, y_obs = xs
    particles_compound, log_weight_prev = state.particles_compound, state.log_weights
    particles_compound_new, sample_new = step_accumulate_sample(particles_compound, xs, proposal, params, carry_transform)

    log_weigth_new = (log_weight_prev
                    + target.logpdf(particles_compound_new)
                    - target.logpdf(particles_compound)
                    - proposal.logpdf(sample_new, particles_compound, params))
    
    state_new = StateSIS(particles_compound=particles_compound_new, log_weights=log_weigth_new)
    return state_new, (sample_new, log_weigth_new)


def eval(key, target, proposal, params, num_particles):
    ...