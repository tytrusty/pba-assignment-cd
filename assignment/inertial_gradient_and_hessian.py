"""
Compute inertial gradient and Hessian for coordinate descent.
"""

import warp as wp


@wp.kernel
def inertial_gradient_and_hessian(
    dt: wp.float64,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    mass: wp.array(dtype=wp.float64),
    q: wp.array(dtype=wp.vec3d),
    q_tilde: wp.array(dtype=wp.vec3d),
    is_pinned: wp.array(dtype=bool),
    # outputs
    gradients: wp.array(dtype=wp.vec3d),
    hessians: wp.array(dtype=wp.mat33d),
):
    """
    Compute inertial gradient and Hessian for particles in a color group.
    
    Energy Function:
    E_inertial_i(q_i) = (1/2) * (q_i - q_tilde_i)^T * m_i * (q_i - q_tilde_i) / dt^2
    
    Parameters
    ----------
    dt : wp.float64
        Time step size
    particle_ids_in_color : wp.array(dtype=wp.int32)
        Array containing the particle indices that belong to the current color group.
        This array has length equal to the number of particles in this color.
        Each thread (tid) accesses particle_ids_in_color[tid] to get the actual particle index.
        This allows parallel processing of all particles in the same color group.
    mass : wp.array(dtype=wp.float64)
        Lumped mass for each particle (1D array, one value per particle)
    q : wp.array(dtype=wp.vec3d)
        Current positions of all particles
    q_tilde : wp.array(dtype=wp.vec3d)
        Predicted positions from momentum integration (inertial target positions)
    is_pinned : wp.array(dtype=bool)
        Boolean array marking which particles are pinned (should not be updated)
    gradients : wp.array(dtype=wp.vec3d)
        Output array to accumulate inertial energy gradients (dE/dq) for each particle
    hessians : wp.array(dtype=wp.mat33d)
        Output array to accumulate inertial energy Hessians (d2E/dq2) for each particle
    
    Useful Functions
    ----------------
    - wp.identity(3): Returns the 3x3 identity matrix

    Algorithm
    ----------
    For each particle in the color group:
    1. Check if particle is pinned; if so, skip
    2. Compute inertial gradient
    3. Compute inertial Hessian
    4. Accumulate both into the output arrays
    """
    tid = wp.tid()
    particle_index = particle_ids_in_color[tid]
    pass

