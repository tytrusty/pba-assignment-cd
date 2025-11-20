import warp as wp


@wp.kernel
def ground_contact_gradient_and_hessian(
    pos: wp.array(dtype=wp.vec3d),
    particle_ids_in_color: wp.array(dtype=int),
    contact_k: wp.float64,
    # outputs: gradients and hessians
    gradients: wp.array(dtype=wp.vec3d),
    hessians: wp.array(dtype=wp.mat33d),
):
    """
    Accumulate ground contact gradients and Hessians for particles in a color group.
    
    This kernel computes the contact energy gradient and Hessian contributions for each particle
    in the current color group. The contact energy prevents particles from penetrating the ground
    plane using a spring-like potential.
    
    Ground plane is at y = 0 with normal pointing up (0, 1, 0).
    
    Energy Function:
    E_contact(q_i) = k_c * d^2 / 2  if d < 0 (penetrating)
                     0                otherwise
    
    where d = y_i (the y-coordinate of the particle, negative when below ground)
        
    Parameters
    ----------
    pos : wp.array(dtype=wp.vec3d)
        Current positions of all particles
    particle_ids_in_color : wp.array(dtype=int)
        Array containing the particle indices that belong to the current color group.
        This array has length equal to the number of particles in this color.
        Each thread (tid) accesses particle_ids_in_color[tid] to get the actual particle index.
        This allows parallel processing of all particles in the same color group.
    contact_k : wp.float64
        Contact stiffness parameter (spring constant for ground contact)
    gradients : wp.array(dtype=wp.vec3d)
        Output array to accumulate contact energy gradients (dE/dq) for each particle
    hessians : wp.array(dtype=wp.mat33d)
        Output array to accumulate contact energy Hessians (d2E/dq2) for each particle
    
    Algorithm
    ----------
    For each particle in the color group:
    1. Compute penetration depth: d = y_coordinate (negative when below ground)
    2. If d < 0 (penetrating):
       a. Compute gradient
       b. Compute Hessian
    3. Otherwise, gradient and Hessian are zero
    4. Accumulate both into the output arrays
    
    Useful Functions
    ----------------
    - wp.outer(a, b): Computes the outer product of two vectors, returning a matrix
    """
    tid = wp.tid()
    particle_index = particle_ids_in_color[tid]
    pass


