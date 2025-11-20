"""
Solve for updated particle coordinates using coordinate descent.
"""

import warp as wp


@wp.kernel
def solve_coordinates(
    particle_ids_in_color: wp.array(dtype=wp.int32),
    q: wp.array(dtype=wp.vec3d),
    is_pinned: wp.array(dtype=bool),
    gradients: wp.array(dtype=wp.vec3d),
    hessians: wp.array(dtype=wp.mat33d)
):
    """
    Solve for updated positions for particles in a color group.
    
    This kernel performs a single Newton step to update particle positions. For each particle,
    it solves the linear system H * delta_q = -gradient, then updates q = q + delta_q.
    
    
    
    Parameters
    ----------
    particle_ids_in_color : wp.array(dtype=wp.int32)
        Array containing the particle indices that belong to the current color group.
        This array has length equal to the number of particles in this color.
        Each thread (tid) accesses particle_ids_in_color[tid] to get the actual particle index.
        This allows parallel processing of all particles in the same color group.
    q : wp.array(dtype=wp.vec3d)
        Current positions of all particles (updated in-place)
    is_pinned : wp.array(dtype=bool)
        Boolean array marking which particles are pinned (should not be updated)
    gradients : wp.array(dtype=wp.vec3d)
        Total energy gradients (inertial + elastic + contact) for each particle.
        These should have been accumulated by the gradient computation kernels.
    hessians : wp.array(dtype=wp.mat33d)
        Total energy Hessians (inertial + elastic + contact) for each particle.
        These should have been accumulated by the Hessian computation kernels.
    
    Useful Functions
    ----------------
    - wp.inverse(M): Computes the inverse of a matrix
    """
    tid = wp.tid()
    particle_index = particle_ids_in_color[tid]
    pass
