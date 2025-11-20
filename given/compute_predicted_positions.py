"""
Compute predicted positions using inertia and external forces.
"""

import warp as wp


@wp.kernel
def compute_predicted_positions(
    dt: wp.float64,
    gravity: wp.array(dtype=wp.vec3d),
    qt: wp.array(dtype=wp.vec3d),
    q: wp.array(dtype=wp.vec3d),
    qdot: wp.array(dtype=wp.vec3d),
    inv_mass: wp.array(dtype=wp.float64),
    external_force: wp.array(dtype=wp.vec3d),
    is_pinned: wp.array(dtype=bool),
    q_tilde: wp.array(dtype=wp.vec3d),
):
    """
    Compute predicted positions for each vertex.
    
    Updates qt (previous position), q (current position), and q_tilde (inertial position).
    Pinned particles keep their positions fixed.
    
    Args:
        dt: Time step
        gravity: Gravity vector
        qt: Previous positions (output - stores old q)
        q: Current positions (output - initial guess for solver)
        qdot: Current velocities
        inv_mass: Inverse particle masses
        external_force: External forces
        is_pinned: Boolean array marking pinned particles
        q_tilde: Inertial positions (output)
    """
    particle = wp.tid()

    qt[particle] = q[particle]
    if is_pinned[particle]:
        q_tilde[particle] = qt[particle]
        return
    v = qdot[particle] + (gravity[0] + external_force[particle] * inv_mass[particle]) * dt

    # initial guess for the position is the inertial position
    q[particle] = q[particle] + v * dt
    q_tilde[particle] = q[particle]



