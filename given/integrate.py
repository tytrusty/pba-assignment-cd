"""
Integrate velocity and update output state.
"""

import warp as wp


@wp.kernel
def integrate_vertex(
    dt: wp.float64,
    qt: wp.array(dtype=wp.vec3d),
    q: wp.array(dtype=wp.vec3d),
    q_out: wp.array(dtype=wp.vec3d),
    qdot_out: wp.array(dtype=wp.vec3d),
):
    """
    Integrate: compute velocity and copy state to output.
    
    Computes final velocity from position change and copies final position to output.
    
    Args:
        dt: Time step
        qt: Previous positions
        q: Current positions
        q_out: Output positions
        qdot_out: Output velocities
    """
    particle = wp.tid()
    q_out[particle] = q[particle]
    qdot_out[particle] = (q[particle] - qt[particle]) / dt
