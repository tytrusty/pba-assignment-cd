"""
Adjacency information for force elements (tets, springs)
This is a simplified version of the adjacency information class in the Newton physics library's VBD solver: https://github.com/newton-physics/newton/blob/8e9cd80558f918549837296e2c27a77f8a8aa4b6/newton/_src/solvers/vbd/solver_vbd.py#L57
"""

import numpy as np
import warp as wp


@wp.struct
class ElementAdjacencyInfo:
    r"""
    - vertex_adjacent_[element]: the flatten adjacency information. Its size is \sum_{i\inV} 2*N_i, where N_i is the
    number of vertex i's adjacent [element]. For each adjacent element it stores 2 information:
        - the id of the adjacent element
        - the order of the vertex in the element, which is essential to compute the force and hessian for the vertex
    - vertex_adjacent_[element]_offsets: stores where each vertex information starts in the  flatten adjacency array.
    Its size is |V|+1 such that the number of vertex i's adjacent [element] can be computed as
    vertex_adjacent_[element]_offsets[i+1]-vertex_adjacent_[element]_offsets[i].
    """

    v_adj_tets: wp.array(dtype=int)
    v_adj_tets_offsets: wp.array(dtype=int)

    def to(self, device):
        if device == self.v_adj_tets.device:
            return self
        else:
            adjacency_gpu = ElementAdjacencyInfo()
            adjacency_gpu.v_adj_tets = self.v_adj_tets.to(device)
            adjacency_gpu.v_adj_tets_offsets = self.v_adj_tets_offsets.to(device)
            return adjacency_gpu


@wp.func
def get_vertex_num_adjacent_tets(adjacency: ElementAdjacencyInfo, vertex: wp.int32):
    return (adjacency.v_adj_tets_offsets[vertex + 1] - adjacency.v_adj_tets_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_tet_id_order(adjacency: ElementAdjacencyInfo, vertex: wp.int32, tet: wp.int32):
    offset = adjacency.v_adj_tets_offsets[vertex]
    return adjacency.v_adj_tets[offset + tet * 2], adjacency.v_adj_tets[offset + tet * 2 + 1]

@wp.kernel
def count_num_adjacent_tets(
    tet_indices: wp.array(dtype=wp.int32, ndim=2), num_vertex_adjacent_tets: wp.array(dtype=wp.int32)
):
    for tet in range(tet_indices.shape[0]):
        v0 = tet_indices[tet, 0]
        v1 = tet_indices[tet, 1]
        v2 = tet_indices[tet, 2]
        v3 = tet_indices[tet, 3]

        num_vertex_adjacent_tets[v0] = num_vertex_adjacent_tets[v0] + 1
        num_vertex_adjacent_tets[v1] = num_vertex_adjacent_tets[v1] + 1
        num_vertex_adjacent_tets[v2] = num_vertex_adjacent_tets[v2] + 1
        num_vertex_adjacent_tets[v3] = num_vertex_adjacent_tets[v3] + 1


@wp.kernel
def fill_adjacent_tets(
    tet_indices: wp.array(dtype=wp.int32, ndim=2),
    vertex_adjacent_tets_offsets: wp.array(dtype=wp.int32),
    vertex_adjacent_tets_fill_count: wp.array(dtype=wp.int32),
    vertex_adjacent_tets: wp.array(dtype=wp.int32),
):
    for tet in range(tet_indices.shape[0]):
        v0 = tet_indices[tet, 0]
        v1 = tet_indices[tet, 1]
        v2 = tet_indices[tet, 2]
        v3 = tet_indices[tet, 3]

        fill_count_v0 = vertex_adjacent_tets_fill_count[v0]
        buffer_offset_v0 = vertex_adjacent_tets_offsets[v0]
        vertex_adjacent_tets[buffer_offset_v0 + fill_count_v0 * 2] = tet
        vertex_adjacent_tets[buffer_offset_v0 + fill_count_v0 * 2 + 1] = 0
        vertex_adjacent_tets_fill_count[v0] = fill_count_v0 + 1

        fill_count_v1 = vertex_adjacent_tets_fill_count[v1]
        buffer_offset_v1 = vertex_adjacent_tets_offsets[v1]
        vertex_adjacent_tets[buffer_offset_v1 + fill_count_v1 * 2] = tet
        vertex_adjacent_tets[buffer_offset_v1 + fill_count_v1 * 2 + 1] = 1
        vertex_adjacent_tets_fill_count[v1] = fill_count_v1 + 1

        fill_count_v2 = vertex_adjacent_tets_fill_count[v2]
        buffer_offset_v2 = vertex_adjacent_tets_offsets[v2]
        vertex_adjacent_tets[buffer_offset_v2 + fill_count_v2 * 2] = tet
        vertex_adjacent_tets[buffer_offset_v2 + fill_count_v2 * 2 + 1] = 2
        vertex_adjacent_tets_fill_count[v2] = fill_count_v2 + 1

        fill_count_v3 = vertex_adjacent_tets_fill_count[v3]
        buffer_offset_v3 = vertex_adjacent_tets_offsets[v3]
        vertex_adjacent_tets[buffer_offset_v3 + fill_count_v3 * 2] = tet
        vertex_adjacent_tets[buffer_offset_v3 + fill_count_v3 * 2 + 1] = 3
        vertex_adjacent_tets_fill_count[v3] = fill_count_v3 + 1

def compute_force_element_adjacency(tet_indices, particle_count):
    """
    Compute adjacency information for force elements (tets and springs).
    
    Args:
        tet_indices: wp.array of tet indices (shape: [num_tets, 4])
        particle_count: number of particles
        
    Returns:
        ForceElementAdjacencyInfo with populated adjacency data
    """
    adjacency = ElementAdjacencyInfo()
    tet_indices = tet_indices.to("cpu")

    with wp.ScopedDevice("cpu"):
        if tet_indices.size:
            # compute adjacent tetrahedra
            # count number of adjacent tets for each vertex
            num_vertex_adjacent_tets = wp.zeros(shape=(particle_count,), dtype=wp.int32)
            wp.launch(kernel=count_num_adjacent_tets, inputs=[tet_indices, num_vertex_adjacent_tets], dim=1)

            # preallocate memory based on counting results
            num_vertex_adjacent_tets = num_vertex_adjacent_tets.numpy()
            vertex_adjacent_tets_offsets = np.empty(shape=(particle_count + 1,), dtype=wp.int32)
            vertex_adjacent_tets_offsets[1:] = np.cumsum(2 * num_vertex_adjacent_tets)[:]
            vertex_adjacent_tets_offsets[0] = 0
            adjacency.v_adj_tets_offsets = wp.array(vertex_adjacent_tets_offsets, dtype=wp.int32)

            vertex_adjacent_tets_fill_count = wp.zeros(shape=(particle_count,), dtype=wp.int32)

            tet_adjacency_array_size = 2 * num_vertex_adjacent_tets.sum()
            # (tet, vertex_order) * num_adj_tets * num_particles
            # vertex order: v0: 0, v1: 1, v2: 2, v3: 3
            adjacency.v_adj_tets = wp.empty(shape=(tet_adjacency_array_size,), dtype=wp.int32)

            wp.launch(
                kernel=fill_adjacent_tets,
                inputs=[
                    tet_indices,
                    adjacency.v_adj_tets_offsets,
                    vertex_adjacent_tets_fill_count,
                    adjacency.v_adj_tets,
                ],
                dim=1,
            )
        else:
            adjacency.v_adj_tets_offsets = wp.empty(shape=(0,), dtype=wp.int32)
            adjacency.v_adj_tets = wp.empty(shape=(0,), dtype=wp.int32)
    return adjacency

