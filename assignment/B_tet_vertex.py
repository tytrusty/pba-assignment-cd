import warp as wp

@wp.func 
def B_tet_vertex(dXinv: wp.mat((4,3), dtype=wp.float64), vertex_order: wp.int32) -> wp.mat((9,3), dtype=wp.float64):
    """
    Constructs the B-matrix for a specific vertex in a tetrahedral finite element.
    
    This Warp function creates a 9x3 matrix that represents the contribution of a single
    vertex's three DOFs to the row-flattened deformation gradient. This is essentially
    selecting the appropriate 3 columns (corresponding to vertex_order) from the full
    9x12 B-matrix that would be constructed for all four vertices of the tetrahedron.
    
    The full B-matrix (9x12) transforms all 12 vertex displacement components (4 vertices x 3 DOFs)
    to 9 deformation gradient components. This function extracts just the 3 columns corresponding
    to one vertex, creating a 9x3 submatrix.
    
    Parameters
    ----------
    dXinv : wp.mat((4,3), dtype=wp.float64)
        This is the D matrix described in class (the inverse reference gradient matrix)
    vertex_order : wp.int32
        The local vertex index within the tetrahedron (0, 1, 2, or 3), indicating which
        vertex's columns to extract from the full B-matrix
    
    Returns
    -------
    wp.mat((9,3), dtype=wp.float64)
        The 9x3 B-matrix for the specified vertex. This matrix is used to compute the
        vertex's contribution to the deformation gradient and its derivatives.
        Used to compute vertex gradient from dE/dF: grad_vertex = B_vertex^T @ dE/dF
    
    Notes
    -----
    This is a Warp function (@wp.func) that can be called from within Warp kernels.
    The structure is similar to constructing the full B_tet matrix but extracts only
    the 3 columns corresponding to one vertex's DOFs.
    """
    return wp.matrix(shape=(9,3), dtype=wp.float64)