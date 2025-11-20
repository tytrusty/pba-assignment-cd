from .adjacency import (
    ElementAdjacencyInfo,
    get_vertex_num_adjacent_tets,
    get_vertex_adjacent_tet_id_order,
    compute_force_element_adjacency,
)
from .lumped_mass import (
    compute_lumped_mass,
)
from .integrate import (
    integrate_vertex,
)
from .compute_predicted_positions import (
    compute_predicted_positions,
)
from .coordinate_descent import (
    CoordinateDescentSolver,
)
