"""
Matching points on the grid to specific grids



"""


import numpy as np

cimport cython
cimport grid_visitors
cimport numpy as np
from grid_visitors cimport (
    GridTreeNode,
    GridTreeNodePadded,
    GridVisitor,
    CountGridCells,
    MaskGridCells,
    ICoordsGrids,
    IResGrids,
    FCoordsGrids,
    FWidthGrids,
)
from libc.stdlib cimport free, malloc

from yt.geometry.selection_routines cimport SelectorObject, _ensure_code
from yt.utilities.lib.fp_utils cimport iclip

cdef class GridTree:
    cdef GridTreeNode *grids
    cdef GridTreeNode *root_grids
    cdef int num_grids
    cdef int num_root_grids
    cdef int num_leaf_grids

cdef class GridTreeSelector:
    cdef GridTree tree
    cdef np.uint8_t[:] mask
    cdef np.uint64_t size
    cdef np.uint64_t cell_count
    cdef np.uint8_t initialized

    cdef void visit_grids(self, GridVisitor visitor, SelectorObject selector)
    cdef void recursively_visit_grid(self,
                          GridVisitor visitor,
                          SelectorObject selector,
                          GridTreeNode *grid)


cdef class MatchPointsToGrids:

    cdef int num_points
    cdef np.float64_t *xp
    cdef np.float64_t *yp
    cdef np.float64_t *zp
    cdef GridTree tree
    cdef np.int64_t *point_grids
    cdef np.uint8_t check_position(self,
                                   np.int64_t pt_index,
                                   np.float64_t x,
                                   np.float64_t y,
                                   np.float64_t z,
                                   GridTreeNode *grid)

    cdef np.uint8_t is_in_grid(self,
			 np.float64_t x,
			 np.float64_t y,
			 np.float64_t z,
			 GridTreeNode *grid)

cdef extern from "platform_dep.h" nogil:
    double rint(double x)
