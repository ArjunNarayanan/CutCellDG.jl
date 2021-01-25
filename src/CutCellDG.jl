module CutCellDG

using LinearAlgebra
using SparseArrays
using NearestNeighbors
using CartesianMesh
using PolynomialBasis
using ImplicitDomainQuadrature
import ImplicitDomainQuadrature: extend

include("cell_map.jl")

end
