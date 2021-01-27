module CutCellDG

using LinearAlgebra
using SparseArrays
using NearestNeighbors
using CartesianMesh
using PolynomialBasis
using ImplicitDomainQuadrature
import ImplicitDomainQuadrature: extend

include("utilities.jl")
include("cell_map.jl")
include("dg_mesh.jl")
include("cut_mesh.jl")
include("cell_quadratures.jl")

end
