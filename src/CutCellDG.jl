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
include("face_quadratures.jl")
include("interface_quadratures.jl")
include("cell_merging.jl")
include("hooke_stiffness.jl")
include("weak_form.jl")
include("interface_conditions.jl")
include("assembly.jl")
include("assemble_displacement_bilinear_forms.jl")
include("assemble_interelement_condition.jl")
include("assemble_interface_condition.jl")

end
