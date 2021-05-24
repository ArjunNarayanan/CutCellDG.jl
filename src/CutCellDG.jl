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
include("cg_mesh.jl")
include("dg_mesh.jl")
include("periodic_bc.jl")
include("levelset.jl")
include("cut_mesh.jl")

include("cell_quadratures.jl")
include("face_quadratures.jl")
include("interface_quadratures.jl")
include("cell_merging.jl")

include("hooke_stiffness.jl")

include("weak_form.jl")
include("interface_conditions.jl")
include("transformation_strain.jl")

include("seed_zero_levelset.jl")
include("levelset_closest_points.jl")
# include("levelset_reinitialize.jl")
include("levelset_propagate.jl")

include("assembly.jl")
include("assemble_hermite_levelset_initializers.jl")
include("assemble_displacement_bilinear_forms.jl")
include("assemble_interelement_condition.jl")
include("assemble_interface_condition.jl")
include("assemble_penalty_displacement_bc.jl")
include("assemble_penalty_displacement_component_bc.jl")
include("assemble_body_force.jl")
include("assemble_traction_force_linear_form.jl")
include("assemble_traction_force_component_linear_form.jl")
include("assemble_transformation_strain.jl")
include("postprocess.jl")


end
