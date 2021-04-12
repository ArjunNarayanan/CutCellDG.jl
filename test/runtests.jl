using SafeTestsets

@safetestset "Test Cell Maps" begin
    include("maps_and_mesh/test_cell_map.jl")
end

@safetestset "Test CG Mesh" begin
    include("maps_and_mesh/test_cg_mesh.jl")
end

@safetestset "Test DG Mesh" begin
    include("maps_and_mesh/test_dg_mesh.jl")
end

@safetestset "Test LevelSet" begin
    include("miscellaneous/test_levelset.jl")
end

@safetestset "Test Cut Mesh" begin
    include("maps_and_mesh/test_cut_mesh.jl")
end

@safetestset "Test Cell Quadratures" begin
     include("quadratures/test_cell_quadratures.jl")
end

@safetestset "Test Face Quadratures" begin
    include("quadratures/test_face_quadratures.jl")
end

@safetestset "Test Interface Quadratures" begin
    include("quadratures/test_interface_quadratures.jl")
end

@safetestset "Test Cell Merging" begin
    include("miscellaneous/test_cell_merging.jl")
end

@safetestset "Test Hooke Stiffness" begin
    include("miscellaneous/test_hooke_stiffness.jl")
end

@safetestset "Test Assembly" begin
    include("miscellaneous/test_assembly.jl")
end

@safetestset "Test Simple Tension" begin
    include("simple_tension/test_simple_tension.jl")
end

@safetestset "Test Cut Simple Tension" begin
    include("simple_tension/test_cut_simple_tension.jl")
end

@safetestset "Test Merged Simple Tension" begin
    include("simple_tension/test_merged_simple_tension.jl")
end

@safetestset "Test Transformation Strain Simple Tension" begin
    include("simple_tension/test_transformation_strain_simple_tension.jl")
end

@safetestset "Test Incoherent Interface Simple Tension" begin
    include("simple_tension/test_incoherent_interface_simple_tension.jl")
end

@safetestset "Test Vertical Plane Interface L2 Displacement Convergence" begin
    include("standard_convergence/test_vertical_plane_interface_convergence.jl")
end

@safetestset "Test Inclined Plane Interface Displacement Convergence" begin
    include("standard_convergence/test_inclined_plane_interface_convergence.jl")
end

@safetestset "Test Curved Interface Displacement Convergence" begin
    include("standard_convergence/test_curved_interface_convergence.jl")
end

@safetestset "Test Displacement + Traction BC Displacement Convergence" begin
    include(
        "standard_convergence/test_displacement_and_traction_bc_convergence.jl",
    )
end

@safetestset "Test Mixed BC Displacement Convergence" begin
    include("standard_convergence/test_mixed_bc_convergence.jl")
end

@safetestset "Test Stress Convergence" begin
    include("standard_convergence/test_stress_convergence.jl")
end

@safetestset "Test Coherent Interface Transformation Strain Displacement Convergence" begin
    include(
        "transformation_strain_convergence/test_transformation_strain_displacement_convergence.jl",
    )
end

@safetestset "Test Coherent Interface Transformation Strain Displacement + Traction BC Displacement Convergence" begin
    include(
        "transformation_strain_convergence/test_transformation_strain_displacement_traction_bc_convergence.jl",
    )
end

@safetestset "Test Coherent Interface Transformation Strain Mixed BC Displacement Convergence" begin
    include(
        "transformation_strain_convergence/test_transformation_strain_mixed_bc_convergence.jl",
    )
end

@safetestset "Test Coherent Interface Stress Convergence" begin
    include(
        "transformation_strain_convergence/test_transformation_strain_L2_stress_convergence.jl",
    )
end

@safetestset "Test Levelset Reinitialize" begin
    include("levelset_routines/test_levelset_reinitialize.jl")
end

@safetestset "Test Closest Point Algorithm on MergedMesh" begin
    include("levelset_routines/test_closest_points_on_merged_mesh.jl")
end

@safetestset "Test Levelset Propagate" begin
    include("levelset_routines/test_levelset_propagate.jl")
end
