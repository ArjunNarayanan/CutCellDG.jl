using SafeTestsets

@safetestset "Test Cell Maps" begin
    include("test_cell_map.jl")
end

@safetestset "Test DG Mesh" begin
    include("test_dg_mesh.jl")
end

@safetestset "Test Cut Mesh" begin
    include("test_cut_mesh.jl")
end

@safetestset "Test Cell Quadratures" begin
    include("test_cell_quadratures.jl")
end

@safetestset "Test Face Quadratures" begin
    include("test_face_quadratures.jl")
end

@safetestset "Test Interface Quadratures" begin
    include("test_interface_quadratures.jl")
end

@safetestset "Test Cell Merging" begin
    include("test_cell_merging.jl")
end

@safetestset "Test Hooke Stiffness" begin
    include("test_hooke_stiffness.jl")
end

@safetestset "Test Assembly" begin
    include("test_assembly.jl")
end

@safetestset "Test Simple Tension" begin
    include("test_simple_tension.jl")
    include("test_cut_simple_tension.jl")
    include("test_merged_simple_tension.jl")
end

@safetestset "Test Coherent Interface Convergence" begin
    include("test_vertical_plane_interface_convergence.jl")
    include("test_inclined_plane_interface_convergence.jl")
    include("test_curved_interface_convergence.jl")
    include("test_stress_convergence.jl")
end

@safetestset "Test Traction BC Convergence" begin
    include("test_displacement_and_traction_bc_convergence.jl")
    include("test_mixed_bc_convergence.jl")
end

@safetestset "Test Incoherent Interface Simple Tension" begin
    include("test_incoherent_interface_simple_tension.jl")
end

@safetestset "Test Transformation Strain" begin
    include("test_transformation_strain_simple_tension.jl")
    include("test_transformation_strain_displacement_convergence.jl")
    include("test_transformation_strain_displacement_traction_bc_convergence.jl")
    include("test_transformation_strain_mixed_bc_convergence.jl")
end
