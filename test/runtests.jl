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
end

@safetestset "Test Simple Tension" begin
    include("test_cut_simple_tension.jl")
    include("test_merged_simple_tension.jl")
end
