using SafeTestsets

@safetestset "Test Cell Maps" begin
    include("test_cell_map.jl")
end

@safetestset "Test DG Mesh" begin
    include("test_dg_mesh.jl")
end
