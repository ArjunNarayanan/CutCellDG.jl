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
