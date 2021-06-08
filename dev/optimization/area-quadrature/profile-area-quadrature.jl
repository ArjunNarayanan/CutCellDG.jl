using ProfileView
using PolynomialBasis
using ImplicitDomainQuadrature
using BenchmarkTools
# using Revise
using CutCellDG
include("../../../test/useful_routines.jl")
include("test-bounds.jl")

interfacecenter = [1.5, 0.5]
interfaceradius = 1.0

levelsetbasis = HermiteTensorProductBasis(2)
interpgrad = InterpolatingPolynomial(2, 2, 3)
numqp = required_quadrature_order(3) + 2
dim, numpts = size(interpolation_points(levelsetbasis))
quad = tensor_product_quadrature(2, 4)
cgmesh = CutCellDG.CGMesh([0.0, 0.0], [1.0, 1.0], [1, 1], numpts)

levelset = CutCellDG.LevelSet(
    x -> -circle_distance_function(x, interfacecenter, interfaceradius)[1],
    cgmesh,
    levelsetbasis,
    quad,
)

CutCellDG.load_coefficients!(levelset, 1)
update_interpolating_gradient!(interpgrad, CutCellDG.interpolater(levelset))

function run_area_quadrature(levelset, interpgrad, numqp; numiter = 100)
    for i = 1:numiter
        pquad = area_quadrature(
            CutCellDG.interpolater(levelset),
            interpgrad,
            +1,
            [-1.0, -1.0],
            [1.0, 1.0],
            numqp,
        )
    end
end

# Run once to compile
# run_area_quadrature(levelset,interpgrad,numqp)

# @profview run_area_quadrature(levelset, interpgrad, numqp)


using IntervalArithmetic
xL,xR = [-1.,-1.],[1.,1.]
box = IntervalBox(xL,xR)
# @btime ImplicitDomainQuadrature.is_suitable(1,interpgrad,box)

curve(x) = ImplicitDomainQuadrature.curvature_measure(interpgrad(x),1,4)
# @btime sign(curve,box,tol=1e-3)
interval_arithmetic_sign_search(curve,box,1e-3,0.0,2)

@btime area_quadrature(CutCellDG.interpolater(levelset),interpgrad)
