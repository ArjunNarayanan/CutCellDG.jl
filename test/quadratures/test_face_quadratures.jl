using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCellDG
include("../useful_routines.jl")

p = [1.0 2.0 3.0]

bp = CutCellDG.extend_to_face(p, 1)
testbp = vcat(p, -ones(3)')
@test allapprox(bp, testbp)

rp = CutCellDG.extend_to_face(p, 2)
testrp = vcat(ones(3)', p)
@test allapprox(rp, testrp)

tp = CutCellDG.extend_to_face(p, 3)
testtp = vcat(p, ones(3)')
@test allapprox(tp, testtp)

lp = CutCellDG.extend_to_face(p, 4)
testlp = vcat(-ones(3)', p)
@test allapprox(testlp, lp)

quad1d = tensor_product_quadrature(1, 4)
points = quad1d.points
bp = vcat(points, -ones(4)')
rp = vcat(ones(4)', points)
tp = vcat(points, ones(4)')
lp = vcat(-ones(4)', points)
bq = CutCellDG.extend_to_face(quad1d, 1)
rq = CutCellDG.extend_to_face(quad1d, 2)
tq = CutCellDG.extend_to_face(quad1d, 3)
lq = CutCellDG.extend_to_face(quad1d, 4)
@test allapprox(bq.points, bp)
@test allapprox(rq.points, rp)
@test allapprox(tq.points, tp)
@test allapprox(lq.points, lp)
@test allapprox(bq.weights, quad1d.weights)
@test allapprox(rq.weights, quad1d.weights)
@test allapprox(tq.weights, quad1d.weights)
@test allapprox(lq.weights, quad1d.weights)


facequads = CutCellDG.face_quadratures(4)
@test allapprox(facequads[1].points, bp)
@test allapprox(facequads[2].points, rp)
@test allapprox(facequads[3].points, tp)
@test allapprox(facequads[4].points, lp)
@test allapprox(facequads[1].weights, quad1d.weights)
@test allapprox(facequads[2].weights, quad1d.weights)
@test allapprox(facequads[3].weights, quad1d.weights)
@test allapprox(facequads[4].weights, quad1d.weights)

polyorder = 1
numqp = 2
basis = TensorProductBasis(2, polyorder)
levelset = InterpolatingPolynomial(1, basis)
x0 = [0.0, 0.0]
meshwidths = [2.0, 1.0]
nelements = [2, 1]
dgmesh = CutCellDG.DGMesh(x0, meshwidths, nelements, basis)
cgmesh = CutCellDG.CGMesh(x0, meshwidths, nelements, basis)

normal = [1.0, 0.0]
xI = [0.5, 0.0]

levelset = CutCellDG.LevelSet(x->plane_distance_function(x,normal,xI),cgmesh,basis)

cutmesh = CutCellDG.CutMesh(dgmesh, levelset)
facequads = CutCellDG.FaceQuadratures(cutmesh, levelset, numqp)
idx = [5, 9, 6, 10, 7, 11, 8, 12, 1, 0, 2, 0, 3, 0, 4, 0]
testfacetoquad = reshape(idx, 2, 4, 2)
@test allequal(testfacetoquad, facequads.facetoquad)
