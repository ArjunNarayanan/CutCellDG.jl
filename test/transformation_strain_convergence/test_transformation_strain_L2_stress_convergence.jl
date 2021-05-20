using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCellDG
include("../useful_routines.jl")
include("circular_bc_transformation_elasticity_solver.jl")

function compute_L2_stress_error(polyorder, nelmts, width)
    K1, K2 = 247.0, 192.0    # Pa
    mu1, mu2 = 126.0, 87.0   # Pa
    lambda1 = lame_lambda(K1, mu1)
    lambda2 = lame_lambda(K2, mu2)
    stiffness = CutCellDG.HookeStiffness(lambda1, mu1, lambda2, mu2)

    theta0 = -0.067
    stiffness = CutCellDG.HookeStiffness(lambda1, mu1, lambda2, mu2)
    transfstress =
        CutCellDG.plane_strain_transformation_stress(lambda1, mu1, theta0)

    meshwidth = [width, width]
    numqp = required_quadrature_order(polyorder) + 2
    penaltyfactor = 1e2

    dx = width / nelmts
    penalty = penaltyfactor / dx * 0.5 * (lambda1 + mu1 + lambda2 + mu2)

    elasticitybasis = LagrangeTensorProductBasis(2, polyorder)
    levelsetbasis = HermiteTensorProductBasis(2)
    quad = tensor_product_quadrature(2, 4)
    dim, nf = size(interpolation_points(levelsetbasis))
    refpoints = interpolation_points(elasticitybasis)

    interfacecenter = [0.5, 0.5]
    interfaceradius = minimum(meshwidth) / 3.0
    outerradius = 2.0
    analyticalsolution = AnalyticalSolution(
        interfaceradius,
        outerradius,
        interfacecenter,
        lambda1,
        mu1,
        lambda2,
        mu2,
        theta0,
    )

    mesh, cellquads, facequads, interfacequads = construct_mesh_and_quadratures(
        meshwidth,
        nelmts,
        elasticitybasis,
        interfacecenter,
        interfaceradius,
        numqp,
    )
    nodaldisplacement = nodal_displacement(
        mesh,
        elasticitybasis,
        cellquads,
        facequads,
        interfacequads,
        stiffness,
        theta0,
        analyticalsolution,
        penalty,
    )

    stresserror = stress_L2_error(
        nodaldisplacement,
        elasticitybasis,
        cellquads,
        stiffness,
        transfstress,
        theta0,
        mesh,
        x -> core_stress(analyticalsolution),
        x -> shell_stress(analyticalsolution, x),
    )

    den = integral_norm_on_mesh(
        x -> exact_stress(analyticalsolution, x),
        cellquads,
        mesh,
        4,
    )

    normalizedstresserr = stresserror ./ den

    return normalizedstresserr
end

function test_quadratic_elements()
    width = 1.0
    polyorder = 2
    powers = [3, 4, 5]
    nelmts = [2^p + 1 for p in powers]

    err = [compute_L2_stress_error(polyorder, ne, width) for ne in nelmts]

    stresserror = Array(transpose(hcat(err...)))
    dx = width ./ nelmts
    stressrates = mapslices(x -> convergence_rate(dx, x), stresserror, dims = 1)

    @test all(stressrates .> 1.8)
end

function test_cubic_elements()
    width = 1.0
    polyorder = 3
    powers = [3, 4, 5]
    nelmts = [2^p + 1 for p in powers]

    err = [compute_L2_stress_error(polyorder, ne, width) for ne in nelmts]

    dx = width ./ nelmts
    stresserror = Array(transpose(hcat(err...)))

    stressrates = mapslices(x -> convergence_rate(dx, x), stresserror, dims = 1)
    @test all(stressrates .> 2.7)
end

test_quadratic_elements()
test_cubic_elements()
