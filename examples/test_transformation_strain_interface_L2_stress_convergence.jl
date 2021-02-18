using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCellDG
include("useful_routines.jl")
include("circular_bc_transformation_elasticity_solver.jl")


function compute_interface_L2_stress_error(polyorder, nelmts, width, penaltyfactor)
    K1, K2 = 247.0, 192.0
    mu1, mu2 = 126.0, 87.0
    lambda1 = lame_lambda(K1, mu1)
    lambda2 = lame_lambda(K2, mu2)
    stiffness = CutCellDG.HookeStiffness(lambda1, mu1, lambda2, mu2)

    theta0 = -0.067
    stiffness = CutCellDG.HookeStiffness(lambda1, mu1, lambda2, mu2)
    transfstress =
        CutCellDG.plane_strain_transformation_stress(lambda1, mu1, theta0)

    meshwidth = [width, width]
    numqp = required_quadrature_order(polyorder) + 2

    dx = width / nelmts
    penalty = penaltyfactor / dx * 0.5 * (lambda1 + mu1 + lambda2 + mu2)

    basis = TensorProductBasis(2, polyorder)
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
        basis,
        interfacecenter,
        interfaceradius,
        numqp,
    )
    nodaldisplacement = nodal_displacement(
        mesh,
        basis,
        cellquads,
        facequads,
        interfacequads,
        stiffness,
        theta0,
        analyticalsolution,
        penalty,
    )
    parenterror, producterror = interface_stress_L2_error(
        nodaldisplacement,
        basis,
        interfacequads,
        stiffness,
        transfstress,
        theta0,
        mesh,
        x -> core_stress(analyticalsolution),
        x -> shell_stress(analyticalsolution, x),
    )

    parentstressnorm = integral_norm_on_interface(
        x -> core_stress(analyticalsolution),
        interfacequads,
        -1,
        mesh,
        4,
    )
    productstressnorm = integral_norm_on_interface(
        x -> shell_stress(analyticalsolution, x),
        interfacequads,
        +1,
        mesh,
        4,
    )

    return parenterror ./ parentstressnorm, producterror ./ productstressnorm
end

width = 1.0
polyorder = 2
penaltyfactor = 1e3
powers = [3,4,5]
nelmts = [2^p for p in powers]

dx = width ./ nelmts

stresserror = [compute_interface_L2_stress_error(polyorder, ne, width, penaltyfactor) for ne in nelmts]

parenterror = [st[1] for st in stresserror]
parenterror = Array(transpose(hcat(parenterror...)))

parentrates = mapslices(x->convergence_rate(dx,x),parenterror,dims=1)
parentrates = parentrates[:,[1,2,4]]

# @test all(parentrates .> 1.4)

producterror = [st[2] for st in stresserror]

producterror = Array(transpose(hcat(producterror...)))
productrates = mapslices(x->convergence_rate(dx,x),producterror,dims=1)

# @test all(productrates .> 1.4)
