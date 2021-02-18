using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCellDG
include("useful_routines.jl")
include("circular_bc_transformation_elasticity_solver.jl")

function compute_L2_stress_error(polyorder,nelmts,width)
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

    stresserror = stress_L2_error(
        nodaldisplacement,
        basis,
        cellquads,
        stiffness,
        transfstress,
        theta0,
        mesh,
        x->core_stress(analyticalsolution),
        x->shell_stress(analyticalsolution,x)
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

width = 1.0
polyorder = 2
powers = [3,4,5]
nelmts = [2^p+1 for p in powers]

err = [compute_L2_stress_error(polyorder,ne,width) for ne in nelmts]

s11err = [er[1] for er in err]
s22err = [er[2] for er in err]
s12err = [er[3] for er in err]
s33err = [er[4] for er in err]

dx = width ./ nelmts

s11rate = convergence_rate(dx,s11err)
s22rate = convergence_rate(dx,s22err)
s12rate = convergence_rate(dx,s12err)
s33rate = convergence_rate(dx,s33err)

@test all(s11rate .> 1.8)
@test all(s22rate .> 1.8)
@test all(s12rate .> 1.8)
@test all(s33rate .> 1.8)



# width = 1.0
# polyorder = 3
# powers = [3,4,5]
# nelmts = [2^p+1 for p in powers]
#
# err = [compute_L2_stress_error(polyorder,ne,width) for ne in nelmts]
#
# s11err = [er[1] for er in err]
# s22err = [er[2] for er in err]
# s12err = [er[3] for er in err]
# s33err = [er[4] for er in err]
#
# dx = width ./ nelmts
#
# s11rate = convergence_rate(dx,s11err)
# s22rate = convergence_rate(dx,s22err)
# s12rate = convergence_rate(dx,s12err)
# s33rate = convergence_rate(dx,s33err)
#
# @test all(s11rate .> 2.7)
# @test all(s22rate .> 2.7)
# @test all(s12rate .> 2.7)
# @test all(s33rate .> 2.7)
