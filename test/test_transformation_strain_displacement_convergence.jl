using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCellDG
include("useful_routines.jl")

function bulk_modulus(l, m)
    return l + 2m / 3
end

function lame_lambda(k, m)
    return k - 2m / 3
end

function analytical_coefficient_matrix(inradius, outradius, ls, ms, lc, mc)
    a = zeros(3, 3)
    a[1, 1] = inradius
    a[1, 2] = -inradius
    a[1, 3] = -1.0 / inradius
    a[2, 1] = 2 * (lc + mc)
    a[2, 2] = -2 * (ls + ms)
    a[2, 3] = 2ms / inradius^2
    a[3, 2] = 2(ls + ms)
    a[3, 3] = -2ms / outradius^2
    return a
end

function analytical_coefficient_rhs(ls, ms, theta0)
    r = zeros(3)
    Ks = bulk_modulus(ls, ms)
    r[2] = -Ks * theta0
    r[3] = Ks * theta0
    return r
end

struct AnalyticalSolution
    inradius::Any
    outradius::Any
    center::Any
    A1c::Any
    A1s::Any
    A2s::Any
    ls::Any
    ms::Any
    lc::Any
    mc::Any
    theta0::Any
    function AnalyticalSolution(
        inradius,
        outradius,
        center,
        ls,
        ms,
        lc,
        mc,
        theta0,
    )
        a = analytical_coefficient_matrix(inradius, outradius, ls, ms, lc, mc)
        r = analytical_coefficient_rhs(ls, ms, theta0)
        coeffs = a \ r
        new(
            inradius,
            outradius,
            center,
            coeffs[1],
            coeffs[2],
            coeffs[3],
            ls,
            ms,
            lc,
            mc,
            theta0,
        )
    end
end

function radial_displacement(A::AnalyticalSolution, r)
    if r <= A.inradius
        return A.A1c * r
    else
        return A.A1s * r + A.A2s / r
    end
end

function (A::AnalyticalSolution)(x)
    relpos = x - A.center
    r = sqrt(relpos' * relpos)
    ur = radial_displacement(A, r)
    if ur ≈ 0.0
        [0.0, 0.0]
    else
        costheta = (x[1] - A.center[1]) / r
        sintheta = (x[2] - A.center[2]) / r
        u1 = ur * costheta
        u2 = ur * sintheta
        return [u1, u2]
    end
end

function onboundary(x, L, W)
    return x[2] ≈ 0.0 || x[1] ≈ L || x[2] ≈ W || x[1] ≈ 0.0
end

function displacement_error(
    width,
    center,
    inradius,
    outradius,
    stiffness,
    theta0,
    nelmts,
    polyorder,
    numqp,
    penaltyfactor,
)
    lambda1, mu1 = CutCellDG.lame_coefficients(stiffness, +1)
    lambda2, mu2 = CutCellDG.lame_coefficients(stiffness, -1)
    transfstress =
        CutCellDG.plane_strain_transformation_stress(lambda1, mu1, theta0)

    dx = width / nelmts
    meanmoduli = 0.5 * (lambda1 + lambda2 + mu1 + mu2)
    penalty = penaltyfactor / dx * meanmoduli

    analyticalsolution = AnalyticalSolution(
        inradius,
        outradius,
        center,
        lambda1,
        mu1,
        lambda2,
        mu2,
        theta0,
    )

    basis = TensorProductBasis(2, polyorder)
    mesh = CutCellDG.DGMesh([0.0, 0.0], [width, width], [nelmts, nelmts], basis)
    levelset = InterpolatingPolynomial(1, basis)
    levelsetcoeffs = CutCellDG.levelset_coefficients(
        x -> -circle_distance_function(x, center, inradius),
        mesh,
    )

    cutmesh = CutCellDG.CutMesh(mesh, levelset, levelsetcoeffs)
    cellquads =
        CutCellDG.CellQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)
    interfacequads =
        CutCellDG.InterfaceQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)
    facequads =
        CutCellDG.FaceQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)

    mergedwithcell, hasmergedcells = CutCellDG.merge_tiny_cells_in_mesh!(
        cutmesh,
        cellquads,
        facequads,
        interfacequads,
    )
    quadareas = CutCellDG.quadrature_areas(cellquads, cutmesh)
    # @assert hasmergedcells
    mergedmesh = CutCellDG.MergedMesh(cutmesh, mergedwithcell)


    sysmatrix = CutCellDG.SystemMatrix()
    sysrhs = CutCellDG.SystemRHS()

    CutCellDG.assemble_displacement_bilinear_forms!(
        sysmatrix,
        basis,
        cellquads,
        stiffness,
        mergedmesh,
    )
    CutCellDG.assemble_bulk_transformation_linear_form!(
        sysrhs,
        transfstress,
        basis,
        cellquads,
        mergedmesh,
    )
    CutCellDG.assemble_interelement_condition!(
        sysmatrix,
        basis,
        facequads,
        stiffness,
        mergedmesh,
        penalty,
        eta,
    )
    CutCellDG.assemble_interelement_transformation_linear_form!(
        sysrhs,
        transfstress,
        basis,
        facequads,
        mergedmesh,
    )

    CutCellDG.assemble_coherent_interface_condition!(
        sysmatrix,
        basis,
        interfacequads,
        stiffness,
        mergedmesh,
        penalty,
        eta,
    )
    CutCellDG.assemble_coherent_interface_transformation_linear_form!(
        sysrhs,
        transfstress,
        basis,
        interfacequads,
        mergedmesh,
    )

    CutCellDG.assemble_penalty_displacement_bc!(
        sysmatrix,
        sysrhs,
        analyticalsolution,
        basis,
        facequads,
        stiffness,
        mergedmesh,
        x -> onboundary(x, L, W),
        penalty,
    )
    CutCellDG.assemble_penalty_displacement_transformation_linear_form!(
        sysrhs,
        transfstress,
        basis,
        facequads,
        mergedmesh,
        x -> onboundary(x, L, W),
    )

    matrix = CutCellDG.sparse_displacement_operator(sysmatrix, mergedmesh)
    rhs = CutCellDG.displacement_rhs_vector(sysrhs, mergedmesh)

    solution = matrix \ rhs
    nodaldisplacement = reshape(solution, 2, :)

    err = mesh_L2_error(
        nodaldisplacement,
        analyticalsolution,
        basis,
        cellquads,
        mergedmesh,
    )

    return err
end




lambda1, mu1 = 100.0, 80.0
lambda2, mu2 = 80.0, 60.0
theta0 = -0.067
stiffness = CutCellDG.HookeStiffness(lambda1, mu1, lambda2, mu2)

width = 1.0
penaltyfactor = 1e3

polyorder = 2
numqp = required_quadrature_order(polyorder) + 2

center = [width / 2, width / 2]
inradius = width / 4
outradius = width

powers = [3, 4, 5]
nelmts = [2^p + 1 for p in powers]

err = [
    displacement_error(
        width,
        center,
        inradius,
        outradius,
        stiffness,
        theta0,
        ne,
        polyorder,
        numqp,
        penaltyfactor,
    ) for ne in nelmts
]

dx = 1.0 ./ nelmts
u1err = [er[1] for er in err]
u2err = [er[2] for er in err]

u1rate = convergence_rate(dx,u1err)
u2rate = convergence_rate(dx,u2err)

@test all(u1rate .> 2.8)
@test all(u2rate .> 2.8)




center = [width, width]
inradius = width / 2
outradius = 2width
powers = [3,4,5]
nelmts = [2^p + 1 for p in powers]

err = [
    displacement_error(
        width,
        center,
        inradius,
        outradius,
        stiffness,
        theta0,
        ne,
        polyorder,
        numqp,
        penaltyfactor,
    ) for ne in nelmts
]

dx = 1.0 ./ nelmts
u1err = [er[1] for er in err]
u2err = [er[2] for er in err]

u1rate = convergence_rate(dx,u1err)
u2rate = convergence_rate(dx,u2err)

@test all(u1rate .> 2.9)
@test all(u2rate .> 2.9)
