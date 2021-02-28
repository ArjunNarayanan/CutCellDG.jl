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

function shell_radial_stress(ls, ms, theta0, A1, A2, r)
    return (ls + 2ms) * (A1 - A2 / r^2) + ls * (A1 + A2 / r^2) -
           (ls + 2ms / 3) * theta0
end

function shell_circumferential_stress(ls, ms, theta0, A1, A2, r)
    return ls * (A1 - A2 / r^2) + (ls + 2ms) * (A1 + A2 / r^2) -
           (ls + 2ms / 3) * theta0
end

function shell_out_of_plane_stress(ls, ms, A1, theta0)
    return 2 * ls * A1 - (ls + 2ms / 3) * theta0
end

function core_in_plane_stress(lc, mc, A1)
    return (lc + 2mc) * A1 + lc * A1
end

function core_out_of_plane_stress(lc, A1)
    return 2 * lc * A1
end

function rotation_matrix(x, r)
    costheta = x[1] / r
    sintheta = x[2] / r
    Q = [
        costheta -sintheta
        sintheta costheta
    ]
    return Q
end

function shell_radial_stress(A::AnalyticalSolution, r)
    return shell_radial_stress(A.ls, A.ms, A.theta0, A.A1s, A.A2s, r)
end

function shell_circumferential_stress(A::AnalyticalSolution, r)
    return shell_circumferential_stress(A.ls, A.ms, A.theta0, A.A1s, A.A2s, r)
end

function shell_stress(A::AnalyticalSolution, x)
    relpos = x - A.center
    r = sqrt(relpos' * relpos)
    Q = rotation_matrix(relpos, r)

    srr = shell_radial_stress(A.ls, A.ms, A.theta0, A.A1s, A.A2s, r)
    stt = shell_circumferential_stress(A.ls, A.ms, A.theta0, A.A1s, A.A2s, r)

    cylstress = [
        srr 0.0
        0.0 stt
    ]

    cartstress = Q * cylstress * Q'
    s11 = cartstress[1, 1]
    s22 = cartstress[2, 2]
    s12 = cartstress[1, 2]
    s33 = shell_out_of_plane_stress(A.ls, A.ms, A.A1s, A.theta0)

    return [s11, s22, s12, s33]
end

function core_in_plane_stress(A::AnalyticalSolution)
    return core_in_plane_stress(A.lc, A.mc, A.A1c)
end

function core_stress(A::AnalyticalSolution)
    s11 = core_in_plane_stress(A.lc, A.mc, A.A1c)
    s33 = core_out_of_plane_stress(A.lc, A.A1c)
    return [s11, s11, 0.0, s33]
end

function exact_stress(A::AnalyticalSolution, x)
    relpos = x - A.center
    r = sqrt(relpos' * relpos)
    if r < A.inradius
        return core_stress(A)
    else
        return shell_stress(A, x)
    end
end

function nodal_displacement(
    mesh,
    basis,
    cellquads,
    facequads,
    interfacequads,
    stiffness,
    theta0,
    boundarydisplacement,
    penalty;
    eta = 1,
)

    L, W = CutCellDG.widths(mesh)
    lambda1, mu1 = CutCellDG.lame_coefficients(stiffness, +1)
    transfstress =
        CutCellDG.plane_strain_transformation_stress(lambda1, mu1, theta0)

    sysmatrix = CutCellDG.SystemMatrix()
    sysrhs = CutCellDG.SystemRHS()

    CutCellDG.assemble_displacement_bilinear_forms!(
        sysmatrix,
        basis,
        cellquads,
        stiffness,
        mesh,
    )
    CutCellDG.assemble_bulk_transformation_linear_form!(
        sysrhs,
        transfstress,
        basis,
        cellquads,
        mesh,
    )
    CutCellDG.assemble_interelement_condition!(
        sysmatrix,
        basis,
        facequads,
        stiffness,
        mesh,
        penalty,
        eta,
    )
    CutCellDG.assemble_interelement_transformation_linear_form!(
        sysrhs,
        transfstress,
        basis,
        facequads,
        mesh,
    )

    CutCellDG.assemble_coherent_interface_condition!(
        sysmatrix,
        basis,
        interfacequads,
        stiffness,
        mesh,
        penalty,
        eta,
    )
    CutCellDG.assemble_coherent_interface_transformation_linear_form!(
        sysrhs,
        transfstress,
        basis,
        interfacequads,
        mesh,
    )

    CutCellDG.assemble_penalty_displacement_bc!(
        sysmatrix,
        sysrhs,
        boundarydisplacement,
        basis,
        facequads,
        stiffness,
        mesh,
        x -> onboundary(x, L, W),
        penalty,
    )
    CutCellDG.assemble_penalty_displacement_transformation_linear_form!(
        sysrhs,
        transfstress,
        basis,
        facequads,
        mesh,
        x -> onboundary(x, L, W),
    )

    matrix = CutCellDG.sparse_displacement_operator(sysmatrix, mesh)
    rhs = CutCellDG.displacement_rhs_vector(sysrhs, mesh)

    solution = matrix \ rhs

    return solution
end

function construct_mesh_and_quadratures(
    meshwidth,
    nelmts,
    basis,
    interfacecenter,
    interfaceradius,
    numqp;
    tinyratio = 0.2,
)
    mesh = CutCellDG.DGMesh([0.0, 0.0], meshwidth, [nelmts, nelmts], basis)

    levelset = InterpolatingPolynomial(1, basis)
    levelsetcoeffs = CutCellDG.levelset_coefficients(
        x -> -circle_distance_function(x, interfacecenter, interfaceradius),
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
        tinyratio = tinyratio,
    )
    mergedmesh = CutCellDG.MergedMesh(cutmesh, mergedwithcell)

    return mergedmesh, cellquads, facequads, interfacequads
end

function construct_unmerged_mesh_and_quadratures(
    meshwidth,
    nelmts,
    basis,
    interfacecenter,
    interfaceradius,
    numqp,
)
    mesh = CutCellDG.DGMesh([0.0, 0.0], meshwidth, [nelmts, nelmts], basis)

    levelset = InterpolatingPolynomial(1, basis)
    levelsetcoeffs = CutCellDG.levelset_coefficients(
        x -> -circle_distance_function(x, interfacecenter, interfaceradius),
        mesh,
    )

    cutmesh = CutCellDG.CutMesh(mesh, levelset, levelsetcoeffs)
    cellquads =
        CutCellDG.CellQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)
    interfacequads =
        CutCellDG.InterfaceQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)
    facequads =
        CutCellDG.FaceQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)

    return cutmesh, cellquads, facequads, interfacequads
end


function product_stress(
    celldisp,
    basis,
    stiffness,
    transfstress,
    theta0,
    point,
    jac,
    vectosymmconverter,
)

    dim = length(vectosymmconverter)
    lambda, mu = CutCellDG.lame_coefficients(stiffness, +1)

    grad = CutCellDG.transform_gradient(gradient(basis, point), jac)
    NK = sum([
        CutCellDG.make_row_matrix(vectosymmconverter[k], grad[:, k]) for
        k = 1:dim
    ])
    symmdispgrad = NK * celldisp

    inplanestress = (stiffness[+1] * symmdispgrad) - transfstress
    s33 =
        lambda * (symmdispgrad[1] + symmdispgrad[2]) -
        (lambda + 2mu / 3) * theta0

    stress = vcat(inplanestress, s33)

    return stress
end

function product_stress_at_reference_points(
    nodaldisplacement,
    basis,
    stiffness,
    transfstress,
    theta0,
    referencepoints,
    referencecellids,
    mesh,
)

    dim = CutCellDG.dimension(mesh)
    dim2, numpts = size(referencepoints)
    @assert dim == dim2
    @assert length(referencecellids) == numpts

    productstress = zeros(4, numpts)
    vectosymmconverter = CutCellDG.vector_to_symmetric_matrix_converter()
    jac = CutCellDG.jacobian(mesh)

    for i = 1:numpts
        cellid = referencecellids[i]
        nodeids = CutCellDG.nodal_connectivity(mesh, +1, cellid)
        celldofs = CutCellDG.element_dofs(nodeids, dim)
        celldisp = nodaldisplacement[celldofs]
        point = referencepoints[:, i]

        productstress[:, i] .= product_stress(
            celldisp,
            basis,
            stiffness,
            transfstress,
            theta0,
            point,
            jac,
            vectosymmconverter,
        )
    end
    return productstress
end


function parent_stress(
    celldisp,
    basis,
    stiffness,
    point,
    jac,
    vectosymmconverter,
)
    dim = length(vectosymmconverter)
    lambda, mu = CutCellDG.lame_coefficients(stiffness, -1)

    grad = CutCellDG.transform_gradient(gradient(basis, point), jac)
    NK = sum([
        CutCellDG.make_row_matrix(vectosymmconverter[k], grad[:, k]) for
        k = 1:dim
    ])
    symmdispgrad = NK * celldisp

    inplanestress = stiffness[-1] * symmdispgrad
    s33 = lambda * (symmdispgrad[1] + symmdispgrad[2])

    stress = vcat(inplanestress, s33)

    return stress
end

function parent_stress_at_reference_points(
    nodaldisplacement,
    basis,
    stiffness,
    referencepoints,
    referencecellids,
    mesh,
)

    dim = CutCellDG.dimension(mesh)
    dim2, numpts = size(referencepoints)
    @assert dim == dim2
    @assert length(referencecellids) == numpts

    parentstress = zeros(4, numpts)
    vectosymmconverter = CutCellDG.vector_to_symmetric_matrix_converter()
    jac = CutCellDG.jacobian(mesh)

    for i = 1:numpts
        cellid = referencecellids[i]
        nodeids = CutCellDG.nodal_connectivity(mesh, -1, cellid)
        celldofs = CutCellDG.element_dofs(nodeids, dim)
        celldisp = nodaldisplacement[celldofs]
        point = referencepoints[:, i]

        parentstress[:, i] .= parent_stress(
            celldisp,
            basis,
            stiffness,
            point,
            jac,
            vectosymmconverter,
        )
    end
    return parentstress
end

function pressure(stress)
    return -1.0 / 3.0 * (stress[1, :] + stress[2, :] + stress[4, :])
end

function deviatoric_stress(stress, press)
    devstress = copy(stress)
    devstress[1, :] .+= press
    devstress[2, :] .+= press
    devstress[4, :] .+= press
    return devstress
end
