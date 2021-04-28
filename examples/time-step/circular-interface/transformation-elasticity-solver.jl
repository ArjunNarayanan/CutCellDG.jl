module TransformationElasticitySolver
using CutCellDG
using LinearAlgebra
include("analytical-solver.jl")
PS = PlaneStrainSolver

function bulk_modulus(l, m)
    return l + 2m / 3
end

function onboundary(x, L, W)
    return x[2] ≈ 0.0 || x[1] ≈ L || x[2] ≈ W || x[1] ≈ 0.0
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

    L, W = CutCellDG.mesh_widths(mesh)
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

function construct_merged_mesh_and_quadratures(
    cutmesh,
    levelset,
    numqp;
    tinyratio = 0.3,
)
    cellquads = CutCellDG.CellQuadratures(cutmesh, levelset, numqp)
    interfacequads = CutCellDG.InterfaceQuadratures(cutmesh, levelset, numqp)
    facequads = CutCellDG.FaceQuadratures(cutmesh, levelset, numqp)

    mergedmesh = CutCellDG.MergedMesh!(
        cutmesh,
        cellquads,
        facequads,
        interfacequads,
        tinyratio = tinyratio,
    )

    return mergedmesh, cellquads, facequads, interfacequads
end

function parent_potential(
    nodaldisplacement,
    basis,
    refpoints,
    refcellids,
    normals,
    mesh,
    stiffness,
    V0,
)
    parentstrain = CutCellDG.parent_strain(
        nodaldisplacement,
        basis,
        refpoints,
        refcellids,
        mesh,
    )
    parentstress = CutCellDG.parent_stress(parentstrain, stiffness)
    parentstrainenergy =
        V0 * CutCellDG.strain_energy(parentstress, parentstrain)

    parentradialtraction =
        CutCellDG.traction_force_at_points(parentstress, normals)
    parentsrr = CutCellDG.traction_component(parentradialtraction, normals)
    parentdilatation = CutCellDG.dilatation(parentstrain)

    parentcompwork = V0 * (1.0 .+ parentdilatation) .* parentsrr

    parentpotential = parentstrainenergy - parentcompwork
    return parentpotential
end

function product_potential(
    nodaldisplacement,
    basis,
    refpoints,
    refcellids,
    normals,
    mesh,
    stiffness,
    theta0,
    V0,
)

    productstrain = CutCellDG.product_elastic_strain(
        nodaldisplacement,
        basis,
        theta0,
        refpoints,
        refcellids,
        mesh,
    )
    productstress = CutCellDG.product_stress(productstrain, stiffness, theta0)
    productstrainenergy =
        V0 * CutCellDG.strain_energy(productstress, productstrain)
    productradialtraction =
        CutCellDG.traction_force_at_points(productstress, normals)
    productsrr = CutCellDG.traction_component(productradialtraction, normals)
    productdilatation = CutCellDG.dilatation(productstrain)
    productcompwork = V0 * (1.0 .+ productdilatation) .* productsrr

    productpotential = productstrainenergy - productcompwork

    return productpotential
end

function potential_difference_at_closest_points(
    querypoints,
    nodaldisplacement,
    basis,
    spatialseedpoints,
    seedcellids,
    mesh,
    levelset,
    stiffness,
    theta0,
    V01,
    V02,
    tol,
    boundingradius,
)

    closestpoints, closestcellids = CutCellDG.closest_points_on_zero_levelset(
        querypoints,
        spatialseedpoints,
        seedcellids,
        levelset,
        tol,
        boundingradius,
    )

    normals = CutCellDG.collect_normals_at_spatial_points(
        closestpoints,
        closestcellids,
        levelset,
    )

    parentclosestrefpoints = CutCellDG.map_to_reference_on_merged_mesh(
        closestpoints,
        closestcellids,
        -1,
        mesh,
    )
    productclosestrefpoints = CutCellDG.map_to_reference_on_merged_mesh(
        closestpoints,
        closestcellids,
        +1,
        mesh,
    )

    parentpotential = parent_potential(
        nodaldisplacement,
        basis,
        parentclosestrefpoints,
        closestcellids,
        normals,
        mesh,
        stiffness,
        V02,
    )
    productpotential = product_potential(
        nodaldisplacement,
        basis,
        productclosestrefpoints,
        closestcellids,
        normals,
        mesh,
        stiffness,
        theta0,
        V01,
    )

    return productpotential - parentpotential
end

function average(v)
    return sum(v) / length(v)
end

function time_step_size(levelsetspeed, dx, CFL)
    s = maximum(abs.(levelsetspeed))
    return CFL * dx / s
end

function potential_difference_at_nodal_coordinates(
    cutmesh,
    basis,
    levelset,
    stiffness,
    theta0,
    boundarydisplacement,
    numqp,
    penalty,
    spatialseedpoints,
    seedcellids,
    V01,
    V02,
    ΔG0,
    tol,
    boundingradius,
)

    mesh, cellquads, facequads, interfacequads =
        construct_merged_mesh_and_quadratures(cutmesh, levelset, numqp)

    nodaldisplacement = nodal_displacement(
        mesh,
        basis,
        cellquads,
        facequads,
        interfacequads,
        stiffness,
        theta0,
        boundarydisplacement,
        penalty,
    )

    querypoints =
        CutCellDG.nodal_coordinates(CutCellDG.background_mesh(levelset))
    potentialdifference =
        (
            ΔG0 .+
            1e9 * potential_difference_at_closest_points(
                querypoints,
                nodaldisplacement,
                basis,
                spatialseedpoints,
                seedcellids,
                mesh,
                levelset,
                stiffness,
                theta0,
                V01,
                V02,
                tol,
                boundingradius,
            )
        ) / abs(ΔG0)

    return potentialdifference
end

function step_levelset(
    levelset,
    levelsetspeed,
    spatialseedpoints,
    seedcellids,
    paddedmesh,
    tol,
    boundingradius,
    CFL,
)

    paddedlevelset = CutCellDG.BoundaryPaddedLevelSet(
        paddedmesh,
        spatialseedpoints,
        seedcellids,
        levelset,
        tol,
        boundingradius,
    )

    dx = minimum(CutCellDG.grid_size(paddedmesh))
    dt = time_step_size(levelsetspeed, dx, CFL)

    newcoeffs =
        CutCellDG.step_first_order_levelset(paddedlevelset, levelsetspeed, dt)

    return newcoeffs
end

function step_interface(
    dgmesh,
    basis,
    levelset,
    stiffness,
    theta0,
    V01,
    V02,
    ΔG0,
    numqp,
    interfacecenter,
    outerradius;
    tol = 1e4eps(),
    boundingradius = 4.5,
    penaltyfactor = 1e3,
    CFL = 0.5,
    maxiter = 50,
)

    cutmesh = CutCellDG.CutMesh(dgmesh, levelset)
    refseedpoints, refseedcellids =
        CutCellDG.seed_zero_levelset(2, levelset, cutmesh)
    spatialseedpoints =
        CutCellDG.map_to_spatial(refseedpoints, refseedcellids, cutmesh)

    lambda1, mu1 = CutCellDG.lame_coefficients(stiffness, +1)
    lambda2, mu2 = CutCellDG.lame_coefficients(stiffness, -1)

    dx = minimum(CutCellDG.element_size(dgmesh))
    penalty = penaltyfactor / dx * 0.5 * (lambda1 + mu1 + lambda2 + mu2)

    interfaceradius =
        average(mapslices(norm, spatialseedpoints .- interfacecenter, dims = 1))
    analyticalsolution = PS.CylindricalSolver(
        interfaceradius,
        outerradius,
        interfacecenter,
        lambda1,
        mu1,
        lambda2,
        mu2,
        theta0,
    )

    potentialdifference = potential_difference_at_nodal_coordinates(
        cutmesh,
        basis,
        levelset,
        refseedpoints,
        refseedcellids,
        spatialseedpoints,
        stiffness,
        theta0,
        V01,
        V02,
        ΔG0,
        numqp,
        penalty,
        analyticalsolution,
        tol,
        boundingradius,
        maxiter,
    )

    newcoeffs = step_levelset(
        levelset,
        potentialdifference,
        refseedpoints,
        refseedcellids,
        spatialseedpoints,
        tol,
        boundingradius,
        CFL,
    )

    return newcoeffs
end


end
