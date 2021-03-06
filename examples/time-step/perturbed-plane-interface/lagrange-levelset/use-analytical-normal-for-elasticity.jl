using PyPlot
using Statistics
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCellDG
include("../../../../test/useful_routines.jl")
include("../transformation-elasticity-solver.jl")

TES = TransformationElasticitySolver

function perturbation(x, frequency, amplitude)
    return amplitude * sin.(2 * pi * frequency * x)
end

perturbed_distancefunction(x, initialposition, frequency, amplitude) =
    plane_distance_function(x, [1.0, 0.0], [initialposition, 0.0]) +
    perturbation(x[2, :], frequency, amplitude)

function perturbed_normal(y, frequency, amplitude)
    ny = 2pi * amplitude * frequency * cos.(2pi * frequency * y)
    magnitude = sqrt.(1.0 .+ ny .^ 2)
    numpts = length(y)
    nx = 1.0 ./ magnitude
    ny = ny ./ magnitude
    normals = vcat(nx', ny')
    return normals
end

function plot_levelset_normal_and_exact_normal(
    ycoords,
    levelsetnormal,
    exactnormal;
    filepath = "",
)
    fig, ax = PyPlot.subplots(2, 1, sharex = true)
    ax[1].plot(ycoords, levelsetnormal[1, :], label = "levelset")
    ax[1].plot(ycoords, exactnormal[1, :], "--", label = "exact")
    ax[1].set_title("X-component of interface normal")
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(ycoords, levelsetnormal[2, :])
    ax[2].plot(ycoords, exactnormal[2, :], "--")
    ax[2].set_title("Y-component of interface normal")
    ax[2].grid()

    if length(filepath) > 0
        fig.savefig(filepath)
        return fig
    else
        return fig
    end
end

function update_cell_interface_normals_tangents_areas!(
    interfacequads,
    cellid,
    normals,
    invjac,
)
    t = CutCellDG.rotate_90(normals)
    s = CutCellDG.scale_area(t, invjac)

    idx = interfacequads.celltoquad[cellid]
    interfacequads.normals[idx] .= normals
    interfacequads.tangents[idx] .= t
    interfacequads.scaleareas[idx] .= s
end

function update_interface_normals!(interfacequads, cutmesh, exactnormalfunc)
    ncells = CutCellDG.number_of_cells(cutmesh)
    cellsign = [CutCellDG.cell_sign(cutmesh, cellid) for cellid = 1:ncells]
    cellids = findall(cellsign .== 0)
    invjac = CutCellDG.inverse_jacobian(cutmesh)

    for cellid in cellids
        cellmap = CutCellDG.cell_map(cutmesh, cellid)
        qps = cellmap(CutCellDG.points(interfacequads[1, cellid]))
        exactnormals = exactnormalfunc(qps[2, :])
        update_cell_interface_normals_tangents_areas!(
            interfacequads,
            cellid,
            exactnormals,
            invjac,
        )
    end
end

function plot_normal_stress(ycoords, parentsrr, productsrr; filepath = "")

    fig, ax = PyPlot.subplots()
    ax.plot(ycoords, parentsrr, label = "parent")
    ax.plot(ycoords, productsrr, label = "product")
    ax.grid()
    ax.set_ylabel(L"\sigma_{nn}")
    ax.set_xlabel("y")
    ax.legend()

    if length(filepath) > 0
        fig.savefig(filepath)
        return fig
    else
        return fig
    end
end


distancefunction(x) =
    perturbed_distancefunction(x, initialposition, frequency, amplitude)


initialposition = 0.5
frequency = 2.5
amplitude = 1e-3
# amplitude = 1e-10
polyorder = 2
nelmts = 17
penaltyfactor = 1e3
meshwidth = [1.0, 1.0]
numqp = required_quadrature_order(polyorder) + 2

K1, K2 = 247.0, 247.0    # GPa
mu1, mu2 = 126.0, 126.0   # GPa

lambda1 = lame_lambda(K1, mu1)
lambda2 = lame_lambda(K2, mu2)
stiffness = CutCellDG.HookeStiffness(lambda1, mu1, lambda2, mu2)
rho1 = 3.93e3           # Kg/m^3
rho2 = 3.68e3           # Kg/m^3
V01 = 1.0 / rho1
V02 = 1.0 / rho2
ΔG0Jmol = -14351.0
molarmass = 0.147
ΔG0 = ΔG0Jmol / molarmass
theta0 = -0.067




numquerypoints = 1000
querypoints = vcat(
    repeat([initialposition], numquerypoints)',
    range(0, 1, length = numquerypoints)',
)
ycoords = querypoints[2, :]



transfstress =
    CutCellDG.plane_strain_transformation_stress(lambda1, mu1, theta0)

elasticitybasis = LagrangeTensorProductBasis(2, polyorder)
levelsetbasis = HermiteTensorProductBasis(2)
quad = tensor_product_quadrature(2, 4)
dim, numpts = size(interpolation_points(levelsetbasis))
basispts = interpolation_points(elasticitybasis)

cgmesh = CutCellDG.CGMesh([0.0, 0.0], meshwidth, [nelmts, nelmts], numpts)
dgmesh = CutCellDG.DGMesh([0.0, 0.0], meshwidth, [nelmts, nelmts], basispts)

# levelset = CutCellDG.LevelSet(distancefunction, cgmesh, levelsetbasis)
levelset = CutCellDG.LevelSet(x->distancefunction(x)[1], cgmesh, levelsetbasis)

elementsize = CutCellDG.element_size(cgmesh)
minelmtsize = minimum(elementsize)
maxelmtsize = maximum(elementsize)
penalty = penaltyfactor / minelmtsize * 0.5 * (lambda1 + mu1 + lambda2 + mu2)
tol = minelmtsize^(polyorder + 1)
boundingradius = 1.5 * maxelmtsize

cutmesh = CutCellDG.CutMesh(dgmesh, levelset)
refseedpoints, seedcellids = CutCellDG.seed_zero_levelset(2, levelset, cutmesh)
spatialseedpoints =
    CutCellDG.map_to_spatial(refseedpoints, seedcellids, cutmesh)

mesh, cellquads, facequads, interfacequads =
    TES.construct_merged_mesh_and_quadratures(cutmesh, levelset, numqp)
update_interface_normals!(
    interfacequads,
    cutmesh,
    x -> perturbed_normal(x, frequency, amplitude),
)

interfacenormals = CutCellDG.collect_interface_normals(interfacequads, cutmesh)
refinterfaceqps, refcellids =
    CutCellDG.collect_interface_quadrature_points(interfacequads, +1, cutmesh)
interfaceqps = CutCellDG.map_to_spatial_on_merged_mesh(
    refinterfaceqps,
    refcellids,
    +1,
    mesh,
)
exactqpnormals = perturbed_normal(interfaceqps[2, :], frequency, amplitude)



nodaldisplacement = TES.nodal_displacement(
    mesh,
    elasticitybasis,
    cellquads,
    facequads,
    interfacequads,
    stiffness,
    theta0,
    penalty,
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
exactnormals = perturbed_normal(closestpoints[2, :], frequency, amplitude)

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


################################################################################
parentstrain = CutCellDG.parent_strain(
    nodaldisplacement,
    elasticitybasis,
    parentclosestrefpoints,
    closestcellids,
    mesh,
)
parentstress = CutCellDG.parent_stress(parentstrain, stiffness)

parentradialtraction = CutCellDG.traction_force_at_points(parentstress, normals)
parentsrr = CutCellDG.traction_component(parentradialtraction, normals)

parentradialtractionalt =
    CutCellDG.traction_force_at_points(parentstress, exactnormals)
parentsrralt =
    CutCellDG.traction_component(parentradialtractionalt, exactnormals)
################################################################################
#
#
#
#
#
# ################################################################################
productstrain = CutCellDG.product_elastic_strain(
    nodaldisplacement,
    elasticitybasis,
    theta0,
    productclosestrefpoints,
    closestcellids,
    mesh,
)
productstress = CutCellDG.product_stress(productstrain, stiffness, theta0)

productradialtraction =
    CutCellDG.traction_force_at_points(productstress, normals)
productsrr = CutCellDG.traction_component(productradialtraction, normals)

productradialtractionalt =
    CutCellDG.traction_force_at_points(productstress, exactnormals)
productsrralt =
    CutCellDG.traction_component(productradialtractionalt, exactnormals)
# ################################################################################
foldername = "examples\\time-step\\perturbed-plane-interface\\analytical-normal-for-elasticity\\"

plot_normal_stress(ycoords,parentsrr,productsrr)
