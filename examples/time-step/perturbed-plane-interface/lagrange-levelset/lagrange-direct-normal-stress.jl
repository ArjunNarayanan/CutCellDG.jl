using PyPlot
using Statistics
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCellDG
include("../../../../test/useful_routines.jl")
include("../transformation-elasticity-solver.jl")
# include("analytical-solver.jl")
TES = TransformationElasticitySolver
# APS = AnalyticalPlaneSolver

function perturbation(x, frequency, amplitude)
    return amplitude * sin.(2 * pi * frequency * x)
end

perturbed_distancefunction(x, initialposition, frequency, amplitude) =
    plane_distance_function(x, [1.0, 0.0], [initialposition, 0.0]) +
    perturbation(x[2, :], frequency, amplitude)

function plot_potential_components(
    ycoords,
    component,
    initialposition,
    frequency,
    amplitude,
    interfacescale,
    pdscale;
    filepath = "",
    ylabel = "",
)
    interfaceposition =
        initialposition .+ perturbation.(ycoords, frequency, amplitude)
    planeinterfacepd = mean(component)

    interfaceylim = interfacescale * amplitude
    pdylim = pdscale

    fig, ax = PyPlot.subplots(2, 1, sharex = true)
    ax[1].plot(ycoords, interfaceposition, color = "black")
    ax[1].set_ylim(
        initialposition - interfaceylim,
        initialposition + interfaceylim,
    )
    ax[1].grid()
    ax[1].set_ylabel("Interface position")

    ax[2].plot(ycoords, component, color = "black")
    ax[2].set_ylim(planeinterfacepd - pdylim, planeinterfacepd + pdylim)
    ax[2].grid()
    ax[2].set_ylabel(ylabel)
    ax[2].set_xlabel("y")
    fig.tight_layout()

    if length(filepath) > 0
        fig.savefig(filepath)
        return fig
    else
        return fig
    end
end

function plot_dilatation_and_normal_stress(
    ycoords,
    parentdilatation,
    productdilatation,
    parentsrr,
    productsrr;
    filepath = "",
)

    fig, ax = PyPlot.subplots(2, 1, sharex = true)
    ax[1].plot(ycoords, 1.0 .+ parentdilatation, label = "parent")
    ax[1].plot(ycoords, 1.0 .+ productdilatation, label = "product")
    ax[1].grid()
    ax[1].legend()
    ax[1].set_ylabel(L"1 + \epsilon_{kk}")

    ax[2].plot(ycoords, parentsrr, label = "parent")
    ax[2].plot(ycoords, productsrr, label = "product")
    ax[2].grid()
    ax[2].set_ylabel(L"\sigma_{nn}")
    ax[2].set_xlabel("y")

    if length(filepath) > 0
        fig.savefig(filepath)
        return fig
    else
        return fig
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
amplitude = 1e-2
# amplitude = 1e-10
elasticityorder = 2
levelsetorder = 2
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

levelsetbasis = LagrangeTensorProductBasis(2,levelsetorder)
solverbasis = LagrangeTensorProductBasis(2, elasticityorder)
basispts = interpolation_points(solverbasis)
dim, numpts = size(interpolation_points(levelsetbasis))
cgmesh = CutCellDG.CGMesh([0.0, 0.0], meshwidth, [nelmts, nelmts], numpts)
dgmesh = CutCellDG.DGMesh([0.0, 0.0], meshwidth, [nelmts, nelmts], basispts)
levelset = CutCellDG.LevelSet(distancefunction, cgmesh, levelsetbasis)

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

nodaldisplacement = TES.nodal_displacement(
    mesh,
    solverbasis,
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
    solverbasis,
    parentclosestrefpoints,
    closestcellids,
    mesh,
)
parentstress = CutCellDG.parent_stress(parentstrain, stiffness)
parentstrainenergy = V02 * CutCellDG.strain_energy(parentstress, parentstrain)

parentradialtraction = CutCellDG.traction_force_at_points(parentstress, normals)
parentsrr = CutCellDG.traction_component(parentradialtraction, normals)
parentdilatation = CutCellDG.dilatation(parentstrain)

parentcompwork = V02 * (1.0 .+ parentdilatation) .* parentsrr
################################################################################





################################################################################
productstrain = CutCellDG.product_elastic_strain(
    nodaldisplacement,
    solverbasis,
    theta0,
    productclosestrefpoints,
    closestcellids,
    mesh,
)
productstress = CutCellDG.product_stress(productstrain, stiffness, theta0)
productstrainenergy =
    V01 * CutCellDG.strain_energy(productstress, productstrain)
productradialtraction =
    CutCellDG.traction_force_at_points(productstress, normals)
productsrr = CutCellDG.traction_component(productradialtraction, normals)
productdilatation = CutCellDG.dilatation(productstrain)
productcompwork = V01 * (1.0 .+ productdilatation) .* productsrr
################################################################################

srrmean = 0.5*(parentsrr + productsrr)

# NOTES: COMPUTE STRAIN ENERGY JUMP AND COMP WORK JUMP
jse = productstrainenergy - parentstrainenergy
jsediff = (maximum(jse) - minimum(jse)) / 2

jcw = productcompwork - parentcompwork
# jcw =
#     (V01 * (1.0 .+ productdilatation) - V02 * (1.0 .+ parentdilatation)) .*
#     srrmean
jcwdiff = (maximum(jcw) - minimum(jcw)) / 2

pd = jse - jcw
pddiff = (maximum(pd) - minimum(pd)) / 2

foldername = "examples\\time-step\\perturbed-plane-interface\\lagrange-levelset\\potential-components\\"


################################################################################

filename = foldername*"lagrange-levelset-"*string(levelsetorder)*"-normal-stress.png"
plot_normal_stress(ycoords,parentsrr,productsrr,filepath=filename)



# interfacescale = 5
# pdscale = 3e-6
# plot_potential_components(
#     ycoords,
#     pd,
#     initialposition,
#     frequency,
#     amplitude,
#     interfacescale,
#     pdscale,
# )


# pdscale = 8e-6
# plot_potential_components(
#     ycoords,
#     jse,
#     initialposition,
#     frequency,
#     amplitude,
#     interfacescale,
#     pdscale,
#     ylabel = "Jump in strain energy",
#     # filepath = foldername * "\\strain-energy-jump.png",
# )

# pdscale = 8e-6
# plot_potential_components(
#     ycoords,
#     jcw,
#     initialposition,
#     frequency,
#     amplitude,
#     interfacescale,
#     pdscale,
#     ylabel = "Jump in compression work",
#     # filepath = foldername * "\\compression-work-jump.png",
# )

# plot_dilatation_and_normal_stress(
#     ycoords,
#     parentdilatation,
#     productdilatation,
#     parentsrr,
#     productsrr,
#     filepath = foldername * "\\dilatation-and-normal-stress.png",
# )
#
# parentsrrdiff = (maximum(parentsrr) - minimum(parentsrr)) / 2
# yscale = 0.3
# plot_potential_components(
#     ycoords,
#     parentsrr,
#     initialposition,
#     frequency,
#     amplitude,
#     interfacescale,
#     yscale,
#     ylabel = L"\mathrm{Parent} \ \sigma_{nn}",
# )
#
#
#
# productsrrdiff = (maximum(productsrr) - minimum(productsrr)) / 2
# yscale = 0.3
# plot_potential_components(
#     ycoords,
#     productsrr,
#     initialposition,
#     frequency,
#     amplitude,
#     interfacescale,
#     yscale,
#     ylabel = L"\mathrm{Product} \ \sigma_{nn}",
# )
