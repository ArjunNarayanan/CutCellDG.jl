using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCellDG
include("../../test/useful_routines.jl")
include("analytical-solver.jl")
PS = PlaneStrainSolver
include("transformation-elasticity-solver.jl")

function lame_lambda(k, m)
    return k - 2m / 3
end

function angular_position(points)
    cpoints = points[1, :] + im * points[2, :]
    return rad2deg.(angle.(cpoints))
end

polyorder = 3
nelmts = 17
penaltyfactor = 1e1

width = 1.0              # mm
K1, K2 = 247.0, 192.0    # GPa
mu1, mu2 = 126.0, 87.0   # GPa

lambda1 = lame_lambda(K1, mu1)
lambda2 = lame_lambda(K2, mu2)
stiffness = CutCellDG.HookeStiffness(lambda1, mu1, lambda2, mu2)

rho1 = 3.93e3           # Kg/m^3
rho2 = 3.68e3           # Kg/m^3
V01 = 1.0 / rho1
V02 = 1.0 / rho2

ΔG0Jmol = -14351.0  # J/mol
molarmass = 0.147
ΔG0 = ΔG0Jmol / molarmass

theta0 = -0.067
transfstress =
    CutCellDG.plane_strain_transformation_stress(lambda1, mu1, theta0)

meshwidth = [width, width]
numqp = required_quadrature_order(polyorder) + 2

dx = width / nelmts
penalty = penaltyfactor / dx * 0.5 * (lambda1 + mu1 + lambda2 + mu2)

basis = TensorProductBasis(2, polyorder)
interfacecenter = [0.0, 0.0]
interfaceradius = 0.5
outerradius = 1.5
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

mesh, cellquads, facequads, interfacequads, levelset =
    construct_mesh_and_quadratures(
        meshwidth,
        nelmts,
        basis,
        x -> -circle_distance_function(x, interfacecenter, interfaceradius),
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

refseedpoints, refseedcellids =
    CutCellDG.seed_zero_levelset_with_interfacequads(interfacequads, mesh)
spatialseedpoints = CutCellDG.map_to_spatial(
    refseedpoints[1, :, :],
    refseedcellids[1, :],
    CutCellDG.background_mesh(mesh),
)

nodalcoordinates =
    CutCellDG.nodal_coordinates(CutCellDG.background_mesh(levelset))

tol = 1e-12
boundingradius = 4.5
refclosestpoints, refclosestcellids =
    CutCellDG.closest_reference_points_on_merged_mesh(
        nodalcoordinates,
        refseedpoints,
        spatialseedpoints,
        refseedcellids,
        levelset,
        mesh,
        tol,
        boundingradius,
    )

parentrefclosestpoints = refclosestpoints[2, :, :]
parentrefclosestcellids = refclosestcellids[2, :]

productrefclosestpoints = refclosestpoints[1, :, :]
productrefclosestcellids = refclosestcellids[1, :]

parentstrain = CutCellDG.parent_strain(
    nodaldisplacement,
    basis,
    parentrefclosestpoints,
    parentrefclosestcellids,
    mesh,
)
productstrain = CutCellDG.product_elastic_strain(
    nodaldisplacement,
    basis,
    theta0,
    productrefclosestpoints,
    productrefclosestcellids,
    mesh,
)

parentstress = CutCellDG.parent_stress(parentstrain,stiffness)
productstress = CutCellDG.product_stress(productstrain,stiffness,theta0)


parentstrainenergy = V02*CutCellDG.strain_energy(parentstress,parentstrain)
productstrainenergy = V01*CutCellDG.strain_energy(productstress,productstrain)

parentradialtraction = CutCellDG.traction_force_at_points(parentstress, normals)
parentsrr = CutCellDG.traction_component(parentradialtraction, normals)

productradialtraction =
    CutCellDG.traction_force_at_points(productstress, normals)
productsrr = CutCellDG.traction_component(productradialtraction, normals)
