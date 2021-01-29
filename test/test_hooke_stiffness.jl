using Test
# using Revise
using CutCellDG
include("useful_routines.jl")

l,m = 1.,2.
stiffness = CutCellDG.plane_strain_voigt_hooke_matrix(l,m)
teststiffness = [5. 1. 0.
                 1. 5. 0.
                 0. 0. 2.]
@test allapprox(teststiffness,stiffness)

stiffness = CutCellDG.HookeStiffness(l,m,l,m)
@test allapprox(teststiffness,stiffness[-1])
@test allapprox(teststiffness,stiffness[+1])

l1,m1 = 1.,2.
l2,m2 = 3.,4.
stiffness = CutCellDG.HookeStiffness(l1,m1,l2,m2)
teststiffness2 = [11.  3.  0.
                   3.  11. 0.
                   0.   0. 4.]
teststiffness1 = [5. 1. 0.
                  1. 5. 0.
                  0. 0. 2.]
@test allapprox(teststiffness1,stiffness[1])
@test allapprox(teststiffness2,stiffness[-1])

@test allapprox((l1,m1),CutCellDG.lame_coefficients(stiffness,1))
@test allapprox((l2,m2),CutCellDG.lame_coefficients(stiffness,-1))
