using PyPlot
using ImplicitDomainQuadrature

xrange = -1:0.2:1
points = ImplicitDomainQuadrature.tensor_product_points(xrange',xrange')

fig,ax = PyPlot.subplots()
ax.scatter(points[1,:],points[2,:])
ax.set_aspect("equal")
ax.grid()
fig.savefig("grid.png")
