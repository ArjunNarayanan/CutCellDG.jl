using Plots
using ImplicitDomainQuadrature

quad = tensor_product_quadrature(2,5)
points = quad.points
rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

fig = plot(legend=false,aspect_ratio=:equal)
plot!(fig,rectangle(2,2,-1,-1),opacity=0.2,fillcolor="red")
scatter!(points[1,:],points[2,:])
folderpath = "examples\\FYRP-Presentation\\"
Plots.savefig(fig,folderpath*"tensor-product-quad.png")
