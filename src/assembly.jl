struct SystemMatrix
    rows
    cols
    vals
    function SystemMatrix(rows,cols,vals)
        @assert length(rows) == length(cols) == length(vals)
        new(rows,cols,vals)
    end
end

function SystemMatrix()
    rows = Int[]
    cols = Int[]
    vals = zeros(0)
    SystemMatrix(rows, cols, vals)
end

function Base.show(io::IO, sysmatrix::SystemMatrix)
    numvals = length(sysmatrix.rows)
    str = "SystemMatrix with $numvals entries"
    print(io, str)
end

function assemble!(matrix, rows, cols, vals)
    @assert length(rows) == length(cols) == length(vals)
    append!(matrix.rows, rows)
    append!(matrix.cols, cols)
    append!(matrix.vals, vals)
end

struct SystemRHS
    rows::Any
    vals::Any
    function SystemRHS(rows, vals)
        @assert length(rows) == length(vals)
        new(rows, vals)
    end
end

function SystemRHS()
    SystemRHS(Int[], zeros(0))
end

function Base.show(io::IO, sysrhs::SystemRHS)
    numvals = length(sysrhs.rows)
    str = "SystemRHS with $numvals entries"
    print(io, str)
end

function assemble!(systemrhs, rows, vals)
    @assert length(rows) == length(vals)
    append!(systemrhs.rows, rows)
    append!(systemrhs.vals, vals)
end

function node_to_dof_id(nodeid, dof, dofspernode)
    return (nodeid - 1) * dofspernode + dof
end

function element_dofs(nodeids, dofspernode)
    numnodes = length(nodeids)
    extnodeids = repeat(nodeids, inner = dofspernode)
    dofs = repeat(1:dofspernode, numnodes)
    edofs = [node_to_dof_id(n, d, dofspernode) for (n, d) in zip(extnodeids, dofs)]
    return edofs
end

function element_dofs_to_operator_dofs(rowdofs, coldofs)
    nr = length(rowdofs)
    nc = length(coldofs)
    rows = repeat(rowdofs, outer = nc)
    cols = repeat(coldofs, inner = nr)
    return rows, cols
end

function assemble_couple_cell_matrix!(sysmatrix, nodeids1, nodeids2, dofspernode, vals)
    edofs1 = element_dofs(nodeids1, dofspernode)
    edofs2 = element_dofs(nodeids2, dofspernode)

    rows, cols = element_dofs_to_operator_dofs(edofs1, edofs2)
    assemble!(sysmatrix, rows, cols, vals)
end

function assemble_cell_matrix!(sysmatrix, nodeids, dofspernode, vals)
    assemble_couple_cell_matrix!(sysmatrix,nodeids,nodeids,dofspernode,vals)
end

function assemble_cell_rhs!(sysrhs, nodeids, dofspernode, vals)
    rows = element_dofs(nodeids, dofspernode)
    assemble!(sysrhs, rows, vals)
end

function sparse_operator(sysmatrix, ndofs)
    return dropzeros!(sparse(sysmatrix.rows, sysmatrix.cols, sysmatrix.vals, ndofs, ndofs))
end

function sparse_displacement_operator(sysmatrix, mesh)
    numnodes = number_of_nodes(mesh)
    dim = dimension(mesh)
    totaldofs = dim*numnodes
    return sparse_operator(sysmatrix, totaldofs)
end
