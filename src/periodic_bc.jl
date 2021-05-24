function make_vertical_periodic!(mesh)
    bottomcellids = bottom_cellids(mesh)
    topcellids = top_cellids(mesh)

    @assert length(bottomcellids) == length(topcellids)
    for idx in 1:length(bottomcellids)
        update_cell_connectivity!(mesh,1,bottomcellids[idx],topcellids[idx])
        update_cell_connectivity!(mesh,3,topcellids[idx],bottomcellids[idx])
    end
end

function make_horizontal_periodic!(mesh)
    leftcellids = left_cellids(mesh)
    rightcellids = right_cellids(mesh)

    @assert length(leftcellids) == length(rightcellids)
    for idx in 1:length(leftcellids)
        update_cell_connectivity!(mesh,4,leftcellids[idx],rightcellids[idx])
        update_cell_connectivity!(mesh,2,rightcellids[idx],leftcellids[idx])
    end
end
