function comment_using_revise(filename)
    lines = readlines(filename)
    idx = findfirst(lines .== "using Revise")
    if isnothing(idx)
        @warn "Could not find \"using Revise\" in file $filename, doing nothing."
    else
        lines[idx] = "# using Revise"
        open(filename, "w") do f
            for l in lines
                println(f, l)
            end
        end
    end
end

function process_folder(folderpath)
    testfiles = filter(x -> startswith(x, "test_"), readdir(folderpath))
    testfiles = (folderpath*"/") .* testfiles
    for f in testfiles
        comment_using_revise(f)
    end
end

foldernames = filter(x->isdir("test/"*x), readdir("test"))
folderpaths = "test/" .* foldernames

for path in folderpaths
    process_folder(path)
end
