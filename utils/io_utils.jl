# I/O helper functions ----------------------------------------------------------------------
function readext(path::String, ext::String)
    dir = readdir(path)
    substr = [split(f, ".") for f in dir]
    id = findall(x -> x[end] == ext, substr)
    return dir[id]
end


function concat_dataset(confs::Vector{DataSet})
    N = length(confs)
    confs_vec = [[confs[i][j] for j = 1:length(confs[i])] for i = 1:N]
    confs_all = reduce(vcat, confs_vec)
    return DataSet(confs_all)
end