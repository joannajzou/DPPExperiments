# load descriptor and energy data
# push!(Base.LOAD_PATH, "../../")

# using PotentialLearning
# using LinearAlgebra, Random, Statistics, StatsBase, Distributions
# using Determinantal
# using DataFrames
# using Zarr

# include(pwd()*"/utils/utils.jl")

# Define directories --------------------------------------------------------------
# elname = "Hf"
inpath = "./experiments/$elname/data/"
data_dir = filter(isdir, readdir(inpath; join=true))


# Load configurations -------------------------------------------------------------
println("loading configurations and energies")
data_arr = [readext(dir, "xyz")[1] for dir in data_dir]

# configurations and reference energies
confs_arr = [load_data(dir*"/"*datafile, ExtXYZ(u"eV", u"Å")) for (dir, datafile) in zip(data_dir, data_arr)]
confs = concat_dataset(confs_arr)
energies_arr = [get_all_energies(conf) for conf in confs_arr]
energies = reduce(vcat, energies_arr)

# indices of datasets
nds = length(energies_arr)
ds_ids = reduce(vcat, [Int.(k*ones(length(e))) for (k,e) in zip(1:nds, energies_arr)])
n_per_ds = [length(e) for e in energies_arr]


# Load descriptors ----------------------------------------------------------------
println("loading descriptors")
z_file = "/globaldescriptors_allconfigs.zarr"
desc_arr = [zopen(dir*z_file)[:,:]' for dir in data_dir]
desc_mat = Matrix(reduce(vcat, desc_arr))
nconf, ndesc = size(desc_mat)

# remove zero-valued descriptors (corresponding to nonpresent elements)
zero_id = findall(x -> iszero(x), [desc_mat[:,i] for i = 1:ndesc])
keep_id = symdiff(zero_id, 1:ndesc)
desc_arr = [desc[:, keep_id] for desc in desc_arr]
desc_mat = desc_mat[:, keep_id]
_, ndesc = size(desc_mat)
desc = [desc_mat[i,:] for i = 1:nconf]


println("DONE")


