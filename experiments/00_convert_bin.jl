## read binary files with POD descriptors and save in zarr files
push!(Base.LOAD_PATH, "../../")

# Define directories -----------------------------------------------------------
elname = "Hf"
inpath = "./experiments/$elname/data/"
data_dir = filter(isdir, readdir(inpath; join=true))

# Write zarr files -------------------------------------------------------------
for dir in data_dir
    try
        run(`python3 experiments/readDesc.py $dir`)
    catch
        println("$dir does not exist")
    end
end
