function init_res_dict(batch_size::Vector, ndiv::Int64)
    res = Dict{Int64, Dict}(bs => Dict{String, Vector}() for bs in batch_size)
    for bs in batch_size
        # res[bs]["cond_num"] = zeros(ndiv)
        # res[bs]["indices"] = Vector{Vector{Float64}}(undef, ndiv)
        # res[bs]["energy_err"] = Vector{Vector{Float64}}(undef, ndiv)
        res[bs]["energy_mae"] = zeros(ndiv)
        res[bs]["energy_rmse"] = zeros(ndiv)
        res[bs]["energy_rsq"] = zeros(ndiv)
        # res[bs]["force_err"] = Vector{Vector{Float64}}(undef, ndiv)
        res[bs]["force_mae"] = zeros(ndiv)
        res[bs]["force_rmse"] = zeros(ndiv)
        res[bs]["force_rsq"] = zeros(ndiv)
    end
    return res
end


function update_res_dict(
    i::Int64,
    res::Dict,
    ds::DataSet,
    lp,
    lb::LBasisPotentialExt,
    ind::Vector, 
)
    res["cond_num"][i] = compute_cond_num(lp)
    # res["indices"][i] = ind

    # get DFT and predicted energies/forces
    energies = get_all_energies(ds)
    forces = get_all_forces_mag(ds) # magnitude
    e_pred = get_all_energies(ds, lb)
    f_pred = get_all_forces_mag(ds, lb)

    # compute errors
    # res["energy_err"][i] = energies - e_pred 
    # res["force_err"][i] = forces - f_pred
    res["energy_mae"][i], res["energy_rmse"][i], res["energy_rsq"][i] = calc_metrics(energies, e_pred)
    res["force_mae"][i], res["force_rmse"][i], res["force_rsq"][i] = calc_metrics(forces, f_pred)
    
    return res
end


function update_res_dict(
    i::Int64,
    res::Dict,
    lp::UnivariateLinearProblem,
    β::Vector,
)

    # get DFT and predicted energies/forces
    energies = lp.dv_data
    e_pred = [β' * B for B in lp.iv_data]

    # compute errors
    res["energy_mae"][i], res["energy_rmse"][i], res["energy_rsq"][i] = calc_metrics(energies, e_pred)
    
    return res
end


function compute_cv_metadata(
    res::Dict,
)
    batches = sort(collect(keys(res)))

    df_conf = DataFrame(
        "batch size" => batches,
        # "DFT energy" => get_all_energies(ds),
        # "E err mean" => mean(res["energy_err"]), # mean error from k-fold CV
        # "E err std" => std(res["energy_err"]), # std of error
        "E mae min" => [minimum(res[bs]["energy_mae"]) for bs in batches],
        "E mae med" => [median(res[bs]["energy_mae"]) for bs in batches],
        "E mae lqt" => [quantile!(res[bs]["energy_mae"], 0.25) for bs in batches],
        "E mae uqt" => [quantile!(res[bs]["energy_mae"], 0.75) for bs in batches],

        "E rmse min" => [minimum(res[bs]["energy_rmse"]) for bs in batches],
        "E rmse med" => [median(res[bs]["energy_rmse"]) for bs in batches],
        "E rmse lqt" => [quantile!(res[bs]["energy_rmse"], 0.25) for bs in batches],
        "E rmse uqt" => [quantile!(res[bs]["energy_rmse"], 0.75) for bs in batches],
        
        "E rsq max" => [maximum(abs.(res[bs]["energy_rsq"])) for bs in batches],
        "E rsq med" => [median(abs.(res[bs]["energy_rsq"])) for bs in batches],
        "E rsq lqt" => [quantile!(abs.(res[bs]["energy_rsq"]), 0.25) for bs in batches],
        "E rsq uqt" => [quantile!(abs.(res[bs]["energy_rsq"]), 0.75) for bs in batches],

        # "DFT force" => get_all_forces_mag(ds),
        # "F err mean" =>  mean(res["force_err"]), # mean error from k-fold CV
        # "F err std" => std(res["force_err"]), # std of error
        "F mae min" => [minimum(res[bs]["force_mae"]) for bs in batches],
        "F mae med" => [median(res[bs]["force_mae"]) for bs in batches],
        "F mae lqt" => [quantile!(res[bs]["force_mae"], 0.25) for bs in batches],
        "F mae uqt" => [quantile!(res[bs]["force_mae"], 0.75) for bs in batches],

        "F rmse min" => [minimum(res[bs]["force_rmse"]) for bs in batches],
        "F rmse med" => [median(res[bs]["force_rmse"]) for bs in batches],
        "F rmse lqt" => [quantile!(res[bs]["force_rmse"], 0.25) for bs in batches],
        "F rmse uqt" => [quantile!(res[bs]["force_rmse"], 0.75) for bs in batches],

        "F rsq max" => [maximum(abs.(res[bs]["force_rsq"])) for bs in batches],
        "F rsq med" => [median(abs.(res[bs]["force_rsq"])) for bs in batches],
        "F rsq lqt" => [quantile!(abs.(res[bs]["force_rsq"]), 0.25) for bs in batches],
        "F rsq uqt" => [quantile!(abs.(res[bs]["force_rsq"]), 0.75) for bs in batches],
    )
    return df_conf
end