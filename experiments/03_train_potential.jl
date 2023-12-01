# Compare model fit with data subset selection
include("./experiments/01_load_data.jl")
using CairoMakie
using JLD

save_dir = "./experiments/$elname/results/"

# Monte Carlo trials ----------------------------------------------------------------------
niter = 100
batch_sizes = [10, 20, 40, 80, 160, 320, 640]

# init results dict
res_srs = init_res_dict(batch_sizes, niter)
res_dp = init_res_dict(batch_sizes, niter)
res_rbf = init_res_dict(batch_sizes, niter)

# split train set
ind_all = rand(1:nconf, nconf)
cut = Int(floor(nconf * 0.7))
train_ind = ind_all[1:cut]
test_ind = ind_all[cut+1:end]
desc_train = desc[train_ind]; e_train = energies[train_ind]
desc_test = desc[test_ind]; e_test = energies[test_ind]

# define LinearProblem
lp_all = LinearProblem(desc, energies)
lp_train = LinearProblem(desc_train, e_train)
lp_test = LinearProblem(desc_test, e_test)

# compute L-ensemble for DPP once
k1 = DotProduct()
precisionmat = inv(Symmetric(cov(desc_train)))
d = Euclidean(precisionmat)
k2 = RBF(d; ℓ=1.0)

K1, L1 = compute_ell_ensemble(desc_train, k1)
K2, L2 = compute_ell_ensemble(desc_train, k2)

# iterate over divisions
for i = 1:niter
    println("----------- iter $i ----------")

    for bs in batch_sizes
        t = @elapsed begin
            # train by DPP
            lp_dp, _ = train_potential(lp_train, L1, bs)
            res_dp[bs] = update_res_dict(i, res_dp[bs], lp_all, lp_dp.β)

            lp_rbf, _ = train_potential(lp_train, L2, bs)
            res_rbf[bs] = update_res_dict(i, res_rbf[bs], lp_all, lp_rbf.β)

            # train by simple random sampling
            lp_srs, _ = train_potential(lp_train, bs)
            res_srs[bs] = update_res_dict(i, res_srs[bs], lp_all, lp_srs.β)

            # JLD.save(save_dir*"DPP_DP_training_results_MC_N=$bs.jld", "res", res_dp[bs])
            # JLD.save(save_dir*"DPP_RBF_training_results_MC_N=$bs.jld", "res", res_rbf[bs])
            # JLD.save(save_dir*"SRS_training_results_MC_N=$bs.jld", "res", res_srs[bs])
        end
        println("Train with batch $bs: $t sec.")
    end
end

JLD.save(save_dir*"DPP_DP_training_results_MC_all.jld", "res", res_dp)
JLD.save(save_dir*"DPP_RBF_training_results_MC_all.jld", "res", res_rbf)
JLD.save(save_dir*"SRS_training_results_MC_all.jld", "res", res_srs)



# plot ---------------------------------------------------------------------------------
df_dp = compute_cv_metadata(res_dp)
df_rbf = compute_cv_metadata(res_rbf)
df_srs = compute_cv_metadata(res_srs)

colors = [:firebrick, :skyblue, :orange]
labels = ["DPP (DP)", "DPP (RBF)", "SRS"]

f1 = plot_metadata((df_dp, df_rbf, df_srs), "E", "mae", labels, colors) # minmax="min")
f2 = plot_metadata((df_dp, df_rbf, df_srs), "E", "rmse", labels, colors) # minmax="min")
f3 = plot_metadata((df_dp, df_rbf, df_srs), "E", "rsq", labels, colors) # minmax="min")
