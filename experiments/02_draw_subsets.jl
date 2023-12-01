# Compares samples from DPP 
include("./experiments/01_load_data.jl")
using CairoMakie

plot_dir = "./experiments/$elname/figures/"

# Compute kernel matrix - dot product -----------------------------------------------------
k1 = DotProduct()
K1, L1 = compute_ell_ensemble(desc, k1)

# compute the dual kernel 
desc_t = [desc_mat[:,i] for i = 1:ndesc]
Kt, Lt = compute_ell_ensemble(desc_t, k1)


# Compute kernel matrix - RBF -------------------------------------------------------------
Cd = Symmetric(cov(desc))
precisionmat = Symmetric(pinv(Cd, 1e-6))
d = Euclidean(precisionmat)
k2 = RBF(d, ℓ=1.0)
K2, L2 = compute_ell_ensemble(desc, k2)




# Plot kernel matrices --------------------------------------------------------------------
fig = Figure(resolution = (1200, 500))
ax1 = Axis(fig[1,1][1,1],
    title="Dot Product Kernel",
    xlabel="desc. dim.",
    ylabel="desc. dim.",
    yticks=(Vector(1:500:1500) .+ (nconf-1500), [string(x) for x in 1500:-500:1]),
)
hm1 = heatmap!(ax1, 1:nconf, 1:nconf, log.(K1[end:-1:1,:]), colorrange=(log(1e-8), log(1)))
Colorbar(fig[1,1][1,2], hm1, label="log(K)")
[vlines!(ax1, x, color=:white) for x in cumsum(n_per_ds)]
[hlines!(ax1, nconf - x, color=:white) for x in cumsum(n_per_ds)]

ax2 = Axis(fig[1,2][1,1],
    title="RBF Kernel",
    xlabel="desc. dim.",
    ylabel="desc. dim.",
    yticks=(Vector(1:500:1500) .+ (nconf-1500), [string(x) for x in 1500:-500:1]),
)
hm2 = heatmap!(ax2, 1:nconf, 1:nconf, log.(K2[end:-1:1,:]), colorrange=(log(1e-8), log(1)))
Colorbar(fig[1,2][1,2], hm2, label="log(K)")
[vlines!(ax2, x, color=:white) for x in cumsum(n_per_ds)]
[hlines!(ax2, nconf - x, color=:white) for x in cumsum(n_per_ds)]
fig


# linear scale
fig = Figure(resolution = (1200, 500))
ax1 = Axis(fig[1,1][1,1],
    title="Dot Product Kernel",
    xlabel="desc. dim.",
    ylabel="desc. dim.",
    yticks=(Vector(1:500:1500) .+ (nconf-1500), [string(x) for x in 1500:-500:1]),
)
hm1 = heatmap!(ax1, 1:nconf, 1:nconf, K1[end:-1:1,:], colorrange=(0,1))
Colorbar(fig[1,1][1,2], hm1, label="K")
[vlines!(ax1, x, color=:white) for x in cumsum(n_per_ds)]
[hlines!(ax1, nconf - x, color=:white) for x in cumsum(n_per_ds)]

ax2 = Axis(fig[1,2][1,1],
    title="RBF Kernel",
    xlabel="desc. dim.",
    ylabel="desc. dim.",
    yticks=(Vector(1:500:1500) .+ (nconf-1500), [string(x) for x in 1500:-500:1]),
)
hm2 = heatmap!(ax2, 1:nconf, 1:nconf, K2[end:-1:1,:], colorrange=(0,1))
Colorbar(fig[1,2][1,2], hm2, label="K")
[vlines!(ax2, x, color=:white) for x in cumsum(n_per_ds)]
[hlines!(ax2, nconf - x, color=:white) for x in cumsum(n_per_ds)]
fig


# Check eigenspectrum --------------------------------------------------------------------
λd = eigvals(Cd)[end:-1:1]
λ1 = eigvals(K1)[end:-1:1]; W1 = eigvecs(K1)
λt = eigvals(Kt)[end:-1:1]; Wt = eigvecs(Kt)
λ2 = eigvals(K2)[end:-1:1]; W2 = eigvecs(K2)

fig = Figure(resolution = (500, 500))
ax = Axis(fig[1,1],
    title="Spectrum of kernel matrix",
    xlabel="dim. (i)",
    ylabel="eigenvalue (λi)",
    yscale=log10,
)
lines!(ax, 1:length(findall(x -> x > 0, λ1)), λ1[findall(x -> x > 0, λ1)], label="primal kernel (DP)")
lines!(ax, 1:length(findall(x -> x > 0, λt)), λt[findall(x -> x > 0, λt)], label="dual kernel (DP)")
lines!(ax, 1:length(findall(x -> x > 0, λ2)), λ2[findall(x -> x > 0, λ2)], label="primal kernel (RBF)")
axislegend(ax)
fig

fig = Figure(resolution = (500, 500))
ax = Axis(fig[1,1],
    title="Spectrum of descriptor covariance matrix",
    xlabel="dim. (i)",
    ylabel="eigenvalue (λi)",
    yscale=log10,
)
lines!(ax, 1:ndesc, λd[findall(x -> x > 0, λd)])
fig


# Evaluate selection ---------------------------------------------------------------------
niter = 500
batch_sizes = [20, 40, 80, 160, 320, 640]

# initialize
ip1 = Dict{Int64, Vector{Float64}}()
ip2 = Dict{Int64, Vector{Float64}}()
conf_ct1 = Dict{Int64, Vector{Int64}}(
    bs => zeros(nconf) for bs in batch_sizes
)
conf_ct2 = Dict{Int64, Vector{Int64}}(
    bs => zeros(nconf) for bs in batch_sizes
)
ds_ct1 = Dict{Int64, Dict}(
    bs => Dict{Int64, Vector{Int64}}(
        n => zeros(niter) for n = 1:nds
    ) for bs in batch_sizes
)
ds_ct2 = Dict{Int64, Dict}(
    bs => Dict{Int64, Vector{Int64}}(
        n => zeros(niter) for n = 1:nds
    ) for bs in batch_sizes
)

for bs in batch_sizes
    println("===== starting batch $bs =====")
    # compute inclusion probabilities
    ip1[bs] = get_inclusion_prob(L1, bs)
    ip2[bs] = get_inclusion_prob(L2, bs)

    # compute dataset selection
    for iter = 1:niter
        println("iter $iter")
        id1 = get_random_subset(L1, bs)
        id2 = get_random_subset(L2, bs)
        ds_rep1 = ds_ids[id1]
        ds_rep2 = ds_ids[id2]

        # count by configuration
        conf_ct1[bs][id1] = conf_ct1[bs][id1] .+ 1
        conf_ct2[bs][id2] = conf_ct2[bs][id2] .+ 1

        # count by dataset
        for n = 1:nds
            ds_ct1[bs][n][iter] = Int(length(findall(x -> x == n, ds_rep1)))
            ds_ct2[bs][n][iter] = Int(length(findall(x -> x == n, ds_rep2)))
        end
    end
end


# Plot inclusion probability ---------------------------------------------------------
figs = Vector{Figure}(undef, length(batch_sizes))
inds = reduce(vcat, [1, cumsum(n_per_ds)])

for (n,bs) in enumerate(batch_sizes)
    figs[n] = Figure(resolution = (1000, 500))
    ax1 = Axis(figs[n][1,1],
        title="k-DPP inclusion prob. (k=$bs, dot product kernel)",
        xlabel="inclusion probability",
        ylabel="empirical probability",
    )
    ax2 = Axis(figs[n][1,2],
        title="k-DPP inclusion prob. (k=$bs, RBF kernel)",
        xlabel="inclusion probability",
        ylabel="empirical probability",
    )

    [scatter!(ax1,
        ip1[bs][inds[j]:inds[j+1]],
        conf_ct1[bs][inds[j]:inds[j+1]] ./ niter,
        label="DS$j")
        for j = 1:nds
    ]
    lrng = LinRange( 
            minimum([minimum(ip1[bs]),
            minimum(conf_ct1[bs] ./ niter)]),
            maximum([maximum(ip1[bs]),
            maximum(conf_ct1[bs] ./ niter)]),
            100)
    lines!(ax1, lrng, lrng, color=:red)
    axislegend(ax1, position=:rb)
    
    [scatter!(ax2,
        ip2[bs][inds[j]:inds[j+1]],
        conf_ct2[bs][inds[j]:inds[j+1]] ./ niter,
        label="DS$j")
        for j = 1:nds
    ]
    lrng = LinRange( 
        minimum([minimum(ip2[bs]),
        minimum(conf_ct2[bs] ./ niter)]),
        maximum([maximum(ip2[bs]),
        maximum(conf_ct2[bs] ./ niter)]),
        100)
    lines!(ax2, lrng, lrng, color=:red)
    axislegend(ax2, position=:rb)
end

[save(plot_dir*"inclusion_prob_$bs.png", f) for (f, bs) in zip(figs, batch_sizes)]


# Plot dataset representation ---------------------------------------------------------
figs2 = Vector{Figure}(undef, length(batch_sizes))

for (n,bs) in enumerate(batch_sizes)
    figs2[n] = Figure(resolution = (1000, 500))
    ax = Axis(figs2[n][1,1],
        title="Dataset Selection in k-DPP (k=$bs)",
        xlabel="dataset",
        ylabel="proportion of selection",
        xticks=1:nds,
    )

    maxp = maximum([maximum(ds_ct1[bs][j] ./ bs ./ n_per_ds[j]) for j = 1:nds])
    srslvl = 1 / nconf
    binrng = LinRange(0, maxp, 50)

    hist!(ax, ds_ct1[bs][1] ./ bs ./ n_per_ds[1],
        scale_to=-0.75,
        offset=1,
        direction=:x,
        color=(:skyblue1, 0.5),
        bins=binrng,
        label="DP kernel")
    hist!(ax, ds_ct2[bs][1] ./ bs ./ n_per_ds[1],
        scale_to=-0.75,
        offset=1,
        direction=:x,
        color=(:magenta, 0.5),
        bins=binrng,
        label="RBF kernel")
    for j = 2:nds 
        hist!(ax, ds_ct1[bs][j] ./ bs ./ n_per_ds[j],
            scale_to=-0.75,
            offset=j, 
            direction=:x,
            color=(:skyblue1, 0.5),
            bins=binrng,
            )
        hist!(ax, ds_ct2[bs][j] ./ bs ./ n_per_ds[j],
            scale_to=-0.75,
            offset=j,
            direction=:x,
            color=(:magenta, 0.5),
            bins=binrng,
            )
    end
    hlines!(ax, srslvl, color=:red, label="SRS ref.")

    axislegend(ax)
end
[save(plot_dir*"dataset_selection_$bs.png", f) for (f, bs) in zip(figs2, batch_sizes)]
