
function plot_metadata(
    dfs::Tuple,
    output::String,
    metric::String,
    labels=Vector{String},
    colors=Vector{Symbol};
    minmax::Union{Bool,String}=false
)
    batches = dfs[1][:,"batch size"]

    type = "$output $metric"

    f = Figure(resolution=(500,500))
    ax = Axis(f[1,1],
        xlabel="sample size (N)",
        ylabel=type,
        xscale=log10,
        yscale=log10,
        xticks=(batches, [string(bs) for bs in sort(batches)]),
        # xgridvisible=false,
        )

    if minmax == false
        for (i, df) in enumerate(dfs)
            scatter!(ax, batches, abs.(df[:,type*" med"]), color=colors[i], markersize=15, label=labels[i])
            rangebars!(ax, batches,
                abs.(df[:,type*" lqt"]),
                abs.(df[:,type*" uqt"]),
                color=colors[i],
                linewidth=3,
            )
        end
    elseif typeof(minmax) == String
        for (i, df) in enumerate(dfs)
            scatter!(ax, batches, df[:,type*" $minmax"], color=colors[i], label=labels[i])
        end
    end
    axislegend(ax, position=:rt)
    return f
end