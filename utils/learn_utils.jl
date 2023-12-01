import PotentialLearning: LinearProblem, UnivariateLinearProblem

# return UnivariateLinearProblem
function LinearProblem(
    descriptors::Vector{Vector{T}},
    energies::Vector{T}
    ) where T <: Real

    dim = length(descriptors[1])
    β = zeros(Float64, dim)
    β0 = zeros(Float64, 1)

    p = UnivariateLinearProblem(
            descriptors,
            energies,
            β,
            β0,
            [1.0], 
            Symmetric(zeros(dim, dim)),
        )
    
    p 
end

## train using simple random sampling
# function train_potential(
#     ds::DataSet,
#     ace::ACE,
#     batch_size::Int64;
#     α=1e-8)

#     # init basis potential
#     lb = LBasisPotentialExt(ace)
#     # define random selector
#     srs = RandomSelector(length(ds), batch_size)
#     # draw subset
#     inds = get_random_subset(srs)
#     # learning problem
#     lp = learn!(lb, ds[inds], α)

#     return lp, lb, inds
# end 

function train_potential(
    lp::UnivariateLinearProblem,
    batch_size::Int64;
    α=1e-8)

    # define random selector
    srs = RandomSelector(length(lp.dv_data), batch_size)
    # draw subset
    inds = get_random_subset(srs)
    # learning problem
    lp_sub = LinearProblem(lp.iv_data[inds], lp.dv_data[inds])

    learn!(lp_sub, α)

    return lp_sub, inds
end 


## train using DPP sampling
# function train_potential(
#     ds::DataSet,
#     ace::ACE,
#     L::EllEnsemble,
#     batch_size::Int64;
#     α=1e-8)

#     # init basis potential
#     lb = LBasisPotentialExt(ace)
#     # compute kDPP
#     rescale!(L, batch_size)
#     dpp = kDPP(L, batch_size)
#     # draw subset
#     inds = get_random_subset(dpp)
#     # learning problem
#     lp = learn!(lb, ds[inds], α)

#     return lp, lb, inds
# end 

function train_potential(
    lp::UnivariateLinearProblem,
    L::EllEnsemble,
    batch_size::Int64;
    α=1e-8)

    # compute kDPP
    rescale!(L, batch_size)
    dpp = kDPP(L, batch_size)
    # draw subset
    inds = get_random_subset(dpp)
    # learning problem
    lp_sub = LinearProblem(lp.iv_data[inds], lp.dv_data[inds])

    learn!(lp_sub, α)

    return lp_sub, inds
end 