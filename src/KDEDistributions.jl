module KDEDistributions

using Distributions
using Random


struct KDEDistribution <: ContinuousUnivariateDistribution 
    data
    kde
    bw
end

Distributions.minimum(d::KDEDistribution) = minimum(d.data)
Distributions.maximum(d::KDEDistribution) = maximum(d.data)

function Distributions.pdf(d::KDEDistribution, x::Real)
    return pdf(d.kde, x)
end

function Distributions.logpdf(d::KDEDistribution, x::Real)
    log(pdf(d, x))
end

function Distributions.rand(d::KDEDistribution)
    x = rand(d.data)
    bw = d.bw
    return bw.*randn() .+ x
end

function Distributions.rand(rng::AbstractRNG, d::KDEDistribution)
    x = rand(rng, d.data)
    bw = d.bw
    return bw.*randn(rng) .+ x
end

export KDEDistribution

end # module KDEDistributions
