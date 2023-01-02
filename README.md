# KDEDistributions.jl

This package is a quick and dirty implementation of using a univariate [Kernel Density Estimate](https://en.wikipedia.org/wiki/Kernel_density_estimation) from [KernelDensity.jl](https://github.com/JuliaStats/KernelDensity.jl) to create a [Turing.jl](https://turing.ml/dev/) compatible distribution. Its development was guided by [this discourse post](https://discourse.julialang.org/t/using-a-posterior-from-a-previous-sample-as-a-prior/59914). It assumes that the kernel is a gaussian. 

The primary contribution of this package is a type, `KDEDistribution`, and specialized methods of `maximum`, `minimum`, `rand`, `pdf`, and `logpdf` which enable using `Turing.jl` to sample from the distribution. 


## Usage
The following is a simple example of how the KDEDistribution can be used to sample from a distribution.

```
using KDEDistributions, Distributions, Turing, KernelDensity

# Generate Data (using normal to make it easy)
data = randn(10000) 

# Create KDE, get bandwidth
U = kde(data)
bw = KernelDensity.default_bandwidth(data)

# Create KDEDistribution
mydist = KDEDistribution(data, U, bw)


@model function test(data, U, bw)
    y ~ KDEDistribution(data, U, bw)
    #y ~ Normal()
    p = Distributions.logpdf(KDEDistribution(data, U, bw), y)
    Turing.Turing.@addlogprob! p
end

sampler = HMC(0.05, 10);
model = test(data, U, bw)

chain = sample(model, HMC(0.05, 10), 10000; progress=true);

new_KDE = kde(chain[:y].data[:,1])

pygui(true)
figure(1);
clf()
PythonPlot.plot(new_KDE.x, new_KDE.density)
plot(U.x, U.density)
legend(["Estimate of Estimator", "Estimator"])

```

