using KDEDistributions, Turing, KernelDensity, Random, Distributions, DynamicHMC, AdvancedHMC, Test

@testset "Test Turing Sampling" begin

    data = randn(10000)

    U = kde(data)
    bw = KernelDensity.default_bandwidth(data)
    mydist = KDEDistribution(data, U, bw)



    @model function test_model(data, U, bw)
        y ~ KDEDistribution(data, U, bw)
        #y ~ Normal()
        p = Distributions.logpdf(KDEDistribution(data, U, bw), y)
        Turing.Turing.@addlogprob! p
    end
    model = test_model(data, U, bw)
    
    @testset verbose=true "Test DynamicHMC" begin
        chain = sample(model, DynamicNUTS(), 100; progress=false);
        @test length(chain[:y].data[:,1]) == 100
    end
    @testset verbose=true "Test SMC" begin
        chain = sample(model, SMC(), 100; progress=false);
        @test length(chain[:y].data[:,1]) == 100
    end
    @testset verbose=true "Test HMC" begin
        chain = sample(model, HMC(0.05, 10), 100; progress=false)
        @test length(chain[:y].data[:, 1]) == 100
    end


end

