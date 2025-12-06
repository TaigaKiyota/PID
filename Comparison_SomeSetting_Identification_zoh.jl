using Random, Distributions
using ControlSystemIdentification, ControlSystemsBase
using LinearAlgebra
using Plots
using LaTeXStrings
using ControlSystems
using StatsPlots

include("function_noise.jl")
include("function_orbit.jl")
include("function_AlgoParam.jl")
include("function_SomeSetting.jl")
using JLD2
using JSON
using Dates
using Base.Threads
Threads.nthreads()

ErrorNorm(A, Aans, A0) = sqrt(sum((A - Aans) .^ 2) / sum((Aans - A0) .^ 2))

Setting_num = 6
simulation_name = "Vanila_parameter6_zoh"

println("simulation_name: ", simulation_name)

dir = "Comparison_SomeSetting/Noise_dynamics/Setting$Setting_num"
if !isdir(dir)
    mkdir(dir)  # フォルダを作成
end
dir = dir * "/" * simulation_name
if !isdir(dir)
    mkdir(dir)  # フォルダを作成
end



dir_experiment_setting = "System_setting/Noise_dynamics/Settings/Setting$Setting_num/VS_ModelBase"
dir_experiment_setting = dir_experiment_setting * "/" * simulation_name
params = JSON.parsefile(dir_experiment_setting * "/params.json")
Ts = params["Ts"]
Num_Samples_per_traj = params["Num_Samples_per_traj"]

h = 0.0005 #時間刻み幅必要に応じて変える
Steps_per_sample = Ts / h
Steps_per_sample = round(Steps_per_sample)

accuracy = params["accuracy"]
T_Sysid = Ts * Num_Samples_per_traj
Num_trajectory = 1
Num_TotalSamples = Num_trajectory * Num_Samples_per_traj
PE_power = params["PE_power"]

Trials = 20
num_of_systems = 20
rng_parent = MersenneTwister(1)
Dict_original_systems = Dict{Any,Any}()
#システムの生成
for iter_system in 1:num_of_systems
    println("iter_system: ", iter_system)
    seed_gen_system = rand(rng_parent, UInt64)
    #seed_attr_system = rand(rng_parent, UInt64)
    seed_attr_system = 42 * iter_system
    original_system = Generate_system(seed_gen_system, seed_attr_system, Setting_num)
    Dict_original_systems["system$iter_system"] = original_system
end
@save dir * "/Dict_original_systems.jld2" Dict_original_systems

#Dict_list_est_system = Dict{Any,Any}()
results_per_system = Vector{Vector{Any}}(undef, num_of_systems)
@threads for iter_system in 1:num_of_systems
    #システム同定
    original_system = Dict_original_systems["system$iter_system"]
    local_list = Vector{Any}(undef, Trials)
    for trial in 1:Trials
        #@info "thread $(threadid()) trial = $trial"
        local_list[trial] = Est_discrete_system(original_system,
            Num_TotalSamples,
            Num_trajectory,
            Steps_per_sample,
            Ts,
            T_Sysid,
            PE_power,
            accuracy=accuracy,
            true_dimension=true)
    end
    #Dict_list_est_system["system$iter_system"] = list_est_system
    results_per_system[iter_system] = local_list
    println("iter_system $iter_system trial$trial has done ")
end
Dict_list_est_system = Dict{Any,Any}()
for iter_system in 1:num_of_systems
    Dict_list_est_system["system$iter_system"] = results_per_system[iter_system]
end
@save dir * "/Dict_list_est_system.jld2" Dict_list_est_system