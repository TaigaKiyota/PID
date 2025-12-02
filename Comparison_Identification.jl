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

Setting_num = 6
simulation_name = "Vanila_parameter4_zoh"
estimated_param = false


@load "System_setting/Noise_dynamics/Settings/Setting$Setting_num/Settings.jld2" Setting
system = Setting["system"]

dir = "System_setting/Noise_dynamics/Settings/Setting$Setting_num/VS_ModelBase"

dir = dir * "/" * simulation_name

params = JSON.parsefile(dir * "/params.json")
Ts = params["Ts"]
Num_Samples_per_traj = params["Num_Samples_per_traj"]
Num_Samples_per_traj = 2000

Steps_per_sample = Ts / system.h
Steps_per_sample = round(Steps_per_sample)

accuracy = params["accuracy"]
T_Sysid = Ts * Num_Samples_per_traj
Num_trajectory = 1
Num_TotalSamples = Num_trajectory * Num_Samples_per_traj
PE_power = params["PE_power"]

Trials = 20
list_est_system = []
for trial in 1:Trials
    println("trial: ", trial)
    est_system = Est_discrete_system(system,
        Num_TotalSamples,
        Num_trajectory,
        Steps_per_sample,
        Ts,
        T_Sysid,
        PE_power,
        accuracy=accuracy)
    push!(list_est_system, est_system)
end

@save dir * "/list_est_system.jld2" list_est_system