using Random, Distributions
using ControlSystemIdentification, ControlSystemsBase
using LinearAlgebra
using Plots
using LaTeXStrings
using ControlSystems
using StatsPlots
using Statistics

include("function_noise.jl")
include("function_orbit.jl")
include("function_SomeSetting.jl")
using JLD2
using JSON
using Dates
using Base.Threads
Threads.nthreads()

Setting_num = 10
simulation_name = "Vanila_parameter_zoh"


println("simulation_name: ", simulation_name)
@load "System_setting/Noise_dynamics/Settings/Setting$Setting_num/Settings.jld2" Setting
system = Setting["system"]

dir_comparison = "Comparison_SomeSetting/Noise_dynamics/Setting$Setting_num"
dir_comparison = dir_comparison * "/" * simulation_name

# パラメータの読み込み（モデルベース側の設定を流用）
dir_experiment_setting = "System_setting/Noise_dynamics/Settings/Setting$Setting_num/VS_ModelBase"
dir_experiment_setting = dir_experiment_setting * "/" * simulation_name
params = JSON.parsefile(dir_experiment_setting * "/params.json")

Ts = params["Ts"]

Trials = 10
num_of_systems = 10

struct Problem_param
    Q1
    Q2
    Q_prime
    last_value
end


@load dir_comparison * "/Dict_list_Kp_seq_ModelFree.jld2" Dict_list_Kp_seq_ModelFree
@load dir_comparison * "/Dict_list_Ki_seq_ModelFree.jld2" Dict_list_Ki_seq_ModelFree


@load dir_comparison * "/Dict_list_Kp_Sysid.jld2" Dict_list_Kp_Sysid
@load dir_comparison * "/Dict_list_Ki_Sysid.jld2" Dict_list_Ki_Sysid
## 相対誤差を計算する関数
ErrorNorm(A, Aans, A0) = sqrt(sum((A - Aans) .^ 2) / sum((Aans - A0) .^ 2))

## ゲイン最適化の結果表示

tau_eval = 100
Iteration_obj_eval = 200
per_system_list_obj_MFree = Vector{Vector{Float64}}(undef, num_of_systems)
per_system_list_obj_SysId = Vector{Vector{Float64}}(undef, num_of_systems)
for iter_system in 1:num_of_systems
    #println("iter_system: ", iter_system)

    ## 最適化問題のパラメータ
    Q1 = 200.0I(system.p)
    Q2 = 20.0I(system.p)
    Q_prime = [system.C'*Q1*system.C zeros(system.n, system.p); zeros(system.p, system.n) Q2]
    Q_prime = Symmetric((Q_prime + Q_prime') / 2)
    prob = Problem_param(Q1, Q2, Q_prime, true)

    list_obj_MFree = Vector{Float64}(undef, Trials)
    list_obj_SysId = Vector{Float64}(undef, Trials)
    for trial in 1:Trials
        println("trial: ", trial)
        Obj_MFree_uhat = obj_mean_continuous(system, prob,
            ((Dict_list_Kp_seq_ModelFree["system$iter_system"])[trial])[end],
            ((Dict_list_Ki_seq_ModelFree["system$iter_system"])[trial])[end],
            system.u_star, tau_eval, Iteration_obj_eval, h=5e-4)
        println("model free: ", Obj_MFree_uhat)
        Obj_SysId = obj_mean_zoh(system, prob,
            Dict_list_Kp_Sysid["system$iter_system"][trial],
            Dict_list_Ki_Sysid["system$iter_system"][trial],
            system.u_star, Ts, tau_eval, Iteration_obj_eval, h=5e-4)
        println("Indirect approach: ", Obj_SysId)
        list_obj_MFree[trial] = Obj_MFree_uhat
        list_obj_SysId[trial] = Obj_SysId
    end
    per_system_list_obj_MFree[iter_system] = list_obj_MFree
    per_system_list_obj_SysId[iter_system] = list_obj_SysId
end


@save dir_comparison * "/per_system_list_obj_MFree.jld2" per_system_list_obj_MFree
@save dir_comparison * "/per_system_list_obj_SysId.jld2" per_system_list_obj_SysId
