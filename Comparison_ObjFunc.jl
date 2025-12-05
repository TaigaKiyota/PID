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

Setting_num = 6
simulation_name = "Vanila_parameter5_zoh"

@load "System_setting/Noise_dynamics/Settings/Setting$Setting_num/Settings.jld2" Setting

dir = "System_setting/Noise_dynamics/Settings/Setting$Setting_num/VS_ModelBase"
dir = dir * "/" * simulation_name
@load dir * "/list_est_system.jld2" list_est_system
system = Setting["system"]

params = JSON.parsefile(dir * "/params.json")
Ts = params["Ts"]
N_GD = params["N_GD"]

Trials = 20

struct Problem_param
    Q1
    Q2
    Q_prime
    last_value
end
Q1 = 200.0I(system.p) # 要変更!!!
Q2 = 20.0I(system.p)  # 要変更!!!
Q_prime = [system.C'*Q1*system.C zeros(system.n, system.p); zeros(system.p, system.n) Q2]
prob = Problem_param(Q1, Q2, Q_prime, true)


#@load dir * "/list_ustar_Sysid.jld2" list_ustar_Sysid
@load dir * "/list_uhat.jld2" list_uhat
#@load dir * "/list_Kp_seq_Sysid.jld2" list_Kp_seq_Sysid
#@load dir * "/list_Ki_seq_Sysid.jld2" list_Ki_seq_Sysid
@load dir * "/list_Kp_seq_ModelFree.jld2" list_Kp_seq_ModelFree
@load dir * "/list_Ki_seq_ModelFree.jld2" list_Ki_seq_ModelFree


@load dir * "/list_Kp_Sysid.jld2" list_Kp_Sysid
@load dir * "/list_Ki_Sysid.jld2" list_Ki_Sysid
## 相対誤差を計算する関数
ErrorNorm(A, Aans, A0) = sqrt(sum((A - Aans) .^ 2) / sum((Aans - A0) .^ 2))

## ゲイン最適化の結果表示

tau_eval = 100
Iteration_obj_eval = 200
list_obj_MFree = []
list_obj_SysId = []
for trial in 1:Trials
    println("trial: ", trial)
    Obj_MFree_uhat = obj_mean_continuous(system, prob, (list_Kp_seq_ModelFree[trial])[end], (list_Ki_seq_ModelFree[trial])[end],
        system.u_star, tau_eval, Iteration_obj_eval, h=5e-4)
    println("model free: ", Obj_MFree_uhat)
    Obj_SysId = obj_mean_zoh(system, prob,
        list_Kp_Sysid[trial], list_Ki_Sysid[trial],
        system.u_star, Ts, tau_eval, Iteration_obj_eval, h=5e-4)
    println("Indirect approach: ", Obj_SysId)
    push!(list_obj_MFree, Obj_MFree_uhat)
    push!(list_obj_SysId, Obj_SysId)
end

boxplot(list_obj_MFree, label="Model Free")
boxplot!(list_obj_SysId, label="System Identification")
savefig(dir * "/Gain_MeanObj_boxplot.png")

@save dir * "/list_obj_MFree_Comparison_ObjFunc.jld2" list_obj_MFree
@save dir * "/list_obj_SysId_Comparison_ObjFunc.jld2" list_obj_SysId
