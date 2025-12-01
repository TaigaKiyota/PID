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
using JLD2
using JSON
using Dates

Setting_num = 6
simulation_name = "Vanila_parameter3_zoh"

@load "System_setting/Noise_dynamics/Settings/Setting$Setting_num/Settings.jld2" Setting

dir = "System_setting/Noise_dynamics/Settings/Setting$Setting_num/VS_ModelBase"
dir = dir * "/" * simulation_name

system = Setting["system"]

params = JSON.parsefile(dir * "/params.json")
Ts = params["Ts"]
N_GD = params["N_GD"]
struct Problem_param
    Q1
    Q2
    Q_prime
    last_value
    N_inner_obj
end
Q1 = 200.0I(system.p) # 要変更!!!
Q2 = 20.0I(system.p)  # 要変更!!!
Q_prime = [system.C'*Q1*system.C zeros(system.n, system.p); zeros(system.p, system.n) Q2]
prob = Problem_param(Q1, Q2, Q_prime, true, params["N_inner_obj"])

@load dir * "/list_ustar_Sysid.jld2" list_ustar_Sysid
@load dir * "/list_uhat.jld2" list_uhat
@load dir * "/list_Kp_seq_Sysid.jld2" list_Kp_seq_Sysid
@load dir * "/list_Ki_seq_Sysid.jld2" list_Ki_seq_Sysid
@load dir * "/list_Kp_seq_ModelFree.jld2" list_Kp_seq_ModelFree
@load dir * "/list_Ki_seq_ModelFree.jld2" list_Ki_seq_ModelFree
## 相対誤差を計算する関数
ErrorNorm(A, Aans, A0) = sqrt(sum((A - Aans) .^ 2) / sum((Aans - A0) .^ 2))

Trials = size(list_ustar_Sysid, 1)
println("Num of Trials: ", Trials)
trial = 5

## ゲイン最適化の結果表示

#ゲイン最適化の結果をシミュレーションで表示
dir_path = dir * "/trial$trial"
if !isdir(dir_path)
    mkdir(dir_path)  # フォルダを作成
end

dir_path = dir_path * "/Gain_Simulation_plot"
if !isdir(dir_path)
    mkdir(dir_path)  # フォルダを作成
end

## シミュレーションによる軌道の確認

x_0 = zeros(system.n)
z_0 = zeros(system.p)
T = 2
h = 1e-4
K_P_Mfree = (list_Kp_seq_ModelFree[trial])[end]
K_I_Mfree = (list_Ki_seq_ModelFree[trial])[end]
println("ModelFree Kp: ", K_P_Mfree)
println("ModelFree Ki: ", K_I_Mfree)

K_P_SysId = (list_Kp_seq_Sysid[trial])[end]
K_I_SysId = (list_Ki_seq_Sysid[trial])[end]

println("SysId Kp: ", K_P_SysId)
println("SysId Ki: ", K_I_SysId)
u_star_Sysid = list_ustar_Sysid[trial]
u_hat = list_uhat[trial]

_, y_s_hat, _ = Orbit_continuous_PI(system, K_P_Mfree, K_I_Mfree, u_hat, x_0, T, h=h)
_, y_s_sysid, _ = Orbit_zoh_PI(system, K_P_SysId, K_I_SysId, u_star_Sysid, x_0, T, Ts=Ts, h=h)
Timeline = 0:system.h:T

plotting = plot(legendfontsize=18, tickfontsize=15, guidefont=18)
plot!(plotting, Timeline[1:end], y_s_hat[1, 1:end], labels="Proposed Method", lw=1.8, lc=:red)
plot!(plotting, Timeline[1:end], y_s_sysid[1, 1:end], labels="Indirect Approach", lw=1.8, lc=:blue)
hline!(plotting, system.y_star, label=L"y^{\star}", lc=:black, lw=3)
xlims!(0, T)
#ylims!(-1, 6)
xlabel!("Time")
ylabel!(L"y")
savefig(plotting, dir_path * "/Compare_Gain_estimatedFF.png")

for i in 1:system.p
    plotting = plot(legendfontsize=15, tickfontsize=15, legend=:right, guidefont=22)
    plot!(plotting, Timeline[1:end], y_s_hat[i, 1:end], labels="Proposed Method", lw=2, lc=:red)
    plot!(plotting, Timeline[1:end], y_s_sysid[i, 1:end], labels="Indirect Approach", lw=2, lc=:blue)
    hline!(plotting, system.y_star, label=L"y^{\star}", lc=:black, lw=4)
    xlims!(0, T)
    #ylims!(-3, 8)
    xlabel!(L"t")
    ylabel!(L"y(t)")
    savefig(plotting, dir_path * "/Compare_Gain_component$(i).png")
end



## ゲイン最適化の結果表示
tau_eval = 500
Iteration_obj_eval = 200
list_obj_MFree = []
list_obj_SysId = []
for trial in 1:Trials
    Obj_MFree_uhat = obj_mean_continuous(system, prob, (list_Kp_seq_ModelFree[trial])[end], (list_Ki_seq_ModelFree[trial])[end],
        system.u_star, tau_eval, Iteration_obj_eval, h=h)

    Obj_SysId = obj_mean_zoh(system, prob,
        (list_Kp_seq_Sysid[trial])[end], (list_Ki_seq_Sysid[trial])[end],
        system.u_star, Ts, tau_eval, Iteration_obj_eval, h=h)
    push!(list_obj_MFree, Obj_MFree_uhat)
    push!(list_obj_SysId, Obj_SysId)
    println("trial $trial has done!")
end

dir_path = dir * "/Gain_meanObj"
if !isdir(dir_path)
    mkdir(dir_path)  # フォルダを作成
end

boxplot(list_obj_MFree,
    tickfontsize=18, yguidefont=font(20), fillcolor=:red, legend=false, outliercolor=:red, markercolor=:red)
boxplot!(list_obj_SysId, fillcolor=:blue, outliercolor=:blue, markercolor=:blue)
xticks!((1:2, ["Proposed method", "Indirect approach"]))
#yticks!([1e-2, 1e-1, 1, 10, 20], ["0.01", "0.1", "1", "10", "20"]),
#ylims!(1e-2, 20)
#ylabel!(L"\|\| u_0 - u^{\star} \|\|")
savefig(dir_path * "/Gain_MeanObj_boxplot.png")

boxplot(list_obj_MFree, tickfontsize=15, bar_width=0.3, yguidefont=font(18), label="Proposed method")
xticks!((1:1, ["Proposed method"]))
#ylabel!(L"\|\| u_0 - u^{\star} \|\|")
savefig(dir_path * "/Gain_MeanObj_boxplot_MFree.png")

boxplot(list_obj_SysId, tickfontsize=15, bar_width=0.3, yguidefont=font(18), label="Indirect approach")
#ylabel!(L"\|\| u_0 - u^{\star} \|\|")
xticks!((1:1, ["Indirect approach"]))
savefig(dir_path * "/Gain_MeanObj_boxplot_SysId.png")

