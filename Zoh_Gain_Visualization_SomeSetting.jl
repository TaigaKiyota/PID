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
ErrorNorm(A, Aans, A0) = sqrt(sum((A - Aans) .^ 2) / sum((Aans - A0) .^ 2))

Setting_num = 10
simulation_name = "Vanila_parameter_zoh"

iter_system = 1
simulation_name_param = "Vanila_parameter_zoh"
dir_comparison = "Comparison_SomeSetting/Noise_dynamics/Setting$Setting_num"
dir_comparison = dir_comparison * "/" * simulation_name_param
@load dir_comparison * "/Dict_original_systems.jld2" Dict_original_systems
system = Dict_original_systems["system$iter_system"]


dir_experiment_setting = "System_setting/Noise_dynamics/Settings/Setting$Setting_num/VS_ModelBase"
dir_experiment_setting = dir_experiment_setting * "/" * simulation_name
params = JSON.parsefile(dir_experiment_setting * "/params.json")
Ts = params["Ts"]

@load dir_comparison * "/list_uhat.jld2" list_uhat
dir_result = "Comparison_SomeSetting/Noise_dynamics/Setting$Setting_num"
dir_result = dir_result * "/" * simulation_name
dir_result = dir_result * "/system" * iter_system
if !isdir(dir)
    mkdir(dir)  # フォルダを作成
end

## FF
@load dir * "/Dict_list_uhat.jld2" Dict_list_uhat
@load dir * "/Dict_list_est_system.jld2" Dict_list_est_system
list_est_system = Dict_list_est_system["system$iter_system"]
list_uhat = Dict_list_uhat["system$iter_system"]
Trials = 10
list_error_ystar_Sysid = []
list_error_ystar_MFree = []
for trial in 1:Trials
    #ustar_sysid = list_ustar_Sysid[trial]
    est_system = list_est_system[trial]
    ustar_sysid = est_system.u_star
    y_inf_sysid = -system.C * (system.A \ (system.B * ustar_sysid))
    push!(list_error_ystar_Sysid, ErrorNorm(y_inf_sysid, system.y_star, zeros(system.m)))

    uhat = list_uhat[trial]
    y_inf_uhat = -system.C * (system.A \ (system.B * uhat))
    push!(list_error_ystar_MFree, ErrorNorm(y_inf_uhat, system.y_star, zeros(system.m)))
end

boxplot(list_error_ystar_MFree,
    tickfontsize=15, yguidefont=font(15), fillcolor=:red, legend=false, fillalpha=0.0, outliercolor=:white, markercolor=:white)
boxplot!(list_error_ystar_Sysid, fillcolor=:blue, fillalpha=0.0, outliercolor=:white, markercolor=:white)
xticks!((1:2, ["Proposed method", "Indirect approach"]))
ylims!(0, 0.35)
savefig(dir_result * "/FF_y_error_boxplot.png")


@load dir_comparison * "/Dict_list_Kp_seq_ModelFree.jld2" Dict_list_Kp_seq_ModelFree
@load dir_comparison * "/Dict_list_Ki_seq_ModelFree.jld2" Dict_list_Ki_seq_ModelFree

@load dir_comparison * "/Dict_list_Kp_Sysid.jld2" Dict_list_Kp_Sysid
@load dir_comparison * "/Dict_list_Ki_Sysid.jld2" Dict_list_Ki_Sysid

## 相対誤差を計算する関数


Trials = 10
println("Num of Trials: ", Trials)
trial = 3

est_system = list_est_system[trial]

## ゲイン最適化の結果表示

#ゲイン最適化の結果をシミュレーションで表示
dir_result_trial = dir_result * "/trial$trial"
if !isdir(dir_result_trial)
    mkdir(dir_result_trial)  # フォルダを作成
end

dir_result_trial = dir_result_trial * "/Gain_Simulation_plot"
if !isdir(dir_result_trial)
    mkdir(dir_result_trial)  # フォルダを作成
end

## シミュレーションによる軌道の確認

x_0 = zeros(system.n)
x_0 = rand(system.rng, system.Dist_x0, system.n)
#Vanila_parameter5_zohでは二回目の初期値の方が対比しやすい
x_0 = rand(system.rng, system.Dist_x0, system.n)
z_0 = zeros(system.p)
T = 10
h = 5e-4
h = 0.0001
K_P_Mfree = (list_Kp_seq_ModelFree[trial])[end]
K_I_Mfree = (list_Ki_seq_ModelFree[trial])[end]

println("ModelFree Kp: ", K_P_Mfree)
println("ModelFree Ki: ", K_I_Mfree)

# システム同定後に最適化アルゴリズムを回したものを使う
K_P_SysId = list_Kp_Sysid[trial]
K_I_SysId = list_Ki_Sysid[trial]

println("SysId Kp: ", K_P_SysId)
println("SysId Ki: ", K_I_SysId)
println("Norm of SysId Kp: ", norm(K_P_SysId))
println("Norm of SysId Ki: ", norm(K_I_SysId))
#u_star_Sysid = list_ustar_Sysid[trial]
#u_hat = list_uhat[trial]

disc_system = ZOH_discrete_system(system, Ts)
println("abs closed loop eigvals of discrete:",
    abs.(eigvals(disc_system.F - disc_system.G * [K_P_SysId K_I_SysId] * disc_system.H))
)
println("abs open loop eigvals of discrete:",
    abs.(eigvals(disc_system.A))
)

_, y_s_sysid, _ = Orbit_zoh_PI(system, K_P_SysId, K_I_SysId, est_system.u_star, x_0, T, Ts=Ts, h=h)

_, y_s_hat, _ = Orbit_continuous_PI(system, K_P_Mfree, K_I_Mfree, list_uhat[trial], x_0, T, h=h)
Timeline = 0:h:T
println(size(y_s_sysid))
println(size(y_s_hat))

plotting = plot(legendfontsize=18, tickfontsize=15, guidefont=18)
plot!(plotting, Timeline[1:end], y_s_hat[1, 1:end], labels="Proposed Method", lw=1.8, lc=:red)
plot!(plotting, Timeline[1:end], y_s_sysid[1, 1:end], labels="Indirect Approach", lw=1.8, lc=:blue)
hline!(plotting, system.y_star, label=L"y^{\star}", lc=:black, lw=3)
xlims!(0, T)
#ylims!(-1, 6)
xlabel!("Time")
ylabel!(L"y")
savefig(plotting, dir_result_trial * "/Compare_Gain_estimatedFF.png")

for i in 1:system.p
    plotting = plot(legendfontsize=15, tickfontsize=15, legend=:best, guidefont=22)
    plot!(plotting, Timeline[1:end], y_s_hat[i, 1:end], labels="Proposed Method", lw=3.5, lc=:red)
    plot!(plotting, Timeline[1:end], y_s_sysid[i, 1:end], labels="Indirect Approach", lw=3.5, lc=:blue)
    hline!(plotting, system.y_star, label=L"y^{\star}", lc=:black, lw=3.5)
    xlims!(0, T)
    ylims!(-35, 35)
    xlabel!(L"t")
    ylabel!(L"y(t)")
    savefig(plotting, dir_result_trial * "/Compare_Gain_component$(i).png")
end



@save dir_comparison * "/per_system_list_obj_MFree.jld2" per_system_list_obj_MFree
@save dir_comparison * "/per_system_list_obj_SysId.jld2" per_system_list_obj_SysId
list_obj_MFree = per_system_list_obj_MFree[iter_system]
list_obj_SysId = per_system_list_obj_SysId[iter_system]

boxplot(list_obj_MFree,
    tickfontsize=15, yguidefont=font(15), legend=false, fillalpha=0.0,
    outliercolor=:white, markercolor=:white)
boxplot!(list_obj_SysId, fillalpha=0.0, outliercolor=:white, markercolor=:white)
xticks!((1:2, ["Proposed method", "Indirect approach"]))
ylims!(0, 0.35)
savefig(dir_result_trial * "/Gain_MeanObj_boxplot.png")


