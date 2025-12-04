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


#@load dir * "/list_ustar_Sysid.jld2" list_ustar_Sysid
@load dir * "/list_uhat.jld2" list_uhat

Trials = 20
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
    tickfontsize=18, yguidefont=font(20), fillcolor=:red, legend=false, outliercolor=:red, markercolor=:red)
boxplot!(list_error_ystar_Sysid, fillcolor=:blue, outliercolor=:blue, markercolor=:blue)
xticks!((1:2, ["Proposed method", "Indirect approach"]))
#yticks!([1e-2, 1e-1, 1, 10, 20], ["0.01", "0.1", "1", "10", "20"]),
#ylims!(1e-2, 20)
#ylabel!(L"\|\| u_0 - u^{\star} \|\|")
savefig(dir * "/FF_y_error_boxplot.png")


#@load dir * "/list_Kp_seq_Sysid.jld2" list_Kp_seq_Sysid
#@load dir * "/list_Ki_seq_Sysid.jld2" list_Ki_seq_Sysid
@load dir * "/list_Kp_seq_ModelFree.jld2" list_Kp_seq_ModelFree
@load dir * "/list_Ki_seq_ModelFree.jld2" list_Ki_seq_ModelFree


@load dir * "/list_Kp_Sysid.jld2" list_Kp_Sysid
@load dir * "/list_Ki_Sysid.jld2" list_Ki_Sysid
## 相対誤差を計算する関数


Trials = 20
println("Num of Trials: ", Trials)
trial = 3

est_system = list_est_system[trial]

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
x_0 = rand(system.rng, system.Dist_x0, system.n)
#Vanila_parameter5_zohでは二回目の初期値の方が対比しやすい
x_0 = rand(system.rng, system.Dist_x0, system.n)
z_0 = zeros(system.p)
T = 0.3
h = 5e-4
K_P_Mfree = (list_Kp_seq_ModelFree[trial])[end]
K_I_Mfree = (list_Ki_seq_ModelFree[trial])[end]
#K_P_Mfree = I(system.p)
#K_I_Mfree = I(system.p)
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
savefig(plotting, dir_path * "/Compare_Gain_estimatedFF.png")

for i in 1:system.p
    plotting = plot(legendfontsize=15, tickfontsize=15, legend=:best, guidefont=22)
    plot!(plotting, Timeline[1:end], y_s_hat[i, 1:end], labels="Proposed Method", lw=3.5, lc=:red)
    plot!(plotting, Timeline[1:end], y_s_sysid[i, 1:end], labels="Indirect Approach", lw=3.5, lc=:blue)
    hline!(plotting, system.y_star, label=L"y^{\star}", lc=:black, lw=3.5)
    xlims!(0, T)
    ylims!(-35, 35)
    xlabel!(L"t")
    ylabel!(L"y(t)")
    savefig(plotting, dir_path * "/Compare_Gain_component$(i).png")
end



@load dir * "/list_obj_MFree_Comparison_ObjFunc.jld2" list_obj_MFree
@load dir * "/list_obj_SysId_Comparison_ObjFunc.jld2" list_obj_SysId

boxplot(list_obj_MFree,
    tickfontsize=18, yguidefont=font(20), fillcolor=:red, legend=false, outliercolor=:red, markercolor=:red)
boxplot!(list_obj_SysId, fillcolor=:blue, outliercolor=:blue, markercolor=:blue)
xticks!((1:2, ["Proposed method", "Indirect approach"]))
#yticks!([1e-2, 1e-1, 1, 10, 20], ["0.01", "0.1", "1", "10", "20"]),
#ylims!(1e-2, 20)
#ylabel!(L"\|\| u_0 - u^{\star} \|\|")
savefig(dir * "/Gain_MeanObj_boxplot.png")

boxplot(list_obj_MFree, tickfontsize=15, bar_width=0.3, yguidefont=font(18), label="Proposed method")
xticks!((1:1, ["Proposed method"]))
#ylabel!(L"\|\| u_0 - u^{\star} \|\|")
savefig(dir * "/Gain_MeanObj_boxplot_MFree.png")

boxplot(list_obj_SysId, tickfontsize=15, bar_width=0.3, yguidefont=font(18), label="Indirect approach")
#ylabel!(L"\|\| u_0 - u^{\star} \|\|")
xticks!((1:1, ["Indirect approach"]))
savefig(dir * "/Gain_MeanObj_boxplot_SysId.png")

