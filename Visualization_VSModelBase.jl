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
simulation_name = "Vanila_parameter2"

@load "System_setting/Noise_dynamics/Settings/Setting$Setting_num/Settings.jld2" Setting

dir = "System_setting/Noise_dynamics/Settings/Setting$Setting_num/VS_ModelBase"
dir = dir * "/" * simulation_name

system = Setting["system"]

params = JSON.parsefile(dir * "/params.json")
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
println("Trials: ", Trials)
## FF推定の結果表示
list_error_u_Sysid = []
list_error_u_MFree = []
for trial in 1:Trials
    push!(list_error_u_Sysid, ErrorNorm(list_ustar_Sysid[trial], system.u_star, zeros(system.m)))
    push!(list_error_u_MFree, ErrorNorm(list_uhat[trial], system.u_star, zeros(system.m)))
end

println("Model-Free FF error", list_error_u_MFree)
println("Mean Model-free: ", mean(list_error_u_MFree))
println("standard error Model-free: ", std(list_error_u_MFree))

println("SysId FF error", list_error_u_Sysid)
println("Mean SysId: ", mean(list_error_u_Sysid))
println("standard error  SysId: ", std(list_error_u_Sysid))


boxplot(list_error_u_MFree,
    tickfontsize=18, yscale=:log10, yguidefont=font(20), fillcolor=:red, legend=false)
boxplot!(list_error_u_Sysid, fillcolor=:blue,)
xticks!((1:2, ["Proposed method", "Indirect approach"]))
yticks!([1e-2, 1e-1, 1, 10, 20], ["0.01", "0.1", "1", "10", "20"]),
ylims!(1e-2, 20)
ylabel!(L"\|\| u_0 - u^{\star} \|\|")
savefig(dir * "/FF_error_boxplot.png")

boxplot(list_error_u_MFree, tickfontsize=15, bar_width=0.3, yguidefont=font(18), label="Proposed method")
xticks!((1:1, ["Proposed method"]))
ylabel!(L"\|\| u_0 - u^{\star} \|\|")
savefig(dir * "/FF_error_boxplot_MFree.png")

boxplot(list_error_u_Sysid, tickfontsize=15, bar_width=0.3, yguidefont=font(18), label="Indirect approach")
ylabel!(L"\|\| u_0 - u^{\star} \|\|")
xticks!((1:1, ["Indirect approach"]))
savefig(dir * "/FF_error_boxplot_SysId.png")

dir_path = dir * "/FF_Simulation_plot"
if !isdir(dir_path)
    mkdir(dir_path)  # フォルダを作成
end

## シミュレーションによる軌道の確認
trial = 5
x_0 = zeros(system.n)
z_0 = zeros(system.p)
T = 5
K_P = 0.1I(system.p)
K_I = 0.01 * I(system.p)

u_star_Sysid = list_ustar_Sysid[trial]
u_hat = list_uhat[trial]

_, y_s_hat, _, Timeline, _, _ = Orbit_PI_noise(system, K_P, K_I, x_0, z_0, T, u_hat)
_, y_s_sysid, _, Timeline, _, _ = Orbit_PI_noise(system, K_P, K_I, x_0, z_0, T, u_star_Sysid)

plotting = plot(legendfontsize=18, tickfontsize=15, guidefont=18)
plot!(plotting, Timeline[1:end], y_s_hat[1, 1:end], labels="Proposed Method", lw=1.8, lc=:red)
plot!(plotting, Timeline[1:end], y_s_sysid[1, 1:end], labels="Indirect Approach", lw=1.8, lc=:blue)
hline!(plotting, system.y_star, label=L"y^{\star}", lc=:black, lw=3)
xlims!(0, T)
#ylims!(-1, 6)
xlabel!("Time")
ylabel!(L"y")
savefig(plotting, dir_path * "/Compare_FF.png")

for i in 1:system.p
    plotting = plot(legendfontsize=15, tickfontsize=15, legend=:right, guidefont=22)
    plot!(plotting, Timeline[1:end], y_s_hat[i, 1:end], labels="Proposed Method", lw=2, lc=:red)
    plot!(plotting, Timeline[1:end], y_s_sysid[i, 1:end], labels="Indirect Approach", lw=2, lc=:blue)
    hline!(plotting, system.y_star, label=L"y^{\star}", lc=:black, lw=4)
    xlims!(0, T)
    #ylims!(-3, 8)
    xlabel!(L"t")
    ylabel!(L"y(t)")
    savefig(plotting, dir_path * "/Compare_FF_component$i.png")
end



## ゲイン最適化の結果表示

list_seq_obj_MFree_uhat = []
list_seq_obj_SysId = []
for trial in 1:Trials
    Obj_seq_MFree_uhat = []
    Obj_seq_SysId = []
    for i in 1:N_GD
        if i <= (size(list_Kp_seq_ModelFree[trial], 1))
            Obj_MFree_uhat = ObjectiveFunction_noise(system, prob, (list_Kp_seq_ModelFree[trial])[i], (list_Ki_seq_ModelFree[trial])[i])
        else
            Obj_MFree_uhat = ObjectiveFunction_noise(system, prob, (list_Kp_seq_ModelFree[trial])[end], (list_Ki_seq_ModelFree[trial])[end])
        end

        if i <= (size(list_Kp_seq_Sysid[trial], 1))
            Obj_SysId = ObjectiveFunction_noise(system, prob, (list_Kp_seq_Sysid[trial])[i], (list_Ki_seq_Sysid[trial])[i])
        else
            Obj_SysId = ObjectiveFunction_noise(system, prob, (list_Kp_seq_Sysid[trial])[end], (list_Ki_seq_Sysid[trial])[end])
        end
        push!(Obj_seq_MFree_uhat, Obj_MFree_uhat)
        push!(Obj_seq_SysId, Obj_SysId)
    end
    push!(list_seq_obj_MFree_uhat, Obj_seq_MFree_uhat)
    push!(list_seq_obj_SysId, Obj_seq_SysId)
end

Concat_Mfree_hat = reduce(hcat, [c for c in list_seq_obj_MFree_uhat])
Mean_Mfree_hat = mean(Concat_Mfree_hat; dims=2)[:]
std_Mfree_hat = std(Concat_Mfree_hat; dims=2)[:]

Concat_MBase_Sysid = reduce(hcat, [c for c in list_seq_obj_SysId])
Mean_MBase_Sysid = mean(Concat_MBase_Sysid; dims=2)[:]
std_MBase_Sysid = std(Concat_MBase_Sysid; dims=2)[:]

plotting = plot(legendfontsize=18, tickfontsize=15, guidefont=18, legend=:right)
# プロット（平均 ± 1σのリボン）
plot!(plotting, 1:N_GD, Mean_Mfree_hat, ribbon=std_Mfree_hat, label="Model Free", lw=3, lc=:red, fillcolor=:red)
plot!(plotting, 1:N_GD, Mean_MBase_Sysid, ribbon=std_MBase_Sysid, label="Indirect Approach", lw=3, lc=:blue, fillcolor=:blue)
xlabel!("Iteration")
ylabel!(L"f(K;u^{\star})")
savefig(plotting, dir * "/Gain_ConvergenceCurve.png")

#ゲイン最適化の結果をシミュレーションで表示
dir_path = dir * "/Gain_Simulation_plot"
if !isdir(dir_path)
    mkdir(dir_path)  # フォルダを作成
end
## シミュレーションによる軌道の確認
trial = 5
x_0 = zeros(system.n)
z_0 = zeros(system.p)
T = 5
K_P_Mfree = (list_Kp_seq_ModelFree[trial])[end]
K_I_Mfree = (list_Ki_seq_ModelFree[trial])[end]

K_P_SysId = (list_Kp_seq_Sysid[trial])[end]
K_I_SysId = (list_Ki_seq_Sysid[trial])[end]

u_star_Sysid = list_ustar_Sysid[trial]
u_hat = list_uhat[trial]

_, y_s_hat, _, Timeline, _, _ = Orbit_PI_noise(system, K_P_Mfree, K_I_SysId, x_0, z_0, T, u_hat)
_, y_s_sysid, _, Timeline, _, _ = Orbit_PI_noise(system, K_P_Mfree, K_I_SysId, x_0, z_0, T, u_star_Sysid)

plotting = plot(legendfontsize=18, tickfontsize=15, guidefont=18)
plot!(plotting, Timeline[1:end], y_s_hat[1, 1:end], labels="Proposed Method", lw=1.8, lc=:red)
plot!(plotting, Timeline[1:end], y_s_sysid[1, 1:end], labels="Indirect Approach", lw=1.8, lc=:blue)
hline!(plotting, system.y_star, label=L"y^{\star}", lc=:black, lw=3)
xlims!(0, T)
#ylims!(-1, 6)
xlabel!("Time")
ylabel!(L"y")
savefig(plotting, dir_path * "/Compare_Gain.png")

for i in 1:system.p
    plotting = plot(legendfontsize=15, tickfontsize=15, legend=:right, guidefont=22)
    plot!(plotting, Timeline[1:end], y_s_hat[i, 1:end], labels="Proposed Method", lw=2, lc=:red)
    plot!(plotting, Timeline[1:end], y_s_sysid[i, 1:end], labels="Indirect Approach", lw=2, lc=:blue)
    hline!(plotting, system.y_star, label=L"y^{\star}", lc=:black, lw=4)
    xlims!(0, T)
    #ylims!(-3, 8)
    xlabel!(L"t")
    ylabel!(L"y(t)")
    savefig(plotting, dir_path * "/Compare_Gain_component$i.png")
end