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
## 相対誤差を計算するもの　
ErrorNorm(A, Aans, A0) = sqrt(sum((A - Aans) .^ 2) / sum((Aans - A0) .^ 2))

Setting_num = 10
simulation_name = "FF_Vanila_parameter_zoh"
estimated_param = false


simulation_name_param = "Vanila_parameter_zoh"
dir_comparison = "Comparison_SomeSetting/Noise_dynamics/Setting$Setting_num"
dir_comparison = dir_comparison * "/" * simulation_name_param
@load dir_comparison * "/Dict_original_systems.jld2" Dict_original_systems

dir_experiment_setting = "System_setting/Noise_dynamics/Settings/Setting$Setting_num/VS_ModelBase"
dir_experiment_setting = dir_experiment_setting * "/" * simulation_name
params = JSON.parsefile(dir_experiment_setting * "/params.json")
Ts = params["Ts"]
Ts = 0.01
println(Ts)

Trials = 10
num_of_systems = 10



## ディレクトリ作成
dir_result = "Comparison_SomeSetting/Noise_dynamics/Setting$Setting_num"

if !isdir(dir_result)
    mkdir(dir_result)  # フォルダを作成
end
dir_result = dir_result * "/" * simulation_name
if !isdir(dir_result)
    mkdir(dir_result)  # フォルダを作成
end

@load dir_result * "/per_system_list_uhat.jld2" per_system_list_uhat
@load dir_result * "/per_system_list_ustar_Sysid.jld2" per_system_list_ustar_Sysid
@load dir_result * "/per_system_estimated_tauu.jld2" per_system_estimated_tauu

#=
for iter_system in 1:num_of_systems
    system = Dict_original_systems["system$iter_system"]
    list_uhat = per_system_list_uhat[iter_system]
    list_ustar_Sysid = per_system_list_ustar_Sysid[iter_system]

    list_error_ystar_Sysid = Vector{Float64}(undef, Trials)
    list_error_ystar_MFree = Vector{Float64}(undef, Trials)
    for trial in 1:Trials
        ustar_sysid = list_ustar_Sysid[trial]
        y_inf_sysid = -system.C * (system.A \ (system.B * ustar_sysid))
        list_error_ystar_Sysid[trial] = ErrorNorm(y_inf_sysid, system.y_star, zeros(system.m))

        uhat = list_uhat[trial]
        y_inf_uhat = -system.C * (system.A \ (system.B * uhat))
        list_error_ystar_MFree[trial] = ErrorNorm(y_inf_uhat, system.y_star, zeros(system.m))
    end
    if !isdir(dir_result * "/system$iter_system")
        mkdir(dir_result * "/system$iter_system")  # フォルダを作成
    end
    boxplot(list_error_ystar_MFree,
        tickfontsize=15, yguidefont=font(15), legend=false, fillalpha=0.0, outliercolor=:white, markercolor=:white)
    boxplot!(list_error_ystar_Sysid, fillalpha=0.0, outliercolor=:white, markercolor=:white)
    xticks!((1:2, ["Proposed method", "Indirect approach"]))
    #ylims!(0, 0.35)
    savefig(dir_result * "/system$iter_system/FF_y_error_boxplot.png")

end
=#
#個別のシステムのスケールを調整したい
iter_system = 1
list_uhat = per_system_list_uhat[iter_system]
list_ustar_Sysid = per_system_list_ustar_Sysid[iter_system]
system = Dict_original_systems["system$iter_system"]
tau_u = per_system_estimated_tauu[iter_system]
Num_Samples_per_traj = (system.m + 1) * tau_u / Ts
Num_Samples_per_traj = Int(trunc(Num_Samples_per_traj))
println("tau_u: ", tau_u)
println("Num_Samples_per_traj: ", Num_Samples_per_traj)

list_error_ystar_Sysid = Vector{Float64}(undef, Trials)
list_error_ystar_MFree = Vector{Float64}(undef, Trials)
for trial in 1:Trials
    ustar_sysid = list_ustar_Sysid[trial]
    y_inf_sysid = -system.C * (system.A \ (system.B * ustar_sysid))
    list_error_ystar_Sysid[trial] = ErrorNorm(y_inf_sysid, system.y_star, zeros(system.m))

    uhat = list_uhat[trial]
    y_inf_uhat = -system.C * (system.A \ (system.B * uhat))
    list_error_ystar_MFree[trial] = ErrorNorm(y_inf_uhat, system.y_star, zeros(system.m))
end

boxplot(list_error_ystar_MFree,
    tickfontsize=15, yguidefont=font(15), legend=false, fillalpha=0.0, outliercolor=:white, markercolor=:white)
boxplot!(list_error_ystar_Sysid, fillalpha=0.0, outliercolor=:white, markercolor=:white)
xticks!((1:2, ["Proposed method", "Model-based Method"]))
ylabel!("Steady-state output relative error", yguidefont=font(14))
ylims!(0, 0.35)
savefig(dir_result * "/system$iter_system/FF_y_error_boxplot.png")

#プロットをする
iter_system = 1
trial = 1

list_uhat = per_system_list_uhat[iter_system]
list_ustar_Sysid = per_system_list_ustar_Sysid[iter_system]
system = Dict_original_systems["system$iter_system"]

K_P = zeros(system.p, system.p)
ustar_sysid = list_ustar_Sysid[trial]
uhat = list_uhat[trial]
T = 3
h = 0.005
x_0 = rand(system.rng, system.Dist_x0, system.n)
x_0 = rand(system.rng, system.Dist_x0, system.n)
x_0 = rand(system.rng, system.Dist_x0, system.n)
x_0 = rand(system.rng, system.Dist_x0, system.n)
x_0 = zeros(system.n)
_, ys_uhat = Orbit_continuous_P(system, K_P, uhat, x_0, T, h=h)
_, ys_sysid = Orbit_continuous_P(system, K_P, ustar_sysid, x_0, T, h=h)
Timeline = 0:h:T

if !isdir(dir_result * "/system$iter_system/trial$trial")
    mkdir(dir_result * "/system$iter_system/trial$trial")  # フォルダを作成
end

for i in 1:system.p
    plotting = plot(legendfontsize=15, tickfontsize=15, legend=:best, guidefont=22)
    plot!(plotting, Timeline[1:end], ys_uhat[i, 1:end], labels="Proposed Method", lw=3.5, lc=:red)
    plot!(plotting, Timeline[1:end], ys_sysid[i, 1:end], labels="Model-based Method", lw=3.5, lc=:blue)
    hline!(plotting, system.y_star, label=L"y^{\star}", lc=:black, lw=3.5)
    xlims!(0, T)
    #ylims!(-35, 35)
    xlabel!(L"t")
    ylabel!(L"y(t)")
    savefig(plotting, dir_result * "/system$iter_system/trial$trial" * "/Compare_FF_component$(i).png")
end
