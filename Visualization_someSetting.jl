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

ErrorNorm(A, Aans, A0) = sqrt(sum((A - Aans) .^ 2) / sum((Aans - A0) .^ 2))

Setting_num = 6
simulation_name = "Vanila_parameter"

dir = "Comparison_SomeSetting/Noise_dynamics/Setting$Setting_num/$simulation_name"

@load dir * "/MFree_u_error_list.jld2" MFree_u_error_list
@load dir * "/MBase_u_error_list.jld2" MBase_u_error_list
@load dir * "/MFree_ObjMean_list.jld2" MFree_ObjMean_list
@load dir * "/MBase_ObjMean_list.jld2" MBase_ObjMean_list

println("FF相対誤差：モデルフリーが勝った数: ", count(x -> x, MFree_u_error_list .< MBase_u_error_list))
println("FF相対誤差：モデルフリー: ", MFree_u_error_list)
println("FF相対誤差：モデルベース: ", MBase_u_error_list)

# 箱ひげ図のプロット
boxplot(MFree_u_error_list,
    tickfontsize=18, yguidefont=font(20), fillcolor=:red, legend=false, outliercolor=:red, markercolor=:red)
boxplot!(MBase_u_error_list, fillcolor=:blue, outliercolor=:blue, markercolor=:blue)
xticks!((1:2, ["Proposed method", "Indirect approach"]))
#yticks!([1e-2, 1e-1, 1, 10, 20], ["0.01", "0.1", "1", "10", "20"]),
#ylims!(1e-2, 20)
ylabel!(L"\|\| u_0 - u^{\star} \|\| / \|\|u^{\star}\|\|")
savefig(dir * "/FF_error_boxplot.png")

println("Mean Model-free: ", mean(MFree_u_error_list))
println("standard error Model-free: ", std(MFree_u_error_list))

println("Mean SysId: ", mean(MBase_u_error_list))
println("standard error  SysId: ", std(MBase_u_error_list))

println("ゲイン最適化：モデルフリーが勝った数： ", count(x -> x, MFree_ObjMean_list .< MBase_ObjMean_list))
println("目的関数：モデルフリー: ", MFree_ObjMean_list)
println("目的関数：モデルベース: ", MBase_ObjMean_list)
println("Mean Model-free: ", mean(MFree_ObjMean_list))
println("standard error Model-free: ", std(MFree_ObjMean_list))

println("Mean SysId: ", mean(MBase_ObjMean_list))
println("standard error  SysId: ", std(MBase_ObjMean_list))

Ratio_ObjMean = MBase_ObjMean_list ./ MFree_ObjMean_list
println("Ratio_ObjMean: ", Ratio_ObjMean)
println("Mean Ratio: ", mean(Ratio_ObjMean))
println("standard error Ratio: ", std(Ratio_ObjMean))