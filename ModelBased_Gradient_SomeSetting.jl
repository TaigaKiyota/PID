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
using Profile

"""
事前に同定済みの複数システムに対して、モデルベースの離散時間勾配法
（`ProjGrad_Discrete_Conststep_ModelBased_Noise`）をまとめて実行するスクリプト。

前提:
- `Comparison_SomeSetting/Noise_dynamics/Setting<Setting_num>/<simulation_name>/Dict_list_est_system.jld2`
  にシステム同定済みモデル（各システムにつき複数トライアル）が保存されている。
- `System_setting/Noise_dynamics/Settings/Setting<Setting_num>/VS_ModelBase/<simulation_name>/params.json`
  に最適化パラメータが保存されている。
出力:
- 同じ `Comparison_SomeSetting/Noise_dynamics/...` 配下に
  `Dict_list_Kp_Sysid.jld2`, `Dict_list_Ki_Sysid.jld2` を保存。
"""

Setting_num = 10
simulation_name = "Vanila_parameter_zoh"

println("simulation_name: ", simulation_name)
@load "System_setting/Noise_dynamics/Settings/Setting$Setting_num/Settings.jld2" Setting
system = Setting["system"]

# 同定済みシステムの読み込み
dir_comparison = "Comparison_SomeSetting/Noise_dynamics/Setting$Setting_num"
dir_comparison = dir_comparison * "/" * simulation_name
@load dir_comparison * "/Dict_list_est_system.jld2" Dict_list_est_system

# パラメータの読み込み（モデルベース側の設定を流用）
dir_experiment_setting = "System_setting/Noise_dynamics/Settings/Setting$Setting_num/VS_ModelBase"
dir_experiment_setting = dir_experiment_setting * "/" * simulation_name
params = JSON.parsefile(dir_experiment_setting * "/params.json")

eta_discrete = params["eta_discrete"]
eta_discrete = 0.0001
epsilon_GD_discrete = params["epsilon_GD_discrete"]
N_GD_discrete = 500000 #500000
projection = params["projection"]
println("eta_discrete: ", eta_discrete)

K_P_disc = 0.01 * I(system.p) #zeros((system.m, system.p))
K_I_disc = 0.01 * I(system.p) #zeros((system.m, system.p))

Q1 = 0.1 * I(system.p)
Q2 = 0.01 * I(system.p)

eps_interval = NaN
M_interval = NaN
N_sample = NaN
N_inner_obj = NaN
tau = NaN
r = NaN
delta = NaN
epsilon_EstGrad = NaN
method_name = NaN

# モデルベース離散勾配法の最適化パラメータ
Opt_discrete = Optimization_param(
  eta_discrete,
  epsilon_GD_discrete,
  epsilon_EstGrad,
  delta,
  eps_interval,
  M_interval,
  N_sample,
  N_inner_obj,
  N_GD_discrete,
  tau,
  r,
  projection,
  method_name
)

struct Problem_param
  Q1
  Q2
  Q_prime
  last_value
end

# 各システムの最終ゲイン列を格納する辞書
Dict_list_Kp_Sysid = Dict{Any,Any}()
Dict_list_Ki_Sysid = Dict{Any,Any}()

num_of_systems = length(Dict_list_est_system)

for iter_system in 1:num_of_systems
  println("iter_system: ", iter_system)
  list_est = Dict_list_est_system["system$iter_system"]
  num_trials = length(list_est)

  local_Kp = Vector{Matrix{Float64}}(undef, num_trials)
  local_Ki = Vector{Matrix{Float64}}(undef, num_trials)

  for trial in 1:num_trials
    est_system = list_est[trial]
    Q_prime_sysid = [est_system.C'*Q1*est_system.C zeros(est_system.n, est_system.p); zeros(est_system.p, est_system.n) Q2]
    Q_prime_sysid = Symmetric((Q_prime_sysid + Q_prime_sysid') / 2)
    prob_sysid = Problem_param(Q1, Q2, Q_prime_sysid, true)

    Kp_seq_SysId, Ki_seq_SysId, _ = ProjGrad_Discrete_Conststep_ModelBased_Noise(
      K_P_disc,
      K_I_disc,
      est_system,
      prob_sysid,
      Opt_discrete,
    )
    local_Kp[trial] = Kp_seq_SysId[end]
    local_Ki[trial] = Ki_seq_SysId[end]
    println("iter_system $iter_system trial $trial has done")
  end

  Dict_list_Kp_Sysid["system$iter_system"] = local_Kp
  Dict_list_Ki_Sysid["system$iter_system"] = local_Ki
end

@save dir_comparison * "/Dict_list_Kp_Sysid.jld2" Dict_list_Kp_Sysid
@save dir_comparison * "/Dict_list_Ki_Sysid.jld2" Dict_list_Ki_Sysid

