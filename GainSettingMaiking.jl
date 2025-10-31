using Random
using LinearAlgebra
using Plots
using LaTeXStrings
using ControlSystems

include("../function.jl")
include("../function_orbit.jl")

rng = MersenneTwister(1)

using JLD2

Setting_num = 5

if Setting_num == 1
    n = 4
    m = 2
    p = m
    y_ans = 5 * ones(p)
    N_GD = 5000
    N_sample = 70
    eta = 0.001
    tau_u = 10

    eps_interval = 0.01
    M_interval = 2
    epsilon = 0.01
    r = 0.005
    h = 0.01
    N_sample = 60
    tau = 20

    K_P0 = (M_interval / 2) * I(p)
    K_I0 = (M_interval / 2) * I(p)
    K_D0 = (M_interval / 2) * I(p)
elseif Setting_num == 2
    n = 4
    m = 2
    p = m
    y_ans = 2 * ones(p)
    N_GD = 5000
    N_sample = 70
    eta = 0.001
    tau_u = 10

    eps_interval = 0.01
    M_interval = 2
    epsilon = 0.01
    r = 0.005
    h = 0.01
    N_sample = 60
    tau = 20

    K_P0 = (M_interval / 2) * I(p)
    K_I0 = (M_interval / 2) * I(p)
    K_D0 = (M_interval / 2) * I(p)
elseif Setting_num == 3
    n = 10
    m = 4
    p = m
    y_ans = 5 * ones(p)
    N_GD = 5000
    eta = 0.001
    tau_u = 20

    eps_interval = 0.01
    M_interval = 2
    epsilon = 0.01
    r = 0.005
    h = 0.01
    N_sample = 60
    tau = 20

    K_P0 = (M_interval / 2) * I(p)
    K_I0 = (M_interval / 2) * I(p)
    K_D0 = (M_interval / 2) * I(p)
elseif Setting_num == 4
    n = 10
    m = 4
    p = m
    y_ans = 2 * ones(p)
    N_GD = 5000
    eta = 0.001
    tau_u = 20

    eps_interval = 0.01
    M_interval = 2
    epsilon = 0.01
    r = 0.005
    h = 0.01
    N_sample = 60
    tau = 20

    K_P0 = (M_interval / 2) * I(p)
    K_I0 = (M_interval / 2) * I(p)
    K_D0 = (M_interval / 2) * I(p)
elseif Setting_num == 5
    n = 10
    m = 4
    p = m
    y_ans = 5 * ones(p)
    N_GD = 5000
    eta = 0.001
    tau_u = 20

    eps_interval = 0.01
    M_interval = 2
    epsilon = 0.01
    r = 0.005
    h = 0.01
    N_sample = 60
    tau = 20

    K_P0 = M_interval * I(p)
    K_I0 = eps_interval * I(p)
    K_D0 = 0 * I(p)
end

#実験するシステムの作成
J = randn(rng, Float64, (n, n))
J = (J - J') / 2
Randmat = randn(rng, Float64, (n, n))
Orth = qr(Randmat).Q
R = Randmat * Randmat'
Randmat = randn(rng, Float64, (n, n))
Hamilton = I(n)
A = (J - R) * Hamilton
B = randn(rng, Float64, (n, m))
C = B' * Hamilton
F = [A zeros(n, p); -C zeros(p, p)]
G = [B; zeros(p, m)]
H = [C zeros(p, p); zeros(p, n) -I(p); C*A zeros(p, p)]
C_aug = [C'*C zeros(n, p); zeros(p, n) zeros(p, p)]
equib = inv([A B; C zeros(p, m)]) * [zeros(n); y_ans]
x_equib = equib[1:n]
u_equib = equib[(n+1):(n+p)]
##フィードフォワードの設計
#ベースとなる誤差の収集
x_0 = Sample_x0(n)
reset = zeros(m)
K_P = I(p)
A_K = A - B * K_P * C
err_state = inv(A_K) * B * (reset - u_equib)
error_zero = -C * (exp(A_K * tau_u) * (x_0 - x_equib + err_state) - err_state)
Ematrix = zeros(m, m)
Umatrix = randn(rng, Float64, (m, m))

#データの収集
for i in 1:m
    global Umatrix
    global Ematrix
    reset = Umatrix[:, i]
    reset = reset ./ norm(reset)
    Umatrix[:, i] = reset
    x_0 = Sample_x0(n)
    err_state = inv(A_K) * B * (reset - u_equib)
    error_i = -C * (exp(A_K * tau_u) * (x_0 - x_equib + err_state) - err_state)
    Ematrix[:, i] = error_i - error_zero
end
u_reset = -Umatrix * inv(Ematrix) * error_zero
println("error of feedforward: ", sqrt(sum((u_reset - u_equib) .^ 2)))

Setting = Dict(
    "u_reset" => u_reset,
    "eta" => eta,
    "N_GD" => N_GD,
    "eps_interval" => eps_interval,
    "M_interval" => M_interval,
    "epsilon" => epsilon,
    "r" => r,
    "h" => h,
    "N_sample" => N_sample,
    "tau_u" => tau_u,
    "tau" => tau,
    "A" => A,
    "B" => B,
    "C" => C,
    "y_ans" => y_ans,
    "K_P0" => K_P0,
    "K_I0" => K_I0,
    "K_D0" => K_D0
)

#println("Setting: ", Setting)
if !isdir("test_file/Gain_result/Settings")
    mkdir("test_file/Gain_result/Settings")  # フォルダを作成
end

@save "test_file/Gain_result/Settings/Setting$Setting_num.jld2" Setting




