using Random
using LinearAlgebra
using Plots
using LaTeXStrings
using ControlSystems
using Zygote
using JLD2
using Distributions
using SparseArrays

include("function_noise.jl")
include("function_orbit.jl")

rng = MersenneTwister(1)


Setting_num = 10

function random_sparse_columns(n, m, rng, nz_per_col)
    A = zeros(n, m)

    for j in 1:m
        rows = randperm(rng, n)[1:nz_per_col]
        vals = randn(rng, nz_per_col)
        A[rows, j] = vals                      # 非ゼロ成分を代入
    end
    return sparse(A)   # 圧縮形式に整形
end

if Setting_num == 1
    n = 4
    m = 2
    p = m

    J = 0.1 * randn(rng, Float64, (n, n))
    J = (J - J') / 2
    Randmat = 0.1 * randn(rng, Float64, (n, n))
    R = Randmat * Randmat'
    Hamilton = I(n)
    A = (J - R) * Hamilton
    B = 0.1 * randn(rng, Float64, (n, m))
    C = B' * Hamilton
    y_star = ones(p)
    W = I(n)
    V = 0.1 * I(p)
    h = 0.005
    Dist_x0 = Uniform(-3, 3)
elseif Setting_num == 2
    n = 4
    m = 2
    p = m

    J = 0.1 * randn(rng, Float64, (n, n))
    J = (J - J') / 2
    Randmat = 0.1 * randn(rng, Float64, (n, n))
    R = Randmat * Randmat'
    Hamilton = I(n)
    A = (J - R) * Hamilton
    B = 0.1 * randn(rng, Float64, (n, m))
    C = B' * Hamilton
    y_star = ones(p)
    W = 0.00001I(n)
    V = 0.00001 * I(p)
    h = 0.005
    Dist_x0 = Uniform(-3, 3)
elseif Setting_num == 3
    n = 40
    m = 4
    p = m
    y_star = 5 * ones(p)

    J = randn(rng, Float64, (n, n))
    J = (J - J') / 2
    Randmat = randn(rng, Float64, (n, n))
    R = Randmat * Randmat'
    Hamilton = I(n)
    A = (J - R) * Hamilton
    B = randn(rng, Float64, (n, m))
    C = B' * Hamilton
    W = 0.01 * I(n)
    V = 0.001 * I(p)
    h = 0.001
    Dist_x0 = Uniform(-3, 3)
elseif Setting_num == 4
    n = 40
    m = 4
    p = m
    y_star = 5 * ones(p)

    J = 0.1 * randn(rng, Float64, (n, n))
    J = (J - J') / 2
    Randmat = 0.1 * randn(rng, Float64, (n, n))
    R = Randmat * Randmat'
    Hamilton = I(n)
    A = (J - R) * Hamilton
    B = 0.1 * randn(rng, Float64, (n, m))
    C = B' * Hamilton
    W = 0.01 * I(n)
    V = 0.0005 * I(p)
    h = 0.005
    Dist_x0 = Uniform(-3, 3)
elseif Setting_num == 5
    n = 20
    m = 4
    p = m
    y_star = 5 * ones(p)

    J = 1.0 * randn(rng, Float64, (n, n))
    J = (J - J') / 2
    Randmat = 2.0 * randn(rng, Float64, (n, n))
    R = Randmat * Randmat'
    Hamilton = 1.0 * I(n)
    A = (J - R) * Hamilton
    B = 3 * randn(rng, Float64, (n, m))
    C = B' * Hamilton
    W = 0.01 * I(n)
    V = 0.0005 * I(p)
    h = 0.005
    Dist_x0 = Uniform(-3, 3)
elseif Setting_num == 6
    # 論文に載せる比較として使用できそう
    n = 30
    m = 4
    p = m
    y_star = 5 * ones(p)

    J = 1.0 * randn(rng, Float64, (n, n))
    J = (J - J') / 2
    Randmat = 2.0 * randn(rng, Float64, (n, n))
    R = Randmat * Randmat'
    Hamilton = 1.0 * I(n)
    A = (J - R) * Hamilton
    B = 3 * randn(rng, Float64, (n, m))
    C = B' * Hamilton
    W = 0.01 * I(n)
    V = 0.0005 * I(p)
    h = 0.0005
    Dist_x0 = Uniform(-3, 3)
elseif Setting_num == 7
    # Setting_num 6から共分散行列をランダムに
    n = 30
    m = 4
    p = m
    y_star = 5 * ones(p)

    J = 1.0 * randn(rng, Float64, (n, n))
    J = (J - J') / 2
    Randmat = 2.0 * randn(rng, Float64, (n, n))
    R = Randmat * Randmat'
    Hamilton = 1.0 * I(n)
    A = (J - R) * Hamilton
    B = 3 * randn(rng, Float64, (n, m))
    C = B' * Hamilton
    Randmat_w = randn(rng, Float64, (n, n))
    Rand_w = Randmat_w * Randmat_w'
    W = 0.001 * Rand_w
    Randmat_p = randn(rng, Float64, (p, p))
    Rand_p = Randmat_p * Randmat_p'
    V = 0.0001 * Rand_p
    h = 0.0005
    Dist_x0 = Uniform(-3, 3)
elseif Setting_num == 8
    # Setting_num 6から共分散行列をランダムに
    n = 20
    m = 2
    p = m
    y_star = 5 * ones(p)

    J = 1.0 * randn(rng, Float64, (n, n))
    J = (J - J') / 2
    Randmat = 2.0 * randn(rng, Float64, (n, n))
    R = Randmat * Randmat'
    Hamilton = 1.0 * I(n)
    A = (J - R) * Hamilton
    B = 3 * randn(rng, Float64, (n, m))
    C = B' * Hamilton
    Randmat_w = randn(rng, Float64, (n, n))
    Rand_w = Randmat_w * Randmat_w'
    W = 0.005 * Rand_w
    Randmat_p = randn(rng, Float64, (p, p))
    Rand_p = Randmat_p * Randmat_p'
    V = 0.0001 * Rand_p
    h = 0.001
    Dist_x0 = Uniform(-3, 3)
elseif Setting_num == 9
    # スパースな観測行列へ
    n = 30
    m = 4
    p = m
    sparse_num = 3
    y_star = 5 * ones(p)

    J = 1.0 * randn(rng, Float64, (n, n))
    J = (J - J') / 2
    Randmat = 2.0 * randn(rng, Float64, (n, n))
    R = Randmat * Randmat'
    Hamilton = 1.0 * I(n)
    A = (J - R) * Hamilton
    B = 3 * randn(rng, Float64, (n, m))
    #B = sparse(B)
    C = B' * Hamilton
    #C = sparse(C)
    Randmat_w = randn(rng, Float64, (n, n))
    Rand_w = Randmat_w * Randmat_w'
    W = 0.05 * I(n)
    Randmat_p = randn(rng, Float64, (p, p))
    Rand_p = Randmat_p * Randmat_p'
    V = 0.0005 * I(p)
    h = 0.001
    Dist_x0 = Uniform(-3, 3)
elseif Setting_num == 10
    # Setting_num 10から時間幅を細かく
    n = 30
    m = 4
    p = m
    y_star = 5 * ones(p)

    J = 1.0 * randn(rng, Float64, (n, n))
    J = (J - J') / 2
    Randmat = 2.0 * randn(rng, Float64, (n, n))
    R = Randmat * Randmat'
    Hamilton = 1.0 * I(n)
    A = (J - R) * Hamilton
    B = 3 * randn(rng, Float64, (n, m))
    C = B' * Hamilton
    Randmat_w = randn(rng, Float64, (n, n))
    Rand_w = Randmat_w * Randmat_w'
    W = 0.001 * Rand_w
    Randmat_p = randn(rng, Float64, (p, p))
    Rand_p = Randmat_p * Randmat_p'
    V = 0.0005 * Rand_p
    h = 0.0001
    Dist_x0 = Uniform(-3, 3)
end

println("eigen value of W: ", eigvals(W))
println("eigen value of V: ", eigvals(V))


equib = Matrix([A B; C zeros(p, m)]) \ [zeros(n); y_star]
x_star = equib[1:n]
u_star = equib[(n+1):(n+p)]

F = [A zeros((n, p)); -C zeros((p, p))]
G = [B; zeros((p, m))]
H = [C zeros(p, p); zeros(p, n) -I(p)]

# 初期点の分布の計算
Sigma0 = zeros((n), (n))
Nsample = 10000
for i in 1:Nsample
    global Sigma0
    x_0 = rand(rng, Dist_x0, n)
    Sigma0 += (x_0 - x_star) * (x_0 - x_star)'
end
Sigma0 = Sigma0 / Nsample

mean_ex0 = zeros(n)
Nsample = 10000
for i in 1:Nsample
    global mean_ex0
    mean_ex0 += rand(rng, Dist_x0, n) - x_star
end
mean_ex0 /= Nsample

V_half = cholesky(V).L
W_half = cholesky(W).L


system = TargetSystem(A,
    B,
    C,
    F,
    G,
    H,
    W,
    V,
    W_half,
    V_half,
    Dist_x0,
    h,
    y_star,
    x_star,
    u_star,
    Sigma0,
    mean_ex0,
    n,
    p,
    m,
    rng)

Setting = Dict(
    "A" => A,
    "B" => B,
    "C" => C,
    "W" => W,
    "V" => V,
    "h" => h,
    "Dist_x0" => Dist_x0,
    "system" => system
)

#println("Setting: ", Setting)
if !isdir("System_setting")
    mkdir("System_setting")  # フォルダを作成
end
if !isdir("System_setting/Noise_dynamics")
    mkdir("System_setting/Noise_dynamics")  # フォルダを作成
end
if !isdir("System_setting/Noise_dynamics/Settings")
    mkdir("System_setting/Noise_dynamics/Settings")  # フォルダを作成
end
if !isdir("System_setting/Noise_dynamics/Settings/Setting$Setting_num")
    mkdir("System_setting/Noise_dynamics/Settings/Setting$Setting_num")  # フォルダを作成
end

@save "System_setting/Noise_dynamics/Settings/Setting$Setting_num/Settings.jld2" Setting




