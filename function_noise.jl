using LinearAlgebra, Logging
include("function_orbit.jl")

mutable struct TargetSystem
    A
    B
    C
    F
    G
    H
    W
    V
    W_half
    V_half
    Dist_x0
    h
    y_star
    x_star
    u_star
    Sigma0
    mean_ex0
    n::Int
    p::Int
    m::Int
    rng
end

# 値の有限性を検査（数値でも配列でもOK）
function check_finite(x, name::AbstractString)
    ok = isa(x, Number) ? isfinite(x) : all(isfinite, x)
    ok || error("NaN/Inf detected in `$name`")
    return x
end

# 真の目的関数
function ObjectiveFunction_noise(system, prob, K_P, K_I)
    F_K = system.F - system.G * [K_P K_I] * system.H
    BK_P = system.B * K_P
    W_tilde = [system.W+BK_P*system.V*BK_P' BK_P*system.V; system.V*BK_P' system.V]
    X = lyap(F_K, W_tilde)
    return sum(X .* prob.Q_prime) + sum(prob.Q1 .* system.V)
end

# 真の勾配
function grad_noise(system, prob, K_P, K_I)
    F_K = system.F - system.G * [K_P K_I] * system.H
    BK_P = system.B * K_P
    W_tilde = [system.W+BK_P*system.V*BK_P' BK_P*system.V; system.V*BK_P' system.V]
    X = lyap(F_K, W_tilde)
    Y = lyap(F_K', prob.Q_prime)
    Z = -2 * system.G' * Y * X * system.H'
    grad_P = Z[:, 1:system.p] + 2 * system.G' * Y * [BK_P; 1I(system.p)] * system.V
    grad_I = Z[:, system.p+1:2*system.p]
    return grad_P, grad_I
end

# シミュレーションによる有限区間打ち切り目的関数
function obj_lastvalue(system, prob, K_P, K_I, x_0, reset, tau)

    #=
    z_0 = zeros(system.p)
    _, y_s, z_s, _, _, _ = Orbit_PI_noise(system, K_P, K_I, x_0, z_0, tau, reset)

    e_last = system.y_star - y_s[:, end]

    Obj_param_mat = [prob.Q1 zeros(system.p, system.p); zeros(system.p, system.p) prob.Q2]
    vec = [e_last; z_s[:, end]]
    obj = vec' * Obj_param_mat * vec
    =#
    z_0 = zeros(system.p)
    Sigma = [system.Sigma0 zeros(system.n, system.p); zeros(system.p, system.n) z_0*z_0']
    F_K = system.F - system.G * [K_P K_I] * system.H
    BK_P = system.B * K_P
    W_tilde = [system.W+BK_P*system.V*BK_P' BK_P*system.V; system.V*BK_P' system.V]
    W_tilde = (W_tilde + W_tilde') / 2

    expFK_tau = exp(F_K * tau)
    #有限区間打ち切りリアプノフ方程式を解く
    X_infty = lyap(F_K, W_tilde)
    X_tau = lyap(F_K, expFK_tau * W_tilde * expFK_tau')
    Var_init = expFK_tau * Sigma * expFK_tau'
    Var_path = X_infty - X_tau
    Variance = Var_init + Var_path
    Variance = (Variance + Variance') / 2
    Var_harf = cholesky(Variance).L

    equib = F_K \ system.G * (reset - system.u_star)
    Expectation = expFK_tau * ([system.mean_ex0; z_0] + equib) - equib

    # 時刻tauでの値をサンプリングする
    noise = randn(system.rng, system.p + system.n)
    ex_z = Expectation + Var_harf * noise
    if all(isfinite, ex_z) != true
        println("ex_z: ", ex_z)
        println("noise: ", noise)
    end
    #check_finite(ex_z, "ex_z")

    observe_noise = randn(system.rng, system.p)
    error = -system.C * ex_z[1:system.n] + system.V_half * observe_noise
    z = ex_z[system.n+1:end]
    return error' * prob.Q1 * error + z' * prob.Q2 * z
end

# シミュレーションによる有限区間打ち切り目的関数
function obj_trunc(system, prob, K_P, K_I, x_0, reset, tau)
    Nstep = Int(tau / system.h)
    z_0 = zeros(system.p)
    _, y_s, z_s, _, _, _ = Orbit_PI_noise(system, K_P, K_I, x_0, z_0, tau, reset)

    e_s = y_s .- system.y_star
    e_s = -e_s

    Obj_param_mat = [prob.Q1 zeros(system.p, system.p); zeros(system.p, system.p) prob.Q2]
    """
    vec = [e_s[:, 1]; z_s[:, 1]]
    obj = (vec' * Obj_param_mat * vec) / 3
    @simd for j in 2:(Nstep-1)
        vec = [e_s[:, Int(j)]; z_s[:, Int(j)]]
        if j % 2 == 1
            obj += 4 * (vec' * Obj_param_mat * vec) / 3
        else
            obj += 2 * (vec' * Obj_param_mat * vec) / 3
        end
    end
    vec = [e_s[:, Nstep]; z_s[:, Nstep]]

    obj += (vec' * Obj_param_mat * vec) / 3
    obj *= system.h
    obj /= tau
    """

    @assert isodd(Nstep + 1) "Simpson則は点数 Nstep+1 を奇数（区間数が偶数）にしてください"
    # 作業バッファ（再利用して割当てを抑える）
    vec = similar(e_s, 2 * system.p)          # 状態を詰める一時ベクトル
    Q_mul_vec = similar(vec)             # M*vec を入れる一時ベクトル

    # 先頭点（係数 1）
    vec[1:system.p] .= @view e_s[:, 1]
    vec[system.p+1:end] .= @view z_s[:, 1]
    mul!(Q_mul_vec, Obj_param_mat, vec)           # tmp = M*vec
    obj = dot(vec, Q_mul_vec)                     # vec' * M * vec

    # 中間点
    @inbounds for j in 2:(Nstep)
        vec[1:system.p] .= @view e_s[:, j]
        vec[system.p+1:end] .= @view z_s[:, j]
        mul!(Q_mul_vec, Obj_param_mat, vec)
        w = (j % 2 == 1) ? 4.0 : 2.0        # Simpson の重み
        obj += w * dot(vec, Q_mul_vec)
    end

    # 最終点（係数 1）
    vec[1:system.p] .= @view e_s[:, Nstep+1]
    vec[system.p+1:end] .= @view z_s[:, Nstep+1]
    mul!(Q_mul_vec, Obj_param_mat, vec)
    obj += dot(vec, Q_mul_vec)

    # 係数は最後にまとめて
    obj *= (system.h / 3)
    obj /= tau
    return obj
end

#軌道を受け取って，有限打ち切り目的関数を計算
function obj_trunc_from_traj(system, prob, y_s, z_s, tau)
    Nstep = Int(tau / system.h)
    e_s = y_s .- system.y_star
    e_s = -e_s
    Obj_param_mat = [prob.Q1 zeros(system.p, system.p); zeros(system.p, system.p) prob.Q2]

    vec = [e_s[:, 1]; z_s[:, 1]]
    obj = (vec' * Obj_param_mat * vec) / 3
    @simd for j in 2:(Nstep-1)
        vec = [e_s[:, Int(j)]; z_s[:, Int(j)]]
        if j % 2 == 1
            obj += 4 * (vec' * Obj_param_mat * vec) / 3
        else
            obj += 2 * (vec' * Obj_param_mat * vec) / 3
        end
    end
    vec = [e_s[:, Nstep]; z_s[:, Nstep]]

    obj += (vec' * Obj_param_mat * vec) / 3
    obj *= system.h
    obj /= tau
    return obj
end

#軌道を受け取って，有限打ち切り目的関数のtau倍を計算
function TauObj_trunc_from_traj(system, prob, y_s, z_s, tau)
    Nstep = Int(tau / system.h)
    e_s = y_s .- system.y_star
    e_s = -e_s
    Obj_param_mat = [prob.Q1 zeros(system.p, system.p); zeros(system.p, system.p) prob.Q2]

    vec = [e_s[:, 1]; z_s[:, 1]]
    obj = (vec' * Obj_param_mat * vec) / 3
    @simd for j in 2:(Nstep-1)
        vec = [e_s[:, Int(j)]; z_s[:, Int(j)]]
        if j % 2 == 1
            obj += 4 * (vec' * Obj_param_mat * vec) / 3
        else
            obj += 2 * (vec' * Obj_param_mat * vec) / 3
        end
    end
    vec = [e_s[:, Nstep]; z_s[:, Nstep]]

    obj += (vec' * Obj_param_mat * vec) / 3
    obj *= system.h

    return obj
end


# ベースラインの用意のために，最小二乗法のためのデータをサンプル．
# この関数では一つの軌道から一つの軌道データを取り出す．制御開始から最初の数秒のみを使う．
function Sample_for_baseline_1traj1obj(system, prob, K_P, K_I, reset, h_samp, D_base)
    # 最初のサンプリング期間(D_base  * h_samp)におけるベクトルを計算する（errors_prior)
    # 軌道をサンプリング．期間は(2*D_base*h_samp)まで
    terminal_baseline = D_base * h_samp
    x_0 = rand(system.rng, system.Dist_x0, system.n)
    z_0 = randn(system.rng, system.p)
    _, y_s, z_s, _, _, _ = Orbit_PI_noise(system, K_P, K_I, x_0, z_0, 2 * terminal_baseline, reset)
    interval_samp = Int(h_samp / system.h)
    errors_prior = zeros(D_base * 2 * system.p) #D個errorのサンプルを撮る
    for j in 1:(D_base)
        error = y_s[:, interval_samp*(j-1)+1] - system.y_star
        errors_prior[(j-1)*2*system.p+1:(j*2*system.p)] = [error; z_s[:, interval_samp*(j-1)+1]]
    end

    errors_prior = kron(errors_prior, errors_prior)

    errors_post = zeros(D_base * 2 * system.p) #D個errorのサンプルを撮る
    for j in 1:(D_base)
        error = y_s[:, interval_samp*D_base+interval_samp*(j-1)+1] - system.y_star
        errors_post[(j-1)*2*system.p+1:(j*2*system.p)] = [error; z_s[:, interval_samp*D_base+interval_samp*(j-1)+1]]
    end
    errors_post = kron(errors_post, errors_post)
    errors = errors_prior - errors_post

    obj_s = TauObj_trunc_from_traj(system, prob, y_s, z_s, terminal_baseline)

    return errors, obj_s
end



# ベースライン計算のための行列PのvecPを返す
# Truncated 目的関数をtau倍したもののベースラインをつくる．
# 一つの軌道から一つのデータを生成し，複数回軌道を回してデータをサンプル
function Baseline_vecP_1traj1obj(system, prob, K_P, K_I, reset, h_samp, D_base, N_base)
    E_mat = zeros(N_base, (D_base * 2 * system.p)^2) #Pのベクトル化が列に並ぶ
    vals = zeros(N_base) #各サンプルごとの目的関数ち
    for i in 1:N_base
        errors, obj_s = Sample_for_baseline_1traj1obj(system, prob, K_P, K_I, reset, h_samp, D_base)
        E_mat[i, :] = errors
        vals[i] = obj_s
    end
    vecP = E_mat \ vals
    return vecP
end

function baseline_grad_obj_1traj_allobj(F_K, G, K_M, x_equib, u_equib, reset, h_samp, D_base, N_base, rng, Dist_x0)
    #何回かサンプルしてex(0)' * P *ex(0) =obj_trunc(ex(0))を満たすようなPを求めてあげたい．
    E_mat = zeros(N_base, (D_base * system.p)^2) #Pのベクトル化が列に並ぶ
    vals = zeros(N_base) #各サンプルごとの目的関数ち
    for i in 1:N_base
        errors, obj_s = Compute_ErrBar(system, reset, h_samp, D_base)
        E_mat[i, :] = errors
        vals[i] = obj_s
    end
    vecP = E_mat \ vals
    return vecP
end

function obj_trunc_Base(system, prob, K_P, K_I, x_0, reset, tau, h_samp, D_base, base_vecP)
    #目的関数とベースラインの計算
    z_0 = zeros(system.p)
    terminal_baseline = D_base * h_samp
    interval_samp = Int(h_samp / system.h)
    #軌道のサンプリング
    _, y_s, z_s, _, _, _ = Orbit_PI_noise(system, K_P, K_I, x_0, z_0, tau + terminal_baseline, reset)
    errors_base1 = zeros(D_base * 2 * system.p)
    for j in 1:(D_base)
        error = y_s[:, interval_samp*(j-1)+1] - system.y_star
        errors_base1[(j-1)*2*system.p+1:(j*2*system.p)] = [error; z_s[:, interval_samp*(j-1)+1]]
    end
    errors_base1 = kron(errors_base1, errors_base1)

    steps_tau = Int(tau / system.h)

    errors_base2 = zeros(D_base * 2 * system.p)
    for j in 1:(D_base)
        error = y_s[:, steps_tau+interval_samp*(j-1)+1] - system.y_star
        errors_base2[(j-1)*2*system.p+1:(j*2*system.p)] = [error; z_s[:, steps_tau+interval_samp*(j-1)+1]]
    end
    errors_base2 = kron(errors_base2, errors_base2)

    #目的関数の計算
    obj = obj_trunc_from_traj(system, prob, y_s, z_s, tau)
    base = (errors_base1 - errors_base2)' * base_vecP
    base /= tau

    return obj, base
end



function grad_obj_est_Base(K_P, K_I, system, N_sample, reset, tau, r, D_base, h_samp)
    """
    モデルフリーな勾配の推定ほう
    ベクトル形式の勾配を返す．
    """

    grad_Ps = zeros(system.p, N_sample)
    grad_Is = zeros(system.p, N_sample)

    grad_P = zeros(system.p)
    grad_I = zeros(system.p)

    Shere_samples = randn(system.rng, Float64, (N_sample, 2 * system.p)) # 球面上のノイズ作成のためのガウシアンノイズ

    #D_base = 2 * (system.n + system.p - 1) + 20
    #h_samp = 2 * system.h

    N_base = Int((system.n + system.p) * (system.n + system.p + 1) / 2) #ベースライン計算に何回反復するか

    base_vecP = Baseline_vecP_1traj1obj(system, prob, K_P, K_I, reset, h_samp, D_base, N_base)

    @simd for i in 1:N_sample
        #zp = Shere_samples[i, :]

        zp = Shere_samples[i, :]
        zp = zp / norm(zp)
        zp = zp * sqrt(2 * system.p)
        U_P = zp[1:system.p]
        U_I = zp[(system.p+1):(2*system.p)]
        x_0 = rand(system.rng, system.Dist_x0, system.n)
        #ある初期点からの摂動を入れたゲインでの目的関数の計算，　
        cost, base = obj_trunc_Base(system, prob, K_P + r * diagm(U_P), K_I + r * diagm(U_I), x_0, reset, tau, h_samp, D_base, base_vecP)

        grad_P += (cost - base) .* U_P
        grad_I += (cost - base) .* U_I

        grad_Ps[:, i] = grad_P ./ (r * i)
        grad_Is[:, i] = grad_I ./ (r * i)
    end
    grad_P = grad_P ./ (r * N_sample)
    grad_I = grad_I ./ (r * N_sample)

    return grad_P, grad_I, grad_Ps, grad_Is
end

function grad_obj_est_WithoutBase(K_P, K_I, system, N_sample, reset, tau, r)
    """
    モデルフリーな勾配の推定ほう
    ベクトル形式の勾配を返す．
    """

    grad_Ps = zeros(system.p, N_sample)
    grad_Is = zeros(system.p, N_sample)

    grad_P = zeros(system.p)
    grad_I = zeros(system.p)

    Shere_samples = randn(system.rng, Float64, (N_sample, 2 * system.p))
    for i in 1:N_sample
        zp = Shere_samples[i, :]
        zp = zp / norm(zp)
        zp = zp * sqrt(2 * system.p)
        U_P = zp[1:system.p]
        U_I = zp[(system.p+1):(2*system.p)]

        #ある初期点からの摂動を入れたゲインでの目的関数の計算，
        cost = 0
        if prob.last_value == false
            for j in 1:prob.N_inner_obj
                x_0 = rand(system.rng, system.Dist_x0, system.n)
                cost += obj_trunc(system, prob, K_P + r * diagm(U_P), K_I + r * diagm(U_I), x_0, reset, tau)
            end
        else
            for j in 1:prob.N_inner_obj
                x_0 = rand(system.rng, system.Dist_x0, system.n)
                cost += obj_lastvalue(system, prob, K_P + r * diagm(U_P), K_I + r * diagm(U_I), x_0, reset, tau)
            end
        end
        cost /= prob.N_inner_obj

        grad_P += cost .* U_P
        grad_I += cost .* U_I

        grad_Ps[:, i] = grad_P ./ (r * i)
        grad_Is[:, i] = grad_I ./ (r * i)
    end
    grad_P = grad_P ./ (r * N_sample)
    grad_I = grad_I ./ (r * N_sample)

    return grad_P, grad_I, grad_Ps, grad_Is
end

function grad_est_TwoPoint(K_P, K_I, system, N_sample, reset, tau, r)
    """
    モデルフリーな勾配の推定ほう
    ベクトル形式の勾配を返す．
    """

    grad_Ps = zeros(system.p, N_sample)
    grad_Is = zeros(system.p, N_sample)

    grad_P = zeros(system.p)
    grad_I = zeros(system.p)

    Shere_samples = randn(system.rng, Float64, (N_sample, 2 * system.p))
    Shere_samples = randn(system.rng, Float64, (N_sample, 2 * system.p))
    for i in 1:N_sample
        zp = Shere_samples[i, :]

        zp = zp / norm(zp)
        zp = zp * sqrt(2 * system.p)
        U_P = zp[1:system.p]
        U_I = zp[(system.p+1):(2*system.p)]
        x_0 = rand(system.rng, system.Dist_x0, system.n)
        #ある初期点からの摂動を入れたゲインでの目的関数の計算，
        cost1 = 0
        cost2 = 0
        if prob.last_value == false
            for i in 1:prob.N_inner_obj
                x_0 = rand(system.rng, system.Dist_x0, system.n)
                cost1 += obj_trunc(system, prob, K_P + r * diagm(U_P), K_I + r * diagm(U_I), x_0, reset, tau)
            end
            cost1 /= prob.N_inner_obj

            for i in 1:prob.N_inner_obj
                x_0 = rand(system.rng, system.Dist_x0, system.n)
                cost2 += obj_trunc(system, prob, K_P + r * diagm(U_P), K_I + r * diagm(U_I), x_0, reset, tau)
            end
            cost2 /= prob.N_inner_obj
        else
            for i in 1:prob.N_inner_obj
                x_0 = rand(system.rng, system.Dist_x0, system.n)
                cost1 += obj_lastvalue(system, prob, K_P + r * diagm(U_P), K_I + r * diagm(U_I), x_0, reset, tau)
            end
            cost1 /= prob.N_inner_obj

            for i in 1:prob.N_inner_obj
                x_0 = rand(system.rng, system.Dist_x0, system.n)
                cost2 += obj_lastvalue(system, prob, K_P - r * diagm(U_P), K_I - r * diagm(U_I), x_0, reset, tau)
            end
            cost2 /= prob.N_inner_obj
        end

        grad_P += (cost1 - cost2) .* U_P
        grad_I += (cost1 - cost2) .* U_I

        grad_Ps[:, i] = grad_P ./ (2.0 * r * i)
        grad_Is[:, i] = grad_I ./ (2.0 * r * i)
    end
    grad_P = grad_P ./ (r * N_sample)
    grad_I = grad_I ./ (r * N_sample)

    return grad_P, grad_I, grad_Ps, grad_Is
end

function grad_est_SimpleBaseline(K_P, K_I, system, N_sample, reset, tau, r)
    """
    モデルフリーな勾配の推定ほう
    ベクトル形式の勾配を返す．
    """

    grad_Ps = zeros(system.p, N_sample)
    grad_Is = zeros(system.p, N_sample)

    grad_P = zeros(system.p)
    grad_I = zeros(system.p)

    Shere_samples = randn(system.rng, Float64, (N_sample, 2 * system.p))
    Shere_samples = randn(system.rng, Float64, (N_sample, 2 * system.p))

    base = 0
    for i in prob.N_inner_obj
        x_0 = rand(system.rng, system.Dist_x0, system.n)
        base += obj_lastvalue(system, prob, K_P, K_I, x_0, reset, tau)
    end
    base /= prob.N_inner_obj

    for i in 1:N_sample
        zp = Shere_samples[i, :]

        zp = zp / norm(zp)
        zp = zp * sqrt(2 * system.p)
        U_P = zp[1:system.p]
        U_I = zp[(system.p+1):(2*system.p)]
        x_0 = rand(system.rng, system.Dist_x0, system.n)
        #ある初期点からの摂動を入れたゲインでの目的関数の計算，
        cost = 0.0

        if prob.last_value == false
            for i in 1:prob.N_inner_obj
                x_0 = rand(system.rng, system.Dist_x0, system.n)
                cost += obj_trunc(system, prob, K_P + r * diagm(U_P), K_I + r * diagm(U_I), x_0, reset, tau)
            end
        else
            for i in 1:prob.N_inner_obj
                x_0 = rand(system.rng, system.Dist_x0, system.n)
                cost += obj_lastvalue(system, prob, K_P + r * diagm(U_P), K_I + r * diagm(U_I), x_0, reset, tau)
            end
        end
        cost /= float(prob.N_inner_obj)

        grad_P += (cost - base) .* U_P
        grad_I += (cost - base) .* U_I

        grad_Ps[:, i] = grad_P ./ (r * i)
        grad_Is[:, i] = grad_I ./ (r * i)
    end
    grad_P = grad_P ./ (r * N_sample)
    grad_I = grad_I ./ (r * N_sample)

    return grad_P, grad_I, grad_Ps, grad_Is
end

function Projection_diagnal_interval(K, Opt, system)
    for i in 1:system.p
        if K[i, i] < Opt.eps_interval
            K[i, i] = Opt.eps_interval
        elseif K[i, i] > Opt.M_interval
            K[i, i] = Opt.M_interval
        end
    end
    return K
end

function ProjectGradient_Gain_Conststep_Noise(K_P, K_I, reset, system, prob, Opt)
    #制約は正の対角行列を想定
    #P制御の無限次元経過後の誤差を最小化する
    f_list = []
    Kp_list = []
    Ki_list = []

    push!(Kp_list, K_P)
    push!(Ki_list, K_I)

    cnt = 0
    # 初期点での目的関数値

    val = ObjectiveFunction_noise(system, prob, K_P, K_I)
    println("目的関数: ", val)
    push!(f_list, val)

    while cnt < Opt.N_GD

        if Opt.method_name == "One_point_WithoutBase"
            grad_P, grad_I, _, _ = grad_obj_est_WithoutBase(K_P, K_I, system, Opt.N_sample, reset, Opt.tau, Opt.r)
        elseif Opt.method_name == "TwoPoint"
            grad_P, grad_I, _, _ = grad_est_TwoPoint(K_P, K_I, system, Opt.N_sample, reset, Opt.tau, Opt.r)
        elseif Opt.method_name == "Onepoint_SimpleBaseline"
            grad_P, grad_I, _, _ = grad_est_SimpleBaseline(K_P, K_I, system, Opt.N_sample, reset, Opt.tau, Opt.r)
        end

        K_P_next = K_P - eta * diagm(grad_P)
        K_I_next = K_I - eta * diagm(grad_I)

        K_P_next = Projection_diagnal_interval(K_P_next, Opt, system)
        K_I_next = Projection_diagnal_interval(K_I_next, Opt, system)

        cnt += 1
        val = ObjectiveFunction_noise(system, prob, K_P, K_I)
        #射影する
        #println(f_val)
        difference = sqrt(sum((K_P_next - K_P) .^ 2) + sum((K_I_next - K_I) .^ 2))
        if (difference < epsilon * eta)
            push!(Kp_list, K_P)
            push!(Ki_list, K_I)
            push!(f_list, val)
            return Kp_list, Ki_list, f_list
        end
        if (cnt % 10 == 0)
            println(cnt)
            println(val)
            #println("勾配の推定", est_grad)
            #println("勾配", 2 * ((C * inv(A_K) * B * K_M)' * (C * inv(A_K) * B * K_M)) * (reset - u_equib))
        end
        K_P = K_P_next
        K_I = K_I_next

        push!(Kp_list, K_P)
        push!(Ki_list, K_I)
        push!(f_list, val)

    end
    return Kp_list, Ki_list, f_list
end




function ProjGrad_Gain_Conststep_ModelBased_Noise(K_P, K_I, system, prob, Opt)
    #P制御の無限次元経過後の誤差を最小化する
    f_list = []
    Kp_list = []
    Ki_list = []

    push!(Kp_list, K_P)
    push!(Ki_list, K_I)

    cnt = 0
    # 初期点での目的関数値

    val = ObjectiveFunction_noise(system, prob, K_P, K_I)
    println("目的関数: ", val)
    push!(f_list, val)

    while cnt < Opt.N_GD

        grad_P, grad_I = grad_noise(system, prob, K_P, K_I)
        grad_P = diag(grad_P)
        grad_I = diag(grad_I)
        K_P_next = K_P - eta * diagm(grad_P)
        K_I_next = K_I - eta * diagm(grad_I)

        K_P_next = Projection_diagnal_interval(K_P_next, Opt, system)
        K_I_next = Projection_diagnal_interval(K_I_next, Opt, system)

        cnt += 1
        val = ObjectiveFunction_noise(system, prob, K_P, K_I)
        #射影する
        #println(f_val)
        difference = sqrt(sum((K_P_next - K_P) .^ 2) + sum((K_I_next - K_I) .^ 2))
        if (difference < epsilon * eta)
            push!(Kp_list, K_P)
            push!(Ki_list, K_I)
            push!(f_list, val)
            return Kp_list, Ki_list, f_list
        end
        if (cnt % 10 == 0)
            println(cnt)
            println(val)
            #println("勾配の推定", est_grad)
            #println("勾配", 2 * ((C * inv(A_K) * B * K_M)' * (C * inv(A_K) * B * K_M)) * (reset - u_equib))
        end
        K_P = K_P_next
        K_I = K_I_next

        push!(Kp_list, K_P)
        push!(Ki_list, K_I)
        push!(f_list, val)

    end
    return Kp_list, Ki_list, f_list
end

