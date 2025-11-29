using LinearAlgebra, Logging, MatrixEquations
include("function_orbit.jl")

mutable struct TargetSystem
    A
    B
    C
    F #A_hat
    G #B_hat
    H #C_hat
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

# 最適化のパラメータ設定
mutable struct Optimization_param
    eta
    epsilon_GD
    epsilon_EstGrad
    delta
    eps_interval
    M_interval
    N_sample
    N_inner_obj
    N_GD
    tau
    r
    projection #制約領域をどのようにするか
    method_name
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

function ObjectiveFunction_discrete_noise(system, prob, K_P, K_I)
    F_K = system.F - system.G * [K_P K_I] * system.H
    BK_P = system.B * K_P
    W_tilde = [system.W+BK_P*system.V*BK_P' BK_P*system.V; system.V*BK_P' system.V]
    X = lyapd(F_K, W_tilde)
    return sum(X .* prob.Q_prime) + sum(prob.Q1 .* system.V)
end

function ObjectiveFunction_zoh_noise(system, system_disc, prob, K_P, K_I, Ts)
    A_zoh = [system.F system.G; zeros((system.m, system.n + system.p + system.m))]
    Q_zoh_pre = [prob.Q_prime zeros((system.n + system.p, system.m)); zeros((system.m, system.n + system.p + system.m))]
    #=
    integration_step = 1e-6
    Num_int = trunc(Ts / integration_step)
    Q_zoh = zeros((system.n + system.p + system.m, system.n + system.p + system.m))
    for i in 0:Num_int
        time = i * integration_step
        if i == 0 || i == Num_int
            Q_zoh += exp(A_zoh' * time) * Q_zoh_pre * exp(A_zoh * time)
        else
            Q_zoh += 2 * exp(A_zoh' * time) * Q_zoh_pre * exp(A_zoh * time)
        end
    end
    Q_zoh = integration_step * Q_zoh / 2
    K = [K_P K_I]
    K_aug = [I(system.p + system.n); K * system_disc.H]
    Q_zoh = K_aug' * Q_zoh * K_aug
    =#
    all_dim = system.p + system.m + system.n
    M = [-A_zoh' Q_zoh_pre;
        zeros(all_dim, all_dim) A_zoh]

    EM = exp(M * Ts)   # 2n × 2n の行列指数

    # ブロックを取り出す
    M12 = EM[1:all_dim, all_dim+1:2*all_dim]
    M22 = EM[all_dim+1:2*all_dim, all_dim+1:2*all_dim]

    Q_zoh = M22' * M12
    K = [K_P K_I]
    K_aug = [I(system.p + system.n); K * system_disc.H]
    Q_zoh = K_aug' * Q_zoh * K_aug

    F_K = system_disc.F - system_disc.G * [K_P K_I] * system_disc.H
    BK_P = system_disc.B * K_P
    W_tilde = [system_disc.W+BK_P*system_disc.V*BK_P' BK_P*system_disc.V; system_disc.V*BK_P' system_disc.V]
    X = lyapd(F_K, W_tilde)
    return sum(X .* Q_zoh) + sum(prob.Q1 .* system_disc.V)
end

function stability_margin(system, K_P, K_I)
    F_K = system.F - system.G * [K_P K_I] * system.H
    eig_closedloop = eigvals(F_K)
    println("eigenvals of closed loop", eig_closedloop)
    stab = -maximum(real(eig_closedloop))
    println("stability margin: ", stab)
    return stab
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

# 真の勾配 discrete
function grad_discrete_noise(system, prob, K_P, K_I)
    # gradient for discrete system
    F_K = system.F - system.G * [K_P K_I] * system.H
    BK_P = system.B * K_P
    W_tilde = [system.W+BK_P*system.V*BK_P' BK_P*system.V; system.V*BK_P' system.V]
    X = lyapd(F_K, W_tilde)
    Y = lyapd(F_K', prob.Q_prime)
    Z = -2 * system.G' * Y * F_K * X * system.H'
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
    if Real(eigmin(Variance)) <= 0.0
        println(eigvals(Variance))
        println(eigvals(F_K))
    end
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

# ZOHの軌道での目的関数のシミュレーション平均
function obj_mean_zoh(system, prob, K_P, K_I, u_hat, Ts, tau, Iteration_obj)
    mean_obj = 0
    for iter in 1:Iteration_obj
        x_0 = rand(system.rng, system.Dist_x0, system.n)
        u_s, y_s, z_s = Orbit_zoh_PI(system, K_P, K_I, u_hat, x_0, tau, Ts=Ts)
        u_s = nothing
        GC.gc()
        mean_obj = mean_obj + obj_trunc_from_traj(system, prob, y_s, z_s, tau)
        y_s = nothing
        z_s = nothing
        GC.gc()
    end
    return mean_obj / Iteration_obj
end

# 連続時間制御器での目的関数のシミュレーション平均
function obj_mean_continuous(system, prob, K_P, K_I, u_hat, tau, Iteration_obj)
    mean_obj = 0
    for iter in 1:Iteration_obj
        x_0 = rand(system.rng, system.Dist_x0, system.n)
        u_s, y_s, z_s = Orbit_continuous_PI(system, K_P, K_I, u_hat, x_0, tau)
        u_s = nothing
        GC.gc()
        mean_obj = mean_obj + obj_trunc_from_traj(system, prob, y_s, z_s, tau)
        y_s = nothing
        z_s = nothing
        GC.gc()
    end
    return mean_obj / Iteration_obj
end

function grad_est_TwoPoint_diag(K_P, K_I, system, prob, Opt, reset)
    """
    モデルフリーな勾配の推定ほう
    ベクトル形式の勾配を返す．
    """

    grad_Ps = zeros(system.p, Opt.N_sample)
    grad_Is = zeros(system.p, Opt.N_sample)

    grad_P = zeros(system.p)
    grad_I = zeros(system.p)

    Sphere_samples = randn(system.rng, Float64, (Opt.N_sample, 2 * system.p))
    Sphere_samples = randn(system.rng, Float64, (Opt.N_sample, 2 * system.p))
    for i in 1:Opt.N_sample
        zp = Sphere_samples[i, :]

        zp = zp / norm(zp)
        zp = zp * sqrt(2 * system.p)
        U_P = zp[1:system.p]
        U_I = zp[(system.p+1):(2*system.p)]
        x_0 = rand(system.rng, system.Dist_x0, system.n)
        #ある初期点からの摂動を入れたゲインでの目的関数の計算，
        cost1 = 0
        cost2 = 0
        if prob.last_value == false
            for i in 1:Opt.N_inner_obj
                x_0 = rand(system.rng, system.Dist_x0, system.n)
                cost1 += obj_trunc(system, prob, K_P + Opt.r * diagm(U_P), K_I + Opt.r * diagm(U_I), x_0, reset, Opt.tau)
            end
            cost1 /= Opt.N_inner_obj

            for i in 1:Opt.N_inner_obj
                x_0 = rand(system.rng, system.Dist_x0, system.n)
                cost2 += obj_trunc(system, prob, K_P + Opt.r * diagm(U_P), K_I + Opt.r * diagm(U_I), x_0, reset, Opt.tau)
            end
            cost2 /= Opt.N_inner_obj
        else
            for i in 1:Opt.N_inner_obj
                x_0 = rand(system.rng, system.Dist_x0, system.n)
                cost1 += obj_lastvalue(system, prob, K_P + Opt.r * diagm(U_P), K_I + Opt.r * diagm(U_I), x_0, reset, Opt.tau)
            end
            cost1 /= Opt.N_inner_obj

            for i in 1:Opt.N_inner_obj
                x_0 = rand(system.rng, system.Dist_x0, system.n)
                cost2 += obj_lastvalue(system, prob, K_P - Opt.r * diagm(U_P), K_I - Opt.r * diagm(U_I), x_0, reset, Opt.tau)
            end
            cost2 /= Opt.N_inner_obj
        end

        grad_P += (cost1 - cost2) .* U_P
        grad_I += (cost1 - cost2) .* U_I

        grad_Ps[:, i] = grad_P ./ (2.0 * Opt.r * i)
        grad_Is[:, i] = grad_I ./ (2.0 * Opt.r * i)
    end
    grad_P = grad_P ./ (Opt.r * Opt.N_sample)
    grad_I = grad_I ./ (Opt.r * Opt.N_sample)

    return grad_P, grad_I, grad_Ps, grad_Is
end

function grad_est_TwoPoint(K_P, K_I, system, prob, Opt, reset)
    """
    モデルフリーな勾配の推定ほう
    ベクトル形式の勾配を返す．
    """

    grad_Ps = zeros(system.m, system.p, Opt.N_sample)
    grad_Is = zeros(system.m, system.p, Opt.N_sample)

    grad_P = zeros(system.m, system.p,)
    grad_I = zeros(system.m, system.p,)

    Sphere_samples = randn(system.rng, Float64, (Opt.N_sample, 2 * system.m * system.p))
    for i in 1:Opt.N_sample
        zp = Sphere_samples[i, :]

        zp = zp / norm(zp)
        zp = zp * sqrt(2 * system.m * system.p)
        U_P = zp[1:system.m*system.p]
        U_P = reshape(U_P, (system.m, system.p))
        #U_P = (U_P + U_P') / 2
        U_I = zp[system.m*system.p+1:end]
        U_I = reshape(U_I, (system.m, system.p))
        #U_I = (U_I + U_I') / 2
        x_0 = rand(system.rng, system.Dist_x0, system.n)
        #ある初期点からの摂動を入れたゲインでの目的関数の計算，
        cost1 = 0
        cost2 = 0
        for i in 1:Opt.N_inner_obj
            x_0 = rand(system.rng, system.Dist_x0, system.n)
            cost1 += obj_lastvalue(system, prob, K_P + Opt.r * U_P, K_I + Opt.r * U_I, x_0, reset, Opt.tau)
        end
        cost1 /= Opt.N_inner_obj

        for i in 1:Opt.N_inner_obj
            x_0 = rand(system.rng, system.Dist_x0, system.n)
            cost2 += obj_lastvalue(system, prob, K_P - Opt.r * U_P, K_I - Opt.r * U_I, x_0, reset, Opt.tau)
        end
        cost2 /= Opt.N_inner_obj

        grad_P += (cost1 - cost2) .* U_P
        grad_I += (cost1 - cost2) .* U_I

        grad_Ps[:, :, i] .= grad_P ./ (2.0 * Opt.r * i)
        grad_Is[:, :, i] .= grad_I ./ (2.0 * Opt.r * i)
    end
    grad_P = grad_P ./ (Opt.r * Opt.N_sample)
    grad_I = grad_I ./ (Opt.r * Opt.N_sample)

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

function Projection_eigenvalues_interval(K, Opt)
    K = (K + K') / 2

    F = eigen(K)                # 対称行列なので固有値は実数
    eigs = F.values
    Q = F.vectors

    # 固有値を [a,b] にクリップ
    eigs_clipped = clamp.(eigs, Opt.eps_interval, Opt.M_interval)

    # Q * Diagonal(λ_clipped) * Q' で再構成
    P = Q * Diagonal(eigs_clipped) * Q'

    # 明示的に対称として扱いたいなら Symmetric を付ける
    return Symmetric(P)
end

function clip_frobenius(A, Opt)
    nA = norm(A)
    if nA >= Opt.M_interval
        return (Opt.M_interval / nA) * A
    elseif nA <= Opt.eps_interval
        return (Opt.eps_interval / nA) * A
    else
        return A
    end
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
        if Opt.projection == "diag"
            grad_P, grad_I, _, _ = grad_est_TwoPoint_diag(K_P, K_I, system, prob, Opt, reset)
            K_P_next = K_P - eta * diagm(grad_P)
            K_I_next = K_I - eta * diagm(grad_I)
            K_P_next = Projection_diagnal_interval(K_P_next, Opt, system)
            K_I_next = Projection_diagnal_interval(K_I_next, Opt, system)
        elseif Opt.projection == "Eigvals"
            grad_P, grad_I, _, _ = grad_est_TwoPoint(K_P, K_I, system, prob, Opt, reset)
            K_P_next = K_P - eta * grad_P
            K_I_next = K_I - eta * grad_I
            K_P_next = Projection_eigenvalues_interval(K_P_next, Opt)
            K_I_next = Projection_eigenvalues_interval(K_I_next, Opt)
        elseif Opt.projection == "Frobenius"
            grad_P, grad_I, _, _ = grad_est_TwoPoint(K_P, K_I, system, prob, Opt, reset)
            K_P_next = K_P - eta * grad_P
            K_I_next = K_I - eta * grad_I
            K_P_next = clip_frobenius(K_P_next, Opt)
            K_I_next = clip_frobenius(K_I_next, Opt)
        end

        cnt += 1
        val = ObjectiveFunction_noise(system, prob, K_P, K_I)
        #射影する
        #println(f_val)
        difference = sqrt(sum((K_P_next - K_P) .^ 2) + sum((K_I_next - K_I) .^ 2))
        if (difference < Opt.epsilon_GD * eta)
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
        if Opt.projection == "diag"
            grad_P, grad_I = grad_noise(system, prob, K_P, K_I)
            K_P_next = K_P - eta * diagm(grad_P)
            K_I_next = K_I - eta * diagm(grad_I)
            K_P_next = Projection_diagnal_interval(K_P_next, Opt, system)
            K_I_next = Projection_diagnal_interval(K_I_next, Opt, system)
        elseif Opt.projection == "Eigvals"
            grad_P, grad_I = grad_noise(system, prob, K_P, K_I)
            K_P_next = K_P - eta * grad_P
            K_I_next = K_I - eta * grad_I
            K_P_next = Projection_eigenvalues_interval(K_P_next, Opt)
            K_I_next = Projection_eigenvalues_interval(K_I_next, Opt)
        elseif Opt.projection == "Frobenius"
            grad_P, grad_I = grad_noise(system, prob, K_P, K_I)
            K_P_next = K_P - eta * grad_P
            K_I_next = K_I - eta * grad_I
            K_P_next = clip_frobenius(K_P_next, Opt)
            K_I_next = clip_frobenius(K_I_next, Opt)
        end
        cnt += 1
        val = ObjectiveFunction_noise(system, prob, K_P, K_I)
        #射影する
        #println(f_val)
        difference = sqrt(sum((K_P_next - K_P) .^ 2) + sum((K_I_next - K_I) .^ 2))
        if (difference < Opt.epsilon_GD * eta)
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

function ProjGrad_Discrete_Conststep_ModelBased_Noise(K_P, K_I, system, prob, Opt)
    # 離散時間モデルベース勾配降下法
    f_list = []
    Kp_list = []
    Ki_list = []

    push!(Kp_list, K_P)
    push!(Ki_list, K_I)

    cnt = 0
    # 初期点での目的関数値

    val = ObjectiveFunction_discrete_noise(system, prob, K_P, K_I)
    println("目的関数: ", val)
    push!(f_list, val)

    while cnt < Opt.N_GD

        if Opt.projection == "diag"
            grad_P, grad_I = grad_discrete_noise(system, prob, K_P, K_I)
            K_P_next = K_P - eta * diagm(grad_P)
            K_I_next = K_I - eta * diagm(grad_I)
            K_P_next = Projection_diagnal_interval(K_P_next, Opt, system)
            K_I_next = Projection_diagnal_interval(K_I_next, Opt, system)
        elseif Opt.projection == "Eigvals"
            grad_P, grad_I = grad_discrete_noise(system, prob, K_P, K_I)
            K_P_next = K_P - eta * grad_P
            K_I_next = K_I - eta * grad_I
            K_P_next = Projection_eigenvalues_interval(K_P_next, Opt)
            K_I_next = Projection_eigenvalues_interval(K_I_next, Opt)
        elseif Opt.projection == "Frobenius"
            grad_P, grad_I = grad_discrete_noise(system, prob, K_P, K_I)
            K_P_next = K_P - eta * grad_P
            K_I_next = K_I - eta * grad_I
            K_P_next = clip_frobenius(K_P_next, Opt)
            K_I_next = clip_frobenius(K_I_next, Opt)
        end

        val = ObjectiveFunction_discrete_noise(system, prob, K_P, K_I)
        #射影する
        #println(f_val)
        difference = sqrt(sum((K_P_next - K_P) .^ 2) + sum((K_I_next - K_I) .^ 2))
        if (difference < Opt.epsilon_GD * Opt.eta)
            println(cnt)
            println(val)
            push!(Kp_list, K_P)
            push!(Ki_list, K_I)
            push!(f_list, val)
            return Kp_list, Ki_list, f_list
        end
        cnt += 1
        if (cnt % 5000 == 0)
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


