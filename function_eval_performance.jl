include("function_orbit.jl")
# シミュレーションによる有限区間打ち切り目的関数
function obj_trunc(system, prob, K_P, K_I, x_0, reset, tau)
    Nstep = Int(tau / system.h)
    z_0 = zeros(system.p)
    _, y_s, z_s, _, _, _ = Orbit_PI_noise(system, K_P, K_I, x_0, z_0, tau, reset)

    e_s = y_s .- system.y_star
    e_s = -e_s

    Obj_param_mat = [prob.Q1 zeros(system.p, system.p); zeros(system.p, system.p) prob.Q2]

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
function obj_trunc_from_traj(system, prob, y_s, z_s, tau, h)
    Nstep = Int(round(tau / h))
    Q1 = prob.Q1
    Q2 = prob.Q2
    y_star = system.y_star

    # 作業用バッファ（1回だけアロケーション）
    e = similar(y_star)  # e = y_star - y
    Q1e = similar(y_star)  # Q1 * e
    Q2z = similar(y_star)  # Q2 * z
    acc = 0

    @inbounds begin
        # --- j = 1 (重み 1) ---
        @views @. e = y_star - y_s[:, 1]
        @views z = z_s[:, 1]

        mul!(Q1e, Q1, e)   # Q1e = Q1 * e
        mul!(Q2z, Q2, z)   # Q2z = Q2 * z
        acc += dot(e, Q1e) + dot(z, Q2z)

        # --- 中点 (j = 2 .. Nstep-1, 重み 4/2) ---
        @simd for j in 2:(Nstep-1)
            @views begin
                @. e = y_star - y_s[:, j]
                z = z_s[:, j]
            end

            mul!(Q1e, Q1, e)
            mul!(Q2z, Q2, z)

            w = isodd(j) ? 4 : 2
            acc += w * (dot(e, Q1e) + dot(z, Q2z))
        end

        # --- j = Nstep (重み 1) ---
        @views @. e = y_star - y_s[:, Nstep]
        @views z = z_s[:, Nstep]

        mul!(Q1e, Q1, e)
        mul!(Q2z, Q2, z)
        acc += dot(e, Q1e) + dot(z, Q2z)
    end

    acc *= h
    acc /= tau
    return acc
end

# ZOHの軌道での目的関数のシミュレーション平均
function obj_mean_zoh(system, prob, K_P, K_I, u_hat, Ts, tau, Iteration_obj; h=0)
    mean_obj = 0
    for iter in 1:Iteration_obj
        x_0 = rand(system.rng, system.Dist_x0, system.n)
        u_s, y_s, z_s = Orbit_zoh_PI(system, K_P, K_I, u_hat, x_0, tau, Ts=Ts, h=h)
        u_s = nothing
        mean_obj = mean_obj + obj_trunc_from_traj(system, prob, y_s, z_s, tau, h)
        y_s = nothing
        z_s = nothing
    end
    return mean_obj / Iteration_obj
end

# 連続時間制御器での目的関数のシミュレーション平均
function obj_mean_continuous(system, prob, K_P, K_I, u_hat, tau, Iteration_obj; h=0)
    mean_obj = 0
    for iter in 1:Iteration_obj
        x_0 = rand(system.rng, system.Dist_x0, system.n)
        u_s, y_s, z_s = Orbit_continuous_PI(system, K_P, K_I, u_hat, x_0, tau, h=h)
        u_s = nothing
        mean_obj = mean_obj + obj_trunc_from_traj(system, prob, y_s, z_s, tau, h)
        y_s = nothing
        z_s = nothing
    end
    return mean_obj / Iteration_obj
end

#########################
# バッファ再利用での高速版
#########################

mutable struct OrbitBuffersZOH{T}
    u_s::Matrix{T}
    y_s::Matrix{T}
    z_s::Matrix{T}
    x::Vector{T}
    z::Vector{T}
    y::Vector{T}
    u_buf::Vector{T}
    w_buf::Vector{T}
    v_buf::Vector{T}
    Ax_buf::Vector{T}
    Bu_buf::Vector{T}
    Wx_buf::Vector{T}
end

mutable struct OrbitBuffersCont{T}
    u_s::Matrix{T}
    y_s::Matrix{T}
    z_s::Matrix{T}
    x::Vector{T}
    z::Vector{T}
    y::Vector{T}
    e::Vector{T}
    u_buf::Vector{T}
    w_buf::Vector{T}
    v_buf::Vector{T}
    Ax_buf::Vector{T}
    Bu_buf::Vector{T}
    Wx_buf::Vector{T}
    Vv_buf::Vector{T}
end

function OrbitBuffersZOH(system, T; Ts=system.h, h=system.h)
    n, m, p = system.n, system.m, system.p
    N = Int(trunc(T / h))
    length = Int(trunc(T / Ts))
    u_s = zeros(eltype(system.A), m, N + 1)
    y_s = zeros(eltype(system.A), p, N + 1)
    z_s = zeros(eltype(system.A), p, N + 1)
    return OrbitBuffersZOH(u_s, y_s, z_s,
        zeros(eltype(system.A), n),
        zeros(eltype(system.A), p),
        zeros(eltype(system.A), p),
        zeros(eltype(system.A), m),
        zeros(eltype(system.A), n),
        zeros(eltype(system.A), p),
        zeros(eltype(system.A), n),
        zeros(eltype(system.A), n),
        zeros(eltype(system.A), n))
end

function OrbitBuffersCont(system, T; h=system.h)
    n, m, p = system.n, system.m, system.p
    N = Int(round(T / h))
    u_s = zeros(eltype(system.A), m, N + 1)
    y_s = zeros(eltype(system.A), p, N + 1)
    z_s = zeros(eltype(system.A), p, N + 1)
    return OrbitBuffersCont(u_s, y_s, z_s,
        zeros(eltype(system.A), n),
        zeros(eltype(system.A), p),
        zeros(eltype(system.A), p),
        zeros(eltype(system.A), p),
        zeros(eltype(system.A), m),
        zeros(eltype(system.A), n),
        zeros(eltype(system.A), p),
        zeros(eltype(system.A), n),
        zeros(eltype(system.A), n),
        zeros(eltype(system.A), n),
        zeros(eltype(system.A), p))
end

# in-place ZOH版
function Orbit_zoh_PI!(buf::OrbitBuffersZOH, system, K_P, K_I, u_hat, x_0, T; Ts=system.h, h=system.h, rng=system.rng)
    n, m, p = system.n, system.m, system.p
    N = Int(trunc(T / h))
    length = Int(trunc(T / Ts))
    N_persample = Int(trunc(Ts / h))

    # ローカル束縛してフィールドアクセスを減らす
    u_s = buf.u_s
    y_s = buf.y_s
    z_s = buf.z_s
    x = buf.x
    z = buf.z
    y = buf.y
    u_buf = buf.u_buf
    w_buf = buf.w_buf
    v_buf = buf.v_buf
    Ax_buf = buf.Ax_buf
    Bu_buf = buf.Bu_buf
    Wx_buf = buf.Wx_buf

    copyto!(x, x_0)
    fill!(z, 0)
    mul!(y, system.C, x)
    @views y_s[:, 1] .= y
    @views z_s[:, 1] .= z
    @views u_s[:, 1] .= 0

    sqrt_h = sqrt(h)
    @inbounds @views for i in 1:length
        # e = y_star - y を v_buf に一時保存して乗算を節約
        @. v_buf = system.y_star - y
        mul!(u_buf, K_P, v_buf)   # u_buf = K_P * e
        mul!(v_buf, K_I, z)       # v_buf = K_I * z
        @. u_buf = u_buf + v_buf + u_hat
        for k in 1:N_persample
            mul!(Ax_buf, system.A, x)
            mul!(Bu_buf, system.B, u_buf)
            randn!(rng, w_buf)
            mul!(Wx_buf, system.W_half, w_buf)
            @. x += h * (Ax_buf + Bu_buf) + sqrt_h * Wx_buf

            randn!(rng, v_buf)
            mul!(v_buf, system.V_half, v_buf)

            y_col = view(y_s, :, (i - 1) * N_persample + k + 1)
            mul!(y_col, system.C, x)
            @. y_col += v_buf
            y .= y_col

            @views u_s[:, (i-1)*N_persample+k+1] .= u_buf
            @views z_s[:, (i-1)*N_persample+k+1] .= z
        end
        @. z = z + (system.y_star - y)
    end
    return u_s, y_s, z_s
end

# in-place 連続時間版
function Orbit_continuous_PI!(buf::OrbitBuffersCont, system, K_P, K_I, u_hat, x_0, T; h=system.h, rng=system.rng)
    n, m, p = system.n, system.m, system.p
    N = Int(round(T / h))

    u_s, y_s, z_s = buf.u_s, buf.y_s, buf.z_s
    x, z, y = buf.x, buf.z, buf.y
    e = buf.e
    u_buf = buf.u_buf
    w_buf = buf.w_buf
    v_buf = buf.v_buf
    Ax_buf = buf.Ax_buf
    Bu_buf = buf.Bu_buf
    Wx_buf = buf.Wx_buf
    Vv_buf = buf.Vv_buf

    copyto!(x, x_0)
    fill!(z, 0)
    mul!(y, system.C, x)
    @views y_s[:, 1] .= y
    @views z_s[:, 1] .= z
    @views u_s[:, 1] .= 0

    sqrt_h = sqrt(h)
    @inbounds @views for k in 1:N
        @. e = system.y_star - y
        mul!(u_buf, K_P, e)
        mul!(Vv_buf, K_I, z)
        @. u_buf = u_buf + Vv_buf + u_hat

        mul!(Ax_buf, system.A, x)
        mul!(Bu_buf, system.B, u_buf)
        randn!(rng, w_buf)
        mul!(Wx_buf, system.W_half, w_buf)
        @. x = x + h * (Ax_buf + Bu_buf) + sqrt_h * Wx_buf

        randn!(rng, v_buf)
        mul!(Vv_buf, system.V_half, v_buf)
        mul!(y, system.C, x)
        @. y = y + Vv_buf

        @. e = system.y_star - y
        @. z = z + h * e

        @views y_s[:, k+1] .= y
        @views u_s[:, k+1] .= u_buf
        @views z_s[:, k+1] .= z
    end
    return u_s, y_s, z_s
end

# 軌道バッファ再利用版の平均評価
function obj_mean_zoh_reuse(system, prob, K_P, K_I, u_hat, Ts, tau, Iteration_obj; h=0, rng=system.rng)
    h = h == 0 ? system.h : h
    work = OrbitBuffersZOH(system, tau; Ts=Ts, h=h)
    mean_obj = 0.0
    for _ in 1:Iteration_obj
        x_0 = rand(rng, system.Dist_x0, system.n)
        Orbit_zoh_PI!(work, system, K_P, K_I, u_hat, x_0, tau; Ts=Ts, h=h, rng=rng)
        mean_obj += obj_trunc_from_traj(system, prob, work.y_s, work.z_s, tau, h)
    end
    return mean_obj / Iteration_obj
end

function obj_mean_continuous_reuse(system, prob, K_P, K_I, u_hat, tau, Iteration_obj; h=0, rng=system.rng)
    h = h == 0 ? system.h : h
    work = OrbitBuffersCont(system, tau; h=h)
    mean_obj = 0.0
    for _ in 1:Iteration_obj
        x_0 = rand(rng, system.Dist_x0, system.n)
        Orbit_continuous_PI!(work, system, K_P, K_I, u_hat, x_0, tau; h=h, rng=rng)
        mean_obj += obj_trunc_from_traj(system, prob, work.y_s, work.z_s, tau, h)
    end
    return mean_obj / Iteration_obj
end
