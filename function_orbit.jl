using LinearAlgebra
function Orbit(A, B, C, K_P, K_I, K_D, x_0, reset, x_equib, y_star, h, T)
    n = size(A, 1)
    m = size(B, 2)
    p = size(C, 1)
    N = T / h

    F = [A zeros((n, p)); -C zeros((p, p))]
    G = [B; zeros((p, m))]
    H = [C zeros(p, p); zeros(p, n) -I(p); C*A zeros(p, p)]
    F_K = computeF_K(F, G, H, K_P, K_I, K_D)

    F_Kinv = inv(F_K)
    K_M = inv(I(p) + K_D * C * B)
    K_Pbar = K_M * K_P
    K_Ibar = K_M * K_I
    K_Dbar = K_M * K_D


    z_s = zeros(p, Int(N) + 1)
    diff_s = zeros(p, Int(N) + 1)
    x_s = zeros(n, Int(N) + 1)
    x_s[:, 1] = x_0
    y_s = zeros(p, Int(N) + 1)
    y_s[:, 1] = C * x_0
    Timeline = zeros(Int(N) + 1)

    ans_aug = zeros(n + p, Int(N) + 1)
    ans_aug[:, 1] = [x_0; zeros(p)]
    aug_init = [x_0; zeros(p)]
    final_vec = B * (K_Pbar * y_star + K_M * reset)
    final_vec = [final_vec; zeros(p)]
    ans_y = zeros(p, Int(N) + 1)
    ans_y[:, 1] = C * x_0
    for i in 1:N
        i = Int(i)
        #=
        k1_x = h * (F_K*[x_s[:, i]; z_s[:, i]]+F_Kinv*final_vec)[1:n]
        k1_z = h * (y_star - C * x_s[:, i])

        # Calculate k2
        k2_x = h * (F_K*[x_s[:, i] + 0.5 * k1_x; z_s[:, i] + 0.5 * k1_z]+F_Kinv*final_vec)[1:n]
        k2_z = h * (y_star - C * (x_s[:, i] + 0.5 * k1_x))

        # Calculate k3
        k3_x = h * (F_K*[x_s[:, i] + 0.5 * k2_x; z_s[:, i] + 0.5 * k2_z]+F_Kinv*final_vec)[1:n]
        k3_z = h * (y_star - C * (x_s[:, i] + 0.5 * k2_x))

        # Calculate k4
        k4_x = h * (F_K*[x_s[:, i] + k3_x; z_s[:, i] + k3_z]+F_Kinv*final_vec)[1:n]
        k4_z = h * (y_star - C * (x_s[:, i] + k3_x))

        # Update x and z
        x_s[:, Int(i + 1)] = x_s[:, i] + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        z_s[:, Int(i + 1)] = z_s[:, i] + (k1_z + 2 * k2_z + 2 * k3_z + k4_z) / 6
        =#

        k1_x = h * (F_K*[x_s[:, i]; z_s[:, i]]+final_vec)[1:n]
        k1_z = h * (y_star - C * x_s[:, i])

        # Calculate k2
        k2_x = h * (F_K*[x_s[:, i] + 0.5 * k1_x; z_s[:, i] + 0.5 * k1_z]+final_vec)[1:n]
        k2_z = h * (y_star - C * (x_s[:, i] + 0.5 * k1_x))

        # Calculate k3
        k3_x = h * (F_K*[x_s[:, i] + 0.5 * k2_x; z_s[:, i] + 0.5 * k2_z]+final_vec)[1:n]
        k3_z = h * (y_star - C * (x_s[:, i] + 0.5 * k2_x))

        # Calculate k4
        k4_x = h * (F_K*[x_s[:, i] + k3_x; z_s[:, i] + k3_z]+final_vec)[1:n]
        k4_z = h * (y_star - C * (x_s[:, i] + k3_x))

        # Update x and z
        x_s[:, Int(i + 1)] = x_s[:, i] + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        z_s[:, Int(i + 1)] = z_s[:, i] + (k1_z + 2 * k2_z + 2 * k3_z + k4_z) / 6

        # Update y
        y_s[:, Int(i + 1)] = C * x_s[:, Int(i + 1)]

        Timeline[Int(i + 1)] = h * i
        ans_aug[:, Int(i + 1)] = exp(F_K * i * h) * (aug_init + inv(F_K) * final_vec) - inv(F_K) * final_vec
        ans_y[:, Int(i + 1)] = C * ans_aug[1:n, Int(i + 1)]
    end
    return x_s, y_s, z_s, Timeline, ans_aug, ans_y
end

function Orbit_nonlinear(F, g, nablaH, K_P, K_I, K_D, x_0, reset, y_star, h, T, n, p)

    m = p
    N = T / h

    z_s = zeros(p, Int(N) + 1)
    x_s = zeros(n, Int(N) + 1)
    x_s[:, 1] = x_0
    y_s = zeros(p, Int(N) + 1)
    y_s[:, 1] = g(x_0)' * nablaH(x_0)
    Timeline = zeros(Int(N) + 1)

    ans_y = zeros(p, Int(N) + 1)
    ans_y[:, 1] = C * x_0
    for i in 1:N
        i = Int(i)
        k1_x = h * (F(x_s[:, i]) + g(x_s[:, i]) +
                    g(x_s[:, i]) * (K_P * (y_s[:, i] - y_star) + K_I * (z_s[:, i]) + K_D * (y_s[:, i] - y_s[:, i-1]) / h + reset))
        k1_z = h * (y_star - g(x_s[:, i])' * nablaH(x_s[:, i]))

        k2_x = h * (F(x_s[:, i] + 0.5 * k1_x) + g(x_s[:, i] + 0.5 * k1_x)
                    + g(x_s[:, i] + 0.5 * k1_x) * (
                        K_P * (g(x_s[:, i] + 0.5 * k1_x)' * nablaH(x_s[:, i] + 0.5 * k1_x) - y_star) +
                        K_I * (z_s[:, i] + 0.5 * k1_z) +
                        K_D * (y_s[:, i] - y_s[:, i-1]) / h + reset))
        k2_z = h * (y_star - g(x_s[:, i] + 0.5 * k1_x)' * nablaH(x_s[:, i] + 0.5 * k1_x))

        k3_x = h * (F(x_s[:, i] + 0.5 * k2_x) + g(x_s[:, i] + 0.5 * k2_x)
                    + g(x_s[:, i] + 0.5 * k2_x) * (
                        K_P * (g(x_s[:, i] + 0.5 * k2_x)' * nablaH(x_s[:, i] + 0.5 * k2_x) - y_star) +
                        K_I * (z_s[:, i] + 0.5 * k2_z) +
                        K_D * (y_s[:, i] - y_s[:, i-1]) / h + reset))
        k3_z = h * (y_star - g(x_s[:, i] + 0.5 * k2_x)' * nablaH(x_s[:, i] + 0.5 * k2_x))

        k4_x = h * (F(x_s[:, i] + k3_x) + g(x_s[:, i] + k3_x)
                    + g(x_s[:, i] + k3_x) * (
                        K_P * (g(x_s[:, i] + k3_x)' * nablaH(x_s[:, i] + k3_x) - y_star) +
                        K_I * (z_s[:, i] + k3_z) +
                        K_D * (y_s[:, i] - y_s[:, i-1]) / h + reset))
        k4_z = h * (y_star - g(x_s[:, i] + k3_x)' * nablaH(x_s[:, i] + k3_x))

        x_s[:, Int(i + 1)] = x_s[:, i] + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        z_s[:, Int(i + 1)] = z_s[:, i] + (k1_z + 2 * k2_z + 2 * k3_z + k4_z) / 6

        # Update y
        y_s[:, Int(i + 1)] = C * x_s[:, Int(i + 1)]

        Timeline[Int(i + 1)] = h * i
    end
    return x_s, y_s, z_s, Timeline
end

function Orbit_nonlinear_p1(F, g, nablaH, K_P, K_I, K_D, x_0, reset, y_star, h, T, n)

    m = 1
    N = T / h

    z_s = zeros(Int(N) + 1)
    x_s = zeros(n, Int(N) + 1)
    x_s[:, 1] = x_0
    y_s = zeros(Int(N) + 1)
    y_s[1] = g(x_0)' * nablaH(x_0)
    Timeline = zeros(Int(N) + 1)
    y_star = y_star[1]

    #=
    k1_x = h * (F(x_s[:, 1]) +
                g(x_s[:, 1]) * (K_P * (-y_s[1] + y_star) + K_I * (z_s[1]) + reset))

    k1_z = h * (y_star - g(x_s[:, 1])' * nablaH(x_s[:, 1]))

    k2_x = h * (F(x_s[:, 1] + 0.5 * k1_x)
                +
                g(x_s[:, 1] + 0.5 * k1_x) * (
        K_P * (-g(x_s[:, 1] + 0.5 * k1_x)' * nablaH(x_s[:, 1] + 0.5 * k1_x) + y_star) +
        K_I * (z_s[1] + 0.5 * k1_z) +
        reset))
    k2_z = h * (y_star - g(x_s[:, 1] + 0.5 * k1_x)' * nablaH(x_s[:, 1] + 0.5 * k1_x))

    k3_x = h * (F(x_s[:, 1] + 0.5 * k2_x)
                +
                g(x_s[:, 1] + 0.5 * k2_x) * (
        K_P * (-g(x_s[:, 1] + 0.5 * k2_x)' * nablaH(x_s[:, 1] + 0.5 * k2_x) + y_star) +
        K_I * (z_s[1] + 0.5 * k2_z) +
        reset))
    k3_z = h * (y_star - g(x_s[:, 1] + 0.5 * k2_x)' * nablaH(x_s[:, 1] + 0.5 * k2_x))

    k4_x = h * (F(x_s[:, 1] + k3_x)
                +
                g(x_s[:, 1] + k3_x) * (
        K_P * (-g(x_s[:, 1] + k3_x)' * nablaH(x_s[:, 1] + k3_x) + y_star) +
        K_I * (z_s[1] + k3_z) +
        reset))
    k4_z = h * (y_star - g(x_s[:, 1] + k3_x)' * nablaH(x_s[:, 1] + k3_x))

    x_s[:, 2] = x_s[:, 1] + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
    z_s[2] = z_s[1] + (k1_z + 2 * k2_z + 2 * k3_z + k4_z) / 6

    # Update y
    y_s[2] = g(x_s[:, 2])' * nablaH(x_s[:, 2])

    Timeline[2] = h * 1

    for i in 2:N
        i = Int(i)
        k1_x = h * (F(x_s[:, i]) + g(x_s[:, i]) +
                    g(x_s[:, i]) * (K_P * (-y_s[i] + y_star) + K_I * (z_s[i]) + K_D * (y_s[i-1] - y_s[i]) / h + reset))
        k1_z = h * (y_star - g(x_s[:, i])' * nablaH(x_s[:, i]))
        k1_d = (g(x_s[:, i-1])' * nablaH(x_s[:, i-1]) - g(x_s[:, i])' * nablaH(x_s[:, i]))

        k2_x = h * (F(x_s[:, i] + 0.5 * k1_x)
                    +
                    g(x_s[:, i] + 0.5 * k1_x) * (
            K_P * (-g(x_s[:, i] + 0.5 * k1_x)' * nablaH(x_s[:, i] + 0.5 * k1_x) + y_star) +
            K_I * (z_s[i] + 0.5 * k1_z) +
            K_D * ((y_s[i-1] - y_s[i]) / h + k1_d) + reset))
        k2_z = h * (y_star - g(x_s[:, i] + 0.5 * k1_x)' * nablaH(x_s[:, i] + 0.5 * k1_x))
        k2_d = (g(x_s[:, i-1] + 0.5 * k1_x)' * nablaH(x_s[:, i-1] + 0.5 * k1_x) - g(x_s[:, i] + 0.5 * k1_x)' * nablaH(x_s[:, i] + 0.5 * k1_x))

        k3_x = h * (F(x_s[:, i] + 0.5 * k2_x)
                    +
                    g(x_s[:, i] + 0.5 * k2_x) * (
            K_P * (-g(x_s[:, i] + 0.5 * k2_x)' * nablaH(x_s[:, i] + 0.5 * k2_x) + y_star) +
            K_I * (z_s[i] + 0.5 * k2_z) +
            K_D * ((y_s[i-1] - y_s[i]) / h + k2_d) + reset))
        k3_z = h * (y_star - g(x_s[:, i] + 0.5 * k2_x)' * nablaH(x_s[:, i] + 0.5 * k2_x))
        k3_d = (g(x_s[:, i-1] + 0.5 * k2_x)' * nablaH(x_s[:, i-1] + 0.5 * k2_x) - g(x_s[:, i] + 0.5 * k2_x)' * nablaH(x_s[:, i] + 0.5 * k2_x))

        k4_x = h * (F(x_s[:, i] + k3_x)
                    +
                    g(x_s[:, i] + k3_x) * (
            K_P * (-g(x_s[:, i] + k3_x)' * nablaH(x_s[:, i] + k3_x) + y_star) +
            K_I * (z_s[i] + k3_z) +
            K_D * ((y_s[i-1] - y_s[i]) / h + k3_d) + reset))
        k4_z = h * (y_star - g(x_s[:, i] + k3_x)' * nablaH(x_s[:, i] + k3_x))

        x_s[:, Int(i + 1)] = x_s[:, i] + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        z_s[Int(i + 1)] = z_s[i] + (k1_z + 2 * k2_z + 2 * k3_z + k4_z) / 6

        # Update y
        y_s[Int(i + 1)] = g(x_s[:, Int(i + 1)])' * nablaH(x_s[:, Int(i + 1)])

        Timeline[Int(i + 1)] = h * i
    end
    =#

    for i in 1:2
        k1_x = h * (F(x_s[:, i]) +
                    g(x_s[:, i]) * (K_P * (-y_s[i] + y_star) + K_I * (z_s[i]) + reset))

        k1_z = h * (y_star - g(x_s[:, i])' * nablaH(x_s[:, i]))

        k2_x = h * (F(x_s[:, i] + 0.5 * k1_x)
                    +
                    g(x_s[:, i] + 0.5 * k1_x) * (
            K_P * (-g(x_s[:, i] + 0.5 * k1_x)' * nablaH(x_s[:, i] + 0.5 * k1_x) + y_star) +
            K_I * (z_s[i] + 0.5 * k1_z) +
            reset))
        k2_z = h * (y_star - g(x_s[:, i] + 0.5 * k1_x)' * nablaH(x_s[:, i] + 0.5 * k1_x))

        k3_x = h * (F(x_s[:, i] + 0.5 * k2_x)
                    +
                    g(x_s[:, i] + 0.5 * k2_x) * (
            K_P * (-g(x_s[:, i] + 0.5 * k2_x)' * nablaH(x_s[:, i] + 0.5 * k2_x) + y_star) +
            K_I * (z_s[i] + 0.5 * k2_z) +
            reset))
        k3_z = h * (y_star - g(x_s[:, i] + 0.5 * k2_x)' * nablaH(x_s[:, i] + 0.5 * k2_x))

        k4_x = h * (F(x_s[:, i] + k3_x)
                    +
                    g(x_s[:, i] + k3_x) * (
            K_P * (-g(x_s[:, i] + k3_x)' * nablaH(x_s[:, i] + k3_x) + y_star) +
            K_I * (z_s[i] + k3_z) +
            reset))
        k4_z = h * (y_star - g(x_s[:, i] + k3_x)' * nablaH(x_s[:, i] + k3_x))

        x_s[:, i+1] = x_s[:, i] + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        z_s[i+1] = z_s[i] + (k1_z + 2 * k2_z + 2 * k3_z + k4_z) / 6

        # Update y
        y_s[i+1] = g(x_s[:, i+1])' * nablaH(x_s[:, i+1])

        Timeline[i+1] = h * i
    end

    for i in 3:N
        i = Int(i)
        k1_x = h * (F(x_s[:, i]) + g(x_s[:, i]) +
                    g(x_s[:, i]) * (K_P * (-y_s[i] + y_star) + K_I * (z_s[i]) + K_D * (y_s[i-2] - y_s[i]) / (2 * h) + reset))
        k1_z = h * (y_star - g(x_s[:, i])' * nablaH(x_s[:, i]))

        k2_x = h * (F(x_s[:, i] + 0.5 * k1_x)
                    +
                    g(x_s[:, i] + 0.5 * k1_x) * (
            K_P * (-g(x_s[:, i] + 0.5 * k1_x)' * nablaH(x_s[:, i] + 0.5 * k1_x) + y_star) +
            K_I * (z_s[i] + 0.5 * k1_z) +
            K_D * (y_s[i-2] - y_s[i]) / (2 * h) + reset))
        k2_z = h * (y_star - g(x_s[:, i] + 0.5 * k1_x)' * nablaH(x_s[:, i] + 0.5 * k1_x))

        k3_x = h * (F(x_s[:, i] + 0.5 * k2_x)
                    +
                    g(x_s[:, i] + 0.5 * k2_x) * (
            K_P * (-g(x_s[:, i] + 0.5 * k2_x)' * nablaH(x_s[:, i] + 0.5 * k2_x) + y_star) +
            K_I * (z_s[i] + 0.5 * k2_z) +
            K_D * (y_s[i-2] - y_s[i]) / (2 * h) + reset))
        k3_z = h * (y_star - g(x_s[:, i] + 0.5 * k2_x)' * nablaH(x_s[:, i] + 0.5 * k2_x))

        k4_x = h * (F(x_s[:, i] + k3_x)
                    +
                    g(x_s[:, i] + k3_x) * (
            K_P * (-g(x_s[:, i] + k3_x)' * nablaH(x_s[:, i] + k3_x) + y_star) +
            K_I * (z_s[i] + k3_z) +
            K_D * (y_s[i-2] - y_s[i]) / (2 * h) + reset))
        k4_z = h * (y_star - g(x_s[:, i] + k3_x)' * nablaH(x_s[:, i] + k3_x))

        x_s[:, Int(i + 1)] = x_s[:, i] + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        z_s[Int(i + 1)] = z_s[i] + (k1_z + 2 * k2_z + 2 * k3_z + k4_z) / 6

        # Update y
        y_s[Int(i + 1)] = g(x_s[:, Int(i + 1)])' * nablaH(x_s[:, Int(i + 1)])

        Timeline[Int(i + 1)] = h * i
    end

    return x_s, y_s, z_s, Timeline
end

function Orbit_nonlinear_p1_P(F, g, nablaH, K_P, K_I, K_D, x_0, reset, y_star, h, T, n)

    m = 1
    N = T / h

    z_s = zeros(Int(N) + 1)
    x_s = zeros(n, Int(N) + 1)
    x_s[:, 1] = x_0
    y_s = zeros(Int(N) + 1)
    y_s[1] = g(x_0)' * nablaH(x_0)
    Timeline = zeros(Int(N) + 1)
    y_star = y_star[1]

    for i in 1:N
        i = Int(i)
        k1_x = h * (F(x_s[:, i]) + g(x_s[:, i]) +
                    g(x_s[:, i]) * (K_P * (-y_s[i] + y_star) + reset))
        k1_z = h * (y_star - g(x_s[:, i])' * nablaH(x_s[:, i]))

        k2_x = h * (F(x_s[:, i] + 0.5 * k1_x)
                    +
                    g(x_s[:, i] + 0.5 * k1_x) * (
            K_P * (-g(x_s[:, i] + 0.5 * k1_x)' * nablaH(x_s[:, i] + 0.5 * k1_x) + y_star)
            +
            reset))
        k2_z = h * (y_star - g(x_s[:, i] + 0.5 * k1_x)' * nablaH(x_s[:, i] + 0.5 * k1_x))

        k3_x = h * (F(x_s[:, i] + 0.5 * k2_x)
                    +
                    g(x_s[:, i] + 0.5 * k2_x) * (
            K_P * (-g(x_s[:, i] + 0.5 * k2_x)' * nablaH(x_s[:, i] + 0.5 * k2_x) + y_star)
            +
            reset))
        k3_z = h * (y_star - g(x_s[:, i] + 0.5 * k2_x)' * nablaH(x_s[:, i] + 0.5 * k2_x))

        k4_x = h * (F(x_s[:, i] + k3_x)
                    +
                    g(x_s[:, i] + k3_x) * (
            K_P * (-g(x_s[:, i] + k3_x)' * nablaH(x_s[:, i] + k3_x) + y_star) +
            reset))
        k4_z = h * (y_star - g(x_s[:, i] + k3_x)' * nablaH(x_s[:, i] + k3_x))

        x_s[:, Int(i + 1)] = x_s[:, i] + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        z_s[Int(i + 1)] = z_s[i] + (k1_z + 2 * k2_z + 2 * k3_z + k4_z) / 6

        # Update y
        y_s[Int(i + 1)] = g(x_s[:, Int(i + 1)])' * nablaH(x_s[:, Int(i + 1)])

        Timeline[Int(i + 1)] = h * i
    end
    return x_s, y_s, z_s, Timeline
end

function Orbit(A, B, C, K_P, K_I, K_D, x_0, reset, x_equib, y_star, h, T)
    n = size(A, 1)
    m = size(B, 2)
    p = size(C, 1)
    N = T / h

    F = [A zeros((n, p)); -C zeros((p, p))]
    G = [B; zeros((p, m))]
    H = [C zeros(p, p); zeros(p, n) -I(p); C*A zeros(p, p)]
    F_K = computeF_K(F, G, H, K_P, K_I, K_D)

    F_Kinv = inv(F_K)
    K_M = inv(I(p) + K_D * C * B)
    K_Pbar = K_M * K_P
    K_Ibar = K_M * K_I
    K_Dbar = K_M * K_D


    z_s = zeros(p, Int(N) + 1)
    diff_s = zeros(p, Int(N) + 1)
    x_s = zeros(n, Int(N) + 1)
    x_s[:, 1] = x_0
    y_s = zeros(p, Int(N) + 1)
    y_s[:, 1] = C * x_0
    Timeline = zeros(Int(N) + 1)

    ans_aug = zeros(n + p, Int(N) + 1)
    ans_aug[:, 1] = [x_0; zeros(p)]
    aug_init = [x_0; zeros(p)]
    final_vec = B * (K_Pbar * y_star + K_M * reset)
    final_vec = [final_vec; zeros(p)]
    ans_y = zeros(p, Int(N) + 1)
    ans_y[:, 1] = C * x_0
    for i in 1:N
        i = Int(i)
        #=
        k1_x = h * (F_K*[x_s[:, i]; z_s[:, i]]+F_Kinv*final_vec)[1:n]
        k1_z = h * (y_star - C * x_s[:, i])

        # Calculate k2
        k2_x = h * (F_K*[x_s[:, i] + 0.5 * k1_x; z_s[:, i] + 0.5 * k1_z]+F_Kinv*final_vec)[1:n]
        k2_z = h * (y_star - C * (x_s[:, i] + 0.5 * k1_x))

        # Calculate k3
        k3_x = h * (F_K*[x_s[:, i] + 0.5 * k2_x; z_s[:, i] + 0.5 * k2_z]+F_Kinv*final_vec)[1:n]
        k3_z = h * (y_star - C * (x_s[:, i] + 0.5 * k2_x))

        # Calculate k4
        k4_x = h * (F_K*[x_s[:, i] + k3_x; z_s[:, i] + k3_z]+F_Kinv*final_vec)[1:n]
        k4_z = h * (y_star - C * (x_s[:, i] + k3_x))

        # Update x and z
        x_s[:, Int(i + 1)] = x_s[:, i] + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        z_s[:, Int(i + 1)] = z_s[:, i] + (k1_z + 2 * k2_z + 2 * k3_z + k4_z) / 6
        =#

        k1_x = h * (F_K*[x_s[:, i]; z_s[:, i]]+final_vec)[1:n]
        k1_z = h * (y_star - C * x_s[:, i])

        # Calculate k2
        k2_x = h * (F_K*[x_s[:, i] + 0.5 * k1_x; z_s[:, i] + 0.5 * k1_z]+final_vec)[1:n]
        k2_z = h * (y_star - C * (x_s[:, i] + 0.5 * k1_x))

        # Calculate k3
        k3_x = h * (F_K*[x_s[:, i] + 0.5 * k2_x; z_s[:, i] + 0.5 * k2_z]+final_vec)[1:n]
        k3_z = h * (y_star - C * (x_s[:, i] + 0.5 * k2_x))

        # Calculate k4
        k4_x = h * (F_K*[x_s[:, i] + k3_x; z_s[:, i] + k3_z]+final_vec)[1:n]
        k4_z = h * (y_star - C * (x_s[:, i] + k3_x))

        # Update x and z
        x_s[:, Int(i + 1)] = x_s[:, i] + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        z_s[:, Int(i + 1)] = z_s[:, i] + (k1_z + 2 * k2_z + 2 * k3_z + k4_z) / 6

        # Update y
        y_s[:, Int(i + 1)] = C * x_s[:, Int(i + 1)]

        Timeline[Int(i + 1)] = h * i
        ans_aug[:, Int(i + 1)] = exp(F_K * i * h) * (aug_init + inv(F_K) * final_vec) - inv(F_K) * final_vec
        ans_y[:, Int(i + 1)] = C * ans_aug[1:n, Int(i + 1)]
    end
    return x_s, y_s, z_s, Timeline, ans_aug, ans_y
end


function Orbit_error(A, B, C, K_P, K_I, K_D, x_0, reset, x_equib, u_equib, y_star, h, T)
    n = size(A, 1)
    m = size(B, 2)
    p = size(C, 1)
    N = T / h

    F = [A zeros((n, p)); -C zeros((p, p))]
    G = [B; zeros((p, m))]
    H = [C zeros(p, p); zeros(p, n) -I(p); C*A zeros(p, p)]
    F_K = computeF_K(F, G, H, K_P, K_I, K_D)
    F_Kinv = inv(F_K)

    z_s = zeros(p, Int(N) + 1)
    diff_s = zeros(p, Int(N) + 1)
    x_s = zeros(n, Int(N) + 1)
    x_s[:, 1] = x_0
    y_s = zeros(p, Int(N) + 1)
    y_s[:, 1] = C * x_0
    Timeline = zeros(Int(N) + 1)

    ans_aug = zeros(n + p, Int(N) + 1)
    ans_aug[:, 1] = [x_0 - x_equib; zeros(p)]
    aug_init = [x_0 - x_equib; zeros(p)]
    ans_y = zeros(p, Int(N) + 1)
    ans_y[:, 1] = -C * (x_0 - x_equib)
    for i in 1:N
        ans_aug[:, Int(i + 1)] = exp(F_K * i * h) * (aug_init + inv(F_K) * G * K_M * (reset - u_equib)) - inv(F_K) * G * K_M * (reset - u_equib)
        ans_y[:, Int(i + 1)] = -C * ans_aug[1:n, Int(i + 1)]
        Timeline[Int(i + 1)] = h * i
    end
    return x_s, y_s, z_s, Timeline, ans_aug, ans_y
end

function Orbit_error_PD(A, B, C, K_P, K_D, x_0, reset, x_equib, u_equib, h, T)
    n = size(A, 1)
    m = size(B, 2)
    p = size(C, 1)
    N = T / h

    K_M = inv(I(p) + K_D * C * B)
    K_Pbar = K_M * K_P
    K_Dbar = K_M * K_D

    A_K = A - B * (K_Pbar * C + K_Dbar * C * A)

    Timeline = zeros(Int(N) + 1)

    ans_ex = zeros(n, Int(N) + 1)
    ans_e = zeros(p, Int(N) + 1)
    ans_e[:, 1] = -C * (x_0 - x_equib)
    for i in 1:N
        ans_ex[:, Int(i + 1)] = exp(A_K * i * h) * (x_0 - x_equib) + inv(A_K) * B * K_M * (reset - u_equib)
        ans_e[:, Int(i + 1)] = -C * ans_ex[1:n, Int(i + 1)]
        Timeline[Int(i + 1)] = h * i
    end
    return Timeline, ans_ex, ans_e
end

function Orbit_PD(A, B, C, K_P, K_D, x_0, reset, y_star, x_equib, u_equib, h, T)
    n = size(A, 1)
    m = size(B, 2)
    p = size(C, 1)
    N = T / h

    K_M = inv(I(p) + K_D * C * B)
    K_Pbar = K_M * K_P
    K_Dbar = K_M * K_D

    A_K = A - B * (K_Pbar * C + K_Dbar * C * A)
    A_Kinv = inv(A_K)
    final_vec = B * (K_Pbar * y_star + K_Dbar * C * A * x_equib + reset)

    Timeline = zeros(Int(N) + 1)

    z_s = zeros(p, Int(N) + 1)

    x_s = zeros(n, Int(N) + 1)
    x_s[:, 1] = x_0
    y_s = zeros(p, Int(N) + 1)
    y_s[:, 1] = C * x_0
    Timeline = zeros(Int(N) + 1)

    ans_y = zeros(p, Int(N) + 1)
    ans_y[:, 1] = C * x_0
    ans_x = zeros(n, Int(N) + 1)
    ans_x[:, 1] = x_0

    for i in 1:N
        i = Int(i)
        # Calculate k1

        k1_x = h * (A * x_s[:, i] - B * (K_Pbar * (y_star - y_s[:, i]) - K_Dbar * C * A * (x_s[:, 1] - x_equib) + reset))
        k1_y = C * k1_x


        # Calculate k2
        k2_x = h * (A * (x_s[:, i] + 0.5 * k1_x) - B * (K_Pbar * (y_star - (y_s[:, i] + 0.5 * k1_y)) - K_Dbar * C * A * (x_s[:, i] + 0.5 * k1_x - x_equib) + reset))
        k2_y = C * k2_x

        # Calculate k3
        k3_x = h * (A * (x_s[:, i] + 0.5 * k2_x) - B * (K_Pbar * (y_star - (y_s[:, i] + 0.5 * k2_y)) - K_Dbar * C * A * (x_s[:, i] + 0.5 * k2_x - x_equib) + reset))
        k3_y = C * k3_x


        # Calculate k4
        k4_x = h * (A * (x_s[:, i] + k3_x) - B * (K_Pbar * (y_star - (y_s[:, i] + k3_y)) - K_Dbar * C * A * (x_s[:, i] + k3_x - x_equib) + reset))
        k4_y = C * k4_x

        x_s[:, Int(i + 1)] = x_s[:, i] + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        # Update y
        y_s[:, Int(i + 1)] = C * x_s[:, Int(i + 1)]
        Timeline[Int(i + 1)] = h * i
        ans_x[:, Int(i + 1)] = exp(A_K * i * h) * (x_0 + A_Kinv * final_vec) - A_Kinv * final_vec
        ans_y[:, Int(i + 1)] = C * ans_x[:, Int(i + 1)]
    end
    return x_s, y_s, Timeline, ans_y
end



function Orbit_PI_noise(system, K_P, K_I, x_0, z_0, T, reset)
    N = T / system.h
    F_K = system.F - system.G * [K_P K_I] * system.H

    z_s = zeros(system.p, Int(N) + 1)
    z_s[:, 1] = z_0
    x_s = zeros(system.n, Int(N) + 1)
    x_s[:, 1] = x_0
    y_s = zeros(system.p, Int(N) + 1)
    y_s[:, 1] = system.C * x_0
    Timeline = zeros(Int(N) + 1)

    ans_aug = zeros(system.n + system.p, Int(N) + 1)
    ans_aug[:, 1] = [x_0; z_0]

    final_vec = system.B * (K_P * system.y_star + reset)
    final_vec = [final_vec; zeros(system.p)]


    ans_y = zeros(system.p, Int(N) + 1)
    ans_y[:, 1] = system.C * x_0
    for i in 1:N
        i = Int(i)

        # Update x and z
        x_s[:, Int(i + 1)] = x_s[:, i] + system.h * (F_K*[x_s[:, i]; z_s[:, i]]+final_vec)[1:system.n] + sqrt(system.h) * system.W_half * randn(system.rng, system.n)
        z_s[:, Int(i + 1)] = z_s[:, i] + system.h * (system.y_star - system.C * x_s[:, i])

        # Update y
        y_s[:, Int(i + 1)] = system.C * x_s[:, Int(i + 1)] + system.V_half * randn(system.rng, system.p)

        Timeline[Int(i + 1)] = system.h * i
    end
    return x_s, y_s, z_s, Timeline, ans_aug, ans_y
end

function Orbit_P_noise(system, K_P, x_0, T, reset)
    N = T / (1.0 * system.h)
    N = round(N)
    x_s = zeros(system.n, Int(N) + 1)
    x_s[:, 1] = x_0
    y_s = zeros(system.p, Int(N) + 1)
    y_s[:, 1] = system.C * x_0
    Timeline = zeros(Int(N) + 1)

    ans_y = zeros(system.p, Int(N) + 1)
    ans_y[:, 1] = system.C * x_0

    ans_y = zeros(system.p, Int(N) + 1)
    ans_y[:, 1] = system.C * x_0
    for i in 1:N
        i = Int(i)
        input = K_P * (system.y_star - y_s[:, Int(i)]) + reset
        # Update x
        x_s[:, Int(i + 1)] = x_s[:, i] + system.h * (system.A * x_s[:, 1] + system.B * input) + sqrt(system.h) * system.W_half * randn(system.rng, system.n)
        # Update y
        y_s[:, Int(i + 1)] = system.C * x_s[:, Int(i + 1)] + system.V_half * randn(system.rng, system.p)

        Timeline[Int(i + 1)] = system.h * i
    end
    return x_s, y_s, Timeline, ans_y
end

function Error_t_P_noise(system, K_P, tau, reset)
    A_K = system.A - system.B * K_P * system.C

    err_state = A_K \ system.B * (reset - system.u_star)
    Expectation_ex = (exp(A_K * tau) * (system.mean_ex0 + err_state) - err_state)

    BK_P = system.B * K_P
    W_tilde = system.W + BK_P * system.V * BK_P'
    W_tilde = (W_tilde + W_tilde') / 2

    expAK_tau = exp(A_K * tau)
    #有限区間打ち切りリアプノフ方程式を解く
    X_infty = lyap(A_K, W_tilde)
    X_tau = lyap(A_K, expAK_tau * W_tilde * expAK_tau')
    Var_init = expAK_tau * system.Sigma0 * expAK_tau'
    Var_path = X_infty - X_tau
    Variance = Var_init + Var_path
    Variance = (Variance + Variance') / 2
    Var_harf = cholesky(Variance).L

    noise = randn(system.rng, system.n)
    ex = Expectation_ex + Var_harf * noise
    observe_noise = randn(system.rng, system.p)
    error = -system.C * ex + system.V_half * observe_noise
    return error
end


function Orbit_Identification_noise(system, x_0, T; PE_power=20, N=0)
    if N == 0
        N = trunc(T / system.h)
    end

    x_s = zeros(system.n, Int(N) + 1)
    x_s[:, 1] = x_0
    u_s = zeros(system.m, Int(N) + 1)
    y_s = zeros(system.p, Int(N) + 1)
    y_s[:, 1] = system.C * x_0
    Timeline = zeros(Int(N) + 1)

    for i in 1:N
        i = Int(i)
        u_s[:, i] = sqrt(PE_power) * randn(system.rng, system.m)

        # Update x and z, Euler-Maruyama method
        x_s[:, Int(i + 1)] = x_s[:, i] + system.h * (system.A * x_s[:, i] + system.B * u_s[:, i]) + sqrt(system.h) * system.W_half * randn(system.rng, system.n)
        # Update y
        y_s[:, Int(i + 1)] = system.C * x_s[:, Int(i + 1)] + system.V_half * randn(system.rng, system.p)

        Timeline[Int(i + 1)] = system.h * i

    end
    return u_s, x_s, y_s, Timeline
end

function Orbit_Identification_noise_succinct(system, x_0, T; Ts=10, PE_power=20, N=0)
    if N == 0
        N = Int(trunc(T / system.h))
    end

    length = Int(trunc(T / Ts))

    N_persample = Int(trunc(Ts / system.h))

    x = x_0
    u_s = zeros(system.m, length + 1)
    y_s = zeros(system.p, length + 1)
    y_s[:, 1] = system.C * x_0

    # 状態と作業用バッファを一度だけ確保
    x = copy(x_0)
    u_buf = zeros(eltype(x_0), system.m)
    w_buf = zeros(eltype(x_0), system.n)  # プロセスノイズ用
    v_buf = zeros(eltype(x_0), system.p)  # 観測ノイズ用
    Ax_buf = similar(x_0)
    Bu_buf = similar(x_0)
    Wx_buf = similar(x_0)

    sqrt_PE = sqrt(PE_power)
    sqrt_h = sqrt(system.h)

    @views for i in 1:length
        # 入力ノイズ：既存バッファに randn! で書き込み
        randn!(system.rng, u_buf)
        @. u_buf = sqrt_PE * u_buf
        u_s[:, i] .= u_buf

        # N_persample ステップ時間発展（Euler–Maruyama）
        for k in 1:N_persample
            # Ax_buf = A*x, Bu_buf = B*u_buf
            mul!(Ax_buf, system.A, x)
            mul!(Bu_buf, system.B, u_buf)

            # Wx_buf = W_half * w_buf
            randn!(system.rng, w_buf)
            mul!(Wx_buf, system.W_half, w_buf)

            @. x += system.h * (Ax_buf + Bu_buf) + sqrt_h * Wx_buf
        end

        # 出力 y
        randn!(system.rng, v_buf)
        # y_col = system.C * x + V_half * v_buf を in-place で
        y_col = view(y_s, :, i + 1)
        mul!(y_col, system.C, x)              # y_col = C*x
        mul!(v_buf, system.V_half, v_buf)     # v_buf = V_half * v_buf
        @. y_col += v_buf
    end
    return u_s, y_s
end

function NotInPlace_Orbit_Identification_noise_succinct(system, x_0, T; Ts=10, PE_power=20, N=0)
    if N == 0
        N = Int(trunc(T / system.h))
    end

    length = Int(trunc(T / Ts))

    N_persample = Int(trunc(Ts / system.h))

    #x_s = zeros(system.n, length + 1)
    #x_s[:, 1] .= x_0
    x = x_0
    u_s = zeros(system.m, length + 1)
    y_s = zeros(system.p, length + 1)
    y_s[:, 1] = system.C * x_0

    @views for i in 1:length
        u_input = sqrt(PE_power) * randn(system.rng, system.m)
        u_s[:, i] .= u_input
        for j in 1:N_persample
            x = x + system.h * (system.A * x + system.B * u_input) +
                sqrt(system.h) * system.W_half * randn(system.rng, system.n)
        end
        # Update y
        y_s[:, i+1] .= system.C * x + system.V_half * randn(system.rng, system.p)

    end
    return u_s, y_s
end

function Orbit_Identification_noiseFree(system, x_0, T)
    N = T / system.h

    x_s = zeros(system.n, Int(N) + 1)
    x_s[:, 1] = x_0
    u_s = zeros(system.m, Int(N) + 1)
    y_s = zeros(system.p, Int(N) + 1)
    y_s[:, 1] = system.C * x_0
    Timeline = zeros(Int(N) + 1)

    for i in 1:N
        i = Int(i)
        u_s[:, i] = sqrt(10) * randn(system.rng, system.m)

        # Update x and z, Euler-Maruyama method
        x_s[:, Int(i + 1)] = x_s[:, i] + h * (system.A * x_s[:, i] + system.B * u_s[:, i])
        # Update y
        y_s[:, Int(i + 1)] = system.C * x_s[:, Int(i + 1)]
        Timeline[Int(i + 1)] = system.h * i

    end
    return u_s, x_s, y_s, Timeline
end