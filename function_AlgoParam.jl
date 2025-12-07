using LinearAlgebra, Logging
include("function_orbit.jl")
include("function_noise.jl")

function Compute_tauu(system, K_P_uhat, epsilon_u)
       A_K_uhat = system.A - system.B * K_P_uhat * system.C
       Z = lyap(A_K_uhat', I(system.n))
       eigvals_Z = eigvals(Z)
       eig_max_Z = maximum(eigvals_Z)
       eig_min_Z = minimum(eigvals_Z)
       C_norm2 = norm(system.C, 2)
       Ak_inv_norm2 = norm(inv(A_K_uhat'), 2)
       B_norm2 = norm(system.B, 2)
       D_norm2 = norm(system.C * inv(A_K_uhat) * system.B, 2)
       min_singular_D = minimum(svdvals(system.C * inv(A_K_uhat) * system.B))

       M1_1 = 2((C_norm2 * (norm(system.mean_ex0) + Ak_inv_norm2 * B_norm2 * norm(system.u_star)))
                /
                (min_singular_D))
       M1_2 = 2 * sqrt(system.m * system.p) *
              ((D_norm2^2) * norm(system.y_star))
       M1 = max(M1_1, M1_2)

       noise_gram = system.W + system.B * K_P_uhat * system.V * (system.B * K_P_uhat)'
       noise_gram = Symmetric(noise_gram)
       Sigma = lyap(A_K_uhat, noise_gram)
       M2_1 = tr(system.Sigma0) / tr(system.C * Sigma * system.C')
       M2_2 = norm(system.Sigma0) / norm(system.C * Sigma * system.C')
       M2 = (C_norm2 * eig_max_Z / eig_min_Z)^2 * max(M2_1, M2_2)

       M3_1 = 2 * sqrt(2 * system.m * system.p) * D_norm2^2

       M3_2 = (norm(system.mean_ex0) + Ak_inv_norm2 * B_norm2 * norm(system.u_star)) /
              norm(system.y_star)
       M3 = (C_norm2 * eig_max_Z / eig_min_Z) * max(M3_1, M3_2)

       tau_u1 = 2 * eig_max_Z * log(
                       max(M1 * eig_max_Z / (eig_min_Z * epsilon_u), M3)
                )
       tau_u2 = eig_max_Z * log(M2)
       tau_u = max(tau_u1, tau_u2)
       return tau_u
end

function Algo_params(system,
       prob,
       epsilon,
       obj_init,
       delta,
       norm_omega,
       K_P_uhat,
       stab=0.001)
       epsilon = epsilon / 5 ## 5等分に注意
       delta = delta / 2 ## 2等分に注意

       #stab:安定余裕は計算できないので初期点の安定余裕で近似する？
       eigvals_W = eigvals(system.W)
       eig_max_W = maximum(eigvals_W)
       eig_min_W = minimum(eigvals_W)

       eigvals_V = eigvals(system.V)
       eig_max_V = maximum(eigvals_V)
       eig_min_V = minimum(eigvals_V)

       eig_max_Qprime = eigmax(prob.Q_prime)

       Ahat_norm2 = norm(system.F, 2)
       Bhat_norm2 = norm(system.G, 2)
       Chat_norm2 = norm(system.H, 2)

       s_frak = 1 / (eig_min_W + eig_min_V)
       println("s_frak: ", s_frak)
       c_frak = eig_max_W + eig_max_V * (1 + (Bhat_norm2^2) * (norm_omega^2))
       println("c_frak: ", c_frak)
       t_frak = eig_max_V * (1 + Bhat_norm2 * norm_omega)
       println("t_frak: ", t_frak)
       b_frak = Bhat_norm2 * (Chat_norm2 * c_frak / (2 * stab) + t_frak) / stab
       println("b_frak: ", b_frak)

       # へシアンリプシッツ
       Hess_Lip1 = 2 * sqrt(system.m * system.p) * 2 * obj_init * s_frak * Bhat_norm2 * Chat_norm2 / stab^2
       Hess_Lip2 = Bhat_norm2 * (b_frak * Chat_norm2 + 3 * t_frak * Bhat_norm2 * Chat_norm2 +
                                 4 * Bhat_norm2 * eig_max_V * stab) + 3 * b_frak * stab
       Hess_lip = Hess_Lip1 * Hess_Lip2

       L_frak = 2 * sqrt(system.m * system.p) * obj_init * s_frak * b_frak * stab
       # サンプル数
       N = (12 * system.p * system.m - 6) * (L_frak^2) + 2 * L_frak * epsilon
       println("N:", N)
       N = (N / (3 * epsilon^2)) * log(4 * system.m * system.p / delta)
       println("N:", N)
       # smoothing parameter
       r = system.m * system.p * sqrt(2 * epsilon / Hess_lip)

       C_2a_frak = 2 * obj_init * s_frak / (eig_min_V * eig_min_W)
       e_frak = Ahat_norm2 + Bhat_norm2 * Chat_norm2 * norm_omega

       # 制御時間
       tau_in_log = 2 * sqrt(system.m * system.p) * eig_max_Qprime * C_2a_frak * e_frak
       tau_in_log = tau_in_log * (tr(system.Sigma0) + c_frak * C_2a_frak) / (epsilon * r)
       tau = C_2a_frak * log(tau_in_log)

       epsilon_u = (epsilon)^(3 / 4) / (e_frak * Bhat_norm2)
       epsilon_u = epsilon_u * ((system.m * system.p) / (4 * Hess_lip))^(1 / 4)
       tau_u = Compute_tauu(system, K_P_uhat, epsilon_u)

       # Nsub
       absolute_const = 2
       sub_gaus = 1.63
       N_sub = (sub_gaus^4) * (8 * (obj_init^2) * s_frak^2 + (eig_max_V^2) * (norm(prob.Q1)^2))
       N_sub = (N_sub / (absolute_const * (r^2) * (epsilon^2))) * log(2 / delta)

       return r, tau, tau_u, N, N_sub
end