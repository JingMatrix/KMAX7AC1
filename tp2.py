import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

# =============================================================================
# Exercice 1: Monte-Carlo pour une intégrale
# =============================================================================
def ex1():
    print("--- Exercice 1 ---")
    
    # 3. Algorithme d'estimation
    def estimate_I(n_samples):
        U = npr.rand(n_samples)
        Y = U * np.exp(-U**3)
        return np.mean(Y)

    I_hat = estimate_I(100000)
    print(f"Estimation de I avec 100,000 échantillons: {I_hat:.6f}")

    # 6. Intervalle de confiance
    def estimate_I_with_CI(n_samples):
        U = npr.rand(n_samples)
        Y = U * np.exp(-U**3)
        I_hat = np.mean(Y)
        sigma_hat = np.std(Y, ddof=1)
        half_width = 1.96 * sigma_hat / np.sqrt(n_samples)
        return I_hat, (I_hat - half_width, I_hat + half_width)

    I_hat_ci, ci = estimate_I_with_CI(100000)
    print(f"Intervalle de confiance à 95%: [{ci[0]:.6f}, {ci[1]:.6f}]")

    # Estimation de n pour une précision de 10^-3
    # On veut 1.96 * sigma / sqrt(n) <= 1e-3 => n >= (1960 * sigma)^2
    # Estimation de sigma
    U_pilot = npr.rand(1000)
    Y_pilot = U_pilot * np.exp(-U_pilot**3)
    sigma_pilot = np.std(Y_pilot)
    n_required = (1960 * sigma_pilot)**2
    print(f"Écart-type estimé: {sigma_pilot:.4f}")
    print(f"Nombre d'échantillons requis pour une précision de 1e-3: {int(n_required):,}")
    
    # Vérifions avec ce n
    I_final, ci_final = estimate_I_with_CI(int(n_required))
    print(f"Valeur finale estimée: {I_final:.6f}")
    print(f"Nouvel intervalle de confiance: [{ci_final[0]:.6f}, {ci_final[1]:.6f}]")
    print(f"Largeur de l'intervalle: {ci_final[1]-ci_final[0]:.6f}")

# =============================================================================
# Exercice 2: Comparaison méthodes d'intégration
# =============================================================================
def ex2():
    print("\n--- Exercice 2 ---")
    
    a, b = 0, 2 * np.pi
    f = lambda x: np.cos(x) * np.exp(-x / 5) + 1
    # 6. Vraie valeur
    true_I = 5/26 * (1 - np.exp(-2*np.pi/5)) + 2*np.pi
    print(f"Vraie valeur de l'intégrale: {true_I:.6f}")
    
    # 7. Implémentation et comparaison
    n_values = np.logspace(1, 4, 50, dtype=int)
    I3_estimates = []
    MC_estimates = []
    MC_lower_bounds = []
    MC_upper_bounds = []

    for n in n_values:
        # Méthode des trapèzes (I3)
        x = np.linspace(a, b, n + 1)
        y = f(x)
        I3 = np.trapz(y, x)
        I3_estimates.append(I3)
        
        # Méthode de Monte-Carlo
        U = a + (b - a) * npr.rand(n)
        Y = f(U)
        I_hat = (b - a) * np.mean(Y)
        sigma_hat = (b-a) * np.std(Y)
        half_width = 1.96 * sigma_hat / np.sqrt(n)
        MC_estimates.append(I_hat)
        MC_lower_bounds.append(I_hat - half_width)
        MC_upper_bounds.append(I_hat + half_width)

    plt.figure(figsize=(10, 6))
    plt.plot(n_values, I3_estimates, label='Méthode des trapèzes (I3)')
    plt.plot(n_values, MC_estimates, label='Monte-Carlo (I_hat)')
    plt.fill_between(n_values, MC_lower_bounds, MC_upper_bounds, color='orange', alpha=0.3, label='IC 95% pour Monte-Carlo')
    plt.axhline(true_I, color='r', linestyle='--', label='Vraie valeur')
    plt.xscale('log')
    plt.xlabel("Nombre de points (n)")
    plt.ylabel("Valeur de l'intégrale")
    plt.title("Comparaison des méthodes d'intégration")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

# =============================================================================
# Exercice 3: Réduction de variance
# =============================================================================
def ex3():
    print("\n--- Exercice 3 ---")
    n_samples = 50000
    
    # Méthode 1: Normale
    X = npr.randn(n_samples)
    Y1 = np.sqrt(2 * np.pi) * np.sin(X**4) * np.exp(-2 * X) * (X > 0)
    var1 = np.var(Y1)
    
    # Méthode 2: Exponentielle
    Y = npr.exponential(1/2, n_samples)
    Y2 = 0.5 * np.sin(Y**4) * np.exp(-Y**2 / 2)
    var2 = np.var(Y2)
    
    print(f"Variance (Normale): {var1:.4f}")
    print(f"Variance (Exponentielle): {var2:.4f}")
    
    # Méthode 3: Importance sampling avec N(lambda, 1)
    lambdas = np.linspace(-3.5, -0.5, 31)
    variances = []
    
    for lmbd in lambdas:
        N = npr.normal(lmbd, 1, n_samples)
        term = np.sin(N**4) * np.exp(-(2 + lmbd) * N) * (N > 0)
        Y3 = np.sqrt(2 * np.pi) * np.exp(lmbd**2 / 2) * term
        variances.append(np.var(Y3))

    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, variances, 'o-')
    plt.xlabel("Lambda (λ)")
    plt.ylabel("Variance de l'estimateur")
    plt.title("Variance en fonction de λ pour l'échantillonnage préférentiel")
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    min_var_lambda = lambdas[np.argmin(variances)]
    print(f"La variance minimale est obtenue pour λ ≈ {min_var_lambda:.2f}")

# =============================================================================
# Exercice 4: Réduction de variance (bis)
# =============================================================================
def ex4():
    print("\n--- Exercice 4 ---")
    n_samples = 100000
    U = npr.rand(n_samples)
    
    # Méthode 1
    Y1 = np.exp(U**2)
    
    # Méthode 2
    Y2 = Y1 - 1 - U**2

    # Méthode 3
    Y3 = Y1 - 1 - U**2 - U**4 / 2

    var1 = np.var(Y1)
    var2 = np.var(Y2)
    var3 = np.var(Y3)

    print(f"Variance méthode 1 (standard): {var1:.6f}")
    print(f"Variance méthode 2 (contrôle 1): {var2:.6f}")
    print(f"Variance méthode 3 (contrôle 2): {var3:.6f}")
    print(f"Facteur de réduction (1 -> 2): {var1/var2:.2f}")
    print(f"Facteur de réduction (2 -> 3): {var2/var3:.2f}")

# =============================================================================
# Exercice 5: Simulation sur la boule unité
# =============================================================================
def ex5():
    print("\n--- Exercice 5 ---")
    
    def simule_boule_unite(n_points):
        points = []
        while len(points) < n_points:
            p = 2 * npr.rand(3) - 1 # point dans [-1,1]^3
            if np.sum(p**2) <= 1:
                points.append(p)
        return np.array(points)

    n_samples = 5000
    points_in_ball = simule_boule_unite(n_samples)
    
    # 5. Densité de X
    X = points_in_ball[:, 0]
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(X, bins=50, density=True, label='Densité empirique de X')
    # Tracé de la vraie densité
    x_th = np.linspace(-1, 1, 100)
    y_th = 3/4 * (1 - x_th**2)
    plt.plot(x_th, y_th, 'r-', label='Densité théorique $f(x)=3/4(1-x^2)$')
    plt.title("Distribution marginale d'une coordonnée")
    plt.legend()
    
    # 6. Densité de X / ||(X,Y,Z)||
    R = np.sqrt(np.sum(points_in_ball**2, axis=1))
    X_normalized = points_in_ball[:, 0] / R
    
    plt.subplot(1, 2, 2)
    plt.hist(X_normalized, bins=50, density=True, label='Densité empirique de X/R')
    plt.ylim(0, 1) # Uniforme sur [-1,1] a une densité de 0.5
    plt.title("Distribution de la coordonnée normalisée (sur la sphère)")
    plt.axhline(0.5, color='r', linestyle='--', label='Densité théorique U[-1,1]')
    plt.legend()
    plt.show()

# =============================================================================
# Exercice 6: Volume d'une sphère
# =============================================================================
def ex6():
    print("\n--- Exercice 6 ---")
    
    def sphere_volume(d, R):
        return (np.pi**(d/2) / math.gamma(d/2 + 1)) * R**d

    dims = np.arange(1, 11)
    
    # 1. Plot des volumes
    plt.figure(figsize=(10,6))
    plt.plot(dims, [sphere_volume(d, 0.5) for d in dims], 'o-', label='R = 0.5')
    plt.plot(dims, [sphere_volume(d, 1.0) for d in dims], 'o-', label='R = 1.0')
    plt.plot(dims, [sphere_volume(d, 2.0) for d in dims], 'o-', label='R = 2.0')
    plt.xlabel("Dimension (d)")
    plt.ylabel("Volume de l'hypersphère")
    plt.title("Volume de l'hypersphère en fonction de la dimension")
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()

    # 4. Monte-Carlo
    def mc_sphere_volume(d, R, n_samples):
        points = -R + 2 * R * npr.rand(n_samples, d)
        radii_sq = np.sum(points**2, axis=1)
        n_in = np.sum(radii_sq <= R**2)
        cube_volume = (2 * R)**d
        return cube_volume * (n_in / n_samples)

    # 6. Comparaison pour d=7
    d = 7
    R = 1.0
    true_vol = sphere_volume(d, R)
    
    print(f"Calcul du volume de la sphère en dimension {d} (R={R})")
    print(f"Vraie valeur: {true_vol:.4f}")
    
    # Déterministe :
    # Pour N=10, le nombre de points est (2*1*10)^7 = 20^7 = 1.28e9 (déjà lent)
    # Pour N=100, le nombre est 200^7, impossible.
    print("La méthode déterministe est impraticable en d=7.")
    
    # Monte-Carlo
    n_mc = 10**6
    vol_mc = mc_sphere_volume(d, R, n_mc)
    print(f"Méthode Monte-Carlo avec {n_mc:,} points: {vol_mc:.4f}")

# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    ex1()
    ex2()
    ex3()
    ex4()
    ex5()
    ex6()
