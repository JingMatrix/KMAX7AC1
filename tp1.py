import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import time
import math

# =============================================================================
# Exercice 1: Simulation d'une variable discrète
# =============================================================================


def simule_discrete_v1(m=1000):
    """ Algorithme 1: ordre initial """
    X = np.zeros(m)
    for i in range(m):
        u = npr.rand()
        if u <= 0.3:
            X[i] = 1
        elif u <= 0.4:
            X[i] = 2
        else:
            X[i] = 3
    return X


def simule_discrete_v2(m=1000):
    """ Algorithme 2: ordre optimisé """
    X = np.zeros(m)
    for i in range(m):
        u = npr.rand()
        if u <= 0.6:
            X[i] = 3
        elif u <= 0.9:
            X[i] = 1
        else:
            X[i] = 2
    return X


def simule_discrete_generic(x_values, p_values, m=1000):
    """ Fonction générique (optimisée) """
    d = len(x_values)
    # Trier par probabilités descendantes pour l'optimisation
    sorted_indices = np.argsort(p_values)[::-1]
    sorted_x = np.array(x_values)[sorted_indices]
    sorted_p = np.array(p_values)[sorted_indices]

    # Calculer les sommes cumulées
    s = np.cumsum(sorted_p)

    X = np.zeros(m)
    for i in range(m):
        u = npr.rand()
        # Trouver la première somme cumulée > u
        k = 0
        while u > s[k]:
            k += 1
        X[i] = sorted_x[k]
    return X


def ex1():
    print("--- Exercice 1 ---")
    # 5. Comparaison de vitesse
    N_samples = 1000000

    start_time = time.time()
    samples_v1 = simule_discrete_v1(N_samples)
    time_v1 = time.time() - start_time

    start_time = time.time()
    samples_v2 = simule_discrete_v2(N_samples)
    time_v2 = time.time() - start_time

    print(f"Temps V1 (original): {time_v1:.4f}s")
    print(f"Temps V2 (optimisé): {time_v2:.4f}s")

    # Vérification par histogramme
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(samples_v1, bins=[0.5, 1.5, 2.5, 3.5], density=True, rwidth=0.8)
    plt.xticks([1, 2, 3])
    plt.title("Histogramme V1")
    plt.axhline(0.3, xmin=0.1, xmax=0.3, color='r', linestyle='--')
    plt.axhline(0.1, xmin=0.4, xmax=0.6, color='r', linestyle='--')
    plt.axhline(0.6, xmin=0.7, xmax=0.9, color='r', linestyle='--')

    plt.subplot(1, 2, 2)
    plt.hist(samples_v2, bins=[0.5, 1.5, 2.5, 3.5], density=True, rwidth=0.8)
    plt.xticks([1, 2, 3])
    plt.title("Histogramme V2")
    plt.show()

    # 7. Application cas générique
    x_i = np.arange(1, 11)
    p_i = x_i / 55.0
    samples_generic = simule_discrete_generic(x_i, p_i, 10000)
    plt.figure()
    plt.hist(samples_generic, bins=np.arange(
        0.5, 11.5, 1), density=True, rwidth=0.8)
    plt.plot(x_i, p_i, 'ro', label='Probabilités théoriques')
    plt.title("Cas générique: p_i = i/55")
    plt.legend()
    plt.show()

# =============================================================================
# Exercice 2: Loi Binomiale
# =============================================================================


def binom1(n, p, m=1000):
    """ Méthode 1: Somme de Bernoulli """
    return np.sum(npr.rand(m, n) < p, axis=1)


def binom2(n, p, m=1000):
    """ Méthode 2: Inversion généralisée """
    # Calculer les probabilités P(X=k)
    q = np.zeros(n + 1)
    q[0] = (1 - p)**n
    for k in range(n):
        ratio = (n - k) / (k + 1) * p / (1 - p)
        q[k+1] = q[k] * ratio

    # Calculer les sommes cumulées
    s = np.cumsum(q)

    # Simulation par inversion
    X = np.zeros(m)
    for i in range(m):
        u = npr.rand()
        k = 0
        while u > s[k]:
            k += 1
        X[i] = k
    return X


def ex2():
    print("\n--- Exercice 2 ---")
    n, p = 20, 0.4
    N_samples = 10000

    start_time = time.time()
    samples_b1 = binom1(n, p, N_samples)
    time_b1 = time.time() - start_time

    start_time = time.time()
    samples_b2 = binom2(n, p, N_samples)
    time_b2 = time.time() - start_time

    print(f"Temps binom1 (Bernoulli): {time_b1:.4f}s")
    print(f"Temps binom2 (Inversion): {time_b2:.4f}s")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(samples_b1, bins=np.arange(-0.5, n + 1.5, 1), density=True)
    plt.title('Binomiale par somme de Bernoulli')

    plt.subplot(1, 2, 2)
    plt.hist(samples_b2, bins=np.arange(-0.5, n + 1.5, 1), density=True)
    plt.title('Binomiale par inversion')
    plt.show()

# =============================================================================
# Exercice 3: Loi géométrique paire
# =============================================================================


def geom_rejet(p, m=1000):
    """ Simule une loi géométrique paire par rejet """
    res = np.zeros(m)
    for i in range(m):
        while True:
            # On utilise la formule d'inversion pour la loi géométrique
            k = math.floor(math.log(npr.rand()) / math.log(1 - p)) + 1
            if k % 2 == 0:
                res[i] = k
                break
    return res


def geom_direct(p, m=1000):
    """ Simule une loi géométrique paire par la méthode directe """
    p_prime = p * (2 - p)
    # Simule H ~ Geom(p_prime)
    H = np.floor(np.log(npr.rand(m)) / np.log(1 - p_prime)) + 1
    return 2 * H


def ex3():
    print("\n--- Exercice 3 ---")
    p = 0.3
    N_samples = 10000

    start_time = time.time()
    samples_g1 = geom_rejet(p, N_samples)
    time_g1 = time.time() - start_time

    start_time = time.time()
    samples_g2 = geom_direct(p, N_samples)
    time_g2 = time.time() - start_time

    print(f"Temps Géométrique par rejet: {time_g1:.4f}s")
    print(f"Temps Géométrique directe: {time_g2:.4f}s")

    plt.figure(figsize=(12, 5))
    bins = np.arange(1.5, 20.5, 2)
    plt.subplot(1, 2, 1)
    plt.hist(samples_g1, bins=bins, density=True)
    plt.title("Géométrique paire (rejet)")
    plt.subplot(1, 2, 2)
    plt.hist(samples_g2, bins=bins, density=True)
    plt.title("Géométrique paire (directe)")
    plt.show()

# =============================================================================
# Exercice 4: Loi sur un ensemble fini (Sac à dos)
# =============================================================================


def choix_uniforme_remplissage(n):
    return (npr.rand(n) < 0.5).astype(int)


def poids_sac(poids_objets, remplissage):
    return np.dot(poids_objets, remplissage)


def remplissage_aleatoire(poids_objets, poids_max):
    while True:
        remplissage = choix_uniforme_remplissage(len(poids_objets))
        if poids_sac(poids_objets, remplissage) <= poids_max:
            return remplissage


def estime_N(poids_objets, poids_max, M=10000):
    n = len(poids_objets)
    M_valide = 0
    for _ in range(M):
        remplissage = choix_uniforme_remplissage(n)
        if poids_sac(poids_objets, remplissage) <= poids_max:
            M_valide += 1

    p_hat = M_valide / M
    N_hat = p_hat * (2**n)
    return N_hat


def ex4():
    print("\n--- Exercice 4 ---")
    poids = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    poids_max = 10

    print("Exemple de remplissage valide:",
          remplissage_aleatoire(poids, poids_max))

    N_estime = estime_N(poids, poids_max, M=100000)
    print(f"Nombre estimé de remplissages valides (N): {N_estime:.2f}")

# =============================================================================
# Exercice 5: Inversion de la fonction de répartition
# =============================================================================


def simule_weibull(m=1000):
    u = npr.rand(m)
    return (-np.log(u))**(1/3)


def simule_cauchy(c, m=1000):
    u = npr.rand(m)
    return c * np.tan(np.pi * (u - 0.5))


def plot_moyennes_empiriques(samples, title):
    moyennes = np.cumsum(samples) / np.arange(1, len(samples) + 1)
    plt.plot(moyennes)
    plt.title(title)
    plt.xlabel("Nombre d'échantillons (N)")
    plt.ylabel("Moyenne empirique")


def ex5():
    print("\n--- Exercice 5 ---")
    N_samples = 10000

    # Weibull
    Y = simule_weibull(N_samples)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_moyennes_empiriques(
        Y, "Convergence de la moyenne empirique (Weibull)")

    # Cauchy
    c = 1.0
    Z = simule_cauchy(c, N_samples)
    plt.subplot(1, 2, 2)
    plot_moyennes_empiriques(
        Z, "Non-convergence de la moyenne empirique (Cauchy)")
    # Limiter l'axe y pour la visibilité
    plt.ylim(-20, 20)
    plt.show()

    # Loi de la moyenne empirique pour Cauchy
    M = 5000  # Nombre de moyennes à calculer
    n_avg = 100  # Taille de chaque échantillon pour la moyenne
    moyennes_cauchy = [np.mean(simule_cauchy(c, n_avg)) for _ in range(M)]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(Z, bins=100, range=(-10, 10), density=True)
    plt.title("Histogramme d'un échantillon Cauchy")
    plt.subplot(1, 2, 2)
    plt.hist(moyennes_cauchy, bins=100, range=(-10, 10), density=True)
    plt.title(f"Histogramme de la moyenne empirique (n={n_avg})")
    plt.show()


# =============================================================================
# Exercice 9: Uniforme sur une Cardioïde
# =============================================================================
def simule_cardioide(a, n_points):
    points = []
    bounding_box_area = (4*a)**2
    n_trials = 0
    while len(points) < n_points:
        n_trials += 1
        x = -2*a + 4*a * npr.rand()
        y = -2*a + 4*a * npr.rand()
        r2 = x**2 + y**2
        if (r2 - a*x)**2 <= a**2 * r2:
            points.append((x, y))

    points = np.array(points)

    # Estimation de l'aire
    p_accept = n_points / n_trials
    estimated_area = p_accept * bounding_box_area
    rejection_prob = 1 - p_accept

    return points, estimated_area, rejection_prob


def ex9():
    print("\n--- Exercice 9 ---")
    a = 1
    N_samples = 1000

    points, est_area, rej_prob = simule_cardioide(a, N_samples)

    # Tracé
    theta = np.linspace(0, 2 * np.pi, 200)
    r = a * (1 + np.cos(theta))
    x_card = r * np.cos(theta)
    y_card = r * np.sin(theta)

    plt.figure(figsize=(8, 8))
    plt.plot(x_card, y_card, 'r-', label='Cardioïde')
    plt.plot([-2*a, 2*a, 2*a, -2*a, -2*a],
             [-2*a, -2*a, 2*a, 2*a, -2*a], 'k--', label='Pavé')
    plt.scatter(points[:, 0], points[:, 1], s=5,
                label=f'{N_samples} points simulés')
    plt.title("Simulation uniforme sur une cardioïde")
    plt.axis('equal')
    plt.legend()
    plt.show()

    true_area = 1.5 * np.pi * a**2
    print(f"Aire théorique : {true_area:.4f}")
    print(f"Aire estimée par Monte-Carlo : {est_area:.4f}")
    print(f"Probabilité de rejet estimée : {rej_prob:.4f}")
    print(f"Probabilité de rejet théorique : {1 - true_area/((4*a)**2):.4f}")


# =============================================================================
# Exercice 12: Méthode de Box-Muller et autres
# =============================================================================
def simule_normale_boxmuller(m, s, n_samples=1000):
    Z = np.zeros(n_samples)
    for i in range(n_samples):
        u1, u2 = npr.rand(2)
        Z[i] = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    return m + s * Z


def simule_normale_rejet(m, s, n_samples=1000):
    Z = np.zeros(n_samples)
    c = np.sqrt(2 * np.pi / np.e)
    for i in range(n_samples):
        while True:
            y = np.tan(np.pi * (npr.rand() - 0.5))  # Cauchy(0,1)
            u = npr.rand()
            if u <= (1 + y**2) * np.exp(-y**2 / 2) / (2 * np.exp(-1/2)):
                Z[i] = y
                break
    return m + s * Z


def simule_normale_tlc(m, s, n_samples=1000):
    # n=12 pour simplifier la variance
    Z = np.sum(npr.rand(n_samples, 12), axis=1) - 6
    return m + s * Z


def ex12():
    print("\n--- Exercice 12 ---")
    m, s = 0, 1
    N = 5000

    start = time.time()
    bm_samples = simule_normale_boxmuller(m, s, N)
    t_bm = time.time() - start

    start = time.time()
    rej_samples = simule_normale_rejet(m, s, N)
    t_rej = time.time() - start

    start = time.time()
    tlc_samples = simule_normale_tlc(m, s, N)
    t_tlc = time.time() - start

    print(f"Temps Box-Muller: {t_bm:.4f}s")
    print(f"Temps Rejet Cauchy: {t_rej:.4f}s")
    print(f"Temps TLC (n=12): {t_tlc:.4f}s")

    plt.figure(figsize=(15, 5))
    x = np.linspace(-4, 4, 200)
    pdf_norm = (1/(s * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x-m)/s)**2)

    plt.subplot(1, 3, 1)
    plt.hist(bm_samples, bins=50, density=True)
    plt.plot(x, pdf_norm, 'r')
    plt.title("Box-Muller")

    plt.subplot(1, 3, 2)
    plt.hist(rej_samples, bins=50, density=True)
    plt.plot(x, pdf_norm, 'r')
    plt.title("Rejet Cauchy")

    plt.subplot(1, 3, 3)
    plt.hist(tlc_samples, bins=50, density=True)
    plt.plot(x, pdf_norm, 'r')
    plt.title("TLC (n=12)")
    plt.show()


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    ex1()
    ex2()
    ex3()
    ex4()
    ex5()
    ex9()
    ex12()
