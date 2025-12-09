import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.stats import norm

# =============================================================================
# Exercice 1: Régression et descente de gradient stochastique
# =============================================================================


def partie1_exo1():
    print("--- Partie 1 / Exercice 1: Régression Linéaire ---")

    # --- Q3: Charger et afficher les données ---
    print("\n[Question 3] Chargement et affichage des données...")
    # NOTE: Le fichier TP6-exo1.txt doit exister dans le même répertoire.
    data = np.loadtxt('TP6-exo1.txt')
    X, Y = data[:, 0], data[:, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, alpha=0.6, label='Données')
    plt.title("Données de régression linéaire")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Q4: Implémenter l'algorithme et observer la convergence ---
    print("\n[Question 4] Implémentation de l'algorithme de Robbins-Monro...")

    def gamma(n, c=0.1, alpha=1):
        """Définit la suite des pas de la forme c / (n+1)^alpha."""
        return c / (n + 1)**alpha

    def robbins_monro_regression(X_data, Y_data, gamma_func):
        a, b = 0.0, 0.0
        a_history, b_history = [a], [b]
        for n in range(len(X_data)):
            g = gamma_func(n)
            error = Y_data[n] - (a * X_data[n] + b)
            a = a + 2 * g * error * X_data[n]
            b = b + 2 * g * error
            a_history.append(a)
            b_history.append(b)
        return a, b, a_history, b_history

    # Comparaison de l'effet de la constante c dans la suite des pas
    plt.figure(figsize=(10, 7))
    for c_val in [0.2, 1.3]:
        _, _, a_h, _ = robbins_monro_regression(
            X, Y, lambda n: gamma(n, c=c_val))
        plt.plot(a_h, label=f'a_n (c={c_val})')

    plt.title(
        "Convergence du paramètre 'a' pour différentes suites de pas $\\gamma_n=c/n^{0.6}$")
    plt.xlabel("Itération (n)")
    plt.ylabel("Valeur du paramètre a_n")
    plt.grid(True)
    plt.legend()
    plt.show()
    print("Observation: une constante 'c' plus grande accélère la convergence au début, mais peut créer plus d'oscillations.")

    # --- Q5: Comparer avec l'estimateur des moindres carrés ---
    print(
        "\n[Question 5] Comparaison avec la méthode des moindres carrés ordinaires...")
    # MCO (Moindres Carrés Ordinaires) est la solution analytique qui minimise
    # la somme des carrés des erreurs pour l'ensemble des données.
    # En anglais : OLS (Ordinary Least Squares).
    a_mco = np.cov(X, Y, ddof=0)[0, 1] / np.var(X)
    b_mco = np.mean(Y) - a_mco * np.mean(X)

    # On relance une simulation pour obtenir les valeurs finales
    a_rm, b_rm, _, _ = robbins_monro_regression(
        X, Y, lambda n: gamma(n, c=0.2))

    print(f"Robbins-Monro final: a_n = {a_rm:.4f}, b_n = {b_rm:.4f}")
    print(f"MCO (solution exacte): a_mco = {a_mco:.4f}, b_mco = {b_mco:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, alpha=0.3, label='Données')
    x_grid = np.array([X.min(), X.max()])
    plt.plot(x_grid, a_rm * x_grid + b_rm, 'r-',
             lw=3, label='Robbins-Monro (final)')
    plt.plot(x_grid, a_mco * x_grid + b_mco, 'g--', lw=3, label='MCO (exact)')
    plt.title("Comparaison des droites de régression")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()


# =============================================================================
# Exercice 2 (Partie 2): Modèle autorégressif
# =============================================================================
def partie2_exo2():
    print("\n--- Partie 2 / Exercice 2: Modèle autorégressif ---")

    theta_star = 0.7
    n_points = 2000

    def simuler_donnees_ar(theta_vrai, n):
        X_sim = np.zeros(n + 1)
        for i in range(n):
            X_sim[i + 1] = theta_vrai * X_sim[i] + npr.normal(0, 1)
        return X_sim

    X = simuler_donnees_ar(theta_star, n_points)

    def estimer_theta_ar(X_data, gamma_func):
        theta = 0.0
        theta_history = [theta]
        for n in range(len(X_data) - 1):
            g = gamma_func(n)
            theta = theta + 2 * g * X_data[n] * \
                (X_data[n + 1] - theta * X_data[n])
            theta_history.append(theta)
        return theta_history

    history = estimer_theta_ar(X, lambda n: 0.1 / (n + 1)**0.8)

    plt.figure(figsize=(8, 6))
    plt.plot(history, label=r'$\theta_n$')
    plt.axhline(theta_star, color='r', linestyle='--',
                label=r'$\theta^*$ (vraie valeur)')
    plt.title(r"Convergence de l'estimateur de $\theta$")
    plt.xlabel("Itération (n)")
    plt.ylabel(r'Valeur de $\theta_n$')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nVérification du Théorème Central Limite (TLC)...")
    n_sims = 2000
    erreurs_finales_normalisees = []

    for _ in range(n_sims):
        X_sim = simuler_donnees_ar(theta_star, n_points)
        def gamma(n): return 0.1 / (n + 1)**0.8
        history_sim = estimer_theta_ar(X_sim, gamma)
        theta_final = history_sim[-1]
        erreurs_finales_normalisees.append(
            (theta_final - theta_star) / np.sqrt(gamma(n_points)))

    plt.figure(figsize=(8, 6))
    plt.hist(erreurs_finales_normalisees, bins=30, density=True,
             label='Distribution des erreurs normalisées')
    mu_err, std_err = np.mean(erreurs_finales_normalisees), np.std(
        erreurs_finales_normalisees)
    x_grid = np.linspace(min(erreurs_finales_normalisees),
                         max(erreurs_finales_normalisees), 200)
    plt.plot(x_grid, norm.pdf(x_grid, mu_err, std_err),
             'r-', lw=2, label='Densité Normale ajustée')
    plt.title(
        r"Distributions de $(\theta_n - \theta^*)/\sqrt{\gamma_n}$" +
        f" et N({
            mu_err:.3f}, {
            std_err:.3f})")
    plt.xlabel("Erreur normalisée")
    plt.ylabel("Densité")
    plt.legend()
    plt.grid(True)
    plt.show()


# =============================================================================
# Exercice 3: Bandit à deux bras
# =============================================================================
def exo3_bandit():
    print("\n--- Exercice 3: Bandit à deux bras ---")

    # =========================================================================
    # Fonctions pour chaque stratégie
    # =========================================================================

    # Stratégie 1: "Greedy" (gourmande)
    def strategie1_greedy(p_A, p_B, n_etapes):
        sA, nA, sB, nB = 0, 0, 0, 0  # succès et tentatives pour A et B
        recompenses = []
        for _ in range(n_etapes):
            # Taux de réussite (avec initialisation +1/+1 pour éviter la
            # division par zéro)
            taux_A = (1 + sA) / (1 + nA)
            taux_B = (1 + sB) / (1 + nB)

            if taux_A >= taux_B:
                # Choisir le bras A
                recompense = 1 if npr.rand() < p_A else 0
                sA += recompense
                nA += 1
            else:
                # Choisir le bras B
                recompense = 1 if npr.rand() < p_B else 0
                sB += recompense
                nB += 1
            recompenses.append(recompense)
        return np.mean(recompenses)

    # Stratégie 2: "Win-Stay, Lose-Switch"
    def strategie2_wsls(p_A, p_B, n_etapes):
        choix_actuel = 'A'  # On commence avec A
        recompenses = []
        for _ in range(n_etapes):
            if choix_actuel == 'A':
                recompense = 1 if npr.rand() < p_A else 0
                if recompense == 0:  # Perdu
                    choix_actuel = 'B'
            else:  # choix_actuel == 'B'
                recompense = 1 if npr.rand() < p_B else 0
                if recompense == 0:  # Perdu
                    choix_actuel = 'A'
            recompenses.append(recompense)
        return np.mean(recompenses)

    # =========================================================================
    # Comparaison des stratégies sur 100 parties
    # =========================================================================
    n_parties = 100
    n_simulations = 1000  # On moyenne sur de nombreuses simulations pour un résultat stable

    print(
        f"\nComparaison du gain moyen sur {n_parties} parties (moyenné sur {n_simulations} simulations)...")

    # --- Cas 1: p_A = 0.9, p_B = 0.1 ---
    pA1, pB1 = 0.9, 0.1
    print(f"\nCas 1: p_A = {pA1}, p_B = {pB1} (Bras A est nettement meilleur)")

    gain_moyen_s1_c1 = np.mean(
        [strategie1_greedy(pA1, pB1, n_parties) for _ in range(n_simulations)])
    gain_moyen_s2_c1 = np.mean(
        [strategie2_wsls(pA1, pB1, n_parties) for _ in range(n_simulations)])

    print(f"  Gain moyen Stratégie 1 (Greedy): {gain_moyen_s1_c1:.4f}")
    print(f"  Gain moyen Stratégie 2 (WSLS):   {gain_moyen_s2_c1:.4f}")
    print(f"  Gain optimal théorique: {max(pA1, pB1):.4f}")

    # --- Cas 2: p_A = 0.45, p_B = 0.55 ---
    pA2, pB2 = 0.45, 0.55
    print(
        f"\nCas 2: p_A = {pA2}, p_B = {pB2} (Bras B est légèrement meilleur)")

    gain_moyen_s1_c2 = np.mean(
        [strategie1_greedy(pA2, pB2, n_parties) for _ in range(n_simulations)])
    gain_moyen_s2_c2 = np.mean(
        [strategie2_wsls(pA2, pB2, n_parties) for _ in range(n_simulations)])

    print(f"  Gain moyen Stratégie 1 (Greedy): {gain_moyen_s1_c2:.4f}")
    print(f"  Gain moyen Stratégie 2 (WSLS):   {gain_moyen_s2_c2:.4f}")
    print(f"  Gain optimal théorique: {max(pA2, pB2):.4f}")

    # =========================================================================
    # Analyse de la convergence de la Stratégie 3
    # =========================================================================
    print("\nAnalyse détaillée de la Stratégie 3 (Robbins-Monro)...")
    p_A, p_B = 0.8, 0.5  # A est clairement le meilleur bras
    n_essais = 2000

    recompenses_rm_run = []
    theta = 0.5
    theta_history = [theta]
    def gamma_func(n): return 0.5 / (n + 10)**0.2

    for n in range(n_essais):
        g = gamma_func(n)
        if npr.rand() < theta:
            recompense = 1 if npr.rand() < p_A else 0
            if recompense == 1:
                theta += g * (1 - theta)
        else:
            recompense = 1 if npr.rand() < p_B else 0
            if recompense == 1:
                theta -= g * theta

        theta = np.clip(theta, 0, 1)
        recompenses_rm_run.append(recompense)
        theta_history.append(theta)

    print(
        f"Convergence de theta_n: la probabilité finale de choisir A est {theta_history[-1]:.4f}")
    print("La convergence vers 1 est observée, comme prédit par la théorie.")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Évolution de theta_n
    ax1.plot(theta_history, label=r'$\theta_n$')
    ax1.axhline(1.0, color='r', linestyle='--', label='Cible (1)')
    ax1.set_title(
        fr"Évolution de $\theta_n$ (Stratégie 3, $p_A={p_A}, p_B={p_B}$)")
    ax1.set_ylabel(r"Probabilité de choisir A ($\theta_n$)")
    ax1.set_ylim(-0.1, 1.1)
    ax1.legend()
    ax1.grid(True)

    # Convergence du taux de réussite global
    taux_reussite_global = np.cumsum(
        recompenses_rm_run) / (np.arange(n_essais) + 1)
    ax2.plot(taux_reussite_global, label='Taux de réussite global')
    ax2.axhline(max(p_A, p_B), color='r', linestyle='--',
                label=f'Gain optimal max($p_A,p_B$) = {max(p_A, p_B)}')
    ax2.set_title("Convergence du Taux de Réussite vers le Gain Optimal")
    ax2.set_xlabel("Essai (n)")
    ax2.set_ylabel("Gain moyen cumulé")
    ax2.set_ylim(bottom=p_B - 0.1)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


# =============================================================================
# Exercice 4: Estimation d'un quantile
# =============================================================================


def exo4_quantile():
    print("\n--- Exercice 4: Estimation d'un quantile ---")
    alpha_quantile = 0.75
    theta_alpha = norm.ppf(alpha_quantile)
    print(
        f"Le quantile théorique q_{alpha_quantile} pour N(0,1) est: {
            theta_alpha:.4f}")
    n_steps = 2000
    n_sims = 500

    def estimer_quantile(n_etapes, gamma_func):
        theta = 0.0
        theta_history = [theta]
        for i in range(n_etapes):
            g = gamma_func(i)
            X_next = npr.normal(0, 1)
            indicator = 1 if X_next <= theta else 0
            theta = theta - g * (indicator - alpha_quantile)
            theta_history.append(theta)
        return theta_history

    # --- Q3: Convergence numérique ---
    history = estimer_quantile(n_steps, lambda i: 1 / (i + 20)**0.4)

    plt.figure(figsize=(8, 6))
    plt.plot(history, label=r'$\theta_n$')
    plt.axhline(theta_alpha, color='r', linestyle='--',
                label=r'$\theta_\alpha$ (théorique)')
    plt.title("Q3: Convergence de l'estimateur du quantile")
    plt.xlabel("Itération (n)")
    plt.ylabel(r"Valeur de $\theta_n$")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Q4: Étude des fluctuations (TLC) ---
    erreurs_finales = []
    for _ in range(n_sims):
        def gamma(i): return 0.1 / (i + 1)**0.6
        h = estimer_quantile(n_steps, gamma)
        erreurs_finales.append(1 / np.sqrt(gamma(n_steps))
                               * (h[-1] - theta_alpha))

    plt.figure(figsize=(8, 6))
    plt.hist(erreurs_finales, bins=30, density=True,
             label='Distribution des erreurs normalisées')
    mu_err, std_err = np.mean(erreurs_finales), np.std(erreurs_finales)
    x_grid = np.linspace(min(erreurs_finales), max(erreurs_finales), 200)
    plt.plot(x_grid, norm.pdf(x_grid, mu_err, std_err),
             'r-', lw=2, label='Densité Normale ajustée')
    plt.title(
        r"Q4: Distributions de $(\theta_n - \theta_\alpha)/\sqrt{\gamma_n}$" +
        f" et N({
            mu_err:.4f}, {
            std_err:.4f})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Q5: Étude de l'erreur L2 ---
    plt.figure(figsize=(8, 6))
    n_steps_l2 = 1000

    # On compare l'influence de l'exposant alpha dans gamma_n = c / n^alpha
    for exposant in [0.6, 0.8, 1.0]:
        erreurs_quadratiques = np.zeros(n_steps_l2)
        for _ in range(n_sims // 5):
            h = estimer_quantile(n_steps_l2, lambda i: 1.0 / (i + 1)**exposant)
            erreurs_quadratiques += (np.array(h[1:]) - theta_alpha)**2

        plt.plot(range(n_steps_l2), erreurs_quadratiques / (n_sims // 5),
                 label=fr'$\gamma_n \propto 1/n^{{{exposant}}}$')

    plt.xscale('log')
    plt.yscale('log')
    plt.title(r"Q5: Erreur $L^2$ moyenne pour différentes suites de pas")
    plt.xlabel("Itération (n)")
    plt.ylabel(r"$\mathbb{E}[(\theta_n - \theta_\alpha)^2]$")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == '__main__':
    # partie1_exo1()
    partie2_exo2()
    # exo3_bandit()
    exo4_quantile()
