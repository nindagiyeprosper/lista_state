import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import *
from scipy.integrate import quad
from math import gamma as gamma_func, factorial
import warnings
warnings.filterwarnings('ignore')

class LoisProbabilite:
    def __init__(self):
        """Initialisation de la classe des lois de probabilité"""
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def afficher_proprietes(self, nom_loi, esperance, variance):
        """Affiche les propriétés d'une loi de probabilité"""
        ecart_type = np.sqrt(variance)
        print(f"\n{nom_loi}:")
        print(f"  Espérance (μ) = {esperance:.6f}")
        print(f"  Variance (σ²) = {variance:.6f}")
        print(f"  Écart-type (σ) = {ecart_type:.6f}")
        return ecart_type
    
    def esperance_tronquee(self, loi, d, **params):
        """Calcule l'espérance tronquée E[X ∧ d] = E[min(X,d)]"""
        try:
            if loi == 'uniforme':
                a, b = params['a'], params['b']
                if d <= a:
                    return d
                elif d >= b:
                    return (a + b) / 2
                else:
                    return (d**2 - a**2) / (2 * (b - a)) + (b - d) * d / (b - a)
                    
            elif loi == 'exponentielle':
                beta = params['beta']
                return (1 - np.exp(-beta * d)) / beta
                
            elif loi == 'normale':
                mu, sigma = params['mu'], params['sigma']
                z = (d - mu) / sigma
                return mu * norm.cdf(z) - sigma * norm.pdf(z) + d * (1 - norm.cdf(z))
                
            elif loi == 'lognormale':
                mu, sigma = params['mu'], params['sigma']
                z = (np.log(d) - mu) / sigma
                return np.exp(mu + sigma**2/2) * norm.cdf(z - sigma) + d * (1 - norm.cdf(z))
                
            elif loi == 'gamma':
                a, beta = params['a'], params['beta']
                # Approximation utilisant la fonction gamma incomplète
                x = beta * d
                return (a / beta) * gamma(a).cdf(x) + d * (1 - gamma(a).cdf(x))
                
            elif loi == 'weibull':
                c, beta = params['c'], params['beta']
                # Approximation
                return (1/beta) * gamma_func(1 + 1/c) * gamma(1 + 1/c).cdf((beta*d)**c) + d * (1 - gamma(1 + 1/c).cdf((beta*d)**c))
                
            elif loi == 'poisson':
                lam = params['lam']
                k_max = min(int(d * 2), 1000)
                k_values = np.arange(0, k_max + 1)
                result = 0
                for k in k_values:
                    if k <= d:
                        result += k * poisson.pmf(k, lam)
                    else:
                        result += d * poisson.pmf(k, lam)
                return result
                
            elif loi == 'binomiale':
                n, p = params['n'], params['p']
                result = 0
                for k in range(0, n + 1):
                    if k <= d:
                        result += k * binom.pmf(k, n, p)
                    else:
                        result += d * binom.pmf(k, n, p)
                return result
                
            else:
                # Méthode générique par intégration/somme
                if loi in ['poisson', 'binomiale', 'bernoulli', 'geometrique', 'logarithmique']:
                    # Lois discrètes
                    x_max = 1000
                    result = 0
                    for k in range(0, x_max + 1):
                        prob = self.probabilite_point(loi, k, **params)
                        result += min(k, d) * prob
                    return result
                else:
                    # Lois continues
                    def integrande(x):
                        return min(x, d) * self.probabilite_point(loi, x, **params)
                    
                    x_min = params.get('x_min', 0)
                    x_max = params.get('x_max', 1000)
                    result, _ = quad(integrande, x_min, x_max)
                    return result
                    
        except Exception as e:
            print(f"Erreur dans le calcul espérance tronquée: {e}")
            return None
    
    def exces_moyen(self, loi, d, **params):
        """Calcule l'excès moyen E[(X-d)+] = E[max(X-d,0)] (fonction stop-loss)"""
        try:
            esperance = self._get_esperance(loi, **params)
            esperance_tronquee = self.esperance_tronquee(loi, d, **params)
            return esperance - esperance_tronquee
            
        except Exception as e:
            print(f"Erreur dans le calcul excès moyen: {e}")
            return None
    
    def esperance_limite(self, loi, d, **params):
        """Calcule l'espérance limite E[X|X>d]"""
        try:
            if loi in ['poisson', 'binomiale', 'bernoulli', 'geometrique', 'logarithmique']:
                # Lois discrètes
                prob_survie = 1 - self._cdf_discrete(loi, d, **params)
                if prob_survie == 0:
                    return d
                
                esperance_conditionnelle = 0
                x_max = 1000
                for k in range(int(d) + 1, x_max + 1):
                    prob = self.probabilite_point(loi, k, **params)
                    esperance_conditionnelle += k * prob
                
                return esperance_conditionnelle / prob_survie
                
            else:
                # Lois continues
                def integrande(x):
                    return x * self.probabilite_point(loi, x, **params)
                
                x_min = max(d, params.get('x_min', 0))
                x_max = params.get('x_max', 1000)
                
                numerateur, _ = quad(integrande, x_min, x_max)
                denominateur = 1 - self._cdf_continue(loi, d, **params)
                
                if denominateur == 0:
                    return d
                
                return numerateur / denominateur
                
        except Exception as e:
            print(f"Erreur dans le calcul espérance limite: {e}")
            return None
    
    def _get_esperance(self, loi, **params):
        """Retourne l'espérance théorique d'une loi"""
        if loi == 'uniforme':
            return (params['a'] + params['b']) / 2
        elif loi == 'exponentielle':
            return 1 / params['beta']
        elif loi == 'normale':
            return params['mu']
        elif loi == 'lognormale':
            return np.exp(params['mu'] + params['sigma']**2/2)
        elif loi == 'gamma':
            return params['a'] / params['beta']
        elif loi == 'weibull':
            return (1/params['beta']) * gamma_func(1 + 1/params['c'])
        elif loi == 'poisson':
            return params['lam']
        elif loi == 'binomiale':
            return params['n'] * params['p']
        elif loi == 'bernoulli':
            return params['p']
        elif loi == 'geometrique':
            return 1 / params['p']
        elif loi == 'beta':
            return params['a'] / (params['a'] + params['b'])
        elif loi == 'pareto':
            xm, alpha = params['xm'], params['alpha']
            return xm * alpha / (alpha - 1) if alpha > 1 else np.inf
        else:
            return None
    
    def _cdf_discrete(self, loi, x, **params):
        """CDF pour les lois discrètes"""
        k = int(x)
        result = 0
        for i in range(0, k + 1):
            result += self.probabilite_point(loi, i, **params)
        return result
    
    def _cdf_continue(self, loi, x, **params):
        """CDF pour les lois continues (approximation)"""
        if loi == 'normale':
            return norm.cdf(x, params['mu'], params['sigma'])
        elif loi == 'exponentielle':
            return expon.cdf(x, scale=1/params['beta'])
        elif loi == 'uniforme':
            a, b = params['a'], params['b']
            if x < a: return 0
            elif x > b: return 1
            else: return (x - a) / (b - a)
        else:
            # Intégration numérique pour les autres lois
            def pdf_func(t):
                return self.probabilite_point(loi, t, **params)
            
            x_min = params.get('x_min', 0)
            result, _ = quad(pdf_func, x_min, x)
            return result

    def probabilite_point(self, loi, x, **params):
        """Calcule P(X=x) pour les lois discrètes ou f(x) pour les lois continues"""
        try:
            if loi == 'uniforme':
                a, b = params['a'], params['b']
                if a <= x <= b:
                    return 1 / (b - a)
                else:
                    return 0.0
                    
            elif loi == 'beta':
                a, b = params['a'], params['b']
                if 0 <= x <= 1:
                    return beta.pdf(x, a, b)
                else:
                    return 0.0
                    
            elif loi == 'exponentielle':
                beta_val = params['beta']
                if x >= 0:
                    return expon.pdf(x, scale=1/beta_val)
                else:
                    return 0.0
                    
            elif loi == 'gamma':
                a, beta_val = params['a'], params['beta']
                if x > 0:
                    return gamma.pdf(x, a, scale=1/beta_val)
                else:
                    return 0.0
                    
            elif loi == 'erlang':
                k, beta_val = params['k'], params['beta']
                if x > 0:
                    return gamma.pdf(x, k, scale=1/beta_val)
                else:
                    return 0.0
                    
            elif loi == 'weibull':
                c, beta_val = params['c'], params['beta']
                if x >= 0:
                    return weibull_min.pdf(x, c, scale=1/beta_val)
                else:
                    return 0.0
                    
            elif loi == 'normale':
                mu, sigma = params['mu'], params['sigma']
                return norm.pdf(x, mu, sigma)
                
            elif loi == 'lognormale':
                mu, sigma = params['mu'], params['sigma']
                if x > 0:
                    return lognorm.pdf(x, sigma, scale=np.exp(mu))
                else:
                    return 0.0
                    
            elif loi == 'invgauss':
                mu, lam = params['mu'], params['lam']
                if x > 0:
                    return invgauss.pdf(x, mu/lam, scale=lam)
                else:
                    return 0.0
                    
            elif loi == 'pareto':
                xm, alpha = params['xm'], params['alpha']
                if x >= xm:
                    return pareto.pdf(x, alpha, scale=xm)
                else:
                    return 0.0
                    
            elif loi == 'loglogistique':
                alpha, beta_val = params['alpha'], params['beta']
                if x > 0:
                    c = beta_val
                    scale = alpha
                    return (c/scale) * (x/scale)**(c-1) / (1 + (x/scale)**c)**2
                else:
                    return 0.0
                    
            elif loi == 'student':
                df = params['df']
                return t.pdf(x, df)
                
            # LOIS DISCRÈTES
            elif loi == 'poisson':
                lam = params['lam']
                k = int(x)
                if k >= 0:
                    return poisson.pmf(k, lam)
                else:
                    return 0.0
                    
            elif loi == 'binomiale':
                n, p = params['n'], params['p']
                k = int(x)
                if 0 <= k <= n:
                    return binom.pmf(k, n, p)
                else:
                    return 0.0
                    
            elif loi == 'bernoulli':
                p = params['p']
                k = int(x)
                if k == 0:
                    return 1 - p
                elif k == 1:
                    return p
                else:
                    return 0.0
                    
            elif loi == 'geometrique':
                p = params['p']
                k = int(x)
                if k >= 1:
                    return geom.pmf(k, p)
                else:
                    return 0.0
                    
            elif loi == 'logarithmique':
                p = params['p']
                k = int(x)
                if k >= 1:
                    return -p**k / (k * np.log(1 - p))
                else:
                    return 0.0
                    
            else:
                print(f"Loi '{loi}' non reconnue")
                return None
                
        except Exception as e:
            print(f"Erreur dans le calcul P(X=x): {e}")
            return None

    # MÉTHODES POUR CHAQUE LOI (versions complètes avec les nouvelles fonctionnalités)
    
    def uniforme(self, a=0, b=1, d=None, x=None, afficher=True, **style):
        """Loi uniforme continue U(a, b)"""
        if x is None:
            x = np.linspace(a-0.5, b+0.5, 1000)
        
        pdf = uniform.pdf(x, a, b-a)
        esperance = (a + b) / 2
        variance = (b - a)**2 / 12
        
        if afficher:
            self._visualiser(x, pdf, f"Uniforme U({a},{b})", **style)
        
        ecart_type = self.afficher_proprietes(f"Uniforme U({a},{b})", esperance, variance)
        
        # Calculs avancés
        if d is None:
            d = esperance
            
        esperance_tronq = self.esperance_tronquee('uniforme', d, a=a, b=b)
        exces_moy = self.exces_moyen('uniforme', d, a=a, b=b)
        esperance_lim = self.esperance_limite('uniforme', d, a=a, b=b)
        
        print(f"  Espérance tronquée E[X∧{d}] = {esperance_tronq:.6f}")
        print(f"  Excès moyen E[(X-{d})+] = {exces_moy:.6f}")
        print(f"  Espérance limite E[X|X>{d}] = {esperance_lim:.6f}")
        
        return esperance, variance, pdf
    
    def exponentielle(self, beta=1, d=None, x=None, afficher=True, **style):
        """Loi exponentielle de paramètre β"""
        if x is None:
            x = np.linspace(0, 10/beta, 1000)
        
        pdf = expon.pdf(x, scale=1/beta)
        esperance = 1 / beta
        variance = 1 / (beta**2)
        
        if afficher:
            self._visualiser(x, pdf, f"Exponentielle β={beta}", **style)
        
        ecart_type = self.afficher_proprietes(f"Exponentielle β={beta}", esperance, variance)
        
        # Calculs avancés
        if d is None:
            d = esperance
            
        esperance_tronq = self.esperance_tronquee('exponentielle', d, beta=beta)
        exces_moy = self.exces_moyen('exponentielle', d, beta=beta)
        esperance_lim = self.esperance_limite('exponentielle', d, beta=beta)
        
        print(f"  Espérance tronquée E[X∧{d}] = {esperance_tronq:.6f}")
        print(f"  Excès moyen E[(X-{d})+] = {exces_moy:.6f}")
        print(f"  Espérance limite E[X|X>{d}] = {esperance_lim:.6f}")
        
        return esperance, variance, pdf
    
    def normale(self, mu=0, sigma=1, d=None, x=None, afficher=True, **style):
        """Loi normale N(μ, σ²)"""
        if x is None:
            x = np.linspace(mu-4*sigma, mu+4*sigma, 1000)
        
        pdf = norm.pdf(x, mu, sigma)
        esperance = mu
        variance = sigma**2
        
        if afficher:
            self._visualiser(x, pdf, f"Normale N({mu},{sigma**2})", **style)
        
        ecart_type = self.afficher_proprietes(f"Normale N({mu},{sigma**2})", esperance, variance)
        
        # Calculs avancés
        if d is None:
            d = mu
            
        esperance_tronq = self.esperance_tronquee('normale', d, mu=mu, sigma=sigma)
        exces_moy = self.exces_moyen('normale', d, mu=mu, sigma=sigma)
        esperance_lim = self.esperance_limite('normale', d, mu=mu, sigma=sigma)
        
        print(f"  Espérance tronquée E[X∧{d}] = {esperance_tronq:.6f}")
        print(f"  Excès moyen E[(X-{d})+] = {exces_moy:.6f}")
        print(f"  Espérance limite E[X|X>{d}] = {esperance_lim:.6f}")
        
        return esperance, variance, pdf
    
    def gamma(self, a=2, beta=1, d=None, x=None, afficher=True, **style):
        """Loi gamma Γ(a, β)"""
        if x is None:
            x = np.linspace(0, 20/beta, 1000)
        
        pdf = gamma.pdf(x, a, scale=1/beta)
        esperance = a / beta
        variance = a / (beta**2)
        
        if afficher:
            self._visualiser(x, pdf, f"Gamma Γ({a},{beta})", **style)
        
        ecart_type = self.afficher_proprietes(f"Gamma Γ({a},{beta})", esperance, variance)
        
        # Calculs avancés
        if d is None:
            d = esperance
            
        esperance_tronq = self.esperance_tronquee('gamma', d, a=a, beta=beta)
        exces_moy = self.exces_moyen('gamma', d, a=a, beta=beta)
        esperance_lim = self.esperance_limite('gamma', d, a=a, beta=beta)
        
        print(f"  Espérance tronquée E[X∧{d}] = {esperance_tronq:.6f}")
        print(f"  Excès moyen E[(X-{d})+] = {exces_moy:.6f}")
        print(f"  Espérance limite E[X|X>{d}] = {esperance_lim:.6f}")
        
        return esperance, variance, pdf
    
    def weibull(self, c=2, beta=1, d=None, x=None, afficher=True, **style):
        """Loi de Weibull W(c, β)"""
        if x is None:
            x = np.linspace(0, 10/beta, 1000)
        
        pdf = weibull_min.pdf(x, c, scale=1/beta)
        esperance = (1/beta) * gamma_func(1 + 1/c)
        variance = (1/beta**2) * (gamma_func(1 + 2/c) - gamma_func(1 + 1/c)**2)
        
        if afficher:
            self._visualiser(x, pdf, f"Weibull W({c},{beta})", **style)
        
        ecart_type = self.afficher_proprietes(f"Weibull W({c},{beta})", esperance, variance)
        
        # Calculs avancés
        if d is None:
            d = esperance
            
        esperance_tronq = self.esperance_tronquee('weibull', d, c=c, beta=beta)
        exces_moy = self.exces_moyen('weibull', d, c=c, beta=beta)
        esperance_lim = self.esperance_limite('weibull', d, c=c, beta=beta)
        
        print(f"  Espérance tronquée E[X∧{d}] = {esperance_tronq:.6f}")
        print(f"  Excès moyen E[(X-{d})+] = {exces_moy:.6f}")
        print(f"  Espérance limite E[X|X>{d}] = {esperance_lim:.6f}")
        
        return esperance, variance, pdf
    
    def poisson(self, lam=3, d=None, k_max=None, afficher=True, **style):
        """Loi de Poisson P(λ)"""
        if k_max is None:
            k_max = max(15, int(lam * 3))
        
        k = np.arange(0, k_max + 1)
        pmf = poisson.pmf(k, lam)
        esperance = lam
        variance = lam
        
        if afficher:
            self._visualiser_discrete(k, pmf, f"Poisson P(λ={lam})", **style)
        
        ecart_type = self.afficher_proprietes(f"Poisson P(λ={lam})", esperance, variance)
        
        # Calculs avancés
        if d is None:
            d = esperance
            
        esperance_tronq = self.esperance_tronquee('poisson', d, lam=lam)
        exces_moy = self.exces_moyen('poisson', d, lam=lam)
        esperance_lim = self.esperance_limite('poisson', d, lam=lam)
        
        print(f"  Espérance tronquée E[X∧{d}] = {esperance_tronq:.6f}")
        print(f"  Excès moyen E[(X-{d})+] = {exces_moy:.6f}")
        print(f"  Espérance limite E[X|X>{d}] = {esperance_lim:.6f}")
        
        return esperance, variance, pmf
    
    def binomiale(self, n=10, p=0.5, d=None, afficher=True, **style):
        """Loi binomiale B(n, p)"""
        k = np.arange(0, n + 1)
        pmf = binom.pmf(k, n, p)
        esperance = n * p
        variance = n * p * (1 - p)
        
        if afficher:
            self._visualiser_discrete(k, pmf, f"Binomiale B({n},{p})", **style)
        
        ecart_type = self.afficher_proprietes(f"Binomiale B({n},{p})", esperance, variance)
        
        # Calculs avancés
        if d is None:
            d = esperance
            
        esperance_tronq = self.esperance_tronquee('binomiale', d, n=n, p=p)
        exces_moy = self.exces_moyen('binomiale', d, n=n, p=p)
        esperance_lim = self.esperance_limite('binomiale', d, n=n, p=p)
        
        print(f"  Espérance tronquée E[X∧{d}] = {esperance_tronq:.6f}")
        print(f"  Excès moyen E[(X-{d})+] = {exces_moy:.6f}")
        print(f"  Espérance limite E[X|X>{d}] = {esperance_lim:.6f}")
        
        return esperance, variance, pmf

    # Méthodes pour les autres lois (structure similaire)
    def beta(self, a=2, b=2, d=None, x=None, afficher=True, **style):
        """Loi bêta B(a, b)"""
        if x is None:
            x = np.linspace(0, 1, 1000)
        
        pdf = beta.pdf(x, a, b)
        esperance = a / (a + b)
        variance = (a * b) / ((a + b)**2 * (a + b + 1))
        
        if afficher:
            self._visualiser(x, pdf, f"Bêta B({a},{b})", **style)
        
        ecart_type = self.afficher_proprietes(f"Bêta B({a},{b})", esperance, variance)
        
        if d is None:
            d = esperance
            
        esperance_tronq = self.esperance_tronquee('beta', d, a=a, b=b)
        exces_moy = self.exces_moyen('beta', d, a=a, b=b)
        esperance_lim = self.esperance_limite('beta', d, a=a, b=b)
        
        print(f"  Espérance tronquée E[X∧{d}] = {esperance_tronq:.6f}")
        print(f"  Excès moyen E[(X-{d})+] = {exces_moy:.6f}")
        print(f"  Espérance limite E[X|X>{d}] = {esperance_lim:.6f}")
        
        return esperance, variance, pdf

    def lognormale(self, mu=0, sigma=1, d=None, x=None, afficher=True, **style):
        """Loi log-normale LN(μ, σ²)"""
        if x is None:
            x = np.linspace(0, 5, 1000)
        
        pdf = lognorm.pdf(x, sigma, scale=np.exp(mu))
        esperance = np.exp(mu + sigma**2/2)
        variance = (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)
        
        if afficher:
            self._visualiser(x, pdf, f"Log-Normale LN({mu},{sigma**2})", **style)
        
        ecart_type = self.afficher_proprietes(f"Log-Normale LN({mu},{sigma**2})", esperance, variance)
        
        if d is None:
            d = esperance
            
        esperance_tronq = self.esperance_tronquee('lognormale', d, mu=mu, sigma=sigma)
        exces_moy = self.exces_moyen('lognormale', d, mu=mu, sigma=sigma)
        esperance_lim = self.esperance_limite('lognormale', d, mu=mu, sigma=sigma)
        
        print(f"  Espérance tronquée E[X∧{d}] = {esperance_tronq:.6f}")
        print(f"  Excès moyen E[(X-{d})+] = {exces_moy:.6f}")
        print(f"  Espérance limite E[X|X>{d}] = {esperance_lim:.6f}")
        
        return esperance, variance, pdf

    def pareto(self, xm=1, alpha=2, d=None, x=None, afficher=True, **style):
        """Loi de Pareto Pareto(xm, α)"""
        if x is None:
            x = np.linspace(xm, xm+10, 1000)
        
        pdf = pareto.pdf(x, alpha, scale=xm)
        esperance = xm * alpha / (alpha - 1) if alpha > 1 else np.inf
        variance = xm**2 * alpha / ((alpha - 1)**2 * (alpha - 2)) if alpha > 2 else np.inf
        
        if afficher:
            self._visualiser(x, pdf, f"Pareto Pareto({xm},{alpha})", **style)
        
        ecart_type = self.afficher_proprietes(f"Pareto Pareto({xm},{alpha})", esperance, variance)
        
        if d is None:
            d = xm * 2
            
        esperance_tronq = self.esperance_tronquee('pareto', d, xm=xm, alpha=alpha)
        exces_moy = self.exces_moyen('pareto', d, xm=xm, alpha=alpha)
        esperance_lim = self.esperance_limite('pareto', d, xm=xm, alpha=alpha)
        
        print(f"  Espérance tronquée E[X∧{d}] = {esperance_tronq:.6f}")
        print(f"  Excès moyen E[(X-{d})+] = {exces_moy:.6f}")
        print(f"  Espérance limite E[X|X>{d}] = {esperance_lim:.6f}")
        
        return esperance, variance, pdf

    # Méthodes pour les autres lois (structure identique)
    def erlang(self, k=2, beta=1, d=None, x=None, afficher=True, **style):
        return self.gamma(k, beta, d, x, afficher, **style)
    
    def invgauss(self, mu=1, lam=1, d=None, x=None, afficher=True, **style):
        """Loi inverse-gaussienne IG(μ, λ)"""
        if x is None:
            x = np.linspace(0.01, 5, 1000)
        
        pdf = invgauss.pdf(x, mu/lam, scale=lam)
        esperance = mu
        variance = mu**3 / lam
        
        if afficher:
            self._visualiser(x, pdf, f"Inverse-Gaussienne IG({mu},{lam})", **style)
        
        ecart_type = self.afficher_proprietes(f"Inverse-Gaussienne IG({mu},{lam})", esperance, variance)
        
        if d is None:
            d = esperance
            
        esperance_tronq = self.esperance_tronquee('invgauss', d, mu=mu, lam=lam)
        exces_moy = self.exces_moyen('invgauss', d, mu=mu, lam=lam)
        esperance_lim = self.esperance_limite('invgauss', d, mu=mu, lam=lam)
        
        print(f"  Espérance tronquée E[X∧{d}] = {esperance_tronq:.6f}")
        print(f"  Excès moyen E[(X-{d})+] = {exces_moy:.6f}")
        print(f"  Espérance limite E[X|X>{d}] = {esperance_lim:.6f}")
        
        return esperance, variance, pdf

    def loglogistique(self, alpha=1, beta=1, d=None, x=None, afficher=True, **style):
        """Loi log-logistique LL(α, β)"""
        if x is None:
            x = np.linspace(0.01, 5, 1000)
        
        c = beta
        scale = alpha
        pdf = (c/scale) * (x/scale)**(c-1) / (1 + (x/scale)**c)**2
        
        esperance = (scale * np.pi / c) / np.sin(np.pi / c) if c > 1 else np.inf
        variance = scale**2 * (2*np.pi/c / np.sin(2*np.pi/c) - (np.pi/c / np.sin(np.pi/c))**2) if c > 2 else np.inf
        
        if afficher:
            self._visualiser(x, pdf, f"Log-Logistique LL({alpha},{beta})", **style)
        
        ecart_type = self.afficher_proprietes(f"Log-Logistique LL({alpha},{beta})", esperance, variance)
        
        if d is None:
            d = scale
            
        esperance_tronq = self.esperance_tronquee('loglogistique', d, alpha=alpha, beta=beta)
        exces_moy = self.exces_moyen('loglogistique', d, alpha=alpha, beta=beta)
        esperance_lim = self.esperance_limite('loglogistique', d, alpha=alpha, beta=beta)
        
        print(f"  Espérance tronquée E[X∧{d}] = {esperance_tronq:.6f}")
        print(f"  Excès moyen E[(X-{d})+] = {exces_moy:.6f}")
        print(f"  Espérance limite E[X|X>{d}] = {esperance_lim:.6f}")
        
        return esperance, variance, pdf

    def student(self, df=10, d=None, x=None, afficher=True, **style):
        """Loi de Student t(df)"""
        if x is None:
            x = np.linspace(-4, 4, 1000)
        
        pdf = t.pdf(x, df)
        esperance = 0 if df > 1 else np.inf
        variance = df / (df - 2) if df > 2 else np.inf
        
        if afficher:
            self._visualiser(x, pdf, f"Student t({df})", **style)
        
        ecart_type = self.afficher_proprietes(f"Student t({df})", esperance, variance)
        
        if d is None:
            d = 0
            
        esperance_tronq = self.esperance_tronquee('student', d, df=df)
        exces_moy = self.exces_moyen('student', d, df=df)
        esperance_lim = self.esperance_limite('student', d, df=df)
        
        print(f"  Espérance tronquée E[X∧{d}] = {esperance_tronq:.6f}")
        print(f"  Excès moyen E[(X-{d})+] = {exces_moy:.6f}")
        print(f"  Espérance limite E[X|X>{d}] = {esperance_lim:.6f}")
        
        return esperance, variance, pdf

    def bernoulli(self, p=0.5, d=None, afficher=True, **style):
        """Loi de Bernoulli Bern(p)"""
        k = np.arange(0, 2)
        pmf = [1-p, p]
        esperance = p
        variance = p * (1 - p)
        
        if afficher:
            self._visualiser_discrete(k, pmf, f"Bernoulli Bern({p})", **style)
        
        ecart_type = self.afficher_proprietes(f"Bernoulli Bern({p})", esperance, variance)
        
        if d is None:
            d = 0.5
            
        esperance_tronq = self.esperance_tronquee('bernoulli', d, p=p)
        exces_moy = self.exces_moyen('bernoulli', d, p=p)
        esperance_lim = self.esperance_limite('bernoulli', d, p=p)
        
        print(f"  Espérance tronquée E[X∧{d}] = {esperance_tronq:.6f}")
        print(f"  Excès moyen E[(X-{d})+] = {exces_moy:.6f}")
        print(f"  Espérance limite E[X|X>{d}] = {esperance_lim:.6f}")
        
        return esperance, variance, pmf

    def geometrique(self, p=0.5, d=None, k_max=20, afficher=True, **style):
        """Loi géométrique G(p)"""
        k = np.arange(1, k_max + 1)
        pmf = geom.pmf(k, p)
        esperance = 1 / p
        variance = (1 - p) / (p**2)
        
        if afficher:
            self._visualiser_discrete(k, pmf, f"Géométrique G({p})", **style)
        
        ecart_type = self.afficher_proprietes(f"Géométrique G({p})", esperance, variance)
        
        if d is None:
            d = esperance
            
        esperance_tronq = self.esperance_tronquee('geometrique', d, p=p)
        exces_moy = self.exces_moyen('geometrique', d, p=p)
        esperance_lim = self.esperance_limite('geometrique', d, p=p)
        
        print(f"  Espérance tronquée E[X∧{d}] = {esperance_tronq:.6f}")
        print(f"  Excès moyen E[(X-{d})+] = {exces_moy:.6f}")
        print(f"  Espérance limite E[X|X>{d}] = {esperance_lim:.6f}")
        
        return esperance, variance, pmf

    def logarithmique(self, p=0.5, d=None, k_max=20, afficher=True, **style):
        """Loi logarithmique Log(p)"""
        k = np.arange(1, k_max + 1)
        pmf = -p**k / (k * np.log(1 - p))
        esperance = -p / ((1 - p) * np.log(1 - p))
        variance = -p * (p + np.log(1 - p)) / ((1 - p)**2 * np.log(1 - p)**2)
        
        if afficher:
            self._visualiser_discrete(k, pmf, f"Logarithmique Log({p})", **style)
        
        ecart_type = self.afficher_proprietes(f"Logarithmique Log({p})", esperance, variance)
        
        if d is None:
            d = 1
            
        esperance_tronq = self.esperance_tronquee('logarithmique', d, p=p)
        exces_moy = self.exces_moyen('logarithmique', d, p=p)
        esperance_lim = self.esperance_limite('logarithmique', d, p=p)
        
        print(f"  Espérance tronquée E[X∧{d}] = {esperance_tronq:.6f}")
        print(f"  Excès moyen E[(X-{d})+] = {exces_moy:.6f}")
        print(f"  Espérance limite E[X|X>{d}] = {esperance_lim:.6f}")
        
        return esperance, variance, pmf

    def _visualiser(self, x, y, titre, **style):
        """Visualisation personnalisable pour les lois continues"""
        fig, ax = plt.subplots(figsize=style.get('figsize', (10, 6)))
        
        color = style.get('color', 'blue')
        linewidth = style.get('linewidth', 2)
        alpha = style.get('alpha', 0.7)
        grid = style.get('grid', True)
        fill = style.get('fill', True)
        
        ax.plot(x, y, color=color, linewidth=linewidth, label=titre)
        if fill:
            ax.fill_between(x, y, alpha=alpha*0.5, color=color)
        
        ax.set_title(f"Loi {titre}", fontsize=14)
        ax.set_xlabel(style.get('xlabel', 'x'), fontsize=12)
        ax.set_ylabel(style.get('ylabel', 'Densité de probabilité f(x)'), fontsize=12)
        
        if grid:
            ax.grid(alpha=0.3)
        
        if style.get('legend', True):
            ax.legend()
        
        plt.tight_layout()
        plt.show()
    
    def _visualiser_discrete(self, k, pmf, titre, **style):
        """Visualisation personnalisable pour les lois discrètes"""
        fig, ax = plt.subplots(figsize=style.get('figsize', (10, 6)))
        
        color = style.get('color', 'skyblue')
        edgecolor = style.get('edgecolor', 'navy')
        alpha = style.get('alpha', 0.7)
        grid = style.get('grid', True)
        
        ax.bar(k, pmf, color=color, edgecolor=edgecolor, alpha=alpha, label=titre)
        
        ax.set_title(f"Loi {titre}", fontsize=14)
        ax.set_xlabel(style.get('xlabel', 'k'), fontsize=12)
        ax.set_ylabel(style.get('ylabel', 'Probabilité P(X=k)'), fontsize=12)
        
        if grid:
            ax.grid(alpha=0.3, axis='y')
        
        if style.get('legend', True):
            ax.legend()
        
        plt.tight_layout()
        plt.show()