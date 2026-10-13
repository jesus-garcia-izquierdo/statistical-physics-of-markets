"""
03_muelle.py — Precios que vuelven: Langevin y Ornstein-Uhlenbeck
=================================================================

Idea física: una partícula browniana atada a un muelle (pozo armónico) en un
fluido viscoso. La ecuación de Langevin sobreamortiguada es

        dx = -theta (x - mu) dt + sigma dW        (proceso de Ornstein-Uhlenbeck)

    theta : rigidez del muelle / fricción  -> rapidez con que vuelve al equilibrio
    mu    : posición de equilibrio
    sigma : intensidad del ruido térmico   -> "temperatura"

En economía: la diferencia de precio entre dos activos muy parecidos (el
"spread") no puede alejarse indefinidamente; algo tira de ella hacia su valor
normal. El precio de UNA acción, en cambio, no tiene muelle (capítulo 1).

ESQUEMA DE TIPOS DE DATOS
-------------------------
    x_sim        : np.ndarray[float64], (3, 150)  trayectorias simuladas desde 3 posiciones
    serie_larga  : np.ndarray[float64], (3000,)   una trayectoria larga para estimar parámetros
    ajuste       : dict[str, float]               {"b", "theta", "vida_media", "mu", "sigma"}
    sp, nq       : np.ndarray[float64], (391,)    S&P 500 y Nasdaq mensuales (1990-2022)
    spread       : np.ndarray[float64], (391,)    ln(Nasdaq) - beta ln(S&P) - c
    b_nulos      : np.ndarray[float64], (1000,)   valores de b obtenidos SIN muelle (azar)
    resultados   : dict[str, float]

ESTRUCTURA DEL PROGRAMA
-----------------------
    1. Simular el OU exacto y ver la relajación            -> figura 3.1
    2. Estimar theta con una regresión lineal y comprobar
       que la distribución estacionaria es la de Boltzmann -> figura 3.2
    3. Datos reales: spread Nasdaq - S&P 500, con su
       hipótesis nula (dos paseos sin muelle)              -> figura 3.3
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import comun

rng = np.random.default_rng(3)
resultados = {}

# ---------------------------------------------------------------------------
# 1. Simulación exacta del OU
# ---------------------------------------------------------------------------
# Entre t y t+dt la solución exacta es un AR(1):
#     x_{t+1} = mu + b (x_t - mu) + ruido,   b = exp(-theta dt),
#     Var(ruido) = sigma^2 (1 - b^2) / (2 theta)
# (no hace falta Euler: no hay error de discretización).
def simular_ou(theta, mu, sigma, x0, n, rng, dt=1.0):
    b = np.exp(-theta * dt)
    sd_ruido = sigma * np.sqrt((1 - b**2) / (2 * theta))
    x = np.empty(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = mu + b * (x[t - 1] - mu) + sd_ruido * rng.standard_normal()
    return x

theta, mu, sigma = 0.05, 0.0, 0.01            # por día; sigma = 1 % diario de ruido
sigma_eq = sigma / np.sqrt(2 * theta)         # anchura de equilibrio (Boltzmann)
resultados["vida_media_teorica_dias"] = np.log(2) / theta
resultados["sigma_equilibrio_%"] = 100 * sigma_eq

pasos = 150
x_sim = np.array([simular_ou(theta, mu, sigma, x0, pasos, rng) for x0 in (0.10, 0.0, -0.07)])
t = np.arange(pasos)

fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6), gridspec_kw={"width_ratios": [1, 1.6]})
xx = np.linspace(-0.12, 0.12, 200)
ax[0].plot(100 * xx, 0.5 * theta * xx**2 * 1e4, color="k", lw=1.2)          # U(x) = theta x^2 / 2 (escala arbitraria)
for x0, c in zip((0.10, 0.0, -0.07), (comun.AZUL, comun.GRIS, comun.NARANJA)):
    ax[0].plot(100 * x0, 0.5 * theta * x0**2 * 1e4, "o", color=c, ms=7)
ax[0].set_xlabel(comun.t("desvío $x-\\mu$ (%)", "deviation $x-\\mu$ (%)"))
ax[0].set_ylabel(comun.t("energía potencial $U(x)$", "potential energy $U(x)$"))
ax[0].set_yticks([])
ax[0].set_title(comun.t("El pozo", "The well"))
for x, c in zip(x_sim, (comun.AZUL, comun.GRIS, comun.NARANJA)):
    ax[1].plot(t, 100 * x, color=c, lw=0.9)
ax[1].plot(t, 100 * 0.10 * np.exp(-theta * t), "k--", lw=0.8, label=comun.t("media: $x_0e^{-\\theta t}$", "mean: $x_0e^{-\\theta t}$"))
ax[1].plot(t, 100 * -0.07 * np.exp(-theta * t), "k--", lw=0.8)
ax[1].axvline(np.log(2) / theta, color=comun.GRIS, lw=0.6, ls=":")
ax[1].annotate(comun.t("vida media\n$\\ln 2/\\theta$ = 13.9 días", "half-life\n$\\ln 2/\\theta$ = 13.9 days"), xy=(np.log(2) / theta, -11), xytext=(40, -11.5),
               fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.6))
ax[1].set_xlabel(comun.t("días", "days"))
ax[1].set_ylabel(comun.t("desvío (%)", "deviation (%)"))
ax[1].set_title(comun.t("La relajación", "Relaxation"))
ax[1].legend(frameon=False, loc="upper right")
comun.guardar(fig, "fig3_pozo")

# ---------------------------------------------------------------------------
# 2. Estimar el muelle a partir de los datos
# ---------------------------------------------------------------------------
def ajustar_ou(x, dt=1.0):
    """Regresión x_{t+1} = a + b x_t. Para el OU, la máxima verosimilitud
    coincide con esta recta de mínimos cuadrados.  b -> theta = -ln(b)/dt."""
    b, a = np.polyfit(x[:-1], x[1:], 1)
    theta_est = -np.log(b) / dt if 0 < b < 1 else 0.0       # b >= 1: no hay muelle
    vida = np.log(2) / theta_est if theta_est > 0 else np.inf
    return {"b": b, "theta": theta_est, "vida_media": vida, "mu": a / (1 - b) if b != 1 else np.nan}

serie_larga = simular_ou(theta, mu, sigma, 0.0, 3000, rng)
ajuste = ajustar_ou(serie_larga)
resultados["b_estimado_sim"] = ajuste["b"]
resultados["theta_estimado_sim"] = ajuste["theta"]
resultados["vida_media_estimada_sim"] = ajuste["vida_media"]
resultados["sigma_eq_medida_%"] = 100 * serie_larga.std()

fig, ax = plt.subplots(figsize=(4.4, 2.6))
ax.hist(100 * serie_larga, bins=50, density=True, color=comun.AZUL, alpha=0.6, label=comun.t("simulación (3000 días)", "simulation (3000 days)"))
xx = np.linspace(-0.13, 0.13, 300)
# Boltzmann: P(x) ∝ exp(-U/kT) con U = theta x^2/2 y kT = sigma^2/2  ->  gaussiana de varianza sigma^2/(2 theta)
ax.plot(100 * xx, stats.norm.pdf(xx, mu, sigma_eq) / 100, color=comun.NARANJA, lw=1.4,
        label="Boltzmann $\\propto e^{-U/k_BT}$")
ax.set_xlabel(comun.t("desvío $x$ (%)", "deviation $x$ (%)"))
ax.set_ylabel(comun.t("densidad", "density"))
ax.legend(frameon=False, fontsize=7)
comun.guardar(fig, "fig3_boltzmann")

# ---------------------------------------------------------------------------
# 3. Datos reales: ¿hay un muelle entre el Nasdaq y el S&P 500?
# ---------------------------------------------------------------------------
fechas, sp, nq = comun.cargar_indices_mensuales()
ln_sp, ln_nq = np.log(sp), np.log(nq)

def spread_y_b(ln_a, ln_b):
    """Ajusta ln_b = beta ln_a + c (cuánto se mueve uno cuando se mueve el otro)
    y devuelve el spread (lo que sobra) y su b."""
    beta, c = np.polyfit(ln_a, ln_b, 1)
    s = ln_b - (beta * ln_a + c)
    return s, ajustar_ou(s)["b"], beta

spread, b_spread, beta = spread_y_b(ln_sp, ln_nq)
aj_spread = ajustar_ou(spread)
aj_precio = ajustar_ou(ln_sp)                 # el S&P 500 solo, sin pareja
resultados["beta_nasdaq_sp"] = beta
resultados["b_spread"] = b_spread
resultados["vida_media_spread_meses"] = aj_spread["vida_media"]
resultados["b_precio_sp"] = aj_precio["b"]
resultados["vida_media_precio_sp_meses"] = aj_precio["vida_media"]

# Hipótesis nula: dos paseos aleatorios INDEPENDIENTES (sin ningún muelle) con
# los mismos pasos mensuales que los índices reales, barajados. Ajustamos beta
# y b exactamente igual que con los datos. ¿Cuántas veces sale un b tan pequeño?
inc_sp, inc_nq = np.diff(ln_sp), np.diff(ln_nq)
b_nulos = np.empty(1000)
for k in range(len(b_nulos)):
    a = np.concatenate([[0], np.cumsum(rng.permutation(inc_sp))])
    bb = np.concatenate([[0], np.cumsum(rng.permutation(inc_nq))])
    b_nulos[k] = spread_y_b(a, bb)[1]
resultados["p_valor_muelle"] = (1 + np.sum(b_nulos <= b_spread)) / (len(b_nulos) + 1)
resultados["b_nulo_mediana"] = np.median(b_nulos)

fig, ax = plt.subplots(2, 1, figsize=(6.2, 3.8), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
ax[0].plot(fechas, sp / sp[0], color=comun.AZUL, lw=1, label="S&P 500")
ax[0].plot(fechas, nq / nq[0], color=comun.NARANJA, lw=1, label="Nasdaq")
ax[0].set_yscale("log")
ax[0].set_ylabel(comun.t("valor (1990 = 1)", "value (1990 = 1)"))
ax[0].legend(frameon=False)
ax[1].plot(fechas, spread, color="k", lw=1)
ax[1].axhline(0, color=comun.GRIS, lw=0.6)
ax[1].axhspan(-spread.std(), spread.std(), color=comun.GRIS, alpha=0.15, lw=0)
ax[1].set_ylabel("spread (log)")
comun.guardar(fig, "fig3_spread")

print("\nRESULTADOS (muelle)")
for k, v in resultados.items():
    print(f"  {k:32s} {v:10.4g}")
