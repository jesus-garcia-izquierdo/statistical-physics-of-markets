"""
04_ising.py — Imitación, burbujas y crashes: el modelo de Ising
================================================================

Idea física: N inversores, cada uno compra (s = +1) o vende (s = -1).
Como los espines de un imán, cada uno tiende a hacer lo que hacen los demás
(acoplamiento J: imitación) y también responde a las noticias comunes
(campo h). La temperatura T mide cuánto decide cada uno por su cuenta.

        "energía" de un inversor:   E_i = -(J m + h) s_i,     m = (1/N) sum_j s_j

    m  (magnetización) = fracción que compra - fracción que vende = exceso de demanda
    Precio: cada día sube en proporción al exceso de demanda, ln P_{t+1} = ln P_t + lambda m_t

Usamos la aproximación de campo medio (todos ven a todos, modelo de
Curie-Weiss): la temperatura crítica es T_c = J.

ESQUEMA DE TIPOS DE DATOS
-------------------------
    s           : np.ndarray[int8], (N,)         decisión de cada inversor (+1/-1): 1 byte basta
    m           : np.ndarray[float64], (dias,)   magnetización al final de cada día
    red         : np.ndarray[int8], (L, L)       espines en una red cuadrada 2D (sólo para la figura)
    precios     : dict[float, np.ndarray]        temperatura -> serie de precios simulada
    resultados  : dict[str, float]

ESTRUCTURA DEL PROGRAMA
-----------------------
    1. Instantáneas de una red 2D a tres temperaturas        -> figura 4.1
    2. Campo medio: m = tanh((J m + h)/T) y energía libre     -> figura 4.2
    3. Tres mercados simulados (T alta, crítica, baja)        -> figura 4.3
    4. Susceptibilidad: respuesta a las noticias              -> figura 4.4 (izq.)
    5. Histéresis: el crash como salto entre dos mínimos      -> figura 4.4 (dcha.)
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import comun

rng = np.random.default_rng(4)
resultados = {}
J = 1.0                                      # imitación; con J = 1, T_c = 1 en campo medio


# ---------------------------------------------------------------------------
# Dinámica: cada inversor revisa su decisión mirando el "campo" que siente
# ---------------------------------------------------------------------------
def un_dia(s, M, T, h, rng):
    """N revisiones al azar (un "día"). Regla de baño térmico (heat bath):
    el inversor i compra con probabilidad  1 / (1 + exp(-2 campo / T)),
    campo = J * (opinión media de LOS DEMÁS) + h.
    Devuelve la nueva suma M = sum_j s_j (int: así no acumulamos error)."""
    N = len(s)
    for i, u in zip(rng.integers(0, N, N), rng.random(N)):
        si = int(s[i])                       # int de Python: evita desbordar el int8 al sumar
        campo = J * (M - si) / N + h
        nuevo = 1 if u < 1.0 / (1.0 + np.exp(-2.0 * campo / T)) else -1
        M += nuevo - si
        s[i] = nuevo
    return M

def simular_mercado(T, dias, N=400, h_ruido=0.02, h_fijo=None, rng=rng):
    """Devuelve m (magnetización diaria). Las noticias h son pequeñas y
    aleatorias cada día (h_ruido), salvo que se dé una lista h_fijo."""
    s = rng.choice(np.array([-1, 1], dtype=np.int8), N)
    M = int(s.sum())
    m = np.empty(dias)
    for d in range(dias):
        h = h_fijo[d] if h_fijo is not None else h_ruido * rng.standard_normal()
        M = un_dia(s, M, T, h, rng)
        m[d] = M / N
    return m


# ---------------------------------------------------------------------------
# 1. Red 2D a tres temperaturas (sólo para ver cómo es cada fase)
# ---------------------------------------------------------------------------
def red_2d(T, L=80, barridos=300, rng=rng):
    """Ising 2D con vecinos próximos (T_c = 2.269 J), partiendo de todos a +1. Actualizamos media red a
    la vez con un patrón de tablero de ajedrez (los vecinos de un blanco son negros)."""
    red = np.ones((L, L), dtype=np.int8)       # partimos de consenso total (todos compran)
    tablero = (np.add.outer(np.arange(L), np.arange(L)) % 2).astype(bool)
    for _ in range(barridos):
        for color in (True, False):
            vecinos = (np.roll(red, 1, 0) + np.roll(red, -1, 0) + np.roll(red, 1, 1) + np.roll(red, -1, 1))
            p = 1 / (1 + np.exp(-2 * J * vecinos / T))
            nuevo = np.where(rng.random((L, L)) < p, 1, -1).astype(np.int8)
            red[tablero == color] = nuevo[tablero == color]
    return red

fig, ax = plt.subplots(1, 3, figsize=(6.2, 2.3))
mapa = ListedColormap([comun.NARANJA, comun.AZUL])
for a, (T, titulo) in zip(ax, ((1.8, comun.t("$T<T_c$: consenso", "$T<T_c$: consensus")), (2.3, comun.t("$T\\approx T_c$: grupos de\ntodos los tamaños", "$T\\approx T_c$: clusters\nof all sizes")),
                               (3.5, comun.t("$T>T_c$: cada uno a lo suyo", "$T>T_c$: everyone\nfor themselves")))):
    red = red_2d(T)
    a.imshow(red, cmap=mapa, vmin=-1, vmax=1, interpolation="nearest")
    a.set_title(f"{titulo}\n$m$ = {red.mean():+.2f}", fontsize=8)
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
fig.text(0.5, 0.0, comun.t("azul: compra (+1)     naranja: vende (−1)", "blue: buy (+1)     orange: sell (−1)"), ha="center", fontsize=8, color=comun.GRIS)
comun.guardar(fig, "fig4_espines")

# ---------------------------------------------------------------------------
# 2. Teoría de campo medio
# ---------------------------------------------------------------------------
def energia_libre(m, T, h=0.0):
    """f(m) = -J m^2/2 - h m - T * entropía(m).  La entropía cuenta de cuántas
    maneras se puede tener una fracción (1+m)/2 de compradores."""
    p, q = (1 + m) / 2, (1 - m) / 2
    entropia = -(p * np.log(p) + q * np.log(q))
    return -0.5 * J * m**2 - h * m - T * entropia

def m_espontanea(T, h=0.0, m0=0.9):
    """Resuelve m = tanh((J m + h)/T) por iteración de punto fijo."""
    m = m0
    for _ in range(5000):
        m = np.tanh((J * m + h) / T)
    return m

mm = np.linspace(-0.999, 0.999, 400)
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
ax[0].plot(mm, mm, "k", lw=1, label="$m$")
for T, c in ((0.7, comun.AZUL), (1.0, comun.GRIS), (1.5, comun.NARANJA)):
    ax[0].plot(mm, np.tanh(J * mm / T), color=c, lw=1.2, label=f"tanh($m/T$), T = {T}")
ax[0].set_xlabel("$m$")
ax[0].set_title(comun.t("Cortes = estados de equilibrio", "Crossings = equilibrium states"))
ax[0].legend(frameon=False, fontsize=7)
for T, c in ((0.7, comun.AZUL), (1.0, comun.GRIS), (1.5, comun.NARANJA)):
    f = energia_libre(mm, T)
    ax[1].plot(mm, f - f.min(), color=c, lw=1.2, label=f"T = {T}")
ax[1].set_ylim(0, 0.25)
ax[1].set_xlabel(comun.t("$m$ (exceso de demanda)", "$m$ (excess demand)"))
ax[1].set_ylabel("$f(m) - f_{\\min}$")
ax[1].set_title(comun.t("Energía libre", "Free energy"))
ax[1].legend(frameon=False, fontsize=7)
comun.guardar(fig, "fig4_campo_medio")
for T in (0.7, 0.9, 0.95):
    resultados[f"m_espontanea_T={T}"] = m_espontanea(T)

# ---------------------------------------------------------------------------
# 3. Tres mercados: el precio sube cada día en proporción al exceso de demanda
# ---------------------------------------------------------------------------
lam = 0.005                                  # sensibilidad del precio: m = 1 -> +0.5 % diario
dias = 1500
temperaturas = (1.5, 1.0, 0.85)
fig, ax = plt.subplots(2, 3, figsize=(6.6, 3.6), sharex=True)
for k, T in enumerate(temperaturas):
    m = simular_mercado(T, dias, N=150)      # 150 inversores: con pocos agentes los vuelcos son visibles
    precio = 100 * np.exp(lam * np.cumsum(m))
    cambios_signo = int(np.sum(np.diff(np.sign(m[np.abs(m) > 0.3])) != 0))    # cruces "de verdad"
    resultados[f"T={T}_desv_m"] = m.std()
    resultados[f"T={T}_autocorr_m"] = np.corrcoef(m[:-1], m[1:])[0, 1]
    resultados[f"T={T}_vuelcos"] = cambios_signo
    ax[0, k].plot(m, lw=0.6, color=comun.AZUL)
    ax[0, k].set_ylim(-1, 1)
    ax[0, k].set_title(f"T = {T}" + (comun.t("  (crítica)", "  (critical)") if T == 1.0 else ""), fontsize=8)
    ax[1, k].plot(precio, lw=0.8, color="k")
    ax[1, k].set_xlabel(comun.t("día", "day"))
ax[0, 0].set_ylabel(comun.t("$m$ (exceso de\ndemanda)", "$m$ (excess\ndemand)"))
ax[1, 0].set_ylabel(comun.t("precio", "price"))
comun.guardar(fig, "fig4_mercados")

# ---------------------------------------------------------------------------
# 4. Susceptibilidad: cuánto responde el mercado a una noticia
# ---------------------------------------------------------------------------
# Teorema de fluctuación-disipación: chi = N Var(m) / T (para T > T_c).
# Cuanto más grandes son las fluctuaciones espontáneas, más responde a un empujón.
T_chi = np.array([1.05, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5])
chi = []
N_chi = 400
for T in T_chi:
    m = simular_mercado(T, 1500, N=N_chi, h_ruido=0.0)
    chi.append(N_chi * m[200:].var() / T)            # descartamos 200 días de "termalización"
chi = np.array(chi)
resultados["chi(T=2.0)"] = chi[T_chi == 2.0][0]
resultados["chi(T=1.1)"] = chi[T_chi == 1.1][0]

# ---------------------------------------------------------------------------
# 5. Histéresis: noticias que mejoran poco a poco y luego empeoran
# ---------------------------------------------------------------------------
T_h = 0.8
h_subida = np.linspace(-0.3, 0.3, 900)        # cambio lento: 900 días en cada sentido
h_rampa = np.concatenate([h_subida, h_subida[::-1]])     # sube y luego baja
m_rampa = simular_mercado(T_h, len(h_rampa), N=400, h_fijo=h_rampa)
# Teoría: rama superior y rama inferior de m = tanh((m + h)/T)
h_teo = np.linspace(-0.3, 0.3, 200)
rama_sup = np.array([m_espontanea(T_h, h, +0.99) for h in h_teo])
rama_inf = np.array([m_espontanea(T_h, h, -0.99) for h in h_teo])
salto = h_subida[np.argmax(m_rampa[:900] > 0)]
resultados["h_del_salto_subida"] = salto
# Teoría: la rama de "vendedores" deja de existir cuando 1 - m^2 = T (pendiente 1 en m = tanh(...))
m_espinodal = -np.sqrt(1 - T_h)
resultados["h_salto_teoria"] = T_h * np.arctanh(m_espinodal) - J * m_espinodal

fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6))
ax[0].plot(T_chi, chi, "o", color=comun.AZUL, ms=4, label=comun.t("simulación", "simulation"))
TT = np.linspace(1.03, 2.6, 200)
ax[0].plot(TT, 1 / (TT - J), color=comun.NARANJA, lw=1.2, label=comun.t("campo medio: $1/(T-T_c)$", "mean field: $1/(T-T_c)$"))
ax[0].set_xlabel(comun.t("temperatura $T$", "temperature $T$"))
ax[0].set_ylabel(comun.t("susceptibilidad $\\chi$", "susceptibility $\\chi$"))
ax[0].set_title(comun.t("Sensibilidad a las noticias", "Sensitivity to news"))
ax[0].legend(frameon=False, fontsize=7)
ax[1].plot(h_rampa[:900], m_rampa[:900], color=comun.AZUL, lw=1, label=comun.t("noticias mejorando", "news improving"))
ax[1].plot(h_rampa[900:], m_rampa[900:], color=comun.NARANJA, lw=1, label=comun.t("noticias empeorando", "news getting worse"))
ax[1].plot(h_teo, rama_sup, "k:", lw=0.8)
ax[1].plot(h_teo, rama_inf, "k:", lw=0.8)
ax[1].set_xlabel(comun.t("noticias $h$", "news $h$"))
ax[1].set_ylabel("$m$")
ax[1].set_title(comun.t("Histéresis", "Hysteresis") + f" (T = {T_h})")
ax[1].legend(frameon=False, fontsize=7, loc="lower right")
comun.guardar(fig, "fig4_respuesta")

print("\nRESULTADOS (Ising)")
for k, v in resultados.items():
    print(f"  {k:32s} {v:10.4g}")
