"""
02_temperatura.py — La volatilidad como temperatura
===================================================

Idea física: el retorno de un día es como la velocidad de una partícula en un
baño térmico. Su dispersión (la volatilidad sigma) hace el papel de la
temperatura: sigma^2 <-> k_B T. Si la temperatura del baño CAMBIA con el
tiempo, la estadística total es una MEZCLA de gaussianas de distinta anchura,
y una mezcla así tiene colas mucho más gruesas que una sola gaussiana.

Comprobamos con Google:
  (a) Hay días "fríos" y "calientes" que vienen agrupados.
  (b) El signo del retorno no tiene memoria, pero su tamaño |r| sí: la
      temperatura es una variable lenta.
  (c) Un ejemplo de juguete con dos temperaturas ya produce colas gruesas.
  (d) Si medimos la temperatura de cada día (modelo GARCH) y dividimos por
      ella, los retornos se vuelven casi gaussianos.
  (e) La temperatura de mañana se puede predecir; la dirección, no.

ESQUEMA DE TIPOS DE DATOS
-------------------------
    r            : np.ndarray[float64], (1046,)   log-retornos diarios
    sigma_movil  : np.ndarray[float64], (1046,)   volatilidad de los 20 días previos (NaN al principio)
    parametros   : tuple[float, float, float]     (omega, alfa, beta) del modelo GARCH
    sigma_t      : np.ndarray[float64], (1046,)   "temperatura" de cada día según GARCH
    sigma_prev   : np.ndarray[float64], (523,)    temperatura prevista para cada día de la 2.a mitad
    grupos       : list[np.ndarray[int]]          índices de los días en cada quintil de temperatura
    resultados   : dict[str, float]               números citados en el documento

ESTRUCTURA DEL PROGRAMA
-----------------------
    1. Termómetro sencillo: volatilidad de los últimos 20 días      -> figura 2.1
    2. Memoria del signo y del tamaño                               -> figura 2.2
    3. Mezcla de dos temperaturas (ejemplo de juguete)              -> figura 2.3
    4. Termómetro con memoria: GARCH(1,1) por máxima verosimilitud  -> figura 2.4
    5. Predicción de la temperatura fuera de muestra (2.a mitad)    -> figura 2.5
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats

import comun

resultados = {}
fechas, precios = comun.cargar_google()
r = comun.log_retornos(precios)
fechas_r = fechas[1:]                       # cada retorno se asocia al día en que termina
n = len(r)
sigma = r.std()

# ---------------------------------------------------------------------------
# 1. Termómetro sencillo: desviación típica de los 20 días anteriores
# ---------------------------------------------------------------------------
ventana = 20                                # ~ un mes de mercado
sigma_movil = np.full(n, np.nan)            # NaN donde aún no hay 20 días de historia
for t in range(ventana, n):
    sigma_movil[t] = r[t - ventana:t].std()
anual = np.sqrt(252)
resultados["temp_min_anual_%"] = 100 * np.nanmin(sigma_movil) * anual
resultados["temp_max_anual_%"] = 100 * np.nanmax(sigma_movil) * anual

fig, ax = plt.subplots(2, 1, figsize=(6.2, 3.6), sharex=True)
ax[0].plot(fechas_r, 100 * r, lw=0.5, color=comun.AZUL)
ax[0].set_ylabel(comun.t("retorno diario (%)", "daily return (%)"))
ax[1].plot(fechas_r, 100 * sigma_movil * anual, lw=1.0, color=comun.NARANJA)
ax[1].set_ylabel(comun.t("volatilidad\nanualizada (%)", "annualized\nvolatility (%)"))
ax[1].set_ylim(0, None)
comun.guardar(fig, "fig2_termometro")

# ---------------------------------------------------------------------------
# 2. El signo no tiene memoria; el tamaño sí
# ---------------------------------------------------------------------------
def autocorrelacion(x, max_retardo):
    x = x - x.mean()
    return np.array([np.mean(x[:-t] * x[t:]) for t in range(1, max_retardo + 1)]) / x.var()

retardos = np.arange(1, 61)
acf_r = autocorrelacion(r, 60)
acf_abs = autocorrelacion(np.abs(r), 60)
banda = 1.96 / np.sqrt(n)
resultados["C_abs(1)"] = acf_abs[0]
resultados["C_abs_media_1a20"] = acf_abs[:20].mean()
resultados["C_r_media_1a20"] = acf_r[:20].mean()

fig, ax = plt.subplots(figsize=(5.0, 2.5))
ax.axhspan(-banda, banda, color=comun.GRIS, alpha=0.2, lw=0, label=comun.t("zona de puro ruido", "pure-noise band"))
ax.plot(retardos, acf_r, "o-", ms=2, lw=0.8, color=comun.AZUL, label=comun.t("signo y tamaño: $r_t$", "sign and size: $r_t$"))
ax.plot(retardos, acf_abs, "s-", ms=2, lw=0.8, color=comun.NARANJA, label=comun.t("sólo tamaño: $|r_t|$", "size only: $|r_t|$"))
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel(comun.t(r"retardo $\tau$ (días)", r"lag $\tau$ (days)"))
ax.set_ylabel(comun.t(r"autocorrelación $C(\tau)$", r"autocorrelation $C(\tau)$"))
ax.legend(frameon=False)
comun.guardar(fig, "fig2_memoria")

# ---------------------------------------------------------------------------
# 3. Ejemplo de juguete: dos temperaturas
# ---------------------------------------------------------------------------
# 80 % de los días son "fríos" (sigma = 1.5 %) y 20 % "calientes" (sigma = 4 %).
p_frio, s_frio, s_cal = 0.8, 0.015, 0.04
var_mezcla = p_frio * s_frio**2 + (1 - p_frio) * s_cal**2
momento4 = 3 * (p_frio * s_frio**4 + (1 - p_frio) * s_cal**4)     # <r^4> de cada gaussiana = 3 sigma^4
s_total = np.sqrt(var_mezcla)
resultados["mezcla_sigma_total_%"] = 100 * s_total
resultados["mezcla_curtosis_exceso"] = momento4 / var_mezcla**2 - 3
caida = -0.08                                                       # una caída del 8 % en un día
p_mezcla = p_frio * stats.norm.cdf(caida, 0, s_frio) + (1 - p_frio) * stats.norm.cdf(caida, 0, s_cal)
p_gauss = stats.norm.cdf(caida, 0, s_total)
resultados["P(caida8%)_mezcla"] = p_mezcla
resultados["P(caida8%)_gauss"] = p_gauss
resultados["cociente_probabilidades"] = p_mezcla / p_gauss

x = np.linspace(-0.15, 0.15, 600)
fig, ax = plt.subplots(figsize=(4.8, 3.1))
ax.plot(100 * x, p_frio * stats.norm.pdf(x, 0, s_frio) / 100, color=comun.AZUL, lw=0.8, ls="--",
        label=comun.t("días fríos (80 %)", "cold days (80 %)"))
ax.plot(100 * x, (1 - p_frio) * stats.norm.pdf(x, 0, s_cal) / 100, color=comun.NARANJA, lw=0.8, ls="--",
        label=comun.t("días calientes (20 %)", "hot days (20 %)"))
ax.plot(100 * x, (p_frio * stats.norm.pdf(x, 0, s_frio) + (1 - p_frio) * stats.norm.pdf(x, 0, s_cal)) / 100,
        color="k", lw=1.4, label=comun.t("mezcla (lo que se observa)", "mixture (what we observe)"))
ax.plot(100 * x, stats.norm.pdf(x, 0, s_total) / 100, color=comun.VERDE, lw=1.2,
        label=comun.t("una sola gaussiana, misma $\\sigma$", "single Gaussian, same $\\sigma$"))
ax.set_yscale("log")
ax.set_ylim(1e-7, 0.5)
ax.set_xlabel(comun.t("retorno diario (%)", "daily return (%)"))
ax.set_ylabel(comun.t("densidad", "density"))
ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
comun.guardar(fig, "fig2_mezcla")

# ---------------------------------------------------------------------------
# 4. Termómetro con memoria: GARCH(1,1)
# ---------------------------------------------------------------------------
#   sigma_t^2 = omega + alfa * r_{t-1}^2 + beta * sigma_{t-1}^2
# La temperatura de hoy = un poco del nivel de fondo (omega)
#                       + el "golpe" de ayer (alfa r^2)
#                       + la inercia de la temperatura de ayer (beta sigma^2).
def varianzas_garch(params, x):
    omega, alfa, beta = params
    v = np.empty(len(x))
    v[0] = x.var()                                  # arrancamos en la varianza media
    for t in range(1, len(x)):
        v[t] = omega + alfa * x[t - 1] ** 2 + beta * v[t - 1]
    return v

def menos_log_verosimilitud(params, x):
    """-ln L para r_t ~ N(0, sigma_t^2). Penalizamos parámetros no físicos."""
    omega, alfa, beta = params
    if omega <= 0 or alfa < 0 or beta < 0 or alfa + beta >= 1:
        return 1e10
    v = varianzas_garch(params, x)
    return 0.5 * np.sum(np.log(2 * np.pi * v) + x**2 / v)

def ajustar_garch(x):
    x = x - x.mean()
    inicio = (0.05 * x.var(), 0.08, 0.90)           # tuple[float]: punto de partida razonable
    res = optimize.minimize(menos_log_verosimilitud, inicio, args=(x,), method="Nelder-Mead",
                            options={"xatol": 1e-10, "fatol": 1e-8, "maxiter": 20000})
    return res.x

omega, alfa, beta = ajustar_garch(r)
persistencia = alfa + beta
resultados["garch_alfa"] = alfa
resultados["garch_beta"] = beta
resultados["garch_persistencia"] = persistencia
resultados["vida_media_temperatura_dias"] = np.log(0.5) / np.log(persistencia)
sigma_t = np.sqrt(varianzas_garch((omega, alfa, beta), r - r.mean()))

z_bruto = (r - r.mean()) / sigma                    # dividiendo por UNA temperatura fija
z_garch = (r - r.mean()) / sigma_t                  # dividiendo por la temperatura de cada día
resultados["curtosis_bruta"] = stats.kurtosis(z_bruto)
resultados["curtosis_tras_GARCH"] = stats.kurtosis(z_garch)
resultados["dias_mas_4sigma_bruto"] = int(np.sum(np.abs(z_bruto) > 4))
resultados["dias_mas_4sigma_GARCH"] = int(np.sum(np.abs(z_garch) > 4))

fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
bordes = np.linspace(-8, 8, 65)
xg = np.linspace(-8, 8, 400)
for a, zz, titulo in ((ax[0], z_bruto, comun.t("dividiendo por una $\\sigma$ fija", "divided by a fixed $\\sigma$")),
                      (ax[1], z_garch, comun.t("dividiendo por la $\\sigma_t$ de cada día", "divided by each day's $\\sigma_t$"))):
    a.hist(zz, bins=bordes, density=True, color=comun.AZUL, alpha=0.6)
    a.plot(xg, stats.norm.pdf(xg), color=comun.NARANJA, lw=1.3)
    a.set_yscale("log")
    a.set_ylim(1e-4, 1)
    a.set_title(f"{titulo}\n" + comun.t("curtosis en exceso", "excess kurtosis") + f" = {stats.kurtosis(zz):.1f}")
    a.set_xlabel(comun.t("retorno (en desviaciones típicas)", "return (in standard deviations)"))
ax[0].set_ylabel(comun.t("densidad", "density"))
comun.guardar(fig, "fig2_normalizar")

# Los seis días más extremos: ¿qué los explica?
extremos = np.argsort(-np.abs(z_bruto))[:6]          # índices de los 6 |z| mayores
print("\n  Los 6 días más extremos (fecha, z con sigma fija, z con sigma_t de GARCH):")
for t in extremos:
    print(f"    {fechas_r[t]}  {z_bruto[t]:+5.1f}  {z_garch[t]:+5.1f}")

# ---------------------------------------------------------------------------
# 5. ¿Se puede predecir la temperatura de mañana? (fuera de muestra)
# ---------------------------------------------------------------------------
# Ajustamos el GARCH con la PRIMERA mitad y lo usamos para predecir la SEGUNDA.
# sigma_prev[t] sólo usa datos hasta el día t-1: es una predicción honesta.
mitad = n // 2
params_1 = ajustar_garch(r[:mitad])
sigma_prev = np.sqrt(varianzas_garch(params_1, r - r[:mitad].mean()))[mitad:]
r_test = r[mitad:]
# Ordenamos los días de la segunda mitad por temperatura PREVISTA y los
# repartimos en 5 grupos iguales (quintiles): de los más "fríos" a los más "calientes".
grupos = np.array_split(np.argsort(sigma_prev), 5)     # list[np.ndarray[int]]
prevista = np.array([sigma_prev[g].mean() for g in grupos])
observada = np.array([np.sqrt(np.mean(r_test[g] ** 2)) for g in grupos])   # temperatura que hubo de verdad
media_signo = np.array([r_test[g].mean() for g in grupos])                 # ¿y la dirección?
resultados["prevista_por_quintil_%"] = ", ".join(f"{100 * v:.2f}" for v in prevista)
resultados["observada_por_quintil_%"] = ", ".join(f"{100 * v:.2f}" for v in observada)
resultados["retorno_medio_por_quintil_%"] = ", ".join(f"{100 * v:+.2f}" for v in media_signo)
resultados["fecha_inicio_2a_mitad"] = str(fechas_r[mitad])

fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.5))
etiquetas = comun.t(["muy\nfrío", "frío", "medio", "caliente", "muy\ncaliente"],
                    ["very\ncold", "cold", "medium", "hot", "very\nhot"])
pos = np.arange(5)
ax[0].bar(pos - 0.18, 100 * prevista, width=0.36, color=comun.GRIS, label=comun.t("prevista el día antes", "predicted the day before"))
ax[0].bar(pos + 0.18, 100 * observada, width=0.36, color=comun.NARANJA, label=comun.t("observada", "observed"))
ax[0].set_xticks(pos, etiquetas)
ax[0].set_ylabel(comun.t("volatilidad diaria (%)", "daily volatility (%)"))
ax[0].set_title(comun.t("La temperatura sí se predice", "Temperature is predictable"))
ax[0].legend(frameon=False)
ax[1].bar(pos, 100 * media_signo, width=0.5, color=comun.AZUL)
ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_xticks(pos, etiquetas)
ax[1].set_ylabel(comun.t("retorno medio (%)", "mean return (%)"))
ax[1].set_title(comun.t("La dirección no", "Direction is not"))
comun.guardar(fig, "fig2_prediccion")

print("\nRESULTADOS (temperatura)")
for k, v in resultados.items():
    print(f"  {k:32s} {v}" if isinstance(v, (str, int)) else f"  {k:32s} {v:10.4g}")
