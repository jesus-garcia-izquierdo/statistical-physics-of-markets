"""
comun.py — Lo que comparten los cuatro scripts
===============================================

Aquí sólo hay tres cosas:
  1. Cargar los datos reales (vienen dentro de matplotlib: no hace falta internet).
  2. Un estilo común para las figuras.
  3. Una función para guardar figuras en la carpeta ``figuras/``.

ESQUEMA DE TIPOS DE DATOS
-------------------------
    fechas   : np.ndarray de np.datetime64[D]   (un día por elemento)
    precios  : np.ndarray de float64            (precio de cierre de cada día)
    retornos : np.ndarray de float64            (r_t = ln P_t - ln P_{t-1}; uno menos que precios)

¿Por qué estos tipos?
  * ``np.ndarray`` de float64 permite operar con toda la serie a la vez
    (``np.diff``, ``np.log``...) sin bucles, y con 16 cifras de precisión.
  * ``datetime64[D]`` guarda fechas como "número de días desde 1970", así que
    restar fechas o calcular el día de la semana es aritmética simple.
"""
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Idioma de las etiquetas de las figuras: "es" (por defecto) o "en".
#   IDIOMA=en python 01_paseo_aleatorio.py   -> figuras en inglés en la carpeta figures/
IDIOMA = os.environ.get("IDIOMA", "es")


def t(es, en):
    """Devuelve el texto en el idioma elegido (str)."""
    return en if IDIOMA == "en" else es


# Carpeta donde se guardan las figuras (se crea si no existe).
CARPETA_FIGURAS = Path(__file__).parent / ("figures" if IDIOMA == "en" else "figuras")
CARPETA_FIGURAS.mkdir(exist_ok=True)

# Los datos de ejemplo que trae matplotlib (descargados en su día de Yahoo Finance).
CARPETA_DATOS = Path(matplotlib.get_data_path()) / "sample_data"

# Paleta de colores: azul y naranja se distinguen bien también con daltonismo.
AZUL, NARANJA, VERDE, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#898781"


# ---------------------------------------------------------------------------
# 1. Datos
# ---------------------------------------------------------------------------
def cargar_google():
    """Precio diario de cierre de Google (GOOG), 19-08-2004 a 14-10-2008.

    Devuelve (fechas, precios). Son 1047 días de mercado: la etapa de
    crecimiento tras la salida a bolsa y el principio de la crisis de 2008.
    """
    datos = np.load(CARPETA_DATOS / "goog.npz")["price_data"]   # array estructurado de numpy
    return datos["date"], datos["adj_close"].astype(np.float64)


def cargar_indices_mensuales():
    """Valor mensual de dos índices bursátiles, 1990-2022 (391 meses).

    * S&P 500  (^GSPC): las 500 mayores empresas de EE. UU.
    * Nasdaq   (^IXIC): sobre todo empresas tecnológicas.

    Devuelve (fechas, sp500, nasdaq).
    """
    import csv
    filas = {}                                    # dict fecha -> (sp500, nasdaq): evita fechas repetidas
    with open(CARPETA_DATOS / "Stocks.csv", encoding="utf-8") as f:
        lector = csv.DictReader(linea for linea in f if not linea.startswith("#"))
        for fila in lector:
            if fila["^GSPC"] and fila["^IXIC"]:   # saltamos filas sin dato
                filas[fila["Date"]] = (float(fila["^GSPC"]), float(fila["^IXIC"]))
    fechas = np.array(sorted(filas), dtype="datetime64[D]")
    valores = np.array([filas[str(d)] for d in fechas])   # matriz (meses x 2)
    return fechas, valores[:, 0], valores[:, 1]


def log_retornos(precios):
    """r_t = ln(P_t / P_{t-1}).

    Para cambios pequeños, r_t es casi el cambio porcentual: si el precio pasa
    de 100 a 102, r = ln(1.02) = 0.0198, es decir, ~2 %. Usamos el logaritmo
    porque así los retornos de días seguidos se SUMAN (como los pasos de un
    caminante), en lugar de multiplicarse.
    """
    return np.diff(np.log(precios))


# ---------------------------------------------------------------------------
# 2. Estilo de las figuras
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120, "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})


def guardar(fig, nombre):
    """Guarda la figura en PDF (para LaTeX) dentro de ``figuras/``."""
    fig.tight_layout()
    fig.savefig(CARPETA_FIGURAS / f"{nombre}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  figura guardada: {CARPETA_FIGURAS.name}/{nombre}.pdf")
