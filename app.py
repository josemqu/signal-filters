from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from filters import kalman_1d, low_pass_butterworth, moving_average


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def _safe_float(x, default: float) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return default


def _infer_unit_from_column_name(col: str) -> str | None:
    col = str(col).strip()
    if not col:
        return None
    if col.endswith(")") and "(" in col:
        unit = col[col.rfind("(") + 1 : -1].strip()
        return unit or None
    if col.endswith("]") and "[" in col:
        unit = col[col.rfind("[") + 1 : -1].strip()
        return unit or None
    return None


def _get_unit_for_column(col: str, *, fallback: str = "") -> str:
    key = f"unit::{col}"
    inferred = _infer_unit_from_column_name(col)
    if key not in st.session_state:
        st.session_state[key] = inferred if inferred is not None else fallback
    return str(st.session_state[key] or "")


def _value_kw(key: str, default):
    if key in st.session_state:
        return {}
    return {"value": default}


def _build_figure(
    t: np.ndarray,
    raw: np.ndarray,
    series_by_name: dict[str, np.ndarray],
    ui_revision: str,
    x_axis_title: str,
    y_axis_title: str,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=t,
            y=raw,
            mode="lines",
            name="Original",
            line=dict(width=2),
        )
    )

    for name, y in series_by_name.items():
        fig.add_trace(
            go.Scatter(
                x=t,
                y=y,
                mode="lines",
                name=name,
                line=dict(width=2),
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=560,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        uirevision=ui_revision,
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        font=dict(size=12),
    )
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)

    return fig


def _init_state() -> None:
    st.session_state.setdefault("preset", "Custom")
    st.session_state.setdefault("enable_ma", True)
    st.session_state.setdefault("ma_window", 21)
    st.session_state.setdefault("enable_lp", True)
    st.session_state.setdefault("fs_hz", 1.0)
    st.session_state.setdefault("cutoff_hz", 0.2)
    st.session_state.setdefault("order", 4)
    st.session_state.setdefault("enable_kf", False)
    st.session_state.setdefault("q", 1e-4)
    st.session_state.setdefault("r", 1e-3)
    st.session_state.setdefault("zoom_revision", 0)


st.set_page_config(
    page_title="Signal Filters Demo",
    page_icon="📈",
    layout="wide",
)

st.title("Signal Filters Demo")

_init_state()

st.markdown(
    """
<style>
/* reduce default paddings/margins */
.block-container { padding-top: 1.0rem; padding-bottom: 1.0rem; padding-left: 1.2rem; padding-right: 1.2rem; max-width: 1400px; }
/* reduce header size */
h1 { font-size: 1.55rem !important; margin-bottom: 0.5rem !important; }
h2 { font-size: 1.15rem !important; }
h3 { font-size: 1.0rem !important; }
/* compact widgets */
div[data-testid="stMetric"] { padding: 0.35rem 0.35rem 0.35rem 0.35rem; }
div[data-testid="stExpander"] details { background: transparent; }
/* make sidebar a bit tighter */
section[data-testid="stSidebar"] { width: 360px !important; }
section[data-testid="stSidebar"] .block-container { padding-top: 0.75rem; }
/* keep chart visible while scrolling (avoid covering title/header) */
div[data-testid="stPlotlyChart"] { position: sticky; top: 4.25rem; z-index: 1; background: transparent; }
</style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Dataset")

    uploaded = st.file_uploader(
        "CSV",
        type=["csv"],
        accept_multiple_files=False,
        help="Subí un CSV con columnas numéricas. Opcionalmente incluí una columna de tiempo (por ejemplo: t en segundos).",
    )

    use_sample = st.toggle("Usar ejemplo", value=(uploaded is None))

    if uploaded is None and not use_sample:
        st.info("Subí un CSV o activá el ejemplo.")
        st.stop()

    if uploaded is not None and not use_sample:
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_csv("data/RAW_ACCELEROMETERS.csv")

    if df.empty:
        st.error("El CSV está vacío.")
        st.stop()

    cols = list(df.columns)

    default_time_idx = 0 if "t" in cols else None
    time_col = st.selectbox(
        "Tiempo",
        options=["(index)"] + cols,
        index=(default_time_idx + 1) if default_time_idx is not None else 0,
        help="Si no tenés columna de tiempo, usá (index) y configurá fs (Hz) manualmente.",
    )

    if time_col == "(index)":
        time_unit = st.text_input(
            "Unidad tiempo",
            value="samples",
            help="Como no hay columna de tiempo, el eje X usa el índice. Podés poner 'samples' o dejarlo vacío.",
        )
    else:
        inferred_t_unit = _infer_unit_from_column_name(time_col)
        if inferred_t_unit is None:
            _get_unit_for_column(time_col, fallback="s")
            time_unit = st.text_input(
                "Unidad tiempo",
                key=f"unit::{time_col}",
                help="No se detectó unidad asegurada en el nombre. Ej: 's', 'ms'.",
            )
        else:
            _get_unit_for_column(time_col, fallback=inferred_t_unit)
            time_unit = st.text_input(
                "Unidad tiempo",
                key=f"unit::{time_col}",
                help="Unidad detectada desde el nombre (podés sobrescribirla).",
            )

    numeric_cols = [c for c in cols if _is_numeric_series(df[c])]
    if not numeric_cols:
        st.error("No hay columnas numéricas para analizar.")
        st.stop()

    default_signal = (
        "X acceleration (Gs)"
        if "X acceleration (Gs)" in numeric_cols
        else numeric_cols[0]
    )
    signal_col = st.selectbox(
        "Señal",
        options=numeric_cols,
        index=numeric_cols.index(default_signal),
        help="Columna a graficar/filtrar.",
    )

    inferred_y_unit = _infer_unit_from_column_name(signal_col)
    if inferred_y_unit is None:
        _get_unit_for_column(signal_col, fallback="")
        signal_unit = st.text_input(
            "Unidad señal",
            key=f"unit::{signal_col}",
            help="Ej: 'm/s^2', 'g', 'Gs'.",
        )
    else:
        _get_unit_for_column(signal_col, fallback=inferred_y_unit)
        signal_unit = st.text_input(
            "Unidad señal",
            key=f"unit::{signal_col}",
            help="Unidad detectada desde el nombre (podés sobrescribirla).",
        )

    with st.expander("Guía rápida", expanded=False):
        st.markdown(
            """
Esta app te deja comparar una **señal original** contra varias salidas de filtros.

- **Moving average**: más ventana = más suavizado.
- **Low-pass (Butterworth)**: `cutoff < fs/2`.
- **Kalman 1D**: `q` más grande reacciona más; `r` más grande suaviza más.
            """
        )

    st.markdown("---")
    st.markdown("### Filtros")

    preset = st.selectbox(
        "Preset",
        options=["Custom", "Suavizado leve", "Suavizado medio", "Suavizado fuerte"],
        key="preset",
        help="Atajo didáctico: carga valores sugeridos.",
    )

    if st.session_state.get("last_preset") != preset:
        st.session_state.last_preset = preset
        if preset == "Suavizado leve":
            st.session_state["enable_ma"] = True
            st.session_state["ma_window"] = 11
            st.session_state["enable_lp"] = True
            st.session_state["cutoff_hz"] = max(
                0.01, float(st.session_state.get("fs_hz", 1.0)) * 0.20
            )
            st.session_state["order"] = 3
            st.session_state["enable_kf"] = False
        elif preset == "Suavizado medio":
            st.session_state["enable_ma"] = True
            st.session_state["ma_window"] = 31
            st.session_state["enable_lp"] = True
            st.session_state["cutoff_hz"] = max(
                0.01, float(st.session_state.get("fs_hz", 1.0)) * 0.12
            )
            st.session_state["order"] = 4
            st.session_state["enable_kf"] = False
        elif preset == "Suavizado fuerte":
            st.session_state["enable_ma"] = True
            st.session_state["ma_window"] = 61
            st.session_state["enable_lp"] = True
            st.session_state["cutoff_hz"] = max(
                0.01, float(st.session_state.get("fs_hz", 1.0)) * 0.06
            )
            st.session_state["order"] = 5
            st.session_state["enable_kf"] = False

    enable_ma = st.toggle(
        "Moving average", key="enable_ma", help="Promedia una ventana de muestras."
    )
    ma_window = st.slider(
        "Ventana MA",
        min_value=1,
        max_value=501,
        step=2,
        disabled=not enable_ma,
        key="ma_window",
        **_value_kw("ma_window", int(st.session_state.get("ma_window", 21))),
    )

    enable_lp = st.toggle(
        "Low-pass (Butterworth)",
        key="enable_lp",
        help="Requiere fs (Hz) y cutoff (Hz) con cutoff < fs/2.",
    )

    default_fs = 1.0
    if time_col != "(index)":
        t_raw = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(t_raw).sum() >= 2:
            dt = np.nanmedian(np.diff(t_raw))
            if np.isfinite(dt) and dt > 0:
                default_fs = 1.0 / dt

    if "fs_hz" not in st.session_state or st.session_state.fs_hz == 1.0:
        st.session_state.fs_hz = float(default_fs)

    fs_hz = st.number_input(
        "fs (Hz)",
        min_value=0.0001,
        step=0.1,
        format="%.6f",
        disabled=not enable_lp,
        key="fs_hz",
        **_value_kw("fs_hz", float(st.session_state.get("fs_hz", 1.0))),
    )

    if "cutoff_hz" not in st.session_state or st.session_state.cutoff_hz <= 0:
        st.session_state.cutoff_hz = min(5.0, float(fs_hz) * 0.2)

    cutoff_hz = st.number_input(
        "cutoff (Hz)",
        min_value=0.0001,
        step=0.1,
        format="%.6f",
        disabled=not enable_lp,
        key="cutoff_hz",
        **_value_kw(
            "cutoff_hz",
            float(st.session_state.get("cutoff_hz", min(5.0, float(fs_hz) * 0.2))),
        ),
    )

    order = st.slider(
        "Orden",
        min_value=1,
        max_value=8,
        step=1,
        disabled=not enable_lp,
        key="order",
        **_value_kw("order", int(st.session_state.get("order", 4))),
    )

    enable_kf = st.toggle("Kalman (1D)", key="enable_kf")
    q = st.number_input(
        "q",
        min_value=1e-12,
        step=1e-4,
        format="%.12f",
        disabled=not enable_kf,
        key="q",
        **_value_kw("q", float(st.session_state.get("q", 1e-4))),
    )
    r = st.number_input(
        "r",
        min_value=1e-12,
        step=1e-4,
        format="%.12f",
        disabled=not enable_kf,
        key="r",
        **_value_kw("r", float(st.session_state.get("r", 1e-3))),
    )

    st.markdown("---")
    st.markdown("### Vista")
    max_points = st.slider(
        "Máx puntos", min_value=500, max_value=50000, value=8000, step=500
    )

    if st.button(
        "Reset zoom", use_container_width=True, help="Resetea el zoom/pan del gráfico."
    ):
        st.session_state.zoom_revision += 1

st.subheader("Gráfico")

a = st.container()

y_raw = pd.to_numeric(df[signal_col], errors="coerce").to_numpy(dtype=float)

if time_col == "(index)":
    t = np.arange(len(df), dtype=float)
else:
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(t).any():
        t = np.arange(len(df), dtype=float)

valid = np.isfinite(t) & np.isfinite(y_raw)
t = t[valid]
y = y_raw[valid]

if t.size == 0:
    st.error("No hay datos válidos para graficar.")
    st.stop()

if time_col != "(index)" and t.size >= 2:
    if np.nanmin(np.diff(t)) <= 0:
        st.warning(
            "La columna de tiempo no es estrictamente creciente. El gráfico se muestra igual, pero fs/cutoff pueden ser engañosos."
        )

missing_pct = 100.0 * (1.0 - float(valid.sum()) / float(len(df)))
dt_median = None
if t.size >= 2:
    dts = np.diff(t)
    dt_median = float(np.nanmedian(dts)) if np.isfinite(dts).any() else None

m1, m2, m3, m4 = st.columns(4)
m1.metric("Muestras", f"{t.size:,}")
m2.metric("Missing", f"{missing_pct:.2f}%")
if dt_median is not None and dt_median > 0:
    m3.metric("dt mediana", f"{dt_median:.6g}")
    m4.metric("fs estimada", f"{(1.0 / dt_median):.6g} Hz")
else:
    m3.metric("dt mediana", "-")
    m4.metric("fs estimada", "-")

if t.size > max_points:
    idx = np.linspace(0, t.size - 1, num=max_points).astype(int)
    t_view = t[idx]
    y_view = y[idx]
else:
    t_view = t
    y_view = y

series: dict[str, np.ndarray] = {}

if enable_ma:
    if ma_window > y_view.size:
        st.warning(
            "Moving average: la ventana es mayor que la cantidad de puntos mostrados. Se ajustó automáticamente."
        )
        ma_window_eff = int(max(1, y_view.size // 2 * 2 + 1))
    else:
        ma_window_eff = int(ma_window)
    series[f"Moving avg (w={ma_window_eff})"] = moving_average(y_view, ma_window_eff)

if enable_lp:
    if float(cutoff_hz) >= 0.5 * float(fs_hz):
        st.error("Low-pass: cutoff debe ser < fs/2 (Nyquist). Bajá cutoff o subí fs.")
    else:
        try:
            series[f"Low-pass (fc={cutoff_hz:g}Hz, order={order})"] = (
                low_pass_butterworth(
                    y_view,
                    fs_hz=float(fs_hz),
                    cutoff_hz=float(cutoff_hz),
                    order=int(order),
                )
            )
        except Exception as e:
            st.warning(f"Low-pass no se pudo calcular: {e}")

if enable_kf:
    try:
        series[f"Kalman (q={_safe_float(q, 0):g}, r={_safe_float(r, 0):g})"] = (
            kalman_1d(
                y_view,
                q=float(q),
                r=float(r),
            )
        )
    except Exception as e:
        st.warning(f"Kalman no se pudo calcular: {e}")

x_label = "index" if time_col == "(index)" else str(time_col)
x_axis_title = f"{x_label} ({time_unit})" if str(time_unit).strip() else x_label

y_label = str(signal_col)
y_axis_title = f"{y_label} ({signal_unit})" if str(signal_unit).strip() else y_label

ui_revision = (
    f"zoom:{st.session_state.zoom_revision}|time:{time_col}|signal:{signal_col}"
)
fig = _build_figure(
    t_view,
    y_view,
    series,
    ui_revision=ui_revision,
    x_axis_title=x_axis_title,
    y_axis_title=y_axis_title,
)
with a:
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Preview de datos"):
    st.dataframe(df.head(50), use_container_width=True)
