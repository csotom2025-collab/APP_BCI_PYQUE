# -*- coding: utf-8 -*-
"""
Genera espectrogramas de grabaciones EEG para las bandas elegidas
(por defecto Beta y Gamma).  VERSION OPTIMIZADA.

Entrada:
    {BASE_DIR}/{usuario}/{tpComando}/{usuario}_{letra}_{trial}.csv

Salida (una imagen por ventana, numeradas -1, -2, -3 ...):
    RAW    -> {SPEC_ROOT}/{usuario}/{tpComando}/{usuario}_{letra}_{trial}-N.png
              (espectrograma "en crudo": sin ejes ni texto, ideal para la CNN)
    VISUAL -> {VISUAL_ROOT}/{usuario}/{tpComando}/{usuario}_{letra}_{trial}-N.png
              (figura matplotlib con titulos, ejes, bandas y colorbar)

Optimizaciones respecto a la version anterior:
  * RAW se arma con numpy + PIL (sin matplotlib): ~100x mas rapido.
  * VISUAL reutiliza UNA sola figura por proceso (solo actualiza los datos).
  * STFT vectorizado para todos los canales en una sola llamada.
  * Procesamiento en paralelo (varios procesos) por grabacion.
  * Opcion de saltar imagenes ya generadas y de limitar las VISUAL.

Todos los parametros se definen en el bloque main.
"""

import os
import re
import sys
import glob
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal


# ----------------------------------------------------------------------------
# Log seguro (evita UnicodeEncodeError en consolas cp1252 de Windows)
# ----------------------------------------------------------------------------
def log(*args):
    msg = " ".join(str(a) for a in args)
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        print(msg.encode(enc, errors="backslashreplace").decode(enc), flush=True)


# ----------------------------------------------------------------------------
# Utilidades de lectura
# ----------------------------------------------------------------------------
def _parse_filename(fname, usuario):
    """Extrae (letra, trial) de {usuario}_{letra}_{trial}.csv"""
    stem = os.path.splitext(os.path.basename(fname))[0]
    prefix = f"{usuario}_"
    if not stem.startswith(prefix):
        return None, None
    resto = stem[len(prefix):]
    if "_" not in resto:
        return None, None
    letra, trial = resto.rsplit("_", 1)
    return letra, trial


def _load_signal_csv(path, channel_names):
    """Carga un CSV y devuelve array (n_channels, n_samples)."""
    df = pd.read_csv(path)
    cols_presentes = [c for c in channel_names if c in df.columns]
    if len(cols_presentes) == len(channel_names):
        return df[channel_names].to_numpy().T
    if df.shape[1] >= len(channel_names):
        return df.iloc[:, :len(channel_names)].to_numpy().T
    raise ValueError(f"No se pudo interpretar el CSV {path}: "
                     f"se esperaban {len(channel_names)} canales.")


def _safe_filename(name):
    return re.sub(r'[<>:"/\\|?*]', "_", name)


# ----------------------------------------------------------------------------
# Preprocesamiento (mismas funciones que eeg_features.py)
# ----------------------------------------------------------------------------
def apply_notch_filter(signals, fs, notch_freq=60.0, Q=30.0):
    b, a = signal.iirnotch(w0=notch_freq, Q=Q, fs=fs)
    return signal.filtfilt(b, a, signals, axis=1)


def apply_bandpass_filter(signals, fs, lowcut=0.5, highcut=40.0, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return signal.filtfilt(b, a, signals, axis=1)


def apply_baseline_correction(signals, fs, baseline_window_s=(0.0, 0.5)):
    start = int(baseline_window_s[0] * fs)
    end = min(int(baseline_window_s[1] * fs), signals.shape[1])
    if end <= start:
        return signals
    return signals - np.mean(signals[:, start:end], axis=1, keepdims=True)


def apply_zscore_normalization(signals):
    mean = np.mean(signals, axis=1, keepdims=True)
    std = np.std(signals, axis=1, keepdims=True)
    std[std == 0] = 1e-6
    return (signals - mean) / std


def crop_p300_window(signals, fs, p300_window_s):
    start = int(p300_window_s[0] * fs)
    end = min(int(p300_window_s[1] * fs), signals.shape[1])
    if start >= end:
        return signals
    return signals[:, start:end]


# ----------------------------------------------------------------------------
# Ventaneo (identico a EEGFeatureExtractor.extract_features)
# ----------------------------------------------------------------------------
def get_windows(n_samples, window_size, overlap):
    step_size = int(window_size * (1 - overlap))
    if step_size <= 0:
        step_size = max(1, window_size // 2)
    n_windows = max(0, (n_samples - window_size) // step_size + 1)
    if n_windows <= 0:
        raise ValueError("Segmento demasiado corto para el window_size y overlap especificados.")
    return [(i * step_size, i * step_size + window_size) for i in range(n_windows)]


# ----------------------------------------------------------------------------
# Bandas / espectrograma
# ----------------------------------------------------------------------------
def resolve_bands(bands, band_ranges, fs, bandpass_high=None):
    """
    Valida bandas y recorta cada rango a Nyquist (y al pasa banda si se da,
    porque por encima del pasa banda la señal ya fue filtrada).
    Retorna lista ordenada de (nombre, f_low, f_high).
    """
    limit = fs / 2.0
    if bandpass_high is not None:
        limit = min(limit, bandpass_high)
    resolved = []
    for b in bands:
        key = b.lower()
        if key not in band_ranges:
            raise ValueError(f"Banda '{b}' desconocida. Opciones: {list(band_ranges)}")
        lo, hi = band_ranges[key]
        if lo >= limit:
            log(f"[AVISO] Banda '{key}' ({lo}-{hi} Hz) esta por encima del limite util ({limit} Hz); se omite.")
            continue
        if hi > limit:
            log(f"[AVISO] Banda '{key}': limite superior {hi} Hz recortado a {limit} Hz "
                f"(min entre Nyquist y el pasa banda).")
            hi = limit
        resolved.append((key, lo, hi))
    if not resolved:
        raise ValueError("Ninguna banda seleccionada es valida para esta configuracion.")
    return sorted(resolved, key=lambda x: x[1])


def compute_band_spectrograms(window_data, fs, resolved_bands, nperseg, noverlap, nfft):
    """
    STFT de TODOS los canales de la ventana en una sola llamada.
    Retorna (times, Sxx_dB[n_ch, filas, tiempo], band_slices, freqs_sel)
    donde freqs_sel[i] es la frecuencia (Hz) de la fila i.
    """
    nperseg = min(nperseg, window_data.shape[1])
    noverlap = min(noverlap, nperseg - 1)
    freqs, times, Sxx = signal.spectrogram(
        window_data, fs=fs, window="hann",
        nperseg=nperseg, noverlap=noverlap, nfft=nfft,
        mode="psd", scaling="density", axis=-1,
    )  # (n_ch, n_freqs, n_times)

    blocks, slices, freq_rows, cursor = [], [], [], 0
    for name, lo, hi in resolved_bands:
        mask = (freqs >= lo) & (freqs <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        blocks.append(Sxx[:, mask, :])
        freq_rows.append(freqs[mask])
        slices.append((name, cursor, cursor + n))
        cursor += n
    if not blocks:
        raise ValueError("Resolucion en frecuencia insuficiente: sube spec_nfft.")
    Sxx_sel = np.concatenate(blocks, axis=1)
    freqs_sel = np.concatenate(freq_rows)
    return times, 10.0 * np.log10(Sxx_sel + 1e-12), slices, freqs_sel


# ----------------------------------------------------------------------------
# Imagen RAW (numpy + PIL, sin matplotlib)
# ----------------------------------------------------------------------------
def save_raw_image(Sxx_db, vmin, vmax, cmap, tile_px, grid_cols, out_path):
    """
    Cada canal = un tile con el espectrograma coloreado (frecuencia baja abajo),
    reescalado a tile_px=(ancho, alto); los tiles se acomodan en una cuadricula.
    """
    n_ch = Sxx_db.shape[0]
    cols = min(grid_cols, n_ch)
    rows = int(np.ceil(n_ch / cols))
    tw, th = tile_px
    cm = matplotlib.colormaps[cmap]
    canvas = Image.new("RGB", (cols * tw, rows * th), (0, 0, 0))

    norm = np.clip((Sxx_db - vmin) / (vmax - vmin + 1e-12), 0.0, 1.0)
    for ch in range(n_ch):
        rgb = (cm(norm[ch])[:, :, :3] * 255).astype(np.uint8)   # (F, T, 3)
        rgb = rgb[::-1]                                          # f baja abajo
        tile = Image.fromarray(rgb).resize((tw, th), Image.BILINEAR)
        canvas.paste(tile, ((ch % cols) * tw, (ch // cols) * th))
    canvas.save(out_path, compress_level=1)


# ----------------------------------------------------------------------------
# Imagen VISUAL (matplotlib, figura reutilizada)
# ----------------------------------------------------------------------------
class VisualRenderer:
    """Construye la figura UNA vez y solo actualiza los datos en cada ventana."""

    def __init__(self, channel_names, grid_cols, cmap, dpi, resolved_bands):
        self.channel_names = channel_names
        self.grid_cols = grid_cols
        self.cmap = cmap
        self.dpi = dpi
        self.resolved_bands = resolved_bands
        self.key = None
        self.fig = None

    def _build(self, Sxx_db, times, slices, freqs_sel, vmin, vmax):
        n_ch = Sxx_db.shape[0]
        cols = min(self.grid_cols, n_ch)
        rows = int(np.ceil(n_ch / cols))
        self.fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows), squeeze=False)
        t0, t1 = float(times[0]), float(times[-1])
        n_rows_f = Sxx_db.shape[1]
        self.images = []
        for ch in range(rows * cols):
            ax = axes[ch // cols][ch % cols]
            if ch >= n_ch:
                ax.axis("off")
                continue
            im = ax.imshow(Sxx_db[ch], origin="lower", aspect="auto", cmap=self.cmap,
                           vmin=vmin, vmax=vmax, extent=(t0, t1, 0, n_rows_f),
                           interpolation="nearest")
            self.images.append(im)
            ax.set_title(str(self.channel_names[ch]), fontsize=9)
            # Eje Y en Hz: se ponen ~6 marcas repartidas en las filas y se
            # etiquetan con la frecuencia real de esa fila (funciona aunque
            # las bandas elegidas no sean contiguas).
            tick_rows = np.unique(np.linspace(0, n_rows_f - 1, 6).astype(int))
            ax.set_yticks(tick_rows + 0.5)
            ax.set_yticklabels([f"{freqs_sel[r]:.0f}" for r in tick_rows], fontsize=7)
            if ch % cols == 0:
                ax.set_ylabel("Frecuencia (Hz)", fontsize=7)
            # nombre de cada banda dentro del grafico
            for name, a, b in slices:
                ax.text(t1, (a + b) / 2, name, color="white", fontsize=6,
                        ha="right", va="center")
            for _, _, b in slices[:-1]:
                ax.axhline(b, color="white", lw=0.8, ls="--")
            ax.tick_params(axis="x", labelsize=7)
            ax.set_xlabel("t (s)", fontsize=7)
        band_txt = ", ".join(f"{n} ({lo:g}-{hi:g} Hz)" for n, lo, hi in self.resolved_bands)
        self.band_txt = band_txt
        self.suptitle = self.fig.suptitle("", fontsize=10)
        self.fig.tight_layout(rect=(0, 0, 0.92, 0.94))
        cax = self.fig.add_axes([0.93, 0.15, 0.015, 0.7])
        self.fig.colorbar(self.images[0], cax=cax, label="Potencia (dB)")

    def save(self, Sxx_db, times, slices, freqs_sel, vmin, vmax, title, out_path):
        key = (Sxx_db.shape, len(times))
        if self.key != key:
            if self.fig is not None:
                plt.close(self.fig)
            self._build(Sxx_db, times, slices, freqs_sel, vmin, vmax)
            self.key = key
        for ch, im in enumerate(self.images):
            im.set_data(Sxx_db[ch])
            im.set_clim(vmin, vmax)
        self.suptitle.set_text(f"{title}\n{self.band_txt}")
        self.fig.savefig(out_path, dpi=self.dpi)


# ----------------------------------------------------------------------------
# Procesamiento de UNA grabacion (se ejecuta en un proceso worker)
# ----------------------------------------------------------------------------
_RENDERER = None   # uno por proceso


def process_recording(task):
    global _RENDERER
    path, raw_dir, vis_dir, make_visual, cfg = task
    out = {"path": path, "status": "ok", "n_imgs": 0, "msg": ""}
    try:
        fs = cfg["fs"]
        signals = _load_signal_csv(path, cfg["channel_names"])

        signals = apply_notch_filter(signals, fs, notch_freq=cfg["notch_freq"])
        signals = apply_bandpass_filter(signals, fs, lowcut=cfg["bandpass_low"],
                                        highcut=cfg["bandpass_high"])
        if cfg["apply_baseline"]:
            signals = apply_baseline_correction(signals, fs, cfg["baseline_window_s"])

        pico = np.max(np.abs(signals))
        if pico > cfg["peak_threshold_uv"]:
            out["status"] = "descartado"
            out["msg"] = f"Bloque descartado por Pico: {pico:.2f} uV en archivo: {path}"
            return out

        signals = apply_zscore_normalization(signals)
        if cfg["use_p300_window_only"]:
            signals = crop_p300_window(signals, fs, cfg["p300_window_s"])

        windows = get_windows(signals.shape[1], cfg["window_size"], cfg["window_overlap"])
        stem = _safe_filename(os.path.splitext(os.path.basename(path))[0])

        if make_visual and _RENDERER is None:
            _RENDERER = VisualRenderer(cfg["channel_names"], cfg["grid_cols"], cfg["cmap"],
                                       cfg["dpi"], cfg["resolved_bands"])

        for i, (start, end) in enumerate(windows, start=1):
            raw_path = os.path.join(raw_dir, f"{stem}-{i}.png") if cfg["save_raw"] else None
            vis_path = os.path.join(vis_dir, f"{stem}-{i}.png") if make_visual else None

            if cfg["skip_existing"]:
                if raw_path and os.path.exists(raw_path):
                    raw_path = None
                if vis_path and os.path.exists(vis_path):
                    vis_path = None
                if raw_path is None and vis_path is None:
                    continue

            times, Sxx_db, slices, freqs_sel = compute_band_spectrograms(
                signals[:, start:end], fs, cfg["resolved_bands"],
                cfg["spec_nperseg"], cfg["spec_noverlap"], cfg["spec_nfft"])
            vmin, vmax = float(Sxx_db.min()), float(Sxx_db.max())

            if raw_path:
                save_raw_image(Sxx_db, vmin, vmax, cfg["cmap"], cfg["raw_tile_px"],
                               cfg["grid_cols"], raw_path)
            if vis_path:
                _RENDERER.save(Sxx_db, times, slices, freqs_sel, vmin, vmax, f"{stem}-{i}", vis_path)
            out["n_imgs"] += 1
        return out
    except Exception as e:
        out["status"] = "error"
        out["msg"] = f"[ERROR] {path}: {type(e).__name__}: {e}"
        return out


# ----------------------------------------------------------------------------
# Pipeline principal
# ----------------------------------------------------------------------------
def generate_spectrograms(base_dir, spec_root, visual_root, usuario, tp_comandos, fs,
                          channel_names, window_size, window_overlap,
                          apply_baseline, baseline_window_s,
                          use_p300_window_only, p300_window_s,
                          notch_freq, bandpass_low, bandpass_high, peak_threshold_uv,
                          bands, band_ranges,
                          spec_nperseg, spec_noverlap, spec_nfft,
                          grid_cols, cmap, dpi, raw_tile_px,
                          save_raw, save_visual, visual_every_n,
                          skip_existing, n_workers, verbose=True):
    resolved_bands = resolve_bands(bands, band_ranges, fs, bandpass_high)

    cfg = dict(
        fs=fs, channel_names=channel_names, window_size=window_size,
        window_overlap=window_overlap, apply_baseline=apply_baseline,
        baseline_window_s=baseline_window_s, use_p300_window_only=use_p300_window_only,
        p300_window_s=p300_window_s, notch_freq=notch_freq, bandpass_low=bandpass_low,
        bandpass_high=bandpass_high, peak_threshold_uv=peak_threshold_uv,
        resolved_bands=resolved_bands, spec_nperseg=spec_nperseg,
        spec_noverlap=spec_noverlap, spec_nfft=spec_nfft, grid_cols=grid_cols,
        cmap=cmap, dpi=dpi, raw_tile_px=raw_tile_px, save_raw=save_raw,
        skip_existing=skip_existing,
    )

    log("Cargando datos de:", base_dir)
    log("Bandas:", [(n, lo, hi) for n, lo, hi in resolved_bands])
    if save_raw:
        log("RAW    en:", spec_root)
    if save_visual:
        log(f"VISUAL en: {visual_root} (1 de cada {visual_every_n} grabaciones)")

    # --- armar lista de tareas ---
    tasks = []
    for tp in tp_comandos:
        carpeta = os.path.join(base_dir, usuario, tp)
        archivos = sorted(glob.glob(os.path.join(carpeta, f"{usuario}_*.csv")))
        if not archivos:
            if verbose:
                log(f"[AVISO] Sin archivos en: {carpeta}")
            continue
        raw_dir = os.path.join(spec_root, usuario, tp)
        vis_dir = os.path.join(visual_root, usuario, tp)
        if save_raw:
            os.makedirs(raw_dir, exist_ok=True)
        if save_visual:
            os.makedirs(vis_dir, exist_ok=True)
        for idx, path in enumerate(archivos):
            letra, _ = _parse_filename(path, usuario)
            if letra is None:
                if verbose:
                    log(f"[AVISO] No se pudo parsear el nombre: {path}")
                continue
            make_visual = bool(save_visual and idx % max(1, visual_every_n) == 0)
            tasks.append((path, raw_dir, vis_dir, make_visual, cfg))

    total = len(tasks)
    log(f"Grabaciones a procesar: {total} | procesos: {n_workers}")
    n_ok, n_fail, n_imgs, done = 0, 0, 0, 0
    t0 = time.time()

    def handle(res):
        nonlocal n_ok, n_fail, n_imgs, done
        done += 1
        if res["status"] == "ok":
            n_ok += 1
            n_imgs += res["n_imgs"]
        else:
            n_fail += 1
            if verbose:
                log(res["msg"])
        if done % 25 == 0 or done == total:
            el = time.time() - t0
            log(f"  [{done}/{total}] {el:.1f}s transcurridos | ~{el / done * (total - done):.0f}s restantes")

    if n_workers <= 1:
        for t in tasks:
            handle(process_recording(t))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for res in ex.map(process_recording, tasks, chunksize=4):
                handle(res)

    log(f"\nGrabaciones OK: {n_ok} | fallidas/descartadas: {n_fail} | "
        f"ventanas generadas: {n_imgs} | tiempo total: {time.time() - t0:.1f}s")


# ----------------------------------------------------------------------------
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # ===================== PARAMETROS =====================
    # --- Rutas / usuario ---
    BASE_DIR = "captures"
    USUARIO = "UserMichel"
    TP_COMANDOS = ["Letters", "Numbers", "Controls"]

    _parent = os.path.dirname(os.path.normpath(BASE_DIR))
    SPEC_ROOT = os.path.join(_parent, "spectrograms")           # imagenes RAW
    VISUAL_ROOT = os.path.join(_parent, "spectrograms_visual")  # figuras matplotlib

    # --- Que guardar ---
    SAVE_RAW = True          # espectrograma limpio (para entrenar)
    SAVE_VISUAL = True       # figura matplotlib con ejes, titulos y colorbar
    VISUAL_EVERY_N = 10      # VISUAL solo para 1 de cada N grabaciones (1 = todas)

    # --- Rendimiento ---
    N_WORKERS = max(1, (os.cpu_count() or 2) - 1)   # procesos en paralelo (1 = sin paralelismo)
    SKIP_EXISTING = True     # no regenerar imagenes que ya existen

    # --- Señal ---
    FS = 128
    CHANNEL_NAMES = ["F3", "FC5", "AF3", "F7", "T7", "P7", "O1", "O2",
                     "P8", "T8", "F8", "AF4", "FC6", "F4"]

    # --- Ventaneo (en muestras, igual que EEGFeatureExtractor) ---
    WINDOW_SIZE = 189
    WINDOW_OVERLAP = 0.89

    # --- Bandas a mostrar ---
    BANDS = ["beta", "gamma"]        # "delta","theta","alpha","beta","gamma"
    BAND_RANGES = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 14.0),
        "beta": (14.0, 30.0),
        "gamma": (30.0, 140.0),      # se recorta a min(Nyquist, BANDPASS_HIGH)
    }

    # --- Preprocesamiento ---
    APPLY_BASELINE = True
    BASELINE_WINDOW_S = (0.0, 0.5)
    USE_P300_WINDOW_ONLY = False
    P300_WINDOW_S = (0.5, 1.2)
    NOTCH_FREQ = 60.0
    BANDPASS_LOW = 0.5
    BANDPASS_HIGH = 40.0
    PEAK_THRESHOLD_UV = 150.0

    # --- Espectrograma (STFT dentro de cada ventana) ---
    SPEC_NPERSEG = 8
    SPEC_NOVERLAP = 7
    SPEC_NFFT = 128

    # --- Imagen ---
    GRID_COLS = 4
    CMAP = "viridis"
    DPI = 100                 # solo afecta a las VISUAL
    RAW_TILE_PX = (128, 128)  # tamaño en pixeles de cada canal en la imagen RAW
    # ======================================================

    generate_spectrograms(
        base_dir=BASE_DIR, spec_root=SPEC_ROOT, visual_root=VISUAL_ROOT,
        usuario=USUARIO, tp_comandos=TP_COMANDOS, fs=FS,
        channel_names=CHANNEL_NAMES,
        window_size=WINDOW_SIZE, window_overlap=WINDOW_OVERLAP,
        apply_baseline=APPLY_BASELINE, baseline_window_s=BASELINE_WINDOW_S,
        use_p300_window_only=USE_P300_WINDOW_ONLY, p300_window_s=P300_WINDOW_S,
        notch_freq=NOTCH_FREQ, bandpass_low=BANDPASS_LOW,
        bandpass_high=BANDPASS_HIGH, peak_threshold_uv=PEAK_THRESHOLD_UV,
        bands=BANDS, band_ranges=BAND_RANGES,
        spec_nperseg=SPEC_NPERSEG, spec_noverlap=SPEC_NOVERLAP, spec_nfft=SPEC_NFFT,
        grid_cols=GRID_COLS, cmap=CMAP, dpi=DPI, raw_tile_px=RAW_TILE_PX,
        save_raw=SAVE_RAW, save_visual=SAVE_VISUAL, visual_every_n=VISUAL_EVERY_N,
        skip_existing=SKIP_EXISTING, n_workers=N_WORKERS,
    )


# IMPORTANTE: este guard es obligatorio en Windows para usar multiprocessing.
if __name__ == "__main__":
    main()