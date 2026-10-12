#!/usr/bin/env python3
"""Count physically available A and corrected B EEG recordings per user/class.

Classification and user/class/trial metadata are read from quality_filter.py's
quality report. This script does not re-evaluate EEG quality or modify files.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, TextIO

QUALITY_REPORT_NAMES = {"quality_report.csv", "quality_report_real.csv"}
C_BLACKLIST_NAMES = {"regrabar_archivos_c.csv", "regrabar_archivos_c_real.csv"}
NON_EEG_AUXILIARY_DIRS = {"graficas", "graficos"}
NON_EEG_AUXILIARY_FILENAMES = {"resultados_clasificacion.csv"}
NON_EEG_AUXILIARY_PREFIXES = ("metricas_", "metrics_")
BALANCE_FIELDS = [
    "usuario", "clase", "a_validos", "b_validos", "total_validos",
    "minimo_requerido", "faltantes", "estado", "c_reportados",
    "a_no_disponibles", "b_no_disponibles",
]
REGRABAR_FIELDS = ["usuario", "clase", "grabaciones_validas", "minimo", "faltantes", "accion"]
USER_FIELDS = ["usuario", "total_clases", "clases_completas", "clases_incompletas", "total_grabaciones_faltantes"]
INCIDENCE_FIELDS = ["usuario", "clase", "trial", "archivo", "clasificacion", "estado", "detalle"]


@dataclass(frozen=True)
class QualityRecord:
    archivo: str
    usuario: str
    clase: str
    trial: str
    clasificacion: str
    n_filas: str = ""
    raw: dict[str, str] | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return self.usuario, self.clase, normalize_trial(self.trial)


@dataclass
class Balance:
    a: int = 0
    b: int = 0
    c: int = 0
    a_missing: int = 0
    b_missing: int = 0


def normalize_trial(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except ValueError:
        pass
    return value


def normalized_headers(headers: Iterable[str | None]) -> dict[str, str]:
    return {h.strip().casefold(): h for h in headers if h is not None}


def get_field(row: dict[str, str], header_map: dict[str, str], name: str) -> str:
    key = header_map.get(name.casefold())
    return str(row.get(key, "") or "").strip() if key else ""


def discover_one(root: Path, names: set[str], explicit: str | Path | None, label: str, option: str, required: bool) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser().resolve(strict=True)
        if not p.is_file():
            raise ValueError(f"{label} no es un archivo: {p}")
        return p
    direct = [root / name for name in sorted(names) if (root / name).is_file()]
    candidates = direct
    if not candidates:
        candidates = sorted(p for p in root.rglob("*") if p.is_file() and p.name.casefold() in names)
    if not candidates:
        report_dir = Path(__file__).resolve().parent
        candidates = [report_dir / name for name in sorted(names) if (report_dir / name).is_file()]
    # Multiple quality reports are common in a work tree; never guess which
    # classification snapshot the caller intended.
    unique = sorted(set(p.resolve() for p in candidates))
    if len(unique) > 1:
        raise ValueError(f"Hay varios {label}; especifique --{option} explícitamente: " + ", ".join(map(str, unique)))
    if unique:
        return unique[0]
    if required:
        raise FileNotFoundError(f"No se encontró {label} bajo {root}; indique su ruta explícitamente.")
    return None


def load_quality_report(path: Path) -> list[QualityRecord]:
    required = ("archivo", "usuario", "clase", "trial", "clasificacion")
    records: list[QualityRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"El quality report está vacío o no tiene cabecera: {path}")
        headers = normalized_headers(reader.fieldnames)
        missing = [name for name in required if name not in headers]
        if missing:
            raise ValueError(
                f"El quality report necesita columnas {required}; faltan {missing}. "
                "No se volverán a inferir usuario/clase/trial desde el nombre del archivo."
            )
        for line, row in enumerate(reader, start=2):
            row = {k: (v or "") for k, v in row.items() if k is not None}
            filename = get_field(row, headers, "archivo")
            user = get_field(row, headers, "usuario")
            label = get_field(row, headers, "clase")
            trial = get_field(row, headers, "trial")
            classification = get_field(row, headers, "clasificacion").upper()
            if filename and is_known_non_eeg_auxiliary(filename):
                print(f"AVISO: se omite auxiliar no EEG en {path}, línea {line}: {filename}", file=sys.stderr)
                continue
            if not all((filename, user, label, trial, classification)):
                raise ValueError(f"Metadatos vacíos en {path}, línea {line}; se requiere archivo/usuario/clase/trial/clasificacion.")
            if classification not in {"A", "B", "C"}:
                raise ValueError(f"Clasificación no reconocida {classification!r} en {path}, línea {line}.")
            records.append(QualityRecord(
                archivo=filename, usuario=user, clase=label, trial=trial,
                clasificacion=classification,
                n_filas=get_field(row, headers, "n_filas"), raw=row,
            ))
    return records


def load_c_blacklist(path: Path | None) -> list[QualityRecord]:
    """Read C identities and metadata; the blacklist is authoritative for C."""
    if path is None:
        return []
    result: list[QualityRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            return []
        headers = normalized_headers(reader.fieldnames)
        required = ("archivo", "usuario", "clase", "trial")
        missing = [field for field in required if field not in headers]
        if missing:
            raise ValueError(f"La lista C necesita columnas {required}; faltan {missing}: {path}")
        for line, row in enumerate(reader, start=2):
            row = {k: (v or "") for k, v in row.items() if k is not None}
            filename = get_field(row, headers, "archivo")
            user = get_field(row, headers, "usuario")
            label = get_field(row, headers, "clase")
            trial = get_field(row, headers, "trial")
            if filename and is_known_non_eeg_auxiliary(filename):
                print(f"AVISO: se omite auxiliar no EEG en lista C {path}, línea {line}: {filename}", file=sys.stderr)
                continue
            if not any((filename, user, label, trial)):
                continue
            if not all((filename, user, label, trial)):
                raise ValueError(f"Metadatos incompletos en lista C, línea {line}: {path}")
            result.append(QualityRecord(filename, user, label, trial, "C"))
    return result


def normalize_report_path(filename: str) -> str:
    return PurePosixPath(str(filename).replace("\\", "/")).as_posix()


def is_known_non_eeg_auxiliary(filename: str) -> bool:
    parts = [part.casefold() for part in PurePosixPath(normalize_report_path(filename)).parts]
    if not parts:
        return False
    basename = parts[-1]
    return (
        any(part in NON_EEG_AUXILIARY_DIRS for part in parts[:-1])
        or basename in NON_EEG_AUXILIARY_FILENAMES
        or basename.startswith(NON_EEG_AUXILIARY_PREFIXES)
    )


def merge_c_blacklist(records: list[QualityRecord], c_records: list[QualityRecord]) -> list[QualityRecord]:
    """Override stale A/B labels by C entries; retain blacklist-only history."""
    c_paths = {normalize_report_path(record.archivo) for record in c_records}
    report_paths = {normalize_report_path(record.archivo) for record in records}
    merged = [
        replace(record, clasificacion="C")
        if normalize_report_path(record.archivo) in c_paths else record
        for record in records
    ]
    appended: set[str] = set()
    for record in c_records:
        path_key = normalize_report_path(record.archivo)
        if path_key not in report_paths and path_key not in appended:
            merged.append(record)
            appended.add(path_key)
    return merged


def safe_relative_parts(filename: str) -> tuple[tuple[str, ...] | None, str]:
    if not filename or "\x00" in filename:
        return None, "ruta vacía o contiene NUL"
    win = PureWindowsPath(filename)
    posix = PurePosixPath(filename.replace("\\", "/"))
    if posix.is_absolute() or win.is_absolute() or win.drive:
        return None, "ruta absoluta/unidad no permitida"
    if any(part in (".", "..") for part in posix.parts) or not posix.parts:
        return None, "ruta relativa insegura"
    return tuple(posix.parts), ""


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def exact_candidates(root: Path, filename: str) -> list[Path]:
    """For basename-only report entries, search exact names; no partial match."""
    matches: list[Path] = []
    for current, dirs, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        dirs[:] = [d for d in dirs if not (current_path / d).is_symlink()]
        if filename in files:
            matches.append(current_path / filename)
    return matches


def locate_physical(root: Path, filename: str) -> tuple[Path | None, str]:
    parts, error = safe_relative_parts(filename)
    if parts is None:
        return None, f"RUTA_INSEGURA: {error}"
    if len(parts) == 1:
        matches = exact_candidates(root, parts[0])
        if len(matches) > 1:
            return None, f"AMBIGUO: {len(matches)} archivos con el mismo nombre exacto"
        if not matches:
            return None, "AUSENTE"
        target = matches[0]
    else:
        target = root.joinpath(*parts)
        if not target.exists() and not target.is_symlink():
            return None, "AUSENTE"
    for parent in target.parents:
        if parent == root:
            break
        if parent.is_symlink():
            return None, "SYMLINK_PADRE"
    if target.is_symlink():
        return None, "SYMLINK"
    try:
        resolved = target.resolve(strict=True)
    except FileNotFoundError:
        return None, "AUSENTE"
    except OSError as exc:
        return None, f"ERROR_RUTA: {exc}"
    if not is_inside(resolved, root):
        return None, "FUERA_DE_RAIZ"
    if not resolved.is_file():
        return None, "NO_ES_ARCHIVO"
    return resolved, ""


def verify_csv_integrity(path: Path, expected_rows: str) -> tuple[bool, str]:
    """Check structural readability only; never evaluate EEG quality metrics."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            if not header or not any(cell.strip() for cell in header):
                return False, "CSV_SIN_CABECERA"
            if any(not cell.strip() for cell in header):
                return False, "CABECERA_CON_COLUMNA_VACIA"
            count = 0
            for row in reader:
                if not row or not any(cell.strip() for cell in row):
                    return False, "FILA_VACIA"
                if len(row) != len(header):
                    return False, f"ANCHO_DE_FILA_INCONSISTENTE en fila {count + 2}"
                for column, value in enumerate(row, start=1):
                    try:
                        number = float(value)
                    except ValueError:
                        return False, f"VALOR_NO_NUMERICO en fila {count + 2}, columna {column}"
                    if not math.isfinite(number):
                        return False, f"VALOR_NO_FINITO en fila {count + 2}, columna {column}"
                count += 1
            if count == 0:
                return False, "CSV_SIN_MUESTRAS"
            if expected_rows:
                expected_float = float(expected_rows)
                if not expected_float.is_integer():
                    return False, f"N_FILAS_REPORTADO_NO_ENTERO:{expected_rows}"
                if count != int(expected_float):
                    return False, f"N_FILAS_NO_COINCIDE: reporte={int(expected_float)}, archivo={count}"
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        return False, f"CSV_NO_LEGIBLE: {type(exc).__name__}: {exc}"
    return True, ""


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def split_expected_classes(values: Iterable[str] | None) -> list[str]:
    if not values:
        return []
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in str(value).split(","):
            label = item.strip()
            if label and label not in seen:
                seen.add(label)
                output.append(label)
    return output


def _candidate_root(captures: Path, explicit: str | Path | None, default_root: bool) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser().resolve(strict=True)
        if not path.is_dir():
            raise NotADirectoryError(f"La carpeta indicada no existe o no es directorio: {path}")
        return path
    if default_root:
        return captures
    return None


def run(
    captures: str | Path,
    *,
    quality_report: str | Path | None = None,
    c_blacklist: str | Path | None = None,
    a_valid: str | Path | None = None,
    b_corrected: str | Path | None = None,
    expected_classes: Iterable[str] | None = None,
    minimum: int = 30,
    output_dir: str | Path = ".",
    overwrite: bool = False,
    out: TextIO = sys.stdout,
) -> dict[str, int | str]:
    if minimum < 1:
        raise ValueError("--minimum debe ser un entero positivo.")
    capture_root = Path(captures).expanduser().resolve(strict=True)
    if not capture_root.is_dir():
        raise NotADirectoryError(f"--captures no es una carpeta: {capture_root}")
    report_path = discover_one(capture_root, QUALITY_REPORT_NAMES, quality_report, "quality report", "quality-report", True)
    c_path = discover_one(capture_root, C_BLACKLIST_NAMES, c_blacklist, "lista C", "c-blacklist", False)
    records = load_quality_report(report_path)
    c_records = load_c_blacklist(c_path)
    records = merge_c_blacklist(records, c_records)
    c_history = [(record.usuario, record.clase) for record in c_records]
    # quality_filter now leaves A and overwrites B in the original captures tree.
    # Separate roots remain available for legacy datasets via explicit options.
    a_root = _candidate_root(capture_root, a_valid, True)
    b_root = _candidate_root(capture_root, b_corrected, True)

    users = {r.usuario for r in records}
    users.update(user for user, _ in c_history)
    observed_classes = {
        r.clase for r in records if r.clase and r.clase.strip().upper() != "UNKNOWN"
    }
    observed_classes.update(label for _, label in c_history if label and label.strip().upper() != "UNKNOWN")
    requested_classes = split_expected_classes(expected_classes)
    classes = requested_classes + sorted(c for c in observed_classes if c not in requested_classes)
    # UNKNOWN is quality_filter's invalid-label sentinel, not a class to collect.
    # A caller may include it explicitly if their application truly uses it.
    if not classes:
        raise ValueError("No se detectaron clases; pase --expected-classes explícitamente.")
    if not users:
        raise ValueError("No se detectaron usuarios en los reportes de calidad o C.")

    groups: dict[tuple[str, str, str], list[QualityRecord]] = defaultdict(list)
    for record in records:
        groups[record.key].append(record)
    balances: dict[tuple[str, str], Balance] = {
        (user, label): Balance() for user in users for label in classes
    }
    incidences: list[dict[str, object]] = []
    duplicate_groups = 0

    for key, group in groups.items():
        user, label, trial = key
        classifications = {r.clasificacion for r in group}
        if len(group) > 1:
            duplicate_groups += 1
        if label.strip().upper() == "UNKNOWN" and label not in requested_classes:
            incidences.append({"usuario": user, "clase": label, "trial": trial,
                "archivo": " | ".join(sorted({r.archivo for r in group})),
                "clasificacion": "|".join(sorted(classifications)),
                "estado": "CLASE_UNKNOWN_IGNORADA", "detalle": "UNKNOWN es una etiqueta inválida/sentinel en el reporte; no se trata como clase esperada."})
            continue
        if label not in classes:
            # Not expected and not observed? With observed classes union this
            # is only possible when an explicit policy excludes a special case.
            continue
        balance = balances[(user, label)]
        if len(classifications) != 1:
            incidences.append({"usuario": user, "clase": label, "trial": trial,
                "archivo": " | ".join(sorted({r.archivo for r in group})),
                "clasificacion": "|".join(sorted(classifications)),
                "estado": "CLASIFICACION_CONFLICTIVA_NO_CONTADA", "detalle": "El mismo usuario/clase/trial aparece con etiquetas diferentes."})
            continue
        classification = next(iter(classifications))
        if classification == "C":
            balance.c += 1
            continue
        physical_root = a_root if classification == "A" else b_root
        if physical_root is None:
            reason = "CARPETA_B_CORREGIDOS_NO_INDICADA"
            physical = None
        else:
            # Duplicate quality rows for the same logical trial count once.
            # If paths differ, inspect only exact paths and accept at most one.
            physical = None
            reasons: list[str] = []
            for record in sorted(group, key=lambda r: r.archivo):
                found, loc_reason = locate_physical(physical_root, record.archivo)
                if found is None:
                    reasons.append(loc_reason)
                    continue
                ok, integrity_reason = verify_csv_integrity(found, record.n_filas)
                if ok:
                    physical = found
                    break
                reasons.append(integrity_reason)
            reason = "; ".join(reasons) if reasons else "ARCHIVO_AUSENTE_O_CORRUPTO"
        if physical is not None:
            if classification == "A":
                balance.a += 1
                valid_status = "A_VALIDO_CONTADO"
            else:
                balance.b += 1
                valid_status = "B_CORREGIDO_VALIDO_CONTADO"
            if len(group) > 1:
                incidences.append({"usuario": user, "clase": label, "trial": trial,
                    "archivo": " | ".join(sorted({r.archivo for r in group})),
                    "clasificacion": classification, "estado": "DUPLICADO_TRIAL_NO_DOBLE_CONTADO",
                    "detalle": f"Se contabilizó una sola vez; archivo físico usado: {physical} ({valid_status})."})
        else:
            if classification == "A":
                balance.a_missing += 1
            else:
                balance.b_missing += 1
            incidence_status = f"{classification}_AUSENTE_O_CORRUPTO"
            incidences.append({"usuario": user, "clase": label, "trial": trial,
                "archivo": " | ".join(sorted({r.archivo for r in group})),
                "clasificacion": classification, "estado": incidence_status, "detalle": reason})

    # Add combinations that exist in the C-only history, and also include any
    # user/class metadata present in the quality report even if their C files
    # were already physically deleted.
    for user, label in c_history:
        if (label.strip().upper() != "UNKNOWN" or label in requested_classes) and label not in classes:
            classes.append(label)
            for existing_user in users:
                balances.setdefault((existing_user, label), Balance())
        users.add(user)
        for existing_class in classes:
            balances.setdefault((user, existing_class), Balance())

    balance_rows: list[dict[str, object]] = []
    regrab_rows: list[dict[str, object]] = []
    user_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    for user in sorted(users):
        complete = incomplete = missing_total = 0
        matrix_row: dict[str, object] = {"usuario": user}
        for label in classes:
            b = balances.setdefault((user, label), Balance())
            valid = b.a + b.b
            missing = max(0, minimum - valid)
            status = "OK" if valid >= minimum else "REGRABAR"
            if status == "OK":
                complete += 1
            else:
                incomplete += 1
                missing_total += missing
                regrab_rows.append({
                    "usuario": user, "clase": label, "grabaciones_validas": valid,
                    "minimo": minimum, "faltantes": missing, "accion": f"REGRABAR {missing}",
                })
            balance_rows.append({
                "usuario": user, "clase": label, "a_validos": b.a, "b_validos": b.b,
                "total_validos": valid, "minimo_requerido": minimum,
                "faltantes": missing, "estado": status, "c_reportados": b.c,
                "a_no_disponibles": b.a_missing, "b_no_disponibles": b.b_missing,
            })
            matrix_row[label] = valid
        matrix_row["total_faltantes"] = missing_total
        matrix_rows.append(matrix_row)
        user_rows.append({
            "usuario": user, "total_clases": len(classes), "clases_completas": complete,
            "clases_incompletas": incomplete, "total_grabaciones_faltantes": missing_total,
        })

    destination = Path(output_dir).expanduser().resolve()
    for protected_root in {capture_root, a_root, b_root} - {None}:
        if is_inside(destination, protected_root):
            raise ValueError(f"--output-dir debe estar fuera de las carpetas de EEG para no modificarlas: {destination}")
    output_names = [
        "reporte_balance_dataset.csv", "regrabar_resumen.csv", "regrabar_por_usuario.csv",
        "matriz_usuario_clase.csv", "dataset_balance_incidencias.csv",
    ]
    existing_outputs = [destination / name for name in output_names if (destination / name).exists()]
    if existing_outputs and not overwrite:
        raise FileExistsError("Ya existen reportes de salida; no se sobrescriben. Use otra carpeta o --overwrite: " + ", ".join(map(str, existing_outputs)))
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "reporte_balance_dataset.csv", BALANCE_FIELDS, balance_rows)
    write_csv(destination / "regrabar_resumen.csv", REGRABAR_FIELDS, regrab_rows)
    write_csv(destination / "regrabar_por_usuario.csv", USER_FIELDS, user_rows)
    matrix_fields = ["usuario", *classes, "total_faltantes"]
    write_csv(destination / "matriz_usuario_clase.csv", matrix_fields, matrix_rows)
    write_csv(destination / "dataset_balance_incidencias.csv", INCIDENCE_FIELDS, incidences)

    total_a = sum(b.a for b in balances.values())
    total_b = sum(b.b for b in balances.values())
    total_valid = total_a + total_b
    total_missing = sum(max(0, minimum - (b.a + b.b)) for b in balances.values())
    ok_count = sum(1 for row in balance_rows if row["estado"] == "OK")
    incomplete_count = len(balance_rows) - ok_count
    print("=" * 54, file=out)
    print("BALANCE DEL DATASET EEG", file=out)
    print("=" * 54, file=out)
    print(f"Mínimo requerido por usuario/clase: {minimum}", file=out)
    print(f"Usuarios analizados: {len(users)}", file=out)
    print(f"Clases por usuario: {len(classes)}", file=out)
    print(f"Combinaciones usuario-clase: {len(balance_rows)}", file=out)
    print(f"  OK:          {ok_count}", file=out)
    print(f"  REGRABAR:    {incomplete_count}", file=out)
    print(f"A válidos físicamente: {total_a}", file=out)
    print(f"B corregidos válidos físicamente: {total_b}", file=out)
    print(f"Total de grabaciones válidas: {total_valid}", file=out)
    print(f"Total de grabaciones faltantes: {total_missing}", file=out)
    print(f"Trials repetidos en quality report (no doble contados): {duplicate_groups}", file=out)
    print("\n" + "=" * 54, file=out)
    print("GRABACIONES QUE DEBEN REHACERSE", file=out)
    print("=" * 54, file=out)
    any_missing = False
    for user in sorted(users):
        rows = [r for r in regrab_rows if r["usuario"] == user]
        if not rows:
            continue
        any_missing = True
        subtotal = 0
        print(f"\n{user}:", file=out)
        for row in rows:
            missing = int(row["faltantes"])
            subtotal += missing
            print(f"  {row['clase']} → faltan {missing}", file=out)
        print(f"Total {user}: {subtotal}", file=out)
    if not any_missing:
        print("No hay combinaciones por debajo del mínimo.", file=out)
    print("\nReportes escritos en:", destination, file=out)
    return {
        "usuarios": len(users), "clases": len(classes), "combinaciones": len(balance_rows),
        "ok": ok_count, "regrabar": incomplete_count, "a_validos": total_a,
        "b_validos": total_b, "total_validos": total_valid,
        "total_faltantes": total_missing, "duplicados": duplicate_groups,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cuenta A físicamente disponibles y B corregidos por usuario/clase; nunca reanaliza calidad EEG."
    )
    parser.add_argument("--captures", required=True, help="Raíz de las señales A/B disponibles; por defecto ambas clases se buscan aquí.")
    parser.add_argument("--quality-report", help="CSV de quality_filter.py (si se omite, se busca bajo --captures y luego junto al checker).")
    parser.add_argument("--c-blacklist", help="Lista C para conservar usuarios/clases con C eliminados; se busca bajo --captures y junto al checker.")
    parser.add_argument("--a-valid", help="Raíz explícita de A válidos; predeterminado: --captures.")
    parser.add_argument("--b-corrected", help="Raíz explícita de B corregidos si están separados; predeterminado: --captures.")
    parser.add_argument("--expected-classes", nargs="+", help="Clases esperadas, separadas por espacios o comas; incluye clases sin archivos.")
    parser.add_argument("--minimum", type=int, default=30, help="Mínimo por usuario/clase (predeterminado: 30).")
    parser.add_argument("--output-dir", default=None, help="Carpeta para los CSV de salida (predeterminado: carpeta chequeo_calidad junto al checker).")
    parser.add_argument("--overwrite", action="store_true", help="Permite reemplazar reportes previos en --output-dir; no modifica capturas.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(
            args.captures, quality_report=args.quality_report, c_blacklist=args.c_blacklist,
            a_valid=args.a_valid, b_corrected=args.b_corrected,
            expected_classes=args.expected_classes, minimum=args.minimum,
            output_dir=args.output_dir or str(Path(__file__).resolve().parent), overwrite=args.overwrite,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
