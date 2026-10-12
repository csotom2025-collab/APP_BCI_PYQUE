#!/usr/bin/env python3
"""Safely remove only EEG files explicitly listed in quality_filter's C blacklist.

Deletion is never performed unless --delete is supplied. The module is standard
library only; matching, preview and deletion logic are intentionally independent
from quality_filter.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable, Iterable, TextIO

EXPECTED_BLACKLIST_NAMES = {
    "regrabar_archivos_c.csv",
    "regrabar_archivos_c_real.csv",
}
NON_EEG_AUXILIARY_DIRS = {"graficas", "graficos"}
NON_EEG_AUXILIARY_FILENAMES = {"resultados_clasificacion.csv"}
NON_EEG_AUXILIARY_PREFIXES = ("metricas_", "metrics_")
FILE_COLUMNS = ("archivo", "file", "filename", "nombre_archivo")
CLASS_COLUMNS = ("clasificacion", "classification", "estado_clasificacion")
ELIMINATED_FIELDS = [
    "fecha_hora_utc", "archivo", "ruta_original", "usuario", "clase", "trial",
    "motivo_C", "canales_afectados", "parametros_C", "tamano_bytes",
    "sha256_antes_de_eliminar", "estado_eliminacion", "ruta_backup", "detalle",
]
NOT_FOUND_FIELDS = [
    "archivo", "usuario", "clase", "trial", "motivo", "estado", "detalle",
]


@dataclass
class BlacklistEntry:
    filename: str
    user: str = ""
    task_class: str = ""
    trial: str = ""
    reason: str = ""
    affected_channels: str = ""
    parameters: str = ""
    explicit_classification: str = ""
    raw: dict[str, str] = field(default_factory=dict)


@dataclass
class Candidate:
    entry: BlacklistEntry
    status: str
    detail: str = ""
    path: Path | None = None
    relative_path: str = ""
    size: int | None = None
    sha256: str = ""
    identity: tuple[int, int, int, int] | None = None


def _normalise_headers(headers: Iterable[str | None]) -> dict[str, str]:
    """Return lower-case, trimmed header -> original header mapping."""
    result: dict[str, str] = {}
    for header in headers:
        if header is not None:
            result[header.strip().casefold()] = header
    return result


def _value(row: dict[str, str], headers: dict[str, str], names: Iterable[str]) -> str:
    for name in names:
        original = headers.get(name.casefold())
        if original is not None:
            value = row.get(original, "")
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""


def _metadata(entry: BlacklistEntry) -> dict[str, str]:
    return {
        "archivo": entry.filename,
        "usuario": entry.user,
        "clase": entry.task_class,
        "trial": entry.trial,
        "motivo": entry.reason,
    }


def load_blacklist(path: Path) -> tuple[list[BlacklistEntry], int]:
    """Read exact target names from the CSV and de-duplicate identical rows.

    A CSV without a classification column is trusted as C-only only when its
    basename is one of quality_filter's known C-list output names. Otherwise,
    each row must explicitly contain classification C or accion=REGRABAR.
    """
    if not path.is_file():
        raise FileNotFoundError(f"No existe el CSV de lista negra: {path}")
    entries: list[BlacklistEntry] = []
    seen: set[str] = set()
    duplicates = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            return [], 0
        headers = _normalise_headers(reader.fieldnames)
        filename_column = next((headers[c] for c in FILE_COLUMNS if c in headers), None)
        if filename_column is None:
            raise ValueError(
                "El CSV no contiene una columna de nombre de archivo; se esperaba "
                "'archivo' (o 'file'/'filename')."
            )
        trusted_c_list = path.name.casefold() in EXPECTED_BLACKLIST_NAMES
        for row in reader:
            row = {k: (v or "") for k, v in row.items() if k is not None}
            filename = str(row.get(filename_column, "")).strip()
            if not filename:
                # Keep invalid rows visible in the preview/report.
                filename = ""
            key = filename
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            classification = _value(row, headers, CLASS_COLUMNS).upper()
            action = _value(row, headers, ("accion", "action")).upper()
            if classification:
                explicit_c = "C" if classification == "C" else classification
            elif action == "REGRABAR" or action.startswith("REGRABAR:"):
                explicit_c = "C"
            elif trusted_c_list:
                explicit_c = "C"
            else:
                explicit_c = ""
            # Preserve parameter-like fields from the C report instead of
            # re-running or duplicating the classification logic.
            param_fields = {}
            for key_name in ("parametros", "parameters", "config", "metricas"):
                v = _value(row, headers, (key_name,))
                if v:
                    param_fields[key_name] = v
            parameters = json.dumps(param_fields, ensure_ascii=False, sort_keys=True) if param_fields else ""
            reason_parts = []
            for key_name in ("codigos_C", "motivo", "razon", "reason"):
                v = _value(row, headers, (key_name,))
                if v and v not in reason_parts:
                    reason_parts.append(v)
            entries.append(BlacklistEntry(
                filename=filename,
                user=_value(row, headers, ("usuario", "user")),
                task_class=_value(row, headers, ("clase", "task_class")),
                trial=_value(row, headers, ("trial", "ensayo")),
                reason=" | ".join(reason_parts),
                affected_channels=_value(row, headers, ("canales_afectados", "affected_channels")),
                parameters=parameters,
                explicit_classification=explicit_c,
                raw=row,
            ))
    return entries, duplicates


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _parse_relative_name(name: str) -> tuple[PurePosixPath | None, str]:
    """Validate an untrusted CSV path without resolving it against the host."""
    if not name:
        return None, "El campo archivo está vacío."
    if "\x00" in name:
        return None, "La ruta contiene un byte NUL."
    # Reject POSIX absolute, UNC and drive-qualified paths, including Windows
    # paths when the program is run on POSIX.
    win = PureWindowsPath(name)
    posix = PurePosixPath(name.replace("\\", "/"))
    if posix.is_absolute() or win.is_absolute() or win.drive:
        return None, "RUTA_INSEGURA_NO_ELIMINADA: se rechazó una ruta absoluta o con unidad."
    if any(part in ("..", ".") for part in posix.parts):
        return None, "RUTA_INSEGURA_NO_ELIMINADA: se rechazó un componente . o ..."
    if not posix.parts:
        return None, "El campo archivo no identifica un archivo."
    return posix, ""


def _walk_exact_basename(root: Path, basename: str) -> tuple[list[Path], list[Path]]:
    """Return exact basename file matches and same-name directory matches.

    Directory symlinks are not followed. Symlink files are returned for a later
    explicit rejection, never followed or deleted.
    """
    files: list[Path] = []
    directories: list[Path] = []
    for current, dirs, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        symlink_dirs = [d for d in dirs if (current_path / d).is_symlink()]
        dirs[:] = [d for d in dirs if d not in symlink_dirs]
        if basename in filenames:
            files.append(current_path / basename)
        if basename in dirs or basename in symlink_dirs:
            directories.append(current_path / basename)
    return files, directories


def _candidate_for(entry: BlacklistEntry, root: Path, blacklist: Path, extensions: set[str]) -> Candidate:
    if entry.explicit_classification != "C":
        return Candidate(entry, "NO_ES_C", "El CSV no identifica explícitamente C ni es una lista C reconocida.")
    relative, error = _parse_relative_name(entry.filename)
    if relative is None:
        status = "RUTA_INSEGURA_NO_ELIMINADA" if "RUTA_INSEGURA" in error else "ARCHIVO_INVALIDO_NO_ELIMINADO"
        return Candidate(entry, status, error)
    parts = [part.casefold() for part in relative.parts]
    basename = parts[-1] if parts else ""
    if (
        any(part in NON_EEG_AUXILIARY_DIRS for part in parts[:-1])
        or basename in NON_EEG_AUXILIARY_FILENAMES
        or basename.startswith(NON_EEG_AUXILIARY_PREFIXES)
    ):
        return Candidate(
            entry, "ARTEFACTO_NO_EEG_NO_ELIMINADO",
            "La ruta coincide con un CSV auxiliar conocido (gráficas/métricas/resultados), no una captura EEG.",
        )

    if len(relative.parts) == 1:
        file_matches, dir_matches = _walk_exact_basename(root, relative.name)
        # More than one exact basename (or a same-name directory) is not
        # uniquely identifiable. Never pick one using user/class/trial.
        if len(file_matches) > 1 or (file_matches and dir_matches):
            return Candidate(entry, "AMBIGUO_NO_ELIMINADO", f"Hay {len(file_matches)} archivos y {len(dir_matches)} carpetas con ese nombre exacto.")
        if not file_matches:
            if dir_matches:
                return Candidate(entry, "CARPETA_NO_ELIMINADA", "Sólo se encontró una carpeta con ese nombre; no se eliminan carpetas.")
            return Candidate(entry, "NO_ENCONTRADO", "No se encontró un archivo con ese nombre exacto.")
        target = file_matches[0]
    else:
        target = root.joinpath(*relative.parts)
        if not target.exists() and not target.is_symlink():
            return Candidate(entry, "NO_ENCONTRADO", "No se encontró esa ruta relativa exacta; no se buscaron nombres alternativos.")

    for parent in target.parents:
        if parent == root:
            break
        if parent.is_symlink():
            return Candidate(entry, "RUTA_INSEGURA_NO_ELIMINADA", "Un componente padre de la ruta es un enlace simbólico.")

    try:
        # resolve(strict=False) detects symlinked parent components even when
        # the final path does not exist. The target itself is also rejected if
        # it is a symlink so unlink never follows an untrusted link.
        resolved = target.resolve(strict=True)
    except FileNotFoundError:
        return Candidate(entry, "NO_ENCONTRADO", "El archivo dejó de existir durante la búsqueda.")
    except OSError as exc:
        return Candidate(entry, "ERROR_VERIFICACION_NO_ELIMINADO", f"No se pudo resolver la ruta: {exc}")
    if target.is_symlink():
        return Candidate(entry, "RUTA_INSEGURA_NO_ELIMINADA", "El objetivo es un enlace simbólico.")
    if not _is_inside(resolved, root):
        return Candidate(entry, "RUTA_INSEGURA_NO_ELIMINADA", "La ruta resuelta queda fuera de --captures.")
    if resolved == blacklist:
        return Candidate(entry, "ARCHIVO_LISTA_NEGRA_NO_ELIMINADO", "Se impide borrar el propio CSV de lista negra.")
    if not resolved.is_file():
        return Candidate(entry, "CARPETA_NO_ELIMINADA", "El objetivo existe pero no es un archivo regular.")
    if resolved.suffix.casefold() not in extensions:
        return Candidate(entry, "EXTENSION_NO_PERMITIDA", f"Extensión {resolved.suffix!r} no incluida en {sorted(extensions)}.")
    try:
        stat = resolved.stat()
        digest = _sha256(resolved)
    except OSError as exc:
        return Candidate(entry, "NO_LEIBLE_NO_ELIMINADO", f"No se pudo leer o calcular SHA-256: {exc}")
    return Candidate(
        entry, "ENCONTRADO", "Coincidencia exacta validada.", resolved,
        resolved.relative_to(root).as_posix(), stat.st_size, digest,
        (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns),
    )


def _append_csv(path: Path, fields: list[str], rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as stream:
            existing_header = next(csv.reader(stream), [])
        if existing_header != fields:
            raise ValueError(f"El reporte existente tiene un esquema distinto; no se sobrescribe: {path}")
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _not_found_row(candidate: Candidate) -> dict[str, str]:
    row = _metadata(candidate.entry)
    row.update({"estado": candidate.status, "detalle": candidate.detail})
    return row


def _backup_destination(candidate: Candidate, backup_root: Path, captures: Path) -> Path:
    assert candidate.path is not None
    backup_root_resolved = backup_root.resolve()
    if _is_inside(backup_root_resolved, captures):
        raise ValueError("La carpeta de respaldo debe estar fuera de --captures.")
    dest = backup_root_resolved.joinpath(*PurePosixPath(candidate.relative_path).parts)
    if not _is_inside(dest.resolve(strict=False), backup_root_resolved):
        raise ValueError("La ruta de respaldo saldría de la carpeta de respaldo.")
    # Refuse symlinked path components and never overwrite prior backups.
    cur = backup_root_resolved
    rel = dest.relative_to(backup_root_resolved)
    for part in rel.parts[:-1]:
        cur = cur / part
        if cur.exists() and cur.is_symlink():
            raise ValueError("La ruta de respaldo contiene un enlace simbólico.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(f"Ya existe un respaldo; no se sobrescribe: {dest}")
    return dest


def _confirm_delete(prompt: Callable[[str], str] | None, out: TextIO) -> bool:
    if prompt is None:
        prompt = input
    try:
        answer = prompt("¿Deseas continuar? [y/N]: ")
    except (EOFError, KeyboardInterrupt):
        print("Confirmación no recibida; no se eliminará ningún archivo.", file=out)
        return False
    return str(answer).strip().casefold() in {"y", "yes", "s", "si", "sí"}


def run(
    captures: str | Path,
    blacklist: str | Path,
    *,
    delete: bool = False,
    dry_run: bool = False,
    yes: bool = False,
    backup: bool = False,
    backup_dir: str | Path | None = None,
    report_dir: str | Path = ".",
    extensions: Iterable[str] = (".csv",),
    confirm: Callable[[str], str] | None = None,
    out: TextIO = sys.stdout,
) -> dict[str, int | str]:
    """Preview or delete eligible C-list files. Returns a concise count summary."""
    if delete and dry_run:
        raise ValueError("--delete y --dry-run son opciones incompatibles.")
    if yes and not delete:
        raise ValueError("--yes sólo puede usarse junto con --delete.")
    if backup and not delete:
        raise ValueError("--backup sólo puede usarse junto con --delete.")
    captures_path = Path(captures).expanduser().resolve(strict=True)
    if not captures_path.is_dir():
        raise NotADirectoryError(f"--captures no es una carpeta: {captures_path}")
    blacklist_path = Path(blacklist).expanduser().resolve(strict=True)
    report_path = Path(report_dir).expanduser().resolve()
    ext_set = {e.casefold() if e.startswith(".") else f".{e.casefold()}" for e in extensions}
    if not ext_set:
        raise ValueError("Se debe permitir al menos una extensión.")
    entries, duplicates = load_blacklist(blacklist_path)
    candidates = [_candidate_for(e, captures_path, blacklist_path, ext_set) for e in entries]
    found = [c for c in candidates if c.status == "ENCONTRADO"]
    not_found = [c for c in candidates if c.status == "NO_ENCONTRADO"]
    ambiguous = [c for c in candidates if c.status == "AMBIGUO_NO_ELIMINADO"]
    blocked = [c for c in candidates if c.status not in {"ENCONTRADO", "NO_ENCONTRADO", "AMBIGUO_NO_ELIMINADO"}]

    mode = "ELIMINACIÓN REAL" if delete else "DRY-RUN"
    print(f"\n{mode}", file=out)
    print("=" * 40, file=out)
    print(f"Filas únicas en lista negra: {len(entries)} (duplicadas omitidas: {duplicates})", file=out)
    print(f"Archivos encontrados y verificados: {len(found)}", file=out)
    print(f"No encontrados: {len(not_found)}", file=out)
    print(f"Ambiguos: {len(ambiguous)}", file=out)
    print(f"Bloqueados por seguridad/validación: {len(blocked)}", file=out)
    print("\nCoincidencias exactas encontradas (preview):", file=out)
    if found:
        for idx, item in enumerate(found, 1):
            print(f"{idx}. {item.relative_path}  SHA-256={item.sha256}", file=out)
    else:
        print("(ninguna)", file=out)
    if not_found:
        print("\nNo encontrado:", file=out)
        for item in not_found:
            print(f"- {item.entry.filename}", file=out)
    if ambiguous:
        print("\nAMBIGUO_NO_ELIMINADO:", file=out)
        for item in ambiguous:
            print(f"- {item.entry.filename}: {item.detail}", file=out)
    if blocked:
        print("\nNo eliminable por validación:", file=out)
        for item in blocked:
            print(f"- {item.entry.filename}: {item.status} — {item.detail}", file=out)

    # Keep an append-only record of absent/ambiguous/unsafe blacklist rows.
    no_found_rows = [_not_found_row(c) for c in candidates if c.status != "ENCONTRADO"]
    _append_csv(report_path / "archivos_C_no_encontrados.csv", NOT_FOUND_FIELDS, no_found_rows)

    deleted_count = 0
    failed_count = 0
    backed_up_count = 0
    if delete and found:
        if yes:
            approved = True
        else:
            print(f"\nADVERTENCIA: se eliminarán permanentemente {len(found)} archivos encontrados.", file=out)
            approved = _confirm_delete(confirm, out)
        if not approved:
            print("ELIMINACIÓN CANCELADA. No se borró ningún archivo.", file=out)
            return {
                "modo": "ELIMINACIÓN REAL CANCELADA", "en_lista_negra": len(entries),
                "encontrados": len(found), "eliminados": 0, "no_encontrados": len(not_found),
                "ambiguos": len(ambiguous), "fallidos": 0, "respaldados": 0,
            }

        backup_root = Path(backup_dir).expanduser().resolve() if backup_dir else captures_path.parent / "backup_eliminados"
        delete_rows: list[dict[str, object]] = []
        for candidate in found:
            assert candidate.path is not None
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            base_row: dict[str, object] = {
                "fecha_hora_utc": now, "archivo": candidate.entry.filename,
                "ruta_original": str(candidate.path), "usuario": candidate.entry.user,
                "clase": candidate.entry.task_class, "trial": candidate.entry.trial,
                "motivo_C": candidate.entry.reason,
                "canales_afectados": candidate.entry.affected_channels,
                "parametros_C": candidate.entry.parameters,
                "tamano_bytes": candidate.size or 0, "sha256_antes_de_eliminar": candidate.sha256,
                "estado_eliminacion": "", "ruta_backup": "", "detalle": "",
            }
            try:
                # Revalidate containment, type, extension, identity and content
                # immediately before any destructive operation.
                current = candidate.path
                if current.is_symlink():
                    raise RuntimeError("El objetivo se convirtió en enlace simbólico tras el preview.")
                resolved = current.resolve(strict=True)
                if not _is_inside(resolved, captures_path) or not resolved.is_file():
                    raise RuntimeError("El objetivo ya no es un archivo regular dentro de --captures.")
                stat = resolved.stat()
                identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
                if identity != candidate.identity or _sha256(resolved) != candidate.sha256:
                    raise RuntimeError("El archivo cambió después del preview; no se elimina.")
                if resolved.suffix.casefold() not in ext_set:
                    raise RuntimeError("La extensión cambió o ya no está permitida.")

                if backup:
                    destination = _backup_destination(candidate, backup_root, captures_path)
                    shutil.copy2(resolved, destination)
                    if destination.stat().st_size != candidate.size or _sha256(destination) != candidate.sha256:
                        try:
                            destination.unlink(missing_ok=True)
                        finally:
                            raise IOError("El respaldo no coincide con el original; original conservado.")
                    backed_up_count += 1
                    base_row["ruta_backup"] = str(destination)
                    # Check again after the copy, before unlinking the source.
                    if _sha256(resolved) != candidate.sha256:
                        raise RuntimeError("El original cambió durante el respaldo; no se elimina.")
                resolved.unlink()
                base_row["estado_eliminacion"] = "ELIMINADO"
                deleted_count += 1
            except Exception as exc:  # per-file failure must never stop safety checks for other rows
                base_row["estado_eliminacion"] = "NO_ELIMINADO_ERROR"
                base_row["detalle"] = f"{type(exc).__name__}: {exc}"
                failed_count += 1
            delete_rows.append(base_row)
        _append_csv(report_path / "archivos_eliminados.csv", ELIMINATED_FIELDS, delete_rows)
    elif delete:
        # A real run with no candidates still creates the requested header-only log.
        _append_csv(report_path / "archivos_eliminados.csv", ELIMINATED_FIELDS, [])

    print("\n" + "=" * 40, file=out)
    print("RESUMEN DE ELIMINACIÓN", file=out)
    print("=" * 40, file=out)
    print(f"Modo: {mode}", file=out)
    print(f"Archivos en lista negra:       {len(entries)}", file=out)
    print(f"Archivos encontrados:          {len(found)}", file=out)
    print(f"Archivos eliminados:           {deleted_count}", file=out)
    print(f"No encontrados:                {len(not_found)}", file=out)
    print(f"Ambiguos:                      {len(ambiguous)}", file=out)
    print(f"No eliminados por error:       {failed_count}", file=out)
    print(f"Archivos respaldados:          {backed_up_count}", file=out)
    if delete:
        print(f"Reporte de eliminaciones:      {report_path / 'archivos_eliminados.csv'}", file=out)
    else:
        print("Reporte de eliminaciones:      no generado (DRY-RUN)", file=out)
    print(f"Reporte de no encontrados:     {report_path / 'archivos_C_no_encontrados.csv'}", file=out)
    return {
        "modo": mode, "en_lista_negra": len(entries), "encontrados": len(found),
        "eliminados": deleted_count, "no_encontrados": len(not_found),
        "ambiguos": len(ambiguous), "fallidos": failed_count, "respaldados": backed_up_count,
        "bloqueados": len(blocked), "duplicados": duplicates,
    }


def _positive_extensions(values: list[str]) -> list[str]:
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Localiza y elimina únicamente EEG enumerados en una lista C de quality_filter.py."
    )
    parser.add_argument("--captures", required=True, help="Carpeta raíz de las capturas EEG.")
    parser.add_argument("--blacklist", help="CSV C generado por quality_filter.py (predeterminado: regrabar_archivos_C.csv junto al script).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Sólo preview. Es el comportamiento predeterminado.")
    mode.add_argument("--delete", action="store_true", help="Solicita confirmación y habilita eliminación física.")
    parser.add_argument("--yes", action="store_true", help="Omite la pregunta interactiva; requiere --delete.")
    parser.add_argument("--backup", action="store_true", help="Respalda y verifica cada archivo antes de eliminarlo.")
    parser.add_argument("--backup-dir", help="Directorio de respaldo; por defecto, sibling backup_eliminados/ fuera de captures.")
    parser.add_argument("--report-dir", default=None, help="Directorio para los CSV de reportes (predeterminado: carpeta chequeo_calidad junto al script).")
    parser.add_argument("--extensions", nargs="+", default=[".csv"], help="Extensiones EEG permitidas (predeterminado: .csv).")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.yes and not args.delete:
        parser.error("--yes sólo puede usarse junto con --delete.")
    blacklist = args.blacklist or str(Path(__file__).resolve().parent / "regrabar_archivos_C.csv")
    report_dir = args.report_dir or str(Path(__file__).resolve().parent)
    try:
        run(
            args.captures, blacklist, delete=args.delete, dry_run=args.dry_run,
            yes=args.yes, backup=args.backup, backup_dir=args.backup_dir,
            report_dir=report_dir, extensions=_positive_extensions(args.extensions),
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
