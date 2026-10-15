import csv
import io
from pathlib import Path

import dataset_balance_checker as checker


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_run_reports_only_requested_user_and_checks_only_their_folder(tmp_path: Path) -> None:
    captures = tmp_path / "captures"
    alice = captures / "Alice"
    bob = captures / "Bob"
    write_csv(alice / "class_a.csv", ["ch1"], [{"ch1": "1"}])
    write_csv(bob / "shared.csv", ["ch1"], [{"ch1": "2"}])
    user_output = tmp_path / "out" / "Alice"
    write_csv(
        user_output / "quality_report.csv",
        ["archivo", "usuario", "clase", "trial", "clasificacion", "n_filas"],
        [
            {"archivo": "Alice/class_a.csv", "usuario": "Alice", "clase": "A", "trial": "0", "clasificacion": "A", "n_filas": "1"},
            {"archivo": "shared.csv", "usuario": "Alice", "clase": "B", "trial": "0", "clasificacion": "A", "n_filas": "1"},
            {"archivo": "Bob/shared.csv", "usuario": "Bob", "clase": "EXTRA", "trial": "0", "clasificacion": "A", "n_filas": "1"},
        ],
    )
    write_csv(
        user_output / "regrabar_archivos_C.csv",
        ["archivo", "usuario", "clase", "trial"],
        [{"archivo": "Bob/deleted.csv", "usuario": "Bob", "clase": "C_ONLY", "trial": "1"}],
    )

    result = checker.run(
        captures,
        usuario="Alice",
        minimum=1,
        output_dir=tmp_path / "out",
        out=io.StringIO(),
    )

    balance_rows = read_csv(user_output / "reporte_balance_dataset.csv")
    incidence_rows = read_csv(user_output / "dataset_balance_incidencias.csv")
    assert result["usuarios"] == 1
    assert {row["usuario"] for row in balance_rows} == {"Alice"}
    assert {row["clase"] for row in balance_rows} == {"A", "B"}
    assert next(row for row in balance_rows if row["clase"] == "A")["a_validos"] == "1"
    assert next(row for row in balance_rows if row["clase"] == "B")["a_no_disponibles"] == "1"
    assert all(row["usuario"] == "Alice" for row in incidence_rows)
    assert not any(row["clase"] == "EXTRA" or row["clase"] == "C_ONLY" for row in balance_rows)

    checker.run(
        captures,
        usuario="Alice",
        minimum=1,
        output_dir=tmp_path / "out",
        out=io.StringIO(),
    )
    assert (user_output / "reporte_balance_dataset.csv").is_file()


def test_cli_accepts_username_without_captures_argument() -> None:
    args = checker.build_parser().parse_args(["Alice"])

    assert args.usuario == "Alice"
    assert args.captures == str(checker.default_captures_dir())
