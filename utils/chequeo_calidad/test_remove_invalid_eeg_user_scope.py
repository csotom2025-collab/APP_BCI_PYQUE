import csv
import io

import remove_invalid_eeg as remover


def _write_blacklist(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["archivo", "usuario", "clase", "trial"])
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_run_only_previews_c_files_for_selected_user(tmp_path):
    captures = tmp_path / "captures"
    user_file = captures / "UserOne" / "Letters" / "same.csv"
    other_file = captures / "OtherUser" / "Letters" / "same.csv"
    user_file.parent.mkdir(parents=True)
    other_file.parent.mkdir(parents=True)
    user_file.write_text("signal\n1\n", encoding="utf-8")
    other_file.write_text("signal\n2\n", encoding="utf-8")
    blacklist = tmp_path / "regrabar_archivos_C.csv"
    _write_blacklist(
        blacklist,
        [
            {"archivo": "same.csv", "usuario": "UserOne", "clase": "A", "trial": "0"},
            {"archivo": "same.csv", "usuario": "OtherUser", "clase": "B", "trial": "0"},
        ],
    )

    result = remover.run(
        captures,
        blacklist,
        usuario="UserOne",
        report_dir=tmp_path / "reports",
        out=io.StringIO(),
    )

    assert result["usuario"] == "UserOne"
    assert result["en_lista_negra"] == 1
    assert result["encontrados"] == 1
    assert user_file.exists()
    assert other_file.exists()
    user_reports = tmp_path / "reports" / "UserOne"
    assert _read_csv(user_reports / "archivos_C_no_encontrados.csv") == []
    remover.run(
        captures,
        blacklist,
        usuario="UserOne",
        report_dir=tmp_path / "reports",
        out=io.StringIO(),
    )
    assert _read_csv(user_reports / "archivos_C_no_encontrados.csv") == []


def test_cli_takes_user_as_only_required_argument():
    args = remover.build_parser().parse_args(["UserOne"])

    assert args.usuario == "UserOne"
    assert args.captures == str(remover.default_captures_dir())


def test_delete_never_touches_another_users_matching_filename(tmp_path):
    captures = tmp_path / "captures"
    user_file = captures / "UserOne" / "Letters" / "same.csv"
    other_file = captures / "OtherUser" / "Letters" / "same.csv"
    user_file.parent.mkdir(parents=True)
    other_file.parent.mkdir(parents=True)
    user_file.write_text("signal\n1\n", encoding="utf-8")
    other_file.write_text("signal\n2\n", encoding="utf-8")
    blacklist = tmp_path / "regrabar_archivos_C.csv"
    _write_blacklist(
        blacklist,
        [
            {"archivo": "same.csv", "usuario": "UserOne", "clase": "A", "trial": "0"},
            {"archivo": "same.csv", "usuario": "OtherUser", "clase": "B", "trial": "0"},
        ],
    )

    result = remover.run(
        captures,
        blacklist,
        usuario="UserOne",
        delete=True,
        yes=True,
        report_dir=tmp_path / "reports",
        out=io.StringIO(),
    )

    assert result["eliminados"] == 1
    assert not user_file.exists()
    assert other_file.exists()
    remover.run(
        captures,
        blacklist,
        usuario="UserOne",
        delete=True,
        yes=True,
        report_dir=tmp_path / "reports",
        out=io.StringIO(),
    )
    user_reports = tmp_path / "reports" / "UserOne"
    assert _read_csv(user_reports / "archivos_eliminados.csv") == []
