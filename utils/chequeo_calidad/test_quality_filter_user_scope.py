import hashlib
import json

import numpy as np
import pandas as pd

import dataset_balance_checker as balance_checker
import quality_filter as checker
import remove_invalid_eeg as invalid_eeg_remover


def _write_signal(path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    values = np.array(
        [checker.DC_REF_REAL_UV[channel] for channel in checker.CANALES]
    ) + rng.normal(0, 8, size=(256, len(checker.CANALES)))
    frame = pd.DataFrame(values, columns=checker.CANALES)
    frame.insert(0, "Tm", 27850 + np.arange(len(frame)) * 7.8125)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_cli_analyzes_and_reports_only_requested_user(tmp_path) -> None:
    captures = tmp_path / "captures"
    requested_file = captures / "UserOne" / "Letters" / "UserOne_A_0.csv"
    other_file = captures / "OtherUser" / "Letters" / "OtherUser_B_0.csv"
    _write_signal(requested_file, 1)
    _write_signal(other_file, 2)
    original_other_hash = hashlib.sha256(other_file.read_bytes()).hexdigest()
    output = tmp_path / "reports"

    assert checker.main(
        ["UserOne", "--captures", str(captures), "--out", str(output), "--silencioso"]
    ) == 0
    user_output = output / "UserOne"
    report = pd.read_csv(user_output / "quality_report.csv")
    summary = json.loads((user_output / "resumen.json").read_text(encoding="utf-8"))
    assert set(report["usuario"]) == {"UserOne"}
    assert summary["usuario"] == "UserOne"
    assert summary["archivos"] == 1
    assert hashlib.sha256(other_file.read_bytes()).hexdigest() == original_other_hash
    assert checker.main(
        ["UserOne", "--captures", str(captures), "--out", str(output), "--silencioso"]
    ) == 0


def test_all_tools_use_the_same_user_report_folder(tmp_path) -> None:
    captures = tmp_path / "captures"
    requested_file = captures / "UserOne" / "Letters" / "UserOne_A_0.csv"
    _write_signal(requested_file, 9)
    reports_root = tmp_path / "reports"

    assert checker.main(
        ["UserOne", "--captures", str(captures), "--out", str(reports_root), "--silencioso"]
    ) == 0
    assert balance_checker.main(
        ["UserOne", "--captures", str(captures), "--output-dir", str(reports_root), "--minimum", "1"]
    ) == 0
    assert invalid_eeg_remover.main(
        ["UserOne", "--captures", str(captures), "--report-dir", str(reports_root)]
    ) == 0

    user_reports = reports_root / "UserOne"
    assert (user_reports / "quality_report.csv").is_file()
    assert (user_reports / "reporte_balance_dataset.csv").is_file()
    assert (user_reports / "archivos_C_no_encontrados.csv").is_file()
