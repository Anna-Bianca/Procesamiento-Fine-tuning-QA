"""Descarga y analiza las filas en espanol de INCLUDE-base-44.

Salidas:
    Bases/0 - Bases crudo/include-base-44/
        datos/<split>.parquet
        analisis/resumen_columnas.csv
        analisis/opciones_<split>_<columna>.csv
        analisis/informe.json

Uso:
    python "Procesamiento/0 - Descarga y analisis/include-base-44/01_descargar_include_base_44_es.py"
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import ClassLabel, DatasetDict, Value, load_dataset


DATASET_ID = "CohereLabs/include-base-44"
CONFIG_NAME = "Spanish"
LANGUAGE_COLUMN = "language"
LANGUAGE_VALUE = "Spanish"
DEFAULT_MAX_OPTIONS = 100
DEFAULT_SAMPLE_SIZE = 5


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    default_output = project_root / "Bases" / "0 - Bases crudo" / "include-base-44"

    parser = argparse.ArgumentParser(
        description="Descarga INCLUDE-base-44 y conserva solamente language=Spanish."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Carpeta de salida (por defecto: {default_output})",
    )
    parser.add_argument(
        "--max-options",
        type=int,
        default=DEFAULT_MAX_OPTIONS,
        help=(
            "Maximo de valores unicos para exportar todas las opciones de una "
            f"columna (por defecto: {DEFAULT_MAX_OPTIONS})."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Cantidad de ejemplos por columna (por defecto: {DEFAULT_SAMPLE_SIZE}).",
    )
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def value_sort_key(value: Any) -> tuple[str, str]:
    return type(value).__name__, str(value)


def write_options_csv(
    path: Path,
    counts: Counter[Any],
    non_null_count: int,
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["valor", "cantidad", "porcentaje_no_nulos"],
        )
        writer.writeheader()
        for value, count in sorted(counts.items(), key=lambda item: value_sort_key(item[0])):
            writer.writerow(
                {
                    "valor": value,
                    "cantidad": count,
                    "porcentaje_no_nulos": round(100 * count / non_null_count, 4)
                    if non_null_count
                    else 0,
                }
            )


def analyze_scalar_column(
    values: list[Any],
    max_options: int,
    sample_size: int,
) -> tuple[dict[str, Any], Counter[Any]]:
    null_count = sum(value is None for value in values)
    non_null_values = [value for value in values if value is not None]
    counts = Counter(non_null_values)
    sorted_values = sorted(counts, key=value_sort_key)

    result: dict[str, Any] = {
        "nulos": null_count,
        "no_nulos": len(non_null_values),
        "valores_unicos": len(counts),
        "ejemplos": [json_safe(value) for value in sorted_values[:sample_size]],
        "exporta_opciones_completas": len(counts) <= max_options,
    }

    numeric_values = [
        value
        for value in non_null_values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if numeric_values and len(numeric_values) == len(non_null_values):
        result["minimo"] = min(numeric_values)
        result["maximo"] = max(numeric_values)

    return result, counts


def filter_spanish(dataset: DatasetDict) -> tuple[DatasetDict, dict[str, Any]]:
    filtered_splits = {}
    filter_report: dict[str, Any] = {}

    for split_name, split_data in dataset.items():
        if LANGUAGE_COLUMN not in split_data.column_names:
            raise ValueError(
                f"La particion {split_name!r} no contiene la columna "
                f"{LANGUAGE_COLUMN!r}."
            )

        original_rows = len(split_data)
        filtered_data = split_data.filter(
            lambda row: row[LANGUAGE_COLUMN] == LANGUAGE_VALUE,
            desc=f"Filtrando {split_name}: {LANGUAGE_COLUMN}={LANGUAGE_VALUE}",
        )
        spanish_rows = len(filtered_data)
        if spanish_rows == 0:
            raise ValueError(
                f"La particion {split_name!r} no contiene filas con "
                f"{LANGUAGE_COLUMN}={LANGUAGE_VALUE!r}."
            )

        filtered_splits[split_name] = filtered_data
        filter_report[split_name] = {
            "filas_originales": original_rows,
            "filas_spanish": spanish_rows,
            "filas_descartadas": original_rows - spanish_rows,
        }

    return DatasetDict(filtered_splits), filter_report


def analyze_dataset(
    dataset: DatasetDict,
    analysis_dir: Path,
    max_options: int,
    sample_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    split_reports: dict[str, Any] = {}

    for split_name, split_data in dataset.items():
        column_reports: dict[str, Any] = {}

        for column_name in split_data.column_names:
            feature = split_data.features[column_name]
            column_report: dict[str, Any] = {
                "tipo_hugging_face": str(feature),
            }

            if isinstance(feature, (Value, ClassLabel)):
                values = split_data[column_name]
                scalar_report, counts = analyze_scalar_column(
                    values,
                    max_options,
                    sample_size,
                )
                column_report.update(scalar_report)

                if scalar_report["exporta_opciones_completas"]:
                    options_filename = f"opciones_{split_name}_{column_name}.csv"
                    write_options_csv(
                        analysis_dir / options_filename,
                        counts,
                        scalar_report["no_nulos"],
                    )
                    column_report["archivo_opciones"] = options_filename
            else:
                sample_rows = split_data.select(
                    range(min(sample_size, len(split_data)))
                )
                column_report.update(
                    {
                        "nulos": None,
                        "no_nulos": None,
                        "valores_unicos": None,
                        "ejemplos": [
                            json_safe(value) for value in sample_rows[column_name]
                        ],
                        "exporta_opciones_completas": False,
                    }
                )

            column_reports[column_name] = column_report
            summary_rows.append(
                {
                    "particion": split_name,
                    "columna": column_name,
                    "tipo_hugging_face": column_report["tipo_hugging_face"],
                    "cantidad_filas": len(split_data),
                    "nulos": column_report.get("nulos"),
                    "valores_unicos": column_report.get("valores_unicos"),
                    "minimo": column_report.get("minimo"),
                    "maximo": column_report.get("maximo"),
                    "archivo_opciones": column_report.get("archivo_opciones", ""),
                }
            )

        split_reports[split_name] = {
            "cantidad_filas": len(split_data),
            "columnas": column_reports,
        }

    return summary_rows, split_reports


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "particion",
        "columna",
        "tipo_hugging_face",
        "cantidad_filas",
        "nulos",
        "valores_unicos",
        "minimo",
        "maximo",
        "archivo_opciones",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.max_options < 1:
        raise ValueError("--max-options debe ser mayor que cero.")
    if args.sample_size < 1:
        raise ValueError("--sample-size debe ser mayor que cero.")

    output_dir = args.output_dir.resolve()
    data_dir = output_dir / "datos"
    analysis_dir = output_dir / "analisis"
    data_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Descargando {DATASET_ID!r}, configuracion {CONFIG_NAME!r}...")
    downloaded_dataset = load_dataset(DATASET_ID, CONFIG_NAME)

    print(f"Validando y filtrando {LANGUAGE_COLUMN}={LANGUAGE_VALUE!r}...")
    dataset, filter_report = filter_spanish(downloaded_dataset)

    print("Guardando particiones en formato Parquet...")
    parquet_files: dict[str, str] = {}
    for split_name, split_data in dataset.items():
        parquet_path = data_dir / f"{split_name}.parquet"
        split_data.to_parquet(str(parquet_path))
        parquet_files[split_name] = str(parquet_path.relative_to(output_dir))

    print("Analizando columnas y opciones...")
    summary_rows, split_reports = analyze_dataset(
        dataset,
        analysis_dir,
        args.max_options,
        args.sample_size,
    )

    report = {
        "dataset_id": DATASET_ID,
        "configuracion": CONFIG_NAME,
        "filtro": {
            "columna": LANGUAGE_COLUMN,
            "valor": LANGUAGE_VALUE,
            "resultado_por_particion": filter_report,
        },
        "archivos_parquet": parquet_files,
        "maximo_opciones_categoricas": args.max_options,
        "particiones": split_reports,
    }

    summary_path = analysis_dir / "resumen_columnas.csv"
    report_path = analysis_dir / "informe.json"
    write_summary_csv(summary_path, summary_rows)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nProceso terminado.")
    print(f"Datos:    {data_dir}")
    print(f"Resumen:  {summary_path}")
    print(f"Informe:  {report_path}")


if __name__ == "__main__":
    main()
