"""Descarga y describe la configuracion en espanol de HEAD-QA v2.

Salidas:
    Bases/0 - Bases crudo/head_qa_v2/
        datos/<split>.parquet
        analisis/resumen_columnas.csv
        analisis/opciones_<split>_<columna>.csv
        analisis/informe.json

Uso:
    python "Procesamiento/0 - Descarga y analisis/head_qa_v2/01_descargar_head_qa_v2_es.py"
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import ClassLabel, Value, load_dataset


DATASET_ID = "alesi12/head_qa_v2"
CONFIG_NAME = "es"
DEFAULT_MAX_OPTIONS = 100
DEFAULT_SAMPLE_SIZE = 5


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    default_output = project_root / "Bases" / "0 - Bases crudo" / "head_qa_v2"

    parser = argparse.ArgumentParser(
        description="Descarga HEAD-QA v2 en espanol y analiza sus columnas."
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
            "Maximo de valores unicos para considerar una columna categorica "
            f"y exportar todas sus opciones (por defecto: {DEFAULT_MAX_OPTIONS})."
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
    """Convierte valores anidados a una representacion serializable y breve."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if key == "bytes" and isinstance(item, bytes):
                converted[key] = f"<{len(item)} bytes>"
            else:
                converted[str(key)] = json_safe(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def is_scalar_feature(feature: Any) -> bool:
    return isinstance(feature, (Value, ClassLabel))


def value_sort_key(value: Any) -> tuple[str, str]:
    return (type(value).__name__, str(value))


def write_options_csv(
    path: Path, counts: Counter[Any], non_null_count: int
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["valor", "cantidad", "porcentaje_no_nulos"]
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
) -> dict[str, Any]:
    null_count = sum(value is None for value in values)
    non_null_values = [value for value in values if value is not None]
    counts = Counter(non_null_values)
    unique_count = len(counts)
    sorted_values = sorted(counts, key=value_sort_key)

    result: dict[str, Any] = {
        "nulos": null_count,
        "no_nulos": len(non_null_values),
        "valores_unicos": unique_count,
        "ejemplos": [json_safe(value) for value in sorted_values[:sample_size]],
        "exporta_opciones_completas": unique_count <= max_options,
    }

    numeric_values = [
        value
        for value in non_null_values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if numeric_values and len(numeric_values) == len(non_null_values):
        result["minimo"] = min(numeric_values)
        result["maximo"] = max(numeric_values)

    return result


def analyze_dataset(
    dataset: Any,
    analysis_dir: Path,
    max_options: int,
    sample_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {"particiones": {}}

    for split_name, split_data in dataset.items():
        split_report: dict[str, Any] = {
            "cantidad_filas": len(split_data),
            "columnas": {},
        }

        for column_name in split_data.column_names:
            feature = split_data.features[column_name]
            column_report: dict[str, Any] = {
                "tipo_hugging_face": str(feature),
            }

            if is_scalar_feature(feature):
                values = split_data[column_name]
                column_report.update(
                    analyze_scalar_column(values, max_options, sample_size)
                )

                if column_report["exporta_opciones_completas"]:
                    non_null_values = [value for value in values if value is not None]
                    counts = Counter(non_null_values)
                    options_filename = f"opciones_{split_name}_{column_name}.csv"
                    write_options_csv(
                        analysis_dir / options_filename,
                        counts,
                        len(non_null_values),
                    )
                    column_report["archivo_opciones"] = options_filename
            else:
                # Las listas, estructuras e imagenes se describen con ejemplos.
                # No se calculan valores unicos porque no representan categorias
                # simples y las imagenes podrian ocupar mucha memoria al decodificarse.
                if type(feature).__name__ == "Image":
                    examples = ["Ejemplos omitidos para evitar decodificar imagenes."]
                else:
                    sample_rows = split_data.select(
                        range(min(sample_size, len(split_data)))
                    )
                    examples = [
                        json_safe(value) for value in sample_rows[column_name]
                    ]
                column_report.update(
                    {
                        "nulos": None,
                        "no_nulos": None,
                        "valores_unicos": None,
                        "ejemplos": examples,
                        "exporta_opciones_completas": False,
                    }
                )

            split_report["columnas"][column_name] = column_report
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

        report["particiones"][split_name] = split_report

    return summary_rows, report


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
    dataset = load_dataset(DATASET_ID, CONFIG_NAME)

    print("Guardando particiones en formato Parquet...")
    parquet_files: dict[str, str] = {}
    for split_name, split_data in dataset.items():
        parquet_path = data_dir / f"{split_name}.parquet"
        split_data.to_parquet(str(parquet_path))
        parquet_files[split_name] = str(parquet_path.relative_to(output_dir))

    print("Analizando columnas y opciones...")
    summary_rows, report = analyze_dataset(
        dataset,
        analysis_dir,
        args.max_options,
        args.sample_size,
    )
    report.update(
        {
            "dataset_id": DATASET_ID,
            "configuracion": CONFIG_NAME,
            "archivos_parquet": parquet_files,
            "maximo_opciones_categoricas": args.max_options,
        }
    )

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
