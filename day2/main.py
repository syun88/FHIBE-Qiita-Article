from __future__ import annotations

import argparse
import ast
import csv
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_FOCUS_DIRS = [
    "data",
    "data/raw",
    "data/raw/fhibe_downsampled",
    "data/processed",
    "data/processed/fhibe_downsampled",
    "data/processed/fhibe_face_crop_align",
    "data/annotator_metadata",
    "data/aggregated_results",
    "data/protocol",
    "results",
]


@dataclass(frozen=True)
class AttributeField:
    column: str
    title: str
    treat_as_list: bool = False


ATTRIBUTE_FIELDS: Sequence[AttributeField] = (
    AttributeField("pronoun", "Self-identified pronoun(s)", True),
    AttributeField("ancestry", "Ancestry bucket(s)", True),
    AttributeField("natural_skin_color", "Natural skin color palette"),
    AttributeField("apparent_skin_color", "Perceived skin color palette"),
    AttributeField("hairstyle", "Hair style category"),
    AttributeField("natural_hair_type", "Natural hair type"),
    AttributeField("apparent_hair_type", "Perceived hair type"),
    AttributeField("natural_hair_color", "Natural hair color(s)", True),
    AttributeField("apparent_hair_color", "Perceived hair color(s)", True),
    AttributeField("facial_hairstyle", "Facial hairstyle(s)", True),
    AttributeField("natural_facial_haircolor", "Natural facial hair color(s)", True),
    AttributeField("apparent_facial_haircolor", "Perceived facial hair color(s)", True),
)

csv.field_size_limit(min(sys.maxsize, 2_147_483_647))


AGE_BUCKETS = [
    (0, 17, "00-17"),
    (18, 24, "18-24"),
    (25, 34, "25-34"),
    (35, 44, "35-44"),
    (45, 54, "45-54"),
    (55, 64, "55-64"),
    (65, 150, "65+"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the FHIBE downsampled dataset structure, metadata and attribute labels.\n"
            "Use --dataset-root to point to the extracted FHIBE archive."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=os.environ.get("FHIBE_DATA_DIR"),
        help="Path to the extracted FHIBE dataset (defaults to $FHIBE_DATA_DIR).",
    )
    parser.add_argument(
        "--focus-dirs",
        nargs="*",
        default=None,
        help="Relative directories (from dataset root) to include in the directory summary.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=4,
        help="Number of sample files/subdirectories to show inside each directory.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top values to show for attribute/camera distributions.",
    )
    return parser.parse_args()


def resolve_dataset_root(raw_path: Path | None) -> Path:
    if raw_path is None:
        raise SystemExit(
            "Dataset path is not set. Provide --dataset-root or export FHIBE_DATA_DIR."
        )
    dataset_root = raw_path.expanduser().resolve()
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")
    return dataset_root


def describe_directories(
    dataset_root: Path, focus_dirs: Sequence[str], max_samples: int
) -> list[dict]:
    summaries: list[dict] = []
    for rel_path in focus_dirs:
        absolute = dataset_root / rel_path
        if not absolute.exists():
            continue
        summaries.append(describe_directory(dataset_root, absolute, max_samples))
    return summaries


def describe_directory(dataset_root: Path, absolute: Path, max_samples: int) -> dict:
    entries = sorted(absolute.iterdir(), key=lambda path: path.name)
    sample_files: list[str] = []
    sample_dirs: list[str] = []
    file_count = 0
    dir_count = 0
    for entry in entries:
        if entry.is_dir():
            dir_count += 1
            if len(sample_dirs) < max_samples:
                sample_dirs.append(entry.name)
        else:
            file_count += 1
            if len(sample_files) < max_samples:
                sample_files.append(entry.name)
    rel = absolute.relative_to(dataset_root)
    return {
        "path": str(rel),
        "files": file_count,
        "dirs": dir_count,
        "sample_files": sample_files,
        "sample_dirs": sample_dirs,
    }


class NumericStats:
    def __init__(self) -> None:
        self.count = 0
        self._min: float | None = None
        self._max: float | None = None
        self._sum = 0.0

    def add(self, value: float | int | None) -> None:
        if value is None:
            return
        value_f = float(value)
        self.count += 1
        self._sum += value_f
        self._min = value_f if self._min is None else min(self._min, value_f)
        self._max = value_f if self._max is None else max(self._max, value_f)

    def to_dict(self) -> dict:
        if self.count == 0:
            return {}
        return {
            "count": self.count,
            "min": self._min,
            "max": self._max,
            "avg": self._sum / self.count,
        }


def summarize_metadata(metadata_csv: Path) -> dict:
    attribute_counters: dict[str, Counter[str]] = {f.column: Counter() for f in ATTRIBUTE_FIELDS}
    camera_models: Counter[str] = Counter()
    subject_ids: set[str] = set()
    row_count = 0
    primary_count = 0
    height_stats = NumericStats()
    width_stats = NumericStats()
    age_stats = NumericStats()
    age_buckets: Counter[str] = Counter()

    with metadata_csv.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row_count += 1
            subject_id = row.get("subject_id", "").strip()
            if subject_id:
                subject_ids.add(subject_id)

            if parse_bool(row.get("is_primary")):
                primary_count += 1

            height_stats.add(parse_float(row.get("image_height")))
            width_stats.add(parse_float(row.get("image_width")))

            manufacturer = (row.get("manufacturer") or "").strip()
            model = (row.get("model") or "").strip()
            if manufacturer or model:
                label = " / ".join(value for value in (manufacturer, model) if value)
                camera_models[label] += 1

            age_value = parse_float(row.get("age"))
            if age_value is not None:
                age_stats.add(age_value)
                age_buckets[bucketize_age(age_value)] += 1

            for field in ATTRIBUTE_FIELDS:
                values = normalize_attribute(row.get(field.column), field.treat_as_list)
                for value in values:
                    attribute_counters[field.column][value] += 1

    return {
        "records": row_count,
        "subjects": len(subject_ids),
        "primary_count": primary_count,
        "height": height_stats.to_dict(),
        "width": width_stats.to_dict(),
        "age": age_stats.to_dict(),
        "age_buckets": age_buckets,
        "camera_models": camera_models,
        "attributes": attribute_counters,
    }


def parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"true", "1", "yes"}


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def normalize_attribute(raw_value: str | None, treat_as_list: bool) -> list[str]:
    if raw_value is None:
        return []
    value = raw_value.strip()
    if not value or value == "[]":
        return []
    if treat_as_list:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = value
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
        parsed_str = str(parsed).strip()
        return [parsed_str] if parsed_str else []
    return [value]


def bucketize_age(age: float) -> str:
    for lower, upper, label in AGE_BUCKETS:
        if lower <= age <= upper:
            return label
    return "unbounded"


def summarize_annotator_table(csv_path: Path) -> dict:
    prefix = "QAannotator" if csv_path.name.startswith("QA") else "annotator"
    age_column = f"{prefix}_age"
    pronoun_column = f"{prefix}_pronoun"
    ancestry_column = f"{prefix}_ancestry"

    pronoun_counter: Counter[str] = Counter()
    ancestry_counter: Counter[str] = Counter()
    ages = NumericStats()
    age_buckets: Counter[str] = Counter()
    row_count = 0

    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not any(row.values()):
                continue
            row_count += 1
            for pronoun in normalize_attribute(row.get(pronoun_column), True):
                pronoun_counter[pronoun] += 1
            for ancestry in normalize_attribute(row.get(ancestry_column), True):
                ancestry_counter[ancestry] += 1
            age_value = parse_float(row.get(age_column))
            if age_value is not None:
                ages.add(age_value)
                age_buckets[bucketize_age(age_value)] += 1

    return {
        "path": csv_path,
        "rows": row_count,
        "pronoun": pronoun_counter,
        "ancestry": ancestry_counter,
        "ages": ages.to_dict(),
        "age_buckets": age_buckets,
    }


def render_directory_summary(dir_summaries: Iterable[dict]) -> None:
    print("## Directory overview")
    for summary in dir_summaries:
        print(
            f"- {summary['path']}: {summary['dirs']} dirs / {summary['files']} files"
        )
        if summary["sample_dirs"]:
            print(f"  - sample dirs: {', '.join(summary['sample_dirs'])}")
        if summary["sample_files"]:
            print(f"  - sample files: {', '.join(summary['sample_files'])}")


def render_metadata_summary(summary: dict, top_n: int) -> None:
    print("\n## Metadata (fhibe_downsampled.csv)")
    print(f"- Images (rows): {summary['records']:,}")
    print(f"- Unique subjects: {summary['subjects']:,}")
    print(f"- Primary hero shots: {summary['primary_count']:,}")

    if summary["height"] and summary["width"]:
        height = summary["height"]
        width = summary["width"]
        print(
            f"- Image resolution (HxW): "
            f"min {height['min']:.0f}x{width['min']:.0f}, "
            f"avg {height['avg']:.0f}x{width['avg']:.0f}, "
            f"max {height['max']:.0f}x{width['max']:.0f}"
        )
    if summary["age"]:
        age = summary["age"]
        print(
            f"- Age range: min {age['min']:.1f}, avg {age['avg']:.1f}, max {age['max']:.1f}"
        )
    if summary["age_buckets"]:
        bucket_line = ", ".join(
            f"{label}: {count} ({count/summary['records']:.1%})"
            for label, count in summary["age_buckets"].most_common()
        )
        print(f"- Age buckets: {bucket_line}")

    print("\n### Camera manufacturers/models")
    for model, count in summary["camera_models"].most_common(top_n):
        pct = (count / summary["records"]) if summary["records"] else 0
        print(f"- {model}: {count} images ({pct:.1%})")

    print("\n### Attribute label coverage")
    attributes = summary["attributes"]
    for field in ATTRIBUTE_FIELDS:
        counter = attributes[field.column]
        total = sum(counter.values())
        if not total:
            print(f"- {field.title}: no annotations.")
            continue
        print(f"- {field.title} ({len(counter)} unique labels, {total} tags)")
        for label, count in counter.most_common(top_n):
            pct = (count / summary["records"]) if summary["records"] else 0
            print(f"  - {label}: {count} ({pct:.1%})")


def render_annotator_summary(title: str, summary: dict, top_n: int) -> None:
    print(f"\n## {title} ({summary['path'].name})")
    print(f"- Annotators: {summary['rows']}")
    if summary["ages"]:
        age = summary["ages"]
        print(
            f"- Age stats: min {age['min']:.1f}, avg {age['avg']:.1f}, max {age['max']:.1f}"
        )
    if summary["age_buckets"]:
        bucket_line = ", ".join(
            f"{label}: {count}" for label, count in summary["age_buckets"].most_common()
        )
        print(f"- Age buckets: {bucket_line}")
    print("- Pronouns:")
    for label, count in summary["pronoun"].most_common(top_n):
        print(f"  - {label}: {count}")
    print("- Ancestry:")
    for label, count in summary["ancestry"].most_common(top_n):
        print(f"  - {label}: {count}")


def main() -> None:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root)
    focus_dirs = args.focus_dirs or DEFAULT_FOCUS_DIRS

    metadata_csv = (
        dataset_root / "data" / "processed" / "fhibe_downsampled" / "fhibe_downsampled.csv"
    )
    if not metadata_csv.exists():
        raise SystemExit(f"Metadata CSV not found: {metadata_csv}")

    directory_summaries = describe_directories(dataset_root, focus_dirs, args.max_samples)
    metadata_summary = summarize_metadata(metadata_csv)

    annotator_dir = dataset_root / "data" / "annotator_metadata"
    annotator_summaries = []
    if annotator_dir.exists():
        for csv_name in ("annotator_demographics.csv", "QAannotator_demographics.csv"):
            candidate = annotator_dir / csv_name
            if candidate.exists():
                annotator_summaries.append(summarize_annotator_table(candidate))

    print("# FHIBE downsampled dataset report")
    render_directory_summary(directory_summaries)
    render_metadata_summary(metadata_summary, args.top_n)
    for summary in annotator_summaries:
        title = (
            "QA annotator demographics"
            if summary["path"].name.startswith("QA")
            else "Annotator demographics"
        )
        render_annotator_summary(title, summary, args.top_n)


if __name__ == "__main__":
    main()
