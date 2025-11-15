from __future__ import annotations

"""FHIBEデータセットのディレクトリ構造と注釈統計を可視化するCLIレポーター。"""

import argparse
import ast
import csv
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:
    Console = None
    Table = None

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


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

IGNORE_BASENAMES = {".DS_Store"}

console = Console(highlight=False, soft_wrap=True) if Console else None
USE_RICH = console is not None and Table is not None


@dataclass(frozen=True)
class AttributeField:
    column: str
    title_en: str
    title_ja: str
    treat_as_list: bool = False

    @property
    def title_bilingual(self) -> str:
        return f"{self.title_en} / {self.title_ja}"


def bilingual(en: str, ja: str) -> str:
    """英語と日本語を同時に表示するためのヘルパー。"""
    return f"{en} / {ja}"


def format_pct(count: int, total: int) -> str:
    """割合を0.1%精度で返す。"""
    if total <= 0:
        return "0.0%"
    return f"{(count / total) * 100:.1f}%"


def make_table(title: str, columns: Sequence[str]) -> Table:
    """指定タイトルと列でRichテーブルを生成する。"""
    if not USE_RICH or Table is None:
        raise RuntimeError("Rich tables are unavailable.")
    table = Table(title=title, header_style="bold cyan")
    for column in columns:
        table.add_column(column)
    return table


def format_list(items: Sequence[str]) -> str:
    """代表リストを折り返し表示しやすい文字列に変換する。"""
    return "\n".join(items) if items else "-"


def append_markdown_table(
    markdown_sections: list[str] | None, title: str, headers: Sequence[str], rows: list[Sequence[str]]
) -> None:
    """tabulateでMarkdownテーブルを生成してセクションに追加する。"""
    if markdown_sections is None or tabulate is None:
        return
    markdown_sections.append(
        f"### {title}\n" + tabulate(rows, headers=headers, tablefmt="github")
    )


def print_rule(title: str) -> None:
    """Richの罫線 or フォールバックを出力する。"""
    if USE_RICH and console is not None:
        console.rule(title)
    else:
        print("\n" + title)
        print("-" * len(title))


ATTRIBUTE_FIELDS: Sequence[AttributeField] = (
    AttributeField("pronoun", "Self-identified pronoun(s)", "自己申告の代名詞", True),
    AttributeField("ancestry", "Ancestry bucket(s)", "祖先カテゴリ", True),
    AttributeField("natural_skin_color", "Natural skin color palette", "自然肌色パレット"),
    AttributeField(
        "apparent_skin_color", "Perceived skin color palette", "見た目肌色パレット"
    ),
    AttributeField("hairstyle", "Hair style category", "ヘアスタイル区分"),
    AttributeField("natural_hair_type", "Natural hair type", "自然な髪質"),
    AttributeField("apparent_hair_type", "Perceived hair type", "見た目の髪質"),
    AttributeField("natural_hair_color", "Natural hair color(s)", "自然な髪色", True),
    AttributeField("apparent_hair_color", "Perceived hair color(s)", "見た目の髪色", True),
    AttributeField("facial_hairstyle", "Facial hairstyle(s)", "顔周りの毛スタイル", True),
    AttributeField(
        "natural_facial_haircolor", "Natural facial hair color(s)", "自然な顔毛の色", True
    ),
    AttributeField(
        "apparent_facial_haircolor", "Perceived facial hair color(s)", "見た目の顔毛の色", True
    ),
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
    """CLIで受け取るパラメータを定義する。"""
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
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Also print GitHub-flavored Markdown tables (requires tabulate).",
    )
    return parser.parse_args()


def resolve_dataset_root(raw_path: Path | None) -> Path:
    """指定パスを展開し、存在確認まで行う。"""
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
    """注目ディレクトリごとの件数やサンプルをまとめる。"""
    summaries: list[dict] = []
    for rel_path in focus_dirs:
        absolute = dataset_root / rel_path
        if not absolute.exists():
            continue
        summaries.append(describe_directory(dataset_root, absolute, max_samples))
    return summaries


def describe_directory(dataset_root: Path, absolute: Path, max_samples: int) -> dict:
    """単一ディレクトリ内のファイル／フォルダ数と代表例を返す。"""
    entries = sorted(absolute.iterdir(), key=lambda path: path.name)
    sample_files: list[str] = []
    sample_dirs: list[str] = []
    file_count = 0
    dir_count = 0
    for entry in entries:
        if entry.name in IGNORE_BASENAMES:
            continue
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
    """数値列の最小・最大・平均を逐次更新で保持する。"""
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
    """画像メタ情報、カメラ、属性ラベル、年齢の統計を集計する。"""
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
    """CSV内表現にかかわらずラベル文字列のリストに正規化する。"""
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
    """アノテータ属性CSVの件数・年齢帯・Pronoun/Ancestry頻度を集計する。"""
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


def render_directory_summary(
    dir_summaries: Iterable[dict], markdown_sections: list[str] | None
) -> None:
    """ディレクトリ構成をRichテーブルとMarkdownで出力する。"""
    title = bilingual("Directory overview", "ディレクトリ概要")
    markdown_rows: list[list[str]] = []
    if USE_RICH:
        table = make_table(
            title,
            [
                bilingual("Path", "パス"),
                bilingual("Dirs/Files", "ディレクトリ/ファイル数"),
                bilingual("Sample dirs", "代表ディレクトリ"),
                bilingual("Sample files", "代表ファイル"),
            ],
        )
    else:
        print_rule(title)
    for summary in dir_summaries:
        dir_counts = f"{summary['dirs']} / {summary['files']}"
        sample_dirs = format_list(summary["sample_dirs"])
        sample_files = format_list(summary["sample_files"])
        if USE_RICH:
            table.add_row(summary["path"], dir_counts, sample_dirs, sample_files)
        else:
            print(f"- {summary['path']}")
            print(f"  - {bilingual('Dirs/Files', '件数')}: {dir_counts}")
            if sample_dirs != "-":
                print(f"  - {bilingual('Sample dirs', '代表ディレクトリ')}: {sample_dirs}")
            if sample_files != "-":
                print(f"  - {bilingual('Sample files', '代表ファイル')}: {sample_files}")
        markdown_rows.append(
            [summary["path"], dir_counts, sample_dirs.replace("\n", ", "), sample_files.replace("\n", ", ")]
        )
    if USE_RICH:
        console.print(table)
    append_markdown_table(
        markdown_sections,
        title,
        [
            bilingual("Path", "パス"),
            bilingual("Dirs/Files", "ディレクトリ/ファイル数"),
            bilingual("Sample dirs", "代表ディレクトリ"),
            bilingual("Sample files", "代表ファイル"),
        ],
        markdown_rows,
    )


def render_metadata_summary(
    summary: dict, top_n: int, markdown_sections: list[str] | None
) -> None:
    """データセット全体の統計・属性別分布をRichテーブルで出力する。"""
    section_title = bilingual("Metadata (fhibe_downsampled.csv)", "メタデータ概要")
    print_rule(section_title)

    stats_headers = [bilingual("Metric", "指標"), bilingual("Value", "値")]
    stats_rows: list[list[str]] = [
        [bilingual("Images (rows)", "画像件数"), f"{summary['records']:,}"],
        [bilingual("Unique subjects", "ユニーク被写体数"), f"{summary['subjects']:,}"],
        [bilingual("Primary hero shots", "主画像件数"), f"{summary['primary_count']:,}"],
    ]
    if summary["height"] and summary["width"]:
        height = summary["height"]
        width = summary["width"]
        stats_rows.append(
            [
                bilingual("Image resolution (HxW)", "画像解像度 (縦×横)"),
                f"min {height['min']:.0f}x{width['min']:.0f}, "
                f"avg {height['avg']:.0f}x{width['avg']:.0f}, "
                f"max {height['max']:.0f}x{width['max']:.0f}",
            ]
        )
    if summary["age"]:
        age = summary["age"]
        stats_rows.append(
            [
                bilingual("Age range", "年齢範囲"),
                f"min {age['min']:.1f}, avg {age['avg']:.1f}, max {age['max']:.1f}",
            ]
        )
    if USE_RICH:
        stats_table = make_table(section_title + " - " + bilingual("Core stats", "基本統計"), stats_headers)
        for metric, value in stats_rows:
            stats_table.add_row(metric, value)
        console.print(stats_table)
    else:
        for metric, value in stats_rows:
            print(f"- {metric}: {value}")
    append_markdown_table(markdown_sections, section_title, stats_headers, stats_rows)

    if summary["age_buckets"]:
        age_title = bilingual("Age buckets", "年齢帯分布")
        age_headers = [
            bilingual("Bucket", "年齢帯"),
            bilingual("Count", "件数"),
            bilingual("Share", "構成比"),
        ]
        age_rows = []
        if USE_RICH:
            age_table = make_table(age_title, age_headers)
        else:
            print(f"\n{age_title}:")
        for label, count in summary["age_buckets"].most_common():
            share = format_pct(count, summary["records"])
            if USE_RICH:
                age_table.add_row(label, str(count), share)
            else:
                print(f"  - {label}: {count} ({share})")
            age_rows.append([label, str(count), share])
        if USE_RICH:
            console.print(age_table)
        append_markdown_table(
            markdown_sections,
            age_title,
            age_headers,
            age_rows,
        )

    camera_title = bilingual("Camera manufacturers/models", "カメラメーカー／モデル")
    camera_headers = [
        bilingual("Model", "機種"),
        bilingual("Images", "件数"),
        bilingual("Share", "構成比"),
    ]
    if USE_RICH:
        camera_table = make_table(camera_title, camera_headers)
    else:
        print(f"\n{camera_title}:")
    camera_rows = []
    for model, count in summary["camera_models"].most_common(top_n):
        share = format_pct(count, summary["records"])
        if USE_RICH:
            camera_table.add_row(model, str(count), share)
        else:
            print(f"- {model}: {count} ({share})")
        camera_rows.append([model, str(count), share])
    if USE_RICH:
        console.print(camera_table)
    append_markdown_table(
        markdown_sections,
        camera_title,
        camera_headers,
        camera_rows,
    )

    attributes = summary["attributes"]
    for field in ATTRIBUTE_FIELDS:
        counter = attributes[field.column]
        total = sum(counter.values())
        if not total:
            message = f"{field.title_bilingual}: no annotations / データなし"
            if USE_RICH and console is not None:
                console.print(f"[yellow]{message}[/yellow]")
            else:
                print(message)
            continue
        attr_headers = [
            bilingual("Label", "ラベル"),
            bilingual("Count", "件数"),
            bilingual("Share", "構成比"),
        ]
        if USE_RICH:
            attr_table = make_table(field.title_bilingual, attr_headers)
        else:
            print(f"\n{field.title_bilingual}:")
        attr_rows = []
        for label, count in counter.most_common(top_n):
            share = format_pct(count, summary["records"])
            if USE_RICH:
                attr_table.add_row(label, str(count), share)
            else:
                print(f"  - {label}: {count} ({share})")
            attr_rows.append([label, str(count), share])
        if USE_RICH:
            console.print(attr_table)
        append_markdown_table(
            markdown_sections,
            field.title_bilingual,
            attr_headers,
            attr_rows,
        )


def render_annotator_summary(
    title: str, summary: dict, top_n: int, markdown_sections: list[str] | None
) -> None:
    """アノテータ統計を表形式で出力する。"""
    section_title = f"{title} ({summary['path'].name})"
    print_rule(section_title)

    info_headers = [bilingual("Metric", "指標"), bilingual("Value", "値")]
    info_rows = [[bilingual("Annotators", "アノテータ数"), str(summary["rows"])]]
    if summary["ages"]:
        age = summary["ages"]
        info_rows.append(
            [
                bilingual("Age stats", "年齢統計"),
                f"min {age['min']:.1f}, avg {age['avg']:.1f}, max {age['max']:.1f}",
            ]
        )
    if USE_RICH:
        info_table = make_table(section_title + " - Info", info_headers)
        for metric, value in info_rows:
            info_table.add_row(metric, value)
        console.print(info_table)
    else:
        for metric, value in info_rows:
            print(f"- {metric}: {value}")
    append_markdown_table(markdown_sections, section_title + " - Info", info_headers, info_rows)

    if summary["age_buckets"]:
        bucket_headers = [
            bilingual("Bucket", "年齢帯"),
            bilingual("Count", "件数"),
        ]
        bucket_rows = []
        if USE_RICH:
            bucket_table = make_table(
                section_title + " - " + bilingual("Age buckets", "年齢帯"), bucket_headers
            )
        else:
            print(f"\n{bilingual('Age buckets', '年齢帯')}:")
        for label, count in summary["age_buckets"].most_common():
            if USE_RICH:
                bucket_table.add_row(label, str(count))
            else:
                print(f"  - {label}: {count}")
            bucket_rows.append([label, str(count)])
        if USE_RICH:
            console.print(bucket_table)
        append_markdown_table(
            markdown_sections,
            section_title + " - " + bilingual("Age buckets", "年齢帯"),
            bucket_headers,
            bucket_rows,
        )

    pronoun_headers = [
        bilingual("Pronouns", "代名詞"),
        bilingual("Count", "件数"),
    ]
    if USE_RICH:
        pronoun_table = make_table(
            section_title + " - " + bilingual("Pronouns", "代名詞"), pronoun_headers
        )
    else:
        print(f"\n{bilingual('Pronouns', '代名詞')}:")
    pronoun_rows = []
    for label, count in summary["pronoun"].most_common(top_n):
        if USE_RICH:
            pronoun_table.add_row(label, str(count))
        else:
            print(f"  - {label}: {count}")
        pronoun_rows.append([label, str(count)])
    if USE_RICH:
        console.print(pronoun_table)
    append_markdown_table(
        markdown_sections,
        section_title + " - " + bilingual("Pronouns", "代名詞"),
        pronoun_headers,
        pronoun_rows,
    )

    ancestry_headers = [
        bilingual("Ancestry", "祖先カテゴリ"),
        bilingual("Count", "件数"),
    ]
    if USE_RICH:
        ancestry_table = make_table(
            section_title + " - " + bilingual("Ancestry", "祖先カテゴリ"), ancestry_headers
        )
    else:
        print(f"\n{bilingual('Ancestry', '祖先カテゴリ')}:")
    ancestry_rows = []
    for label, count in summary["ancestry"].most_common(top_n):
        if USE_RICH:
            ancestry_table.add_row(label, str(count))
        else:
            print(f"  - {label}: {count}")
        ancestry_rows.append([label, str(count)])
    if USE_RICH:
        console.print(ancestry_table)
    append_markdown_table(
        markdown_sections,
        section_title + " - " + bilingual("Ancestry", "祖先カテゴリ"),
        ancestry_headers,
        ancestry_rows,
    )


def main() -> None:
    """CLI→集計→Markdown描画の実行フローを束ねるエントリーポイント。"""
    args = parse_args()
    dataset_root = resolve_dataset_root(args.dataset_root)
    focus_dirs = args.focus_dirs or DEFAULT_FOCUS_DIRS
    if args.markdown and tabulate is None:
        raise SystemExit(
            "Markdownテーブル出力には tabulate が必要です。`pip install tabulate` を実行してください。"
        )
    markdown_sections: list[str] | None = [] if args.markdown else None

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

    print_rule("# " + bilingual("FHIBE downsampled dataset report", "FHIBEダウンサンプル版レポート"))
    render_directory_summary(directory_summaries, markdown_sections)
    render_metadata_summary(metadata_summary, args.top_n, markdown_sections)
    for summary in annotator_summaries:
        title = (
            bilingual("QA annotator demographics", "QAアノテータ属性")
            if summary["path"].name.startswith("QA")
            else bilingual("Annotator demographics", "アノテータ属性")
        )
        render_annotator_summary(title, summary, args.top_n, markdown_sections)

    if markdown_sections:
        print_rule("Markdown tables (copy & paste)")
        for block in markdown_sections:
            print(block)
            print()


if __name__ == "__main__":
    main()
