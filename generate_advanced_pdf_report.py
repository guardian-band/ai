"""Render one validated aggregate-results JSON artifact as a PDF report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Circle, Drawing, Line, String

from src.evaluation.reporting import (
    DISPLAY_METRICS,
    ReportingValidationError,
    load_aggregation_summary,
)


METRIC_LABELS = {
    "macro_ap": "Macro AP",
    "micro_ap": "Micro AP",
    "macro_auroc": "AUROC",
    "precision_at_k": "Precision@5",
    "recall_at_k": "Recall@5",
    "ndcg_at_k": "NDCG@5",
}


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        page_count = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.saveState()
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748B"))
            self.drawString(24, 18, "Polypharmacy benchmark evidence report")
            self.drawRightString(816, 18, f"Page {self._pageNumber} of {page_count}")
            self.restoreState()
            super().showPage()
        super().save()


def _format_stat(metric: dict[str, float | None]) -> str:
    mean = metric.get("mean")
    if mean is None:
        return "N/A"
    std = metric.get("std")
    lower = metric.get("lower_ci")
    upper = metric.get("upper_ci")
    if std is None or lower is None or upper is None:
        return f"{mean:.6f}"
    return f"{mean:.6f} +/- {std:.6f} [{lower:.6f}, {upper:.6f}]"


def _summary_chart(summary: dict) -> Drawing | None:
    points = []
    for cohort in summary["cohorts"]:
        metric = cohort["metrics"]["macro_ap"]
        if metric["mean"] is None:
            continue
        lower = metric["lower_ci"]
        upper = metric["upper_ci"]
        if lower is None or upper is None:
            continue
        points.append((
            f"{cohort['model_type']}/{cohort['scenario']}",
            float(metric["mean"]),
            max(0.0, float(metric["mean"]) - float(lower)),
            max(0.0, float(upper) - float(metric["mean"])),
        ))
    if not points:
        return None
    labels, means, lower_errors, upper_errors = zip(*points)
    width, height = 680, 220
    chart = Drawing(width, height)
    chart.add(String(8, height - 16, "Cohort Macro AP with 95% bootstrap intervals", fontSize=11, fillColor=colors.HexColor("#0F172A")))
    left, right, bottom, top = 48, width - 12, 36, height - 32
    all_values = list(means) + [float(mean) - float(error) for mean, error in zip(means, lower_errors)] + [float(mean) + float(error) for mean, error in zip(means, upper_errors)]
    min_value = min(0.0, min(all_values))
    max_value = max(1.0, max(all_values))
    scale = (top - bottom) / max(max_value - min_value, 1e-9)
    chart.add(Line(left, bottom, right, bottom, strokeColor=colors.HexColor("#64748B")))
    chart.add(Line(left, bottom, left, top, strokeColor=colors.HexColor("#64748B")))
    step = (right - left) / max(len(points), 1)
    for index, (label, mean, lower_error, upper_error) in enumerate(points):
        x = left + step * (index + 0.5)
        y = bottom + (mean - min_value) * scale
        low_y = bottom + (mean - lower_error - min_value) * scale
        high_y = bottom + (mean + upper_error - min_value) * scale
        chart.add(Line(x, low_y, x, high_y, strokeColor=colors.HexColor("#1E40AF"), strokeWidth=1.5))
        chart.add(Line(x - 4, low_y, x + 4, low_y, strokeColor=colors.HexColor("#1E40AF")))
        chart.add(Line(x - 4, high_y, x + 4, high_y, strokeColor=colors.HexColor("#1E40AF")))
        chart.add(Circle(x, y, 3.5, fillColor=colors.HexColor("#1E40AF"), strokeColor=colors.HexColor("#1E40AF")))
        chart.add(String(x, 18, label, fontSize=6, textAnchor="middle", fillColor=colors.HexColor("#334155")))
    return chart


def _build_story(summary: dict, source_path: Path, summary_sha256: str, generated_utc: str):
    styles = getSampleStyleSheet()
    title = ParagraphStyle("ReportTitle", parent=styles["Title"], fontSize=20, leading=24, alignment=1, textColor=colors.HexColor("#0F172A"))
    heading = ParagraphStyle("Heading", parent=styles["Heading2"], fontSize=13, leading=16, textColor=colors.HexColor("#1E40AF"), spaceBefore=10)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9, leading=12, textColor=colors.HexColor("#334155"))
    small = ParagraphStyle("Small", parent=body, fontSize=7, leading=9)
    table_header = ParagraphStyle("TableHeader", parent=small, fontName="Helvetica-Bold", textColor=colors.white)

    champion = summary["champion"]["model_type"]
    story = [
        Paragraph("Polypharmacy Benchmark Evidence Report", title),
        Spacer(1, 10),
        Paragraph("This document is a rendering of one validated aggregation JSON artifact. It does not calculate or replace benchmark metrics.", body),
        Paragraph(
            f"Benchmark config: {summary['benchmark_config']}<br/>"
            f"Expected scenarios: {', '.join(summary['expected_scenarios'])}<br/>"
            f"Expected seeds: {', '.join(str(seed) for seed in summary['expected_seeds'])}<br/>"
            f"Benchmark-selected champion under the stated rule: <b>{champion}</b>", body,
        ),
        Paragraph(f"Champion rule: {summary['champion_rule']}", small),
        Spacer(1, 8),
        Paragraph("Scope and limitations", heading),
        Paragraph(
            "The listed supported baselines are prevalence, logistic, and symmetric MLP. "
            "Current inputs use deterministic ID-derived temporary features. Advanced and unified configurations are unavailable. "
            "There is no external or clinical validation in this artifact; conclusions reflect only the benchmark manifests represented in the aggregation.",
            body,
        ),
        Paragraph("Cohort metrics", heading),
    ]

    headers = ["Model", "Scenario"] + [METRIC_LABELS[name] for name in DISPLAY_METRICS]
    rows = [[Paragraph(str(header), table_header) for header in headers]]
    for cohort in summary["cohorts"]:
        rows.append([
            Paragraph(str(cohort["model_type"]), small),
            Paragraph(str(cohort["scenario"]), small),
        ] + [Paragraph(_format_stat(cohort["metrics"][name]), small) for name in DISPLAY_METRICS])
    table = Table(rows, repeatRows=1, colWidths=[58, 58, 100, 100, 100, 100, 100, 100])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E40AF")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 6),
        ("LEADING", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.extend([table, Spacer(1, 10)])

    chart = _summary_chart(summary)
    if chart is not None:
        story.extend([KeepTogether([Paragraph("Summary chart", heading), chart]), Spacer(1, 8)])

    story.extend([
        PageBreak(),
        Paragraph("Provenance appendix", heading),
        Paragraph(f"Aggregation JSON source: {source_path}", small),
        Paragraph(f"Aggregation JSON SHA-256: {summary_sha256}", small),
        Paragraph(f"Report generation UTC: {generated_utc}", small),
        Paragraph(f"Benchmark config path: {summary['benchmark_config']}", small),
    ])
    for cohort in summary["cohorts"]:
        story.append(Paragraph(
            f"{cohort['model_type']} / {cohort['benchmark_id']} / {cohort['scenario']} — "
            f"seeds: {', '.join(str(seed) for seed in cohort['seeds'])}; "
            f"run IDs: {', '.join(cohort['run_ids'])}",
            small,
        ))
    return story


def build_pdf_report(summary_path: str | Path, output_path: str | Path) -> Path:
    """Validate summary first, then atomically render the requested PDF."""

    source_path = Path(summary_path).resolve()
    summary = load_aggregation_summary(source_path)
    summary_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    generated_utc = datetime.now(timezone.utc).isoformat()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="report-", suffix=".pdf", dir=output_path.parent, delete=False) as handle:
            temporary_path = Path(handle.name)
        doc = SimpleDocTemplate(
            str(temporary_path),
            pagesize=landscape(A4),
            leftMargin=24,
            rightMargin=24,
            topMargin=28,
            bottomMargin=30,
            title="Polypharmacy Benchmark Evidence Report",
            author="Polypharmacy evidence pipeline",
        )
        doc.build(_build_story(summary, source_path, summary_sha256, generated_utc), canvasmaker=NumberedCanvas)
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an aggregate benchmark summary as a PDF")
    parser.add_argument("--summary", required=True, help="Aggregation JSON emitted by aggregate_results.py")
    parser.add_argument("--output", required=True, help="Destination PDF path")
    args = parser.parse_args()
    try:
        build_pdf_report(args.summary, args.output)
    except ReportingValidationError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
