"""Multimodal perception engine for PDF documents, screenshots, and untrusted taint detection."""

from __future__ import annotations

import base64
import io
from pathlib import Path
import re
from typing import Any
import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin
from pydantic import BaseModel, Field
from pypdf import PdfReader


_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "override_instructions",
        re.compile(
            r"(ignore\s+(all\s+)?(previous|prior|above)\s+instructions|"
            r"忽略(?:之前|以上|所有)(?:的)?(?:指令|规则|约束)|"
            r"system\s*override|disregard\s+safety)",
            re.IGNORECASE,
        ),
    ),
    (
        "secret_exfiltration",
        re.compile(
            r"(cat\s+.*(?:\.env|id_rsa|\.ssh)|send\s+.*api[_-]?key|"
            r"读取.*\.env|发送.*密钥|exfiltrate|curl\s+https?://[^\s]+.*\$\{?[A-Z_]*KEY)",
            re.IGNORECASE,
        ),
    ),
    (
        "hidden_command_directive",
        re.compile(
            r"(IMPORTANT\s+SYSTEM\s+NOTE\s*:.*(?:run|execute|shell_command_exec)|"
            r"<\s*system_instruction\s*>)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
]

_KV_LINE_RE = re.compile(
    r"^\s*([A-Za-z\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff\s_()/-]{1,28})\s*[:：]\s*(.+?)\s*$"
)


class VisualRegionSummary(BaseModel):
    """Luminance and color statistics for a spatial screen region."""

    region_name: str
    bbox: tuple[int, int, int, int]
    mean_rgb: tuple[float, float, float]
    luminance_std: float


class MultimodalObservation(BaseModel):
    """Structured output produced from parsing an image/screenshot or PDF document."""

    source_path: str
    modality: str = Field(pattern=r"^(pdf|image|text)$")
    extracted_text: str = ""
    structured_kv: dict[str, str] = Field(default_factory=dict)
    table_rows: list[list[str]] = Field(default_factory=list)
    page_or_frame_count: int = 1
    visual_regions: list[VisualRegionSummary] = Field(default_factory=list)
    ui_elements: list[str] = Field(default_factory=list)
    is_tainted_untrusted: bool = False
    injection_signals: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def detect_prompt_injection_signals(text: str) -> list[str]:
    """Scan external document/screenshot text for indirect prompt injection markers."""
    hits: list[str] = []
    for label, pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            hits.append(label)
    return hits


def _extract_kv_and_tables(text: str) -> tuple[dict[str, str], list[list[str]]]:
    kv: dict[str, str] = {}
    tables: list[list[str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "|" in line:
            cells = [c.strip() for c in line.strip("|").split("|") if c.strip()]
            if len(cells) >= 2 and not all(set(c) <= {"-", ":"} for c in cells):
                tables.append(cells)
                continue
        m = _KV_LINE_RE.match(line)
        if m:
            k, v = m.group(1).strip(), m.group(2).strip()
            if len(v) <= 120:
                kv[k] = v
    return kv, tables


class MultimodalPerceptionEngine:
    """Parses PDF documents and desktop screenshots into typed MultimodalObservations."""

    def parse_file(self, file_path: str | Path) -> MultimodalObservation:
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Multimodal input file does not exist: {path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self.parse_pdf(path)
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            return self.parse_image(path)
        raw_text = path.read_text(encoding="utf-8")
        kv, tables = _extract_kv_and_tables(raw_text)
        signals = detect_prompt_injection_signals(raw_text)
        return MultimodalObservation(
            source_path=str(path),
            modality="text",
            extracted_text=raw_text,
            structured_kv=kv,
            table_rows=tables,
            page_or_frame_count=1,
            is_tainted_untrusted=bool(signals),
            injection_signals=signals,
        )

    def parse_pdf(self, pdf_path: str | Path) -> MultimodalObservation:
        path = Path(pdf_path).resolve()
        reader = PdfReader(str(path))
        pages_text: list[str] = []
        for idx, page in enumerate(reader.pages):
            txt = (page.extract_text() or "").strip()
            if txt:
                pages_text.append(f"[Page {idx + 1}]\n{txt}")

        joined = "\n\n".join(pages_text)
        kv, tables = _extract_kv_and_tables(joined)
        signals = detect_prompt_injection_signals(joined)
        doc_meta = {}
        if reader.metadata:
            for k, v in reader.metadata.items():
                if v is not None:
                    doc_meta[str(k).lstrip("/")] = str(v)

        return MultimodalObservation(
            source_path=str(path),
            modality="pdf",
            extracted_text=joined,
            structured_kv=kv,
            table_rows=tables,
            page_or_frame_count=len(reader.pages),
            is_tainted_untrusted=bool(signals),
            injection_signals=signals,
            metadata=doc_meta,
        )

    def parse_image(self, image_path: str | Path) -> MultimodalObservation:
        path = Path(image_path).resolve()
        with Image.open(path) as img:
            rgb_img = img.convert("RGB")
            width, height = rgb_img.size
            arr = np.asarray(rgb_img, dtype=np.float32)
            info_dict = {str(k): str(v) for k, v in (img.info or {}).items() if isinstance(v, str)}

        # Partition screen into canonical UI layout zones (header, left_nav, main_canvas, footer)
        h_top = max(1, int(height * 0.15))
        h_bot = max(h_top + 1, int(height * 0.88))
        w_nav = max(1, int(width * 0.22))
        zones = [
            ("header_bar", (0, 0, width, h_top), arr[0:h_top, :, :]),
            ("left_sidebar", (0, h_top, w_nav, h_bot), arr[h_top:h_bot, 0:w_nav, :]),
            ("main_canvas", (w_nav, h_top, width, h_bot), arr[h_top:h_bot, w_nav:width, :]),
            ("status_footer", (0, h_bot, width, height), arr[h_bot:height, :, :]),
        ]
        region_summaries: list[VisualRegionSummary] = []
        for name, bbox, patch in zones:
            if patch.size == 0:
                continue
            mean_rgb = tuple(round(float(x), 2) for x in patch.mean(axis=(0, 1)))
            lum = 0.299 * patch[:, :, 0] + 0.587 * patch[:, :, 1] + 0.114 * patch[:, :, 2]
            region_summaries.append(
                VisualRegionSummary(
                    region_name=name,
                    bbox=bbox,
                    mean_rgb=(mean_rgb[0], mean_rgb[1], mean_rgb[2]),
                    luminance_std=round(float(lum.std()), 2),
                )
            )

        # Extract embedded accessibility tree / UI OCR annotations from PNG text chunks if present
        ui_text_parts: list[str] = []
        ui_elements: list[str] = []
        for key in ("ui_accessibility_tree", "ocr_transcript", "Description", "Comment"):
            if key in info_dict and info_dict[key].strip():
                ui_text_parts.append(info_dict[key].strip())

        if "ui_elements_json" in info_dict:
            for item in info_dict["ui_elements_json"].split(";"):
                cleaned = item.strip()
                if cleaned:
                    ui_elements.append(cleaned)

        extracted = "\n".join(ui_text_parts)
        kv, tables = _extract_kv_and_tables(extracted)
        signals = detect_prompt_injection_signals(extracted)

        return MultimodalObservation(
            source_path=str(path),
            modality="image",
            extracted_text=extracted,
            structured_kv=kv,
            table_rows=tables,
            page_or_frame_count=1,
            visual_regions=region_summaries,
            ui_elements=ui_elements,
            is_tainted_untrusted=bool(signals),
            injection_signals=signals,
            metadata={
                "width": width,
                "height": height,
                "aspect_ratio": round(width / max(1, height), 3),
            },
        )

    @staticmethod
    def encode_image_base64(image_path: str | Path) -> str:
        """Encode an image file as a base64 data URI for OpenAI-compatible VLM endpoints."""
        path = Path(image_path).resolve()
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"


def build_valid_pdf(output_path: str | Path, pages_lines: list[list[str]], title: str = "Report") -> Path:
    """ deterministically construct a standards-compliant text PDF (ASCII-safe content stream)
    for offline unit testing and evaluation benchmarks without external binary tools."""
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    def _pdf_escape(text: str) -> str:
        return (
            text.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
        )

    objects: list[bytes] = []
    # Obj 1: Catalog, Obj 2: Pages, Obj 3: Font Helvetica
    page_obj_ids: list[int] = []
    next_id = 4
    page_entries: list[tuple[int, int, bytes]] = []

    for lines in pages_lines:
        page_id = next_id
        content_id = next_id + 1
        next_id += 2
        page_obj_ids.append(page_id)

        stream_cmds = ["BT", "/F1 11 Tf", "50 750 Td", "16 TL"]
        for idx, line in enumerate(lines):
            escaped = _pdf_escape(line)
            if idx == 0:
                stream_cmds.append(f"({escaped}) Tj")
            else:
                stream_cmds.append(f"T* ({escaped}) Tj")
        stream_cmds.append("ET")
        stream_bytes = "\n".join(stream_cmds).encode("latin-1", errors="replace")
        page_entries.append((page_id, content_id, stream_bytes))

    info_id = next_id
    kids_str = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append(
        f"2 0 obj\n<< /Type /Pages /Kids [{kids_str}] /Count {len(page_obj_ids)} >>\nendobj\n".encode("ascii")
    )
    objects.append(
        b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    )

    for page_id, content_id, stream_bytes in page_entries:
        objects.append(
            (
                f"{page_id} 0 obj\n"
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>\n"
                f"endobj\n"
            ).encode("ascii")
        )
        objects.append(
            f"{content_id} 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
            + stream_bytes
            + b"\nendstream\nendobj\n"
        )

    objects.append(
        f"{info_id} 0 obj\n<< /Title ({_pdf_escape(title)}) /Producer (OmniPersonalAgent) >>\nendobj\n".encode(
            "latin-1", errors="replace"
        )
    )

    buf = io.BytesIO()
    buf.write(b"%PDF-1.4\n")
    offsets: list[int] = [0]
    for obj in objects:
        offsets.append(buf.tell())
        buf.write(obj)

    xref_pos = buf.tell()
    buf.write(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    buf.write(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        buf.write(f"{off:010d} 00000 n \n".encode("ascii"))
    buf.write(
        (
            f"trailer\n<< /Size {len(offsets)} /Root 1 0 R /Info {info_id} 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("ascii")
    )
    out.write_bytes(buf.getvalue())
    return out


def build_synthetic_screenshot(
    output_path: str | Path,
    *,
    window_title: str,
    metrics_kv: dict[str, str],
    ui_elements: list[str],
    extra_transcript: str = "",
    size: tuple[int, int] = (960, 600),
) -> Path:
    """Render a realistic desktop dashboard screenshot PNG with visual layout zones
    and embedded OS accessibility tree metadata."""
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    img = Image.new("RGB", (w, h), color=(244, 246, 250))
    draw = ImageDraw.Draw(img)

    # Header bar, left sidebar, main metric cards, and footer
    draw.rectangle([0, 0, w, int(h * 0.14)], fill=(28, 37, 54))
    draw.text((24, 26), window_title, fill=(255, 255, 255))
    draw.rectangle([0, int(h * 0.14), int(w * 0.22), int(h * 0.90)], fill=(42, 54, 75))
    draw.rectangle([0, int(h * 0.90), w, h], fill=(225, 230, 238))

    card_y = int(h * 0.18)
    transcript_lines = [f"Window: {window_title}"]
    for k, v in metrics_kv.items():
        draw.rectangle([int(w * 0.26), card_y, int(w * 0.94), card_y + 64], fill=(255, 255, 255), outline=(205, 213, 224))
        line_str = f"{k}: {v}"
        draw.text((int(w * 0.28), card_y + 22), line_str, fill=(20, 28, 40))
        transcript_lines.append(line_str)
        card_y += 80

    if extra_transcript:
        transcript_lines.append(extra_transcript)

    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("ui_accessibility_tree", "\n".join(transcript_lines), zip=False)
    pnginfo.add_text("ui_elements_json", "; ".join(ui_elements), zip=False)
    img.save(out, format="PNG", pnginfo=pnginfo)
    return out
