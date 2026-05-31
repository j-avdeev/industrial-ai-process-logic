from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "SLIDES.md"
DEFAULT_OUTPUT = ROOT / "SLIDES.pdf"
PAGE_WIDTH = 960
PAGE_HEIGHT = 540


def escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def parse_slides(source: Path) -> list[tuple[str, list[str]]]:
    slides: list[tuple[str, list[str]]] = []
    title: str | None = None
    bullets: list[str] = []
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            if title is not None:
                slides.append((title, bullets))
            title = re.sub(r"^Slide \d+:\s*", "", line[3:])
            bullets = []
        elif line.startswith("- ") and title is not None:
            bullets.append(line[2:])
    if title is not None:
        slides.append((title, bullets))
    if not slides:
        raise SystemExit(f"No slides found in {source}")
    if len(slides) > 10:
        raise SystemExit(f"Submission deck has {len(slides)} slides; maximum is 10")
    return slides


def wrap_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if len(candidate) <= max_chars:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def page_stream(title: str, bullets: list[str], index: int, total: int) -> bytes:
    commands = [
        "q",
        "0.96 0.97 0.96 rg",
        f"0 0 {PAGE_WIDTH} {PAGE_HEIGHT} re f",
        "0.05 0.24 0.32 rg",
        f"0 {PAGE_HEIGHT - 22} {PAGE_WIDTH} 22 re f",
        "0.10 0.10 0.10 rg",
        "BT /F1 32 Tf 52 456 Td",
        f"({escape_pdf_text(title)}) Tj",
        "ET",
    ]
    y = 384
    for bullet in bullets:
        wrapped = wrap_text(bullet, 76)
        commands.extend(
            [
                "0.05 0.24 0.32 rg",
                f"62 {y + 5} 8 8 re f",
                "0.13 0.13 0.13 rg",
                f"BT /F1 20 Tf 88 {y} Td",
                f"({escape_pdf_text(wrapped[0])}) Tj",
                "ET",
            ]
        )
        y -= 29
        for line in wrapped[1:]:
            commands.extend(
                [
                    "0.13 0.13 0.13 rg",
                    f"BT /F1 20 Tf 88 {y} Td",
                    f"({escape_pdf_text(line)}) Tj",
                    "ET",
                ]
            )
            y -= 29
        y -= 13
    commands.extend(
        [
            "0.40 0.40 0.40 rg",
            "BT /F1 13 Tf 52 36 Td (Industrial AI Process Logic) Tj ET",
            f"BT /F1 13 Tf 858 36 Td ({index}/{total}) Tj ET",
            "Q",
        ]
    )
    return ("\n".join(commands) + "\n").encode("ascii")


def build_pdf(slides: list[tuple[str, list[str]]]) -> bytes:
    objects: list[bytes] = []
    pages_object_id = 2
    font_object_id = 3
    page_ids: list[int] = []

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for index, (title, bullets) in enumerate(slides, start=1):
        page_id = len(objects) + 1
        content_id = page_id + 1
        stream = page_stream(title, bullets, index, len(slides))
        page = (
            f"<< /Type /Page /Parent {pages_object_id} 0 R "
            f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 {font_object_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        ).encode("ascii")
        content = b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"endstream"
        objects.append(page)
        objects.append(content)
        page_ids.append(page_id)

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_id, payload in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{object_id} 0 obj\n".encode("ascii"))
        output.extend(payload)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the submission slide PDF from SLIDES.md.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Slide markdown source")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT), help="PDF output path")
    args = parser.parse_args()
    source = Path(args.source)
    if not source.is_absolute():
        source = ROOT / source
    output = Path(args.out)
    if not output.is_absolute():
        output = ROOT / output
    pdf = build_pdf(parse_slides(source.resolve()))
    output.resolve().write_bytes(pdf)
    print(f"Wrote {output.resolve().relative_to(ROOT)} ({len(pdf)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
