"""Dependency-free XLSX export for offline SystemCycle measurements.

This module formats already validated measurement tables. It does not read
profiling files, calculate timings, or execute spreadsheet formulas. Formula
cells must be explicit ``{"formula": "=...", "value": cached_value}`` objects;
ordinary strings are always text, including strings beginning with ``=``.
"""

from __future__ import annotations

import math
import os
import re
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT = "http://schemas.openxmlformats.org/package/2006/content-types"
XML_HEAD = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
_BAD_XML = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")
_BAD_SHEET = re.compile(r"[\\/*?:\[\]]")
_METADATA_KEYS = ("baseline_revision", "source_revision", "frequency_hz", "notes")


def _text(value: str, context: str) -> str:
    if _BAD_XML.search(value):
        raise ValueError(f"{context}: invalid XML character")
    if len(value) > 32767:
        raise ValueError(f"{context}: Excel text limit of 32767 characters exceeded")
    return value


def _value(value: Any, context: str) -> None:
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{context}: non-finite numeric value")
        return
    if isinstance(value, str):
        _text(value, context)
        return
    raise ValueError(f"{context}: unsupported cell value {type(value).__name__}")


def _cell(value: Any, context: str) -> None:
    if isinstance(value, dict):
        if set(value) - {"formula", "value", "number_format"}:
            raise ValueError(f"{context}: unsupported formula cell field")
        formula = value.get("formula")
        if not isinstance(formula, str) or not formula.lstrip("= ").strip():
            raise ValueError(f"{context}: formula must be a nonempty string")
        _text(formula, context)
        _value(value.get("value"), context)
        if "number_format" in value:
            if not isinstance(value["number_format"], str) or not value["number_format"]:
                raise ValueError(f"{context}: number_format must be a nonempty string")
            _text(value["number_format"], context)
        cached = value.get("value")
        if isinstance(cached, int) and not isinstance(cached, bool) and len(str(abs(cached))) > 15:
            raise ValueError(f"{context}: formula cache exceeds Excel integer precision; use relative ticks")
    else:
        _value(value, context)


def column_name(index: int) -> str:
    """Return a one-based Excel column label."""
    if not isinstance(index, int) or not 1 <= index <= 16384:
        raise ValueError("column index must be between 1 and 16384")
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _normalize_table(name: str, table: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(name, str) or not name or len(name) > 31 or _BAD_SHEET.search(name):
        raise ValueError(f"invalid Excel sheet name: {name!r}")
    _text(name, "sheet name")
    if name.startswith("'") or name.endswith("'"):
        raise ValueError("sheet name must not begin or end with an apostrophe")
    headers, rows = table.get("headers"), table.get("rows")
    if not isinstance(headers, (list, tuple)) or not 1 <= len(headers) <= 16384:
        raise ValueError(f"{name}: headers must contain 1..16384 strings")
    if not all(isinstance(item, str) for item in headers):
        raise ValueError(f"{name}: headers must be strings")
    for item in headers:
        _text(item, name)
    if not isinstance(rows, (list, tuple)) or len(rows) > 1048575:
        raise ValueError(f"{name}: rows must be a sequence of at most 1048575 rows")
    for number, row in enumerate(rows, 2):
        if not isinstance(row, (list, tuple)) or len(row) != len(headers):
            raise ValueError(f"{name}: row {number} is not rectangular ({len(headers)} columns required)")
        for col, value in enumerate(row, 1):
            _cell(value, f"{name}!{column_name(col)}{number}")
    freeze = table.get("freeze_column", 1)
    if not isinstance(freeze, int) or isinstance(freeze, bool) or not 0 <= freeze <= len(headers):
        raise ValueError(f"{name}: freeze_column must be between 0 and column count")
    widths = table.get("widths")
    if widths is None:
        widths = [min(18.0, max(12.0, len(header) + 2.0)) if col < freeze else 19.0
                  for col, header in enumerate(headers)]
    if not isinstance(widths, (list, tuple)) or len(widths) != len(headers):
        raise ValueError(f"{name}: widths must match headers")
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not 1 <= v <= 255
           for v in widths):
        raise ValueError(f"{name}: column widths must be finite values in 1..255")
    number_formats = table.get("number_formats")
    if number_formats is None:
        number_formats = ["0.000" if any(unit in header.lower() for unit in ("µs", "μs", "_us", "(us)", "latency"))
                          else None for header in headers]
    if not isinstance(number_formats, (list, tuple)) or len(number_formats) != len(headers):
        raise ValueError(f"{name}: number_formats must match headers")
    for number_format in number_formats:
        if number_format is not None:
            if not isinstance(number_format, str) or not number_format:
                raise ValueError(f"{name}: number_formats entries must be nonempty strings or None")
            _text(number_format, name)
    return {"name": name, "headers": headers, "rows": rows, "freeze_column": freeze,
            "widths": widths, "wrap": bool(table.get("wrap", False)), "number_formats": number_formats}


class _Styles:
    def __init__(self) -> None:
        self.formats = {"0.000": 164}
        # numFmtId, fontId, fillId, wrap, identifier. Index 1 is the table header.
        self.entries = [(0, 0, 0, False, False), (0, 1, 2, True, False),
                        (1, 0, 0, False, False), (164, 0, 0, False, False)]

    def index(self, value: Any, wrap: bool, column_format: str | None = None, identifier: bool = False) -> int:
        number_format = column_format
        if isinstance(value, dict):
            number_format = value.get("number_format", column_format)
            value = value.get("value")
        if number_format is None:
            number_format = "0.000" if isinstance(value, float) else "0" if isinstance(value, int) else "General"
        if number_format in ("General", "0", "@"):
            fmt = {"General": 0, "0": 1, "@": 49}[number_format]
        else:
            fmt = self.formats.setdefault(number_format, 164 + len(self.formats))
        entry = (fmt, 0, 0, wrap, identifier and type(value) is int)
        if entry not in self.entries:
            self.entries.append(entry)
        return self.entries.index(entry)

    def xml(self) -> bytes:
        root = ET.Element("styleSheet", xmlns=MAIN)
        formats = ET.SubElement(root, "numFmts", count=str(len(self.formats)))
        for code, number in self.formats.items():
            ET.SubElement(formats, "numFmt", numFmtId=str(number), formatCode=code)
        fonts = ET.SubElement(root, "fonts", count="2")
        for bold, color in ((False, "FF172B3A"), (True, "FFFFFFFF")):
            font = ET.SubElement(fonts, "font")
            ET.SubElement(font, "sz", val="10")
            if bold:
                ET.SubElement(font, "b")
            ET.SubElement(font, "color", rgb=color)
            ET.SubElement(font, "name", val="Arial")
            ET.SubElement(font, "family", val="2")
        fills = ET.SubElement(root, "fills", count="3")
        for pattern in ("none", "gray125", "solid"):
            fill = ET.SubElement(ET.SubElement(fills, "fill"), "patternFill", patternType=pattern)
            if pattern == "solid":
                ET.SubElement(fill, "fgColor", rgb="FF173D62")
                ET.SubElement(fill, "bgColor", indexed="64")
        borders = ET.SubElement(root, "borders", count="1")
        border = ET.SubElement(borders, "border")
        for edge in ("left", "right", "top", "bottom", "diagonal"):
            ET.SubElement(border, edge)
        style_xfs = ET.SubElement(root, "cellStyleXfs", count="1")
        ET.SubElement(style_xfs, "xf", numFmtId="0", fontId="0", fillId="0", borderId="0")
        xfs = ET.SubElement(root, "cellXfs", count=str(len(self.entries)))
        for number, (fmt, font, fill, wrap, identifier) in enumerate(self.entries):
            xf = ET.SubElement(xfs, "xf", numFmtId=str(fmt), fontId=str(font), fillId=str(fill),
                               borderId="0", xfId="0", applyNumberFormat="1", applyAlignment="1")
            alignment = {"vertical": "center", "wrapText": "1" if wrap else "0"}
            if number == 1 or identifier:
                alignment["horizontal"] = "center"
            ET.SubElement(xf, "alignment", **alignment)
        cell_styles = ET.SubElement(root, "cellStyles", count="1")
        ET.SubElement(cell_styles, "cellStyle", name="Normal", xfId="0", builtinId="0")
        return _xml(root)


def _xml(element: ET.Element) -> bytes:
    return XML_HEAD.encode("utf-8") + ET.tostring(element, encoding="utf-8")


def _inline_cell(address: str, value: str, style: int) -> str:
    return (f'<c r="{address}" s="{style}" t="inlineStr"><is><t xml:space="preserve">'
            f'{escape(value)}</t></is></c>')


def _cell_xml(address: str, item: Any, style: int) -> str:
    if item is None:
        return ""
    if isinstance(item, dict):
        formula = item["formula"].strip()
        if formula.startswith("="):
            formula = formula[1:]
        value = item.get("value")
        kind = ' t="str"' if value is None or isinstance(value, str) else ' t="b"' if isinstance(value, bool) else ""
        cached = "" if value is None else "1" if value is True else "0" if value is False else str(value)
        return f'<c r="{address}" s="{style}"{kind}><f>{escape(formula)}</f><v>{escape(cached)}</v></c>'
    if isinstance(item, str):
        return _inline_cell(address, item, style)
    if isinstance(item, int) and not isinstance(item, bool) and len(str(abs(item))) > 15:
        return _inline_cell(address, str(item), style)
    if isinstance(item, bool):
        return f'<c r="{address}" s="{style}" t="b"><v>{int(item)}</v></c>'
    return f'<c r="{address}" s="{style}"><v>{item}</v></c>'


def _row_height(row: Sequence[Any], widths: Sequence[float], wrap: bool) -> float:
    if not wrap:
        return 22.0
    lines = 1
    for item, width in zip(row, widths):
        if isinstance(item, dict):
            item = item.get("value")
        if not isinstance(item, str):
            continue
        count = 0
        for line in item.split("\n"):
            size = sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in line)
            count += max(1, math.ceil(size / max(1, width - 2)))
        lines = max(lines, count)
    return min(240.0, max(22.0, lines * 15.0 + 6.0))


def _write_sheet(archive: ZipFile, index: int, table: Mapping[str, Any], styles: _Styles) -> None:
    with archive.open(f"xl/worksheets/sheet{index}.xml", "w") as stream:
        def write(value: str) -> None:
            stream.write(value.encode("utf-8"))

        width = len(table["headers"])
        last_row = len(table["rows"]) + 1
        freeze = table["freeze_column"]
        pane = "bottomRight" if freeze else "bottomLeft"
        # Excel does not permit a pane outside its last column.
        top_left = f"{column_name(min(freeze + 1, 16384))}2"
        write(XML_HEAD + f'<worksheet xmlns="{MAIN}"><dimension ref="A1:{column_name(width)}{last_row}"/>')
        write('<sheetViews><sheetView workbookViewId="0" showGridLines="0">')
        write(f'<pane xSplit="{freeze}" ySplit="1" topLeftCell="{top_left}" activePane="{pane}" state="frozen"/>')
        write(f'<selection pane="{pane}" activeCell="{top_left}" sqref="{top_left}"/></sheetView></sheetViews>')
        write('<sheetFormatPr defaultRowHeight="22"/><cols>')
        for col, value in enumerate(table["widths"], 1):
            write(f'<col min="{col}" max="{col}" width="{value}" customWidth="1"/>')
        write('</cols><sheetData><row r="1" ht="55" customHeight="1">')
        for col, value in enumerate(table["headers"], 1):
            write(_inline_cell(f"{column_name(col)}1", value, 1))
        write('</row>')
        for number, row in enumerate(table["rows"], 2):
            height = _row_height(row, table["widths"], table["wrap"])
            write(f'<row r="{number}" ht="{height}" customHeight="1">')
            for col, value in enumerate(row, 1):
                style = styles.index(value, table["wrap"], table["number_formats"][col - 1], col <= freeze)
                write(_cell_xml(f"{column_name(col)}{number}", value, style))
            write('</row>')
        write('</sheetData><pageMargins left="0.25" right="0.25" top="0.5" bottom="0.5" header="0.2" footer="0.2"/></worksheet>')


def _relationships(items: Sequence[tuple[str, str, str]]) -> bytes:
    root = ET.Element("Relationships", xmlns=PKG_REL)
    for ident, kind, target in items:
        ET.SubElement(root, "Relationship", Id=ident, Type=kind, Target=target)
    return _xml(root)


def _write_package(archive: ZipFile, tables: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any]) -> None:
    types = ET.Element("Types", xmlns=CONTENT)
    ET.SubElement(types, "Default", Extension="rels", ContentType="application/vnd.openxmlformats-package.relationships+xml")
    ET.SubElement(types, "Default", Extension="xml", ContentType="application/xml")
    parts = [("/xl/workbook.xml", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"),
             ("/xl/styles.xml", "application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"),
             ("/docProps/core.xml", "application/vnd.openxmlformats-package.core-properties+xml"),
             ("/docProps/app.xml", "application/vnd.openxmlformats-officedocument.extended-properties+xml"),
             ("/docProps/custom.xml", "application/vnd.openxmlformats-officedocument.custom-properties+xml")]
    parts.extend((f"/xl/worksheets/sheet{i}.xml", "application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml")
                 for i in range(1, len(tables) + 1))
    for part, content_type in parts:
        ET.SubElement(types, "Override", PartName=part, ContentType=content_type)
    archive.writestr("[Content_Types].xml", _xml(types))
    archive.writestr("_rels/.rels", _relationships([
        ("rId1", REL + "/officeDocument", "xl/workbook.xml"),
        ("rId2", "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties", "docProps/core.xml"),
        ("rId3", REL + "/extended-properties", "docProps/app.xml"),
        ("rId4", REL + "/custom-properties", "docProps/custom.xml")]))
    workbook = ET.Element("workbook", xmlns=MAIN, attrib={"xmlns:r": REL})
    views = ET.SubElement(workbook, "bookViews")
    ET.SubElement(views, "workbookView", activeTab="0")
    sheets = ET.SubElement(workbook, "sheets")
    for index, table in enumerate(tables, 1):
        ET.SubElement(sheets, "sheet", name=table["name"], sheetId=str(index), attrib={"r:id": f"rId{index}"})
    ET.SubElement(workbook, "calcPr", calcId="191029", fullCalcOnLoad="1", forceFullCalc="1")
    archive.writestr("xl/workbook.xml", _xml(workbook))
    archive.writestr("xl/_rels/workbook.xml.rels", _relationships([
        *[(f"rId{i}", REL + "/worksheet", f"worksheets/sheet{i}.xml") for i in range(1, len(tables) + 1)],
        (f"rId{len(tables)+1}", REL + "/styles", "styles.xml")]))
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    core = (XML_HEAD + '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:creator>DeepEP-LL</dc:creator>'
            '<dc:title>SystemCycle 阶段耗时</dc:title>'
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created></cp:coreProperties>')
    archive.writestr("docProps/core.xml", core.encode("utf-8"))
    app = ET.Element("Properties", xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties")
    ET.SubElement(app, "Application").text = "DeepEP-LL offline SystemCycle exporter"
    archive.writestr("docProps/app.xml", _xml(app))
    custom = ET.Element("Properties", xmlns="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties",
                        attrib={"xmlns:vt": "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"})
    for number, (key, value) in enumerate(metadata.items(), 2):
        prop = ET.SubElement(custom, "property", fmtid="{D5CDD505-2E9C-101B-9397-08002B2CF9AE}", pid=str(number), name=key)
        ET.SubElement(prop, "vt:lpwstr").text = value
    archive.writestr("docProps/custom.xml", _xml(custom))
    styles = _Styles()
    for index, table in enumerate(tables, 1):
        _write_sheet(archive, index, table, styles)
    archive.writestr("xl/styles.xml", styles.xml())


def export_tables(tables: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
                  path: str | os.PathLike[str], *, metadata: Mapping[str, Any] | None = None) -> Path:
    """Create a new workbook from named rectangular tables without overwriting.

    ``tables`` is either an insertion-ordered mapping of sheet name to table or
    a sequence of tables with a ``name`` key. Each table has ``headers``,
    ``rows``, and optional ``freeze_column``, ``widths``, ``number_formats`` and
    ``wrap`` fields. Unit-labelled microsecond columns default to ``0.000``;
    individual formula cells can override the column number format.
    The absolute output path's parent must already exist. A completed sibling
    temporary file is atomically linked into place; an existing destination
    (including a symlink) is never replaced. Filesystems without hard-link
    support fail explicitly, leaving no partial destination.
    """
    if isinstance(tables, Mapping):
        prepared = [_normalize_table(name, table) for name, table in tables.items()]
    else:
        prepared = [_normalize_table(table.get("name"), table) for table in tables]
    names = [table["name"].casefold() for table in prepared]
    if not names or len(set(names)) != len(names):
        raise ValueError("workbook needs at least one sheet and unique sheet names")
    properties: dict[str, str] = {}
    for key in _METADATA_KEYS:
        if metadata and key in metadata:
            value = metadata[key]
            if key == "notes" and isinstance(value, (list, tuple)):
                value = "\n".join(str(item) for item in value)
            properties[key] = _text(str(value), f"metadata.{key}")
    destination = Path(path)
    if not destination.is_absolute():
        raise ValueError("output path must be absolute and resolved by the caller")
    parent = destination.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("output parent must be a directory")
    destination = parent / destination.name
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite workbook: {destination}")
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=parent)
    temporary = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w+b") as file:
            with ZipFile(file, "w", compression=ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
                _write_package(archive, prepared, properties)
            file.flush()
            os.fsync(file.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def export_workbook(payload: Mapping[str, Any], path: str | os.PathLike[str]) -> Path:
    """Export the standard performance/core timings/definitions/raw workbook.

    Required keys are ``performance``, ``operations``, ``definitions`` and
    ``raw``. ``operations`` contains dispatch and/or combine tables. Optional
    metadata properties are baseline_revision, source_revision, frequency_hz
    and notes. Values are written as provided; no samples are invented.
    """
    operations = payload.get("operations")
    if not isinstance(operations, Mapping) or not operations or set(operations) - {"dispatch", "combine"}:
        raise ValueError("operations must contain dispatch and/or combine tables")
    tables: dict[str, dict[str, Any]] = {}
    tables["性能对照"] = {"freeze_column": 1, **payload["performance"]}
    for operation in ("dispatch", "combine"):
        if operation in operations:
            table = operations[operation]
            tables[f"{operation.title()}逐核耗时"] = {"freeze_column": min(8, len(table["headers"])), **table}
    definitions = payload["definitions"]
    tables["阶段说明"] = {"freeze_column": 1, "wrap": True,
                         "widths": [37.0, 40.0, *([72.0] * max(0, len(definitions["headers"]) - 2))][:len(definitions["headers"])],
                         **definitions}
    raw = payload["raw"]
    tables["原始Cycle"] = {"freeze_column": min(3, len(raw["headers"])),
                          "widths": [18.0 if i < 3 else 22.0 for i in range(len(raw["headers"]))], **raw}
    return export_tables(tables, path, metadata=payload.get("metadata"))
