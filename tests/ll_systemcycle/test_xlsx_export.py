"""CPU-only checks for the deployed standard-library OOXML writer."""

from __future__ import annotations

import importlib.util
import posixpath
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pytest


spec = importlib.util.spec_from_file_location("ll_systemcycle_xlsx", Path(__file__).with_name("xlsx_export.py"))
assert spec and spec.loader
xlsx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xlsx)
NS = {"s": xlsx.MAIN, "r": xlsx.PKG_REL}


def payload():
    return {
        "metadata": {"baseline_revision": "baseline-test", "source_revision": "source-test",
                     "frequency_hz": 1000000000, "notes": ["仅CPU合成测试", "不是设备测量"]},
        "performance": {"headers": ["实验组", "算子", "耗时 (µs)", "状态"],
                        "rows": [["合成检查", "dispatch", None, "未测"]]},
        "operations": {
            "dispatch": {
                "headers": ["实验组", "Rank", "服务器", "本地卡", "核号", "角色", "轮次", "任务数", "耗时 (µs)"],
                "rows": [["合成检查", 0, "local", 0, 0, "发送后接收", 1, 0,
                          {"formula": "=('原始Cycle'!E2)/1000", "value": 1.25, "number_format": "0.000"}]],
            },
            "combine": {"headers": ["Rank", "核号", "耗时 (µs)"], "rows": [[0, 0, 0.0]]},
        },
        "definitions": {"headers": ["阶段", "换算", "边界", "包含关系"],
                        "rows": [["耗时", "delta / 1000", "仅同核差分。\n提交并不等同完成。", "细分包含在主项"]]},
        "raw": {"headers": ["算子", "Rank", "核号", "绝对Cycle", "delta ticks"],
                "rows": [["dispatch", 0, 0, "18446744073709551615", 1250]]},
    }


def xml_part(archive, name):
    return ET.fromstring(archive.read(name))


def cell(root, address):
    return root.find(f".//s:c[@r='{address}']", NS)


def test_full_workbook_relationships_values_and_layout(tmp_path):
    folder = tmp_path / "含 空格目录"
    folder.mkdir()
    path = xlsx.export_workbook(payload(), folder / "阶段报告.xlsx")
    assert path.is_file()
    with ZipFile(path) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            if name.endswith((".xml", ".rels")):
                xml_part(archive, name)
        workbook = xml_part(archive, "xl/workbook.xml")
        assert [item.attrib["name"] for item in workbook.find("s:sheets", NS)] == [
            "性能对照", "Dispatch逐核耗时", "Combine逐核耗时", "阶段说明", "原始Cycle"]
        rels = xml_part(archive, "xl/_rels/workbook.xml.rels")
        relationships = {item.attrib["Id"]: item.attrib for item in rels}
        for sheet in workbook.find("s:sheets", NS):
            reference = sheet.attrib[f"{{{xlsx.REL}}}id"]
            target = posixpath.normpath(posixpath.join("xl", relationships[reference]["Target"]))
            assert target in archive.namelist()
        core_sheet = xml_part(archive, "xl/worksheets/sheet2.xml")
        assert core_sheet.find("s:sheetViews/s:sheetView", NS).attrib["showGridLines"] == "0"
        assert core_sheet.find("s:sheetViews/s:sheetView/s:pane", NS).attrib["topLeftCell"] == "I2"
        assert core_sheet.find("s:sheetData/s:row", NS).attrib["ht"] == "55"
        assert cell(core_sheet, "F2").find("s:is/s:t", NS).text == "发送后接收"
        formula = cell(core_sheet, "I2")
        assert formula.find("s:f", NS).text == "('原始Cycle'!E2)/1000"
        assert formula.find("s:v", NS).text == "1.25"
        raw = xml_part(archive, "xl/worksheets/sheet5.xml")
        assert cell(raw, "D2").attrib["t"] == "inlineStr"
        assert cell(raw, "D2").find("s:is/s:t", NS).text == "18446744073709551615"
        assert cell(raw, "E2").find("s:v", NS).text == "1250"
        performance = xml_part(archive, "xl/worksheets/sheet1.xml")
        assert cell(performance, "C2") is None
        combine = xml_part(archive, "xl/worksheets/sheet3.xml")
        assert cell(combine, "C2").find("s:v", NS).text == "0.0"
        definitions = xml_part(archive, "xl/worksheets/sheet4.xml")
        assert float(definitions.find("s:sheetData/s:row[@r='2']", NS).attrib["ht"]) > 22
        styles = xml_part(archive, "xl/styles.xml")
        assert styles.find("s:numFmts/s:numFmt", NS).attrib["formatCode"] == "0.000"
        assert styles.find("s:fonts/s:font/s:name", NS).attrib["val"] == "Arial"
        assert styles.find("s:fills/s:fill[3]/s:patternFill/s:fgColor", NS).attrib["rgb"] == "FF173D62"
        assert str(tmp_path).encode() not in archive.read("docProps/core.xml")
        assert "不是设备测量" in archive.read("docProps/custom.xml").decode()


def test_single_operation_has_four_sheets(tmp_path):
    data = payload()
    del data["operations"]["combine"]
    path = xlsx.export_workbook(data, tmp_path / "one.xlsx")
    with ZipFile(path) as archive:
        root = xml_part(archive, "xl/workbook.xml")
        assert len(root.find("s:sheets", NS)) == 4


def test_strings_are_never_formulas_and_caches_keep_types(tmp_path):
    values = ["=HYPERLINK(\"https://example.invalid\",\"text\")", "+SUM(A1)", "@SUM(A1)",
              "-1+1", "  中文 & < >  ", None, 0, 0.0, 18446744073709551615,
              {"formula": '=IF(A2="x","",1)', "value": None},
              {"formula": '="中文"', "value": "中文"},
              {"formula": "=0", "value": 0}, {"formula": "=TRUE()", "value": True}]
    path = xlsx.export_tables({"安全文本": {"headers": [str(i) for i in range(len(values))], "rows": [values]}},
                              tmp_path / "types.xlsx")
    with ZipFile(path) as archive:
        root = xml_part(archive, "xl/worksheets/sheet1.xml")
        for col, expected in enumerate(values[:5], 1):
            item = cell(root, f"{xlsx.column_name(col)}2")
            assert item.attrib["t"] == "inlineStr"
            assert item.find("s:f", NS) is None
            assert item.find("s:is/s:t", NS).text == expected
        assert cell(root, "F2") is None
        assert cell(root, "G2").find("s:v", NS).text == "0"
        assert cell(root, "H2").find("s:v", NS).text == "0.0"
        assert cell(root, "I2").find("s:is/s:t", NS).text == "18446744073709551615"
        assert cell(root, "J2").attrib["t"] == "str"
        assert cell(root, "J2").find("s:v", NS).text is None
        assert cell(root, "K2").find("s:v", NS).text == "中文"
        assert cell(root, "L2").find("s:v", NS).text == "0"
        assert cell(root, "M2").attrib["t"] == "b"


@pytest.mark.parametrize("bad", [
    {"headers": ["A", "B"], "rows": [[1]]},
    {"headers": ["A"], "rows": [[1, 2]]},
    {"headers": [], "rows": []},
    {"headers": ["A"], "rows": [[float("nan")]]},
    {"headers": ["A"], "rows": [[float("inf")]]},
    {"headers": ["A"], "rows": [["bad\x00xml"]]},
    {"headers": ["A"], "rows": [[{"formula": "=", "value": 1}]]},
    {"headers": ["A"], "rows": [[{"formula": "=1", "value": 18446744073709551615}]]},
    {"headers": ["A"], "rows": [[1]], "freeze_column": -1},
    {"headers": ["A"], "rows": [[1]], "widths": [float("nan")]},
])
def test_invalid_tables_rejected_before_any_output(tmp_path, bad):
    with pytest.raises(ValueError):
        xlsx.export_tables({"数据": bad}, tmp_path / "invalid.xlsx")
    assert list(tmp_path.iterdir()) == []


def test_existing_file_is_not_overwritten(tmp_path):
    path = tmp_path / "existing.xlsx"
    path.write_bytes(b"original data")
    with pytest.raises(FileExistsError):
        xlsx.export_workbook(payload(), path)
    assert path.read_bytes() == b"original data"
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_publish_rejects_destination_created_during_export(tmp_path, monkeypatch):
    path = tmp_path / "race.xlsx"
    real_link = xlsx.os.link

    def race(source, destination):
        destination.write_bytes(b"created concurrently")
        return real_link(source, destination)

    monkeypatch.setattr(xlsx.os, "link", race)
    with pytest.raises(FileExistsError):
        xlsx.export_workbook(payload(), path)
    assert path.read_bytes() == b"created concurrently"
    assert list(tmp_path.iterdir()) == [path]


def test_failed_package_write_cleans_only_own_temporary(tmp_path, monkeypatch):
    retained = tmp_path / "history.txt"
    retained.write_bytes(b"keep")

    def fail(*args):
        raise RuntimeError("injected writer failure")

    monkeypatch.setattr(xlsx, "_write_package", fail)
    with pytest.raises(RuntimeError, match="injected"):
        xlsx.export_workbook(payload(), tmp_path / "failed.xlsx")
    assert list(tmp_path.iterdir()) == [retained]


def test_relative_paths_duplicate_names_and_missing_parent_rejected(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        xlsx.export_workbook(payload(), "relative.xlsx")
    with pytest.raises(ValueError, match="unique"):
        xlsx.export_tables([{"name": "Same", "headers": ["A"], "rows": []},
                            {"name": "same", "headers": ["A"], "rows": []}], tmp_path / "duplicate.xlsx")
    with pytest.raises(FileNotFoundError):
        xlsx.export_workbook(payload(), tmp_path / "absent" / "failed.xlsx")
    assert list(tmp_path.iterdir()) == []


def test_independent_openpyxl_readback_when_available(tmp_path):
    """Optional interoperability check; deployed exporter has no dependency."""
    openpyxl = pytest.importorskip("openpyxl")
    path = xlsx.export_workbook(payload(), tmp_path / "readback.xlsx")
    workbook = openpyxl.load_workbook(path, read_only=False, data_only=False)
    assert workbook["Dispatch逐核耗时"]["I2"].value == "=('原始Cycle'!E2)/1000"
    assert workbook["Dispatch逐核耗时"].freeze_panes == "I2"
    assert workbook["Dispatch逐核耗时"]["I2"].number_format == "0.000"
    assert workbook["原始Cycle"]["D2"].value == "18446744073709551615"
    assert workbook["原始Cycle"]["D2"].data_type == "s"
    assert workbook["性能对照"]["C2"].value is None
    workbook.close()
    cached = openpyxl.load_workbook(path, read_only=True, data_only=True)
    assert cached["Dispatch逐核耗时"]["I2"].value == 1.25
    assert cached["Combine逐核耗时"]["C2"].value == 0
    cached.close()
