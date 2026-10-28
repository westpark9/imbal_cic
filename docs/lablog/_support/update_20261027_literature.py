"""Apply the source-checked literature update without moving historical rows.

Run from the project root, or pass --root. Search decisions and evidence are in
evidence/references/20261027/literature_update.json; this script does no research.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

import openpyxl
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


DATE = "2026-10-27"
LATEST = "20261027_최신목록"
NEW = "20261027_신규IDS"
AUDIT = "20261027_게재점검"
SCREENED = "20261027_검색검토"
ADDITIONAL_CHAPTER = {
    "LotBoNC": 3, "TFM-Systematic-Survey": 3, "TabFM / TabFM+": 2,
    "TabPFN-3.5": 1, "StreamTFM-Study": 2, "TFM-Ensemble-Diversity": 2,
}
FIELDS = [
    "chapter", "abb", "title", "summary", "type", "grade", "date", "code",
    "motivation", "goal", "methods", "datasets", "baselines", "experiments",
    "metrics", "findings", "limitations", "source", "year", "url",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell_signature(cell):
    """Preserve every historical value, annotation and style unless authorized."""
    return (
        cell.value, copy(cell._style),
        (cell.comment.text, cell.comment.author) if cell.comment else None,
        (cell.hyperlink.target, cell.hyperlink.location) if cell.hyperlink else None,
    )


def date_key(value):
    match = re.search(r"(20\d{2})(?:-(\d{2}))?(?:-(\d{2}))?", str(value))
    return int("".join((part or "00") for part in match.groups())) if match else 0


def table_sheet(workbook, name, headers, notes, widths):
    if name in workbook:
        del workbook[name]
    sheet = workbook.create_sheet(name)
    sheet.sheet_view.showGridLines = False
    sheet.sheet_view.zoomScale = 85
    sheet.sheet_properties.pageSetUpPr.fitToPage = True
    sheet.page_setup.orientation = "landscape"
    sheet.page_setup.paperSize = sheet.PAPERSIZE_A3
    sheet.page_setup.fitToWidth = 1
    for row, note in enumerate(notes, 1):
        sheet.merge_cells(start_row=row, start_column=1, end_row=row, end_column=len(headers))
        cell = sheet.cell(row, 1, note)
        cell.font = Font(name="맑은 고딕", size=11, bold=row == 1,
                         color="FFFFFF" if row == 1 else "244763")
        cell.fill = PatternFill("solid", fgColor="244763" if row == 1 else "EAF0F5")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        sheet.row_dimensions[row].height = 28 if row == 1 else 32
    header = len(notes) + 1
    for col, value in enumerate(headers, 1):
        cell = sheet.cell(header, col, value)
        cell.font = Font(name="맑은 고딕", size=10, color="FFFFFF", bold=True)
        cell.fill = PatternFill("solid", fgColor="244763")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        sheet.column_dimensions[get_column_letter(col)].width = widths.get(col, 42)
    sheet.row_dimensions[header].height = 38
    sheet.freeze_panes = f"D{header + 1}"
    sheet.print_title_rows = f"1:{header}"
    return sheet, header


def append_row(sheet, values, source_cells=None, accent=None):
    sheet.append(values)
    row = sheet.max_row
    for col, cell in enumerate(sheet[row], 1):
        cell.font = Font(name="맑은 고딕", size=10, color="243746")
        if source_cells and col <= len(source_cells):
            original = source_cells[col - 1]
            cell.font = copy(original.font)
            cell.number_format = original.number_format
            if original.comment:
                cell.comment = copy(original.comment)
        cell.fill = PatternFill("solid", fgColor=accent or ("F1F5F9" if row % 2 else "FFFFFF"))
        cell.border = Border(bottom=Side(style="hair", color="D6DFE6"))
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        if isinstance(cell.value, str) and re.fullmatch(r"https?://\S+", cell.value):
            cell.hyperlink = cell.value
            font = copy(cell.font)
            font.color, font.underline = "0563C1", "single"
            cell.font = font
    sheet.row_dimensions[row].height = 110
    return row


def finish_table(sheet, header):
    sheet.auto_filter.ref = f"A{header}:{get_column_letter(sheet.max_column)}{sheet.max_row}"
    sheet.print_options.horizontalCentered = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    refs = root / "docs/lablog/evidence/references"
    evidence = refs / "20261027"
    target = refs / "tabpfn_papers_survey.xlsx"
    plan = json.loads((evidence / "literature_update.json").read_text())
    inventory = json.loads((evidence / "inventory_before.json").read_text())
    updates = {p["abb"]: p for p in plan["updates"]}
    related = {p["abb"]: p for p in plan["related_version_notes"]}
    original = {p["abb"]: p for p in inventory}
    assert len(original) == len(inventory) == 134
    preprints = [p for p in inventory if p["grade"] == "프리프린트"]
    assert len(preprints) == 61 and len(updates) == 12
    before_hash = sha(target)
    audit_path = evidence / "workbook_validation.json"
    previous_audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    backup = Path(tempfile.mkdtemp(prefix="literature_20261027_")) / target.name
    shutil.copy2(target, backup)
    workbook = openpyxl.load_workbook(target)
    historical = [s for s in workbook if s.title not in {LATEST, NEW, AUDIT, SCREENED}]
    snapshots = {
        s.title: {(c.row, c.column): cell_signature(c) for row in s for c in row}
        for s in historical
    }
    original_merges = {s.title: str(s.merged_cells) for s in historical}
    changed, authorized = [], set()

    def update_cell(sheet, row, col, value, note=None, url=False):
        cell = sheet.cell(row, col)
        old = cell.value
        cell.value = value
        if url:
            cell.hyperlink = value
        if note:
            marker = f"[{DATE} 게재 점검]"
            prior = cell.comment.text if cell.comment else ""
            if marker not in prior:
                cell.comment = Comment((prior + "\n\n" if prior else "") + marker + "\n" + note, "Literature review")
        authorized.add((sheet.title, row, col))
        if old != value:
            changed.append({"sheet": sheet.title, "cell": cell.coordinate, "before": old, "after": value})

    for abb, item in updates.items():
        prior = original[abb]
        sheet, row = workbook[prior["sheet"]], prior["row"]
        assert sheet.cell(row, 2).value == abb
        note = item["note"] + "\n근거: " + "\n".join(item["evidence"]) + "\n기존 preprint: " + str(prior["url"])
        for col, key in [(6, "grade"), (18, "source"), (19, "year"), (20, "url")]:
            update_cell(sheet, row, col, item[key], note, url=col == 20)
        if item.get("published_title"):
            update_cell(sheet, row, 3, sheet.cell(row, 3).value,
                        "정식 게재 제목: " + item["published_title"] + "\n" + note)

    for abb, item in related.items():
        prior = original[abb]
        sheet, row = workbook[prior["sheet"]], prior["row"]
        update_cell(sheet, row, 18, prior["source"] + "; " + item["source_append"],
                    item["note"] + "\n" + "\n".join(item["evidence"]))

    for abb in ["WFE-Tab", "TabPFN-IIoT"]:
        item = original[abb]
        sheet, row = workbook[item["sheet"]], item["row"]
        old = sheet.cell(row, 20).value
        clean = old.strip().rstrip("\\")
        if clean != old:
            update_cell(sheet, row, 20, clean, "URL 끝의 공백/역슬래시를 제거했다. 문헌 분류는 유지한다.", url=True)
    sheet = workbook["Lit TabPFN"]
    sorting = sheet.cell(2, 3).value.split(" [2026-10-27 갱신]")[0]
    update_cell(sheet, 2, 3, sorting + " [2026-10-27 갱신] 행 번호를 참조하는 과거 검토 기록을 위해 이 시트의 행 위치는 유지. 통합 재정렬과 신규 문헌은 20261027_최신목록 참조.")

    base_headers = [workbook["Lit TabPFN"].cell(5, c).value for c in range(1, 21)]
    headers = base_headers + ["이번 점검 상태", "확인일", "공식 근거 URL", "기존 preprint URL", "원본 위치", "이번 연구와의 관계·Discussion", "검토 범위·비고"]
    common_widths = {1: 6, 2: 25, 3: 66, 4: 43, 5: 30, 6: 16, 7: 24, 8: 30, 18: 56, 19: 9, 20: 48, 21: 30, 22: 14, 23: 54, 24: 45, 25: 28}
    latest, latest_header = table_sheet(workbook, LATEST, headers, [
        "2026-10-27 최신 문헌 목록 | 137편 = 기존 134편 + 신규 IDS 3편 | 기존 preprint 61편 점검: 게재/채택 확인 12편, 미확인 49편",
        "정렬: Ch.1 공개일 최신순; Ch.2·3 출처등급 → Year → 공개일 최신순. 기존 메모·취소선 유지. 게재일 열은 기존 공개일 의미를 유지하고, 정식 게재일은 게재점검 시트에서 구별한다.",
        "갱신 12편은 학회/저널 8편(WWW Companion Demo 포함) + 워크숍 4편이다. 나머지 기존 비-preprint의 상태·내용을 전부 재검증한 목록은 아니다. 채택 미확인은 미채택을 뜻하지 않는다.",
    ], common_widths)
    rows = []
    for prior in inventory:
        abb = prior["abb"]
        source = workbook[prior["sheet"]]
        source_cells = [source.cell(prior["row"], c) for c in range(1, 21)]
        values = [c.value for c in source_cells]
        if prior["sheet"] == "20261021_추가문헌":
            values[0] = ADDITIONAL_CHAPTER[abb]
        item = updates.get(abb)
        discussion = source.cell(prior["row"], 21).value if source.max_column >= 21 else None
        prior_review = source.cell(prior["row"], 22).value if source.max_column >= 22 else None
        if item:
            values[2] = item.get("published_title", values[2])
            state, urls, note = "게재/채택 확인 → " + item["grade"], item["evidence"], item["note"]
        elif abb in related:
            state, urls, note = "관련 워크숍판 확인; 본 보고서 채택 미확인", related[abb]["evidence"], related[abb]["note"]
        elif prior["grade"] == "프리프린트":
            state, urls = "채택 미확인 (분류 유지)", [prior["url"]] if prior["url"] else []
            note = plan["remaining_notes"].get(abb, "검색·공개 메타데이터에서 공식 채택을 확인하지 못했다. 색인 누락·접근 제한 가능성이 있어 미채택으로 단정하지 않는다.")
        else:
            state, urls, note = "기존 기록 유지 (이번 게재 재검증 범위 밖)", [], "기존 내용 검토 수준을 유지한다."
        values += [state, DATE if prior["grade"] == "프리프린트" else None, "\n".join(urls) or None,
                   prior["url"] if prior["grade"] == "프리프린트" else None,
                   f"{prior['sheet']}!{prior['row']}", discussion,
                   note + ("\n이전 검토: " + str(prior_review) if prior_review else "")]
        rows.append((values, source_cells, "E9F3FF" if item else None))

    new_sheet, new_header = table_sheet(workbook, NEW, headers, [
        "이번에 추가한 TabPFN + IDS 문헌 3편 | 2026-10-27 확인 | 새로 발견한 문헌이며 전부 2026년 최초 발표라는 뜻은 아니다.",
        "직접 TabPFN을 사용한 IDS 연구만 포함했다. 공식 초록·논문 전문·저자 설명의 확인 범위를 구별한다. 인접 보안 연구 및 TabICL 중심 후보는 검색검토 시트에 별도로 남겼다.",
    ], common_widths)
    for item in plan["new_papers"]:
        assert item["abb"] not in original
        values = [item.get(key) for key in FIELDS]
        state = "신규 추가 · 학회 채택 확인 (전문 미확보)" if item["abb"] == "Conformal-TabPFN-IDS" else "신규 추가 · 정식 게재 확인"
        values += [state, DATE, "\n".join(item["evidence"]), None,
                   f"{NEW}!{new_sheet.max_row + 1}", item["discussion"], item["level"]]
        append_row(new_sheet, values, accent="E8F5EC")
        rows.append((values, None, "E8F5EC"))
    rank = {"학회/저널": 0, "워크숍/블로그": 1, "프리프린트": 2}
    rows.sort(key=lambda r: (r[0][0], 0 if r[0][0] == 1 else rank.get(r[0][5], 9),
                            0 if r[0][0] == 1 else -int(r[0][18]), -date_key(r[0][6]), r[0][1]))
    for values, source_cells, accent in rows:
        append_row(latest, values, source_cells, accent)
    for sheet, header in [(latest, latest_header), (new_sheet, new_header)]:
        finish_table(sheet, header)
        for row in range(header + 1, sheet.max_row + 1):
            for col in [1, 19]:
                sheet.cell(row, col).alignment = Alignment(horizontal="right", vertical="top")

    audit, audit_header = table_sheet(workbook, AUDIT, [
        "Abb.", "기존 제목", "이전 등급", "확인 결과", "현재 등급", "현재 출처", "정식 게재/행사일", "게재본/공식 링크", "기존 preprint 링크", "확인 근거 URL", "원본 위치", "확인일", "판단 근거·한계",
    ], [
        "기존 preprint 61편 점검 | 12편 게재/채택으로 갱신 | 49편 채택 미확인(그중 TabPFN-2.5 관련 워크숍판 1편 확인)",
        "출판사·공식 논문집·학회 채택 목록과 제목/저자를 대조했다. OpenReview 제출, 저장소 DOI, 검색 서비스의 학회 태그만으로 채택 처리하지 않았다. 원래 공개일은 변경하지 않았다.",
        "미확인 항목의 추가 확인 순서: OpenReview 결정 → 공식 논문집/출판사 → 저자 페이지. 검색 색인·API 제한으로 빠질 수 있어 49편이 모두 미채택이라는 뜻은 아니다. 검색 기록은 20261027/search_log.json에 보존했다.",
    ], {1: 26, 2: 65, 3: 17, 4: 34, 5: 17, 6: 60, 7: 29, 8: 48, 9: 48, 10: 60, 11: 30, 12: 14, 13: 85})
    for prior in sorted(preprints, key=lambda p: (p["abb"] not in updates, p["abb"] not in related, p["abb"].lower())):
        abb, item = prior["abb"], updates.get(prior["abb"])
        sheet, row = workbook[prior["sheet"]], prior["row"]
        note = item["note"] if item else related[abb]["note"] if abb in related else plan["remaining_notes"].get(abb, "공식 채택 미확인. 검색·메타데이터의 부재는 미채택 증거가 아니다.")
        urls = item["evidence"] if item else related[abb]["evidence"] if abb in related else [prior["url"]]
        append_row(audit, [abb, prior["title"], prior["grade"],
                          "게재/채택 확인" if item else "관련 워크숍판 확인; 보고서는 미확인" if abb in related else "채택 미확인",
                          sheet.cell(row, 6).value, sheet.cell(row, 18).value,
                          item["venue_date"] if item else None, item["url"] if item else None,
                          prior["url"], "\n".join(u for u in urls if u), f"{sheet.title}!{row}", DATE, note],
                   accent="E9F3FF" if item else None)
    finish_table(audit, audit_header)
    candidates, candidates_header = table_sheet(workbook, SCREENED,
        ["검색 후보 제목", "링크", "이번 처리", "범위/판단 이유", "확인 근거", "확인일"],
        ["신규 IDS 3편과 구별한 검색 후보 8편 | 2026-10-27 확인",
         "네트워크 IDS 직접 실험, TabPFN 실제 사용, 공식 채택을 각각 확인한다. 인접 보안 문헌은 제외 이유를 남기고, 확인되지 않은 모델 사용·게재 상태를 추정하지 않는다."],
        {1: 70, 2: 55, 3: 42, 4: 85, 5: 55, 6: 14})
    for item in plan["screened_candidates"]:
        append_row(candidates, [item["title"], item["url"], item["status"], item["reason"], item.get("evidence"), DATE])
    finish_table(candidates, candidates_header)
    for name in [SCREENED, AUDIT, NEW, LATEST]:
        workbook.move_sheet(name, offset=-workbook.index(workbook[name]))
    workbook.active = 0

    with tempfile.NamedTemporaryFile(suffix=".xlsx", prefix=".literature-", dir=refs, delete=False) as handle:
        staged = Path(handle.name)
    try:
        workbook.save(staged)
        with zipfile.ZipFile(staged) as archive:
            assert archive.testzip() is None
        checked = openpyxl.load_workbook(staged)
        for name, snapshot in snapshots.items():
            assert str(checked[name].merged_cells) == original_merges[name]
            for (row, col), signature in snapshot.items():
                actual = cell_signature(checked[name].cell(row, col))
                if (name, row, col) not in authorized:
                    assert actual == signature, (name, row, col, "unintended edit")
                else:
                    assert actual[1] == signature[1], (name, row, col, "historical style changed")
        assert checked[LATEST].max_row - latest_header == 137
        assert checked[NEW].max_row - new_header == 3
        assert checked[AUDIT].max_row - audit_header == 61
        ids = [checked[LATEST].cell(r, 2).value for r in range(latest_header + 1, checked[LATEST].max_row + 1)]
        assert len(set(ids)) == len(ids)
        titles = [re.sub(r"\W+", "", str(checked[LATEST].cell(r, 3).value).lower())
                  for r in range(latest_header + 1, checked[LATEST].max_row + 1)]
        assert len(set(titles)) == len(titles)
        for item in plan["updates"]:
            prior = original[item["abb"]]
            cell = checked[prior["sheet"]].cell(prior["row"], 20)
            assert cell.value == cell.hyperlink.target == item["url"]
        strikes = 0
        for row in range(latest_header + 1, checked[LATEST].max_row + 1):
            abb = checked[LATEST].cell(row, 2).value
            if abb in original:
                old = original[abb]
                assert checked[LATEST].cell(row, 3).font.strike == checked[old["sheet"]].cell(old["row"], 3).font.strike
                strikes += bool(checked[LATEST].cell(row, 3).font.strike)
        checked.close()
        # A concurrently edited workbook must never be replaced silently.
        assert sha(target) == before_hash, "Workbook changed during generation; staged output retained only in temp."
        staged.replace(target)
    finally:
        staged.unlink(missing_ok=True)
    audit_result = {
        "checked_on": DATE, "workbook": str(target.relative_to(root)),
        "before_sha256": before_hash, "after_sha256": sha(target),
        "temporary_backup": str(backup), "papers_before": 134, "papers_after": 137,
        "preprints_checked": 61, "status_upgrades": 12,
        "upgrades_by_grade": dict(Counter(p["grade"] for p in plan["updates"])),
        "acceptance_unconfirmed": 49, "related_version_notes": 1, "new_ids_papers": 3,
        "screened_candidates": 8, "preserved_struck_title_rows": strikes,
        "historical_value_changes": changed,
        "validation": ["OOXML ZIP integrity", "openpyxl reload", "all historical cells/styles/comments preserved outside explicit update list",
                       "historical merged ranges preserved", "137 unique paper identifiers and titles", "61 original preprints covered",
                       "3 new IDS papers", "12 publication links and hyperlink targets agree", "strikethrough preserved in consolidated view"],
        "generated_sheets": [LATEST, NEW, AUDIT, SCREENED],
        "source_files": {p.name: sha(p) for p in evidence.glob("*.json") if p.name != "workbook_validation.json"},
    }
    if previous_audit and previous_audit["after_sha256"] == before_hash and not changed:
        # Regenerating views must not erase the original metadata change log.
        for key in ["before_sha256", "temporary_backup", "historical_value_changes"]:
            audit_result[key] = previous_audit[key]
        audit_result["view_rebuilds"] = previous_audit.get("view_rebuilds", []) + [
            {"input_sha256": before_hash, "temporary_backup": str(backup)}
        ]
    audit_path.write_text(json.dumps(audit_result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit_result.items() if k not in {"historical_value_changes", "source_files"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
