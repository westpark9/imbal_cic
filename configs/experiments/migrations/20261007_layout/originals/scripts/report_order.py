#!/usr/bin/env python3
"""Table rules for lablog/html_report (user rules, 2026-10-01; keep applying to every tab):

1. Class tables list classes by test sample count, descending (footer rows such as macro/weighted/합계 stay last).
2. Numeric cells are right-aligned (ones digit / decimal point aligned via tabular figures). A column counts as
   numeric when at least one body cell is a number and every other cell is a number or a neutral mark (—, TBD, ·).
   Row-header cells and tables that use colspan/rowspan are left alone.

Builders import `support_order`, `permute` and `apply_rules`; static tabs are processed in place:

    python scripts/report_order.py lablog/html_report/foo.html [...]        # apply both rules in place
    python scripts/report_order.py --check lablog/html_report/*.html         # exit 1 if any rule is violated
"""
import re
import sys
from html import unescape
from pathlib import Path

FIRST = re.compile(r'^(클래스|class)\b', re.I)
COUNT = re.compile(r'(test\s*(수|행|표본\s*수|rows|n)|support|표본\s*수|샘플\s*수)', re.I)
EMBEDDED = re.compile(r'테스트 행 수|test rows', re.I)
FOOTER = re.compile(r'^(macro|weighted|합계|total|평균|mean|micro)', re.I)
TABLE = re.compile(r'<table\b.*?</table>', re.S)
ROW = re.compile(r'<tr\b.*?</tr>', re.S)
CELL = re.compile(r'<t[dh]\b[^>]*>(.*?)</t[dh]>', re.S)
CELL_TAG = re.compile(r'<(t[dh])\b([^>]*)>', re.S)
NEUTRAL = re.compile(r'^(|—|–|-|·|TBD|n/a|N/A|\?)$')
UNITS = re.compile(r'(GiB|GB|MB|KB|초|행|건|개|회|배|시간|분|epochs?|steps?|%p|%|(?<=\d)\s*[MkK]\b|\bs\b|\bh\b|\bx\b|×|\bpt\b)')
NUMERIC = re.compile(r'^[\d.,\s+\-−–—→←↔/±~≈<>:;∞†‡*]+$')
STYLE_TAG = '<style data-report-rule="num-align">td[data-num],th[data-num]{text-align:right;font-variant-numeric:tabular-nums}</style>'


# ---------------------------------------------------------------- rule 1: class order
def support_order(supports):
    """Indices sorted by support descending; ties keep the original order."""
    return sorted(range(len(supports)), key=lambda i: (-int(supports[i]), i))


def permute(items, order):
    return [items[i] for i in order]


def _text(cell_html):
    return unescape(re.sub(r'<[^>]+>', '', cell_html)).strip()


def _count(text):
    m = re.search(r'\d[\d,]*', text)
    return int(m.group(0).replace(',', '')) if m else None


def _sort_table(table_html):
    rows = ROW.findall(table_html)
    if len(rows) < 3:
        return table_html, False
    header_cells = [_text(c) for c in CELL.findall(rows[0])]
    if not header_cells or not FIRST.search(header_cells[0]):
        return table_html, False
    embedded = bool(EMBEDDED.search(header_cells[0]))
    col = next((i for i, h in enumerate(header_cells[1:], 1) if COUNT.search(h)), None)
    if col is None and not embedded:
        return table_html, False
    body = rows[1:]
    keyed = []
    for r in body:
        cells = [_text(c) for c in CELL.findall(r)]
        if not cells or FOOTER.search(cells[0]):
            keyed.append((None, r)); continue
        paren = re.search(r'\(([^)]*)\)', cells[0])
        value = _count(paren.group(1)) if embedded and paren else (_count(cells[col]) if col is not None and col < len(cells) else None)
        keyed.append((value, r))
    sortable = [(v, r) for v, r in keyed if v is not None]
    rest = [r for v, r in keyed if v is None]
    ordered = [r for _, r in sorted(sortable, key=lambda t: -t[0])] + rest
    if ordered == body:
        return table_html, False
    out, pos = [], 0
    for old, new in zip(body, ordered):
        i = table_html.index(old, pos)
        out.append(table_html[pos:i]); out.append(new); pos = i + len(old)
    out.append(table_html[pos:])
    return ''.join(out), True


def sort_class_tables_html(text):
    changed = 0
    def repl(m):
        nonlocal changed
        new, did = _sort_table(m.group(0))
        changed += did
        return new
    return TABLE.sub(repl, text), changed


# ---------------------------------------------------------------- rule 2: numeric alignment
def is_numeric_text(text):
    t = re.sub(r'\([^)]*\)', ' ', text)          # "(e3)", "(26.7%)" annotations do not decide
    t = UNITS.sub(' ', t).strip()
    return bool(re.search(r'\d', t)) and bool(NUMERIC.match(t))


def _mark_table(table_html):
    """Add data-num to every cell of numeric columns (header included). Returns (html, changed)."""
    if re.search(r'\b(colspan|rowspan)\s*=', table_html):
        return table_html, False
    rows = ROW.findall(table_html)
    if len(rows) < 2:
        return table_html, False
    parsed = []
    for r in rows:
        cells = [(m.group(1), m.group(2), c) for m, c in zip(CELL_TAG.finditer(r), CELL.findall(r))]
        parsed.append(cells)
    width = max(len(c) for c in parsed)
    # header rows = leading rows made only of <th>; body = the rest
    header_n = 0
    for cells in parsed:
        if cells and all(tag == 'th' for tag, _, _ in cells):
            header_n += 1
        else:
            break
    body = parsed[header_n:]
    if not body:
        return table_html, False
    numeric_cols = set()
    for j in range(width):
        values = [(cells[j][0], _text(cells[j][2])) for cells in body if j < len(cells)]
        if not values or any(tag == 'th' for tag, _ in values):
            continue
        texts = [t for _, t in values]
        if any(is_numeric_text(t) for t in texts) and all(is_numeric_text(t) or NEUTRAL.match(t) for t in texts):
            numeric_cols.add(j)
    if not numeric_cols:
        return table_html, False
    changed = False
    out, pos = [], 0
    for ri, r in enumerate(rows):
        i = table_html.index(r, pos)
        out.append(table_html[pos:i])
        cells = parsed[ri]
        new_row, rpos, j = [], 0, 0
        for m in CELL_TAG.finditer(r):
            tag, attrs = m.group(1), m.group(2)
            new_row.append(r[rpos:m.start()])
            if j in numeric_cols and 'data-num' not in attrs and (tag == 'td' or ri < header_n):
                new_row.append(f'<{tag}{attrs} data-num="">'); changed = True
            else:
                new_row.append(m.group(0))
            rpos = m.end(); j += 1
        new_row.append(r[rpos:])
        out.append(''.join(new_row)); pos = i + len(r)
    out.append(table_html[pos:])
    return ''.join(out), changed


def align_numeric_cells_html(text):
    changed = 0
    def repl(m):
        nonlocal changed
        new, did = _mark_table(m.group(0))
        changed += did
        return new
    text = TABLE.sub(repl, text)
    if STYLE_TAG not in text and '<table' in text:
        if '</head>' in text:
            text = text.replace('</head>', STYLE_TAG + '\n</head>', 1)
        else:
            text = STYLE_TAG + '\n' + text
        changed += 1
    return text, changed


def apply_rules(text):
    """Both rules; returns the processed HTML (builders call this right before writing a tab)."""
    text, _ = sort_class_tables_html(text)
    text, _ = align_numeric_cells_html(text)
    return text


def main(argv):
    check = '--check' in argv
    paths = [Path(a) for a in argv if a != '--check']
    problems = []
    for p in paths:
        text = p.read_text(encoding='utf-8')
        sorted_text, unsorted = sort_class_tables_html(text)
        aligned_text, unaligned = align_numeric_cells_html(sorted_text)
        if check:
            if unsorted: problems.append(f'{p}: {unsorted} class table(s) not in test-count order')
            if unaligned: problems.append(f'{p}: {unaligned} table(s)/style without numeric right-alignment')
        elif unsorted or unaligned:
            p.write_text(aligned_text, encoding='utf-8')
            print(f'{p}: reordered {unsorted} class table(s), aligned {unaligned} table(s)')
        else:
            print(f'{p}: already conforms')
    if check:
        print('\n'.join(problems) if problems else 'all tables conform: class order by test count (desc), numeric cells right-aligned')
        return 1 if problems else 0
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
