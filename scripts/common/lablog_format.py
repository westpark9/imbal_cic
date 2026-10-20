"""한국어 실험 기록의 LaTeX 서식. Markdown은 변환 입력일 뿐 파일로 저장하지 않는다."""
from pathlib import Path
import re
import html as html_lib
from lxml import html
from markdown_it import MarkdownIt
MD = MarkdownIt('commonmark').enable('table')
def inline_math(state, silent):
    if state.src[state.pos] != '$' or state.src.startswith('$$', state.pos): return False
    end=state.src.find('$',state.pos+1)
    if end < 0 or '\n' in state.src[state.pos+1:end]: return False
    if not silent:
        tok=state.push('math_inline','math',0);tok.content=state.src[state.pos+1:end]
    state.pos=end+1;return True

def block_math(state,start,end,silent):
    first=state.src[state.bMarks[start]+state.tShift[start]:state.eMarks[start]].strip()
    if not first.startswith('$$'):return False
    if silent:return True
    lines=[first[2:]];last=start
    if not lines[0].endswith('$$'):
        for last in range(start+1,end):
            line=state.src[state.bMarks[last]+state.tShift[last]:state.eMarks[last]].strip()
            lines.append(line)
            if line.endswith('$$'):break
        else:return False
    content='\n'.join(lines)
    if content.endswith('$$'):content=content[:-2]
    tok=state.push('math_block','math',0);tok.content=content;tok.block=True
    state.line=last+1;return True

def clean(s):
    replacements = {'⚠️': '[주의]', '⚠': '[주의]', '✅': '[확인]', '❌': '[제외]', '✓': '[확인]',
                    '→': ' → ', '↔': ' ↔ ', '\u200b': '', '\ufe0f': '', '❗': '[주의]', '🔴': '[주의]',
                    '🟢': '[확인]', '📌': '', '📊': '', '📁': '', '🔥': '', '🟡': '[검토]',
                    '✗':'[제외]', '✕':'[제외]', '≲':'<~', '⓪':'0'}
    for a, b in replacements.items(): s = s.replace(a, b)
    return s

def esc(s):
    return ''.join({'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
                    '_': r'\_\allowbreak{}', '/': r'/\allowbreak{}', '+': r'+\allowbreak{}', '=': r'=\allowbreak{}',
                    '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
                    '≥': r'\ensuremath{\geq}', '≤': r'\ensuremath{\leq}', '≠': r'\ensuremath{\neq}',
                    '→': r'\ensuremath{\rightarrow}', '←': r'\ensuremath{\leftarrow}',
                    '↔': r'\ensuremath{\leftrightarrow}', '≈': r'\ensuremath{\approx}',
                    '∈': r'\ensuremath{\in}', '∞': r'\ensuremath{\infty}', '∑': r'\ensuremath{\sum}',
                    '∝': r'\ensuremath{\propto}', '∅': r'\ensuremath{\varnothing}',
                    }.get(c, c) for c in clean(str(s)))

def mono(s):
    # URL 매크로의 math 모드는 한글을 누락할 수 있다. 긴 경로는 텍스트 안에서 줄을 나눈다.
    out=[]
    for i,c in enumerate(clean(s)):
        out.append(esc(c))
        if c in '/_,.{}=+-' or (i+1)%8==0:out.append(r'\allowbreak{}')
    return r'\texttt{' + ''.join(out) + '}'

def inline(tokens):
    out = []
    for t in tokens or []:
        if t.type == 'text': out.append(esc(t.content))
        elif t.type == 'math_inline': out.append(r'\('+t.content+r'\)')
        elif t.type == 'code_inline': out.append(mono(t.content))
        elif t.type in ('softbreak', 'hardbreak'): out.append(' ')
        elif t.type == 'strong_open': out.append(r'\textbf{')
        elif t.type == 'em_open': out.append(r'\emph{')
        elif t.type in ('strong_close', 'em_close'): out.append('}')
        elif t.type == 'link_open': out.append('')
        elif t.type == 'link_close': out.append('')
        elif t.type == 'image': out.append(esc('[그림: ' + t.content + ']'))
        elif t.type == 'html_inline': out.append(esc(html_lib.unescape(re.sub('<[^>]+>', '', t.content))))
        elif t.content: out.append(esc(t.content))
    return ''.join(out)

def table(rows, aligns=None):
    if not rows: return ''
    n = max(map(len, rows)); rows = [r + [''] * (n-len(r)) for r in rows]
    # 긴 역사 표도 여러 페이지로 이어진다. 본문 숫자를 이미지로 축소하지 않는다.
    lens = [max(7, min(38, max(len(re.sub(r'\\\w+\{|[{}]', '', r[i])) for r in rows))) for i in range(n)]
    weights = [.5/n+.5*x/sum(lens) for x in lens]
    specs = []
    for i,w in enumerate(weights):
        right = aligns and aligns[i] == 'right'
        style = r'\raggedleft' if right else r'\raggedright'
        specs.append(r'>{'+style+r'\arraybackslash}p{\dimexpr '+f'{w:.5f}'+r'\linewidth-2\tabcolsep\relax}')
    landscape = n >= 8
    result = ('\\begin{landscape}\n' if landscape else '') + '\\begingroup\n'
    result += ('\\scriptsize\n' if n >= 8 else '\\footnotesize\n')
    result += '\\begin{longtable}{'+''.join(specs)+'}\n\\toprule\n'
    header = ' & '.join(rows[0]) + r' \\ \midrule' + '\n'
    result += header + '\\endfirsthead\n' + header + '\\endhead\n'
    result += '\n'.join('{}' + ' & '.join(r) + r' \\' for r in rows[1:])
    result += '\n\\bottomrule\n\\end{longtable}\n\\endgroup\n'
    if landscape: result += '\\end{landscape}\n'
    return result

def convert_md(text, source):
    tokens = MD.parse(text); out = []; i = 0; first_heading = True
    while i < len(tokens):
        t = tokens[i]
        if t.type == 'heading_open':
            content = tokens[i+1]
            if first_heading and t.tag == 'h1':
                first_heading = False; i += 3; continue
            level = int(t.tag[1]); command = {1:'subsection',2:'subsection',3:'subsubsection'}.get(level,'paragraph')
            title = inline(content.children)
            for a,b in {'Goal':'목적', 'Setup':'설정', 'Runs':'실행', 'Takeaway':'관찰과 판단', 'Files to inspect first when resuming':'재개 시 확인할 파일', 'Results':'결과'}.items():
                if title == a: title = b
            out.append('\\'+command+'{'+title+'}\n'); i += 3; continue
        if t.type == 'table_open':
            rows=[]; row=[]; cell=[]; aligns=[]; current_align=[]
            i += 1
            while tokens[i].type != 'table_close':
                q=tokens[i]
                if q.type == 'tr_open': row=[]; current_align=[]
                elif q.type in ('td_open','th_open'):
                    current_align.append('right' if 'right' in (q.attrGet('style') or '') else 'left')
                elif q.type == 'inline': row.append(inline(q.children))
                elif q.type == 'tr_close':
                    rows.append(row)
                    if not aligns: aligns = current_align
                i+=1
            # 과거 Markdown의 열 정렬 표시보다 실제 숫자 셀의 정렬을 우선한다.
            aligns=[]
            for j in range(len(rows[0])):
                vals=[re.sub(r'\\(?:textbf|emph)\{([^}]+)\}',r'\1',r[j]) for r in rows[1:] if len(r)>j and r[j]]
                numeric=sum(bool(re.fullmatch(r'[\d\s.,+−\-/%p()\\]+',v)) for v in vals)
                aligns.append('right' if vals and numeric/len(vals)>.65 else 'left')
            out.append(table(rows,aligns)); i += 1; continue
        if t.type == 'inline':
            images = [q for q in (t.children or []) if q.type == 'image']
            if images:
                from experiment_paths import repo_root
                root = repo_root() / 'docs/lablog'
                for q in images:
                    path = (Path(source).parent / q.attrGet('src')).resolve()
                    if path.is_file() and path.is_relative_to(root):
                        out.append(r'\reportfigure{' + str(path.relative_to(root)) + '}{' + esc(q.content) + '}\n')
                    else:
                        raise FileNotFoundError(f'보고서 그림을 찾을 수 없음: {path}')
            else:
                out.append(inline(t.children))
        elif t.type == 'math_block': out.append('\n\\[\n'+t.content+'\n\\]\n')
        elif t.type in ('paragraph_close','blockquote_close'): out.append('\n\n')
        elif t.type == 'list_item_open': out.append('\\par\\noindent '+r'\textbullet\enspace ')
        elif t.type == 'list_item_close': out.append('\n')
        elif t.type in ('fence','code_block'):
            content=clean(t.content).replace(r'\end{Verbatim}', '[Verbatim 종료]')
            out.append('\n\\begin{Verbatim}[breaklines=true,breakanywhere=true,fontsize=\\scriptsize]\n'+content+'\n\\end{Verbatim}\n')
        elif t.type == 'hr': out.append('\n\\medskip\\hrule\\medskip\n')
        elif t.type == 'html_block':
            try:
                tree=html.fragment_fromstring(t.content,create_parent=True)
                for bad in tree.xpath('.//script|.//style'): bad.drop_tree()
                s=' '.join(tree.itertext()).strip()
                if s: out.append(esc(s)+'\n\n')
            except Exception: out.append(esc(re.sub('<[^>]+>', '', t.content))+'\n')
        i+=1
    return ''.join(out)
MD.inline.ruler.before('escape','math_inline',inline_math)
MD.block.ruler.before('paragraph','math_block',block_math)
