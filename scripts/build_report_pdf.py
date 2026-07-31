#!/usr/bin/env python3
"""final report build pipeline (v3) — 표지·목차·러닝 헤더 조판으로 제출용 PDF를 만든다.

`outputs/reports/final_report_v3.md` → `share/최종보고서_LendingClub_Sharpe_v3_20260731.pdf`.
레이아웃·팔레트는 팀 표지본(네이비 #1e3a5e·#335179, 라이트 블루그레이 배경)을 따르고,
LaTeX 수식($$…$$)은 pandoc `--mathml`로 조판한다. 목차 쪽번호는 실제 조판 결과에서 역산한다.

3-pass: ① 본문 조판(pandoc+Chrome, 목차 쪽번호 placeholder) ② 실측 쪽번호로 목차 재조판
③ 러닝 헤더/푸터 오버레이 합성 + PDF 메타데이터.

실행: `python3 scripts/build_report_pdf.py` (어느 CWD에서든 동작. 중간 산출물은 임시 폴더)
의존: pandoc · Google Chrome · PyPDF2 — 조판 전용 의존성으로 분석 파이프라인과 무관하며
macOS 시스템 python3로 충분하다(scikit-learn 불필요).
"""
import re, subprocess, sys, tempfile
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "outputs/reports/final_report_v3.md"
SP = Path(tempfile.mkdtemp(prefix="report_pdf_build_"))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT = REPO / "share" / "최종보고서_LendingClub_Sharpe_v3_20260731.pdf"

HEADER_L = "서울대학교 핀테크 전문가 과정 「통계, 데이터 사이언스」"
HEADER_R = "팀 프로젝트 최종 보고서"

TITLE_HTML = """
<div id="titlepage">
  <p class="inst">서울대학교 핀테크 전문가 과정 「통계, 데이터 사이언스」</p>
  <h1 class="doctitle">Lending Club 신용평가모형과<br>Sharpe Ratio 극대화 승인 전략</h1>
  <p class="subtitle">팀 프로젝트 최종 보고서</p>
  <p class="authors">강권재 · 류성환 · 안예환 · 유명곤 · 이지희 · 최지원</p>
  <p class="submitdate">제출일: 2026-07-31</p>
</div>
<hr class="titlerule">
"""

CSS = """
@page { size: A4; margin: 24mm 20mm 20mm 20mm; }
@page :first { margin-top: 14mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Times New Roman", "AppleMyungjo", "Apple SD Gothic Neo", serif;
       font-size: 10pt; line-height: 1.78; color: #262b36; text-align: justify; margin: 0; }
#title-block-header { display: none; }
hr { display: none; }

/* ---- 표지 ---- */
#titlepage { text-align: center; margin-top: 0; }
#titlepage .inst { font-family: "Apple SD Gothic Neo", sans-serif; font-size: 9pt; color: #335179;
                   letter-spacing: .08em; margin-bottom: 5mm; }
#titlepage .doctitle { font-family: "Apple SD Gothic Neo", sans-serif; font-weight: 800;
                       font-size: 19pt; line-height: 1.35; margin: 0 0 4mm; color: #13171f;
                       text-align: center; }
#titlepage .subtitle { font-family: "Apple SD Gothic Neo", sans-serif; font-size: 9.5pt;
                       color: #3a404a; margin-bottom: 5mm; }
#titlepage .authors { font-family: "Apple SD Gothic Neo", sans-serif; font-size: 10.5pt;
                      color: #2a2e38; margin-bottom: 1.5mm; }
#titlepage .submitdate { font-family: "Apple SD Gothic Neo", sans-serif; font-size: 8.5pt;
                         color: #5a616f; margin: 0; }
hr.titlerule { display: block; border: 0; border-top: 1.2px solid #b8bdc7; margin: 4mm 0 3.5mm; }

/* ---- 요약 (표지면에 함께) ---- */
h2#요약 { text-align: center !important; border: 0; color: #13171f; font-size: 12pt; letter-spacing: .5em;
          text-indent: .5em; margin: 0 0 3mm; page-break-before: avoid; }

/* ---- 제목 (고딕) ---- */
h1, h2, h3, h4 { font-family: "Apple SD Gothic Neo", sans-serif; font-weight: 700;
                 color: #1e3a5e; page-break-after: avoid; text-align: left; }
h2 { font-size: 13.5pt; margin: 8mm 0 3.5mm; }
h3 { font-size: 11pt; margin: 6mm 0 2.5mm; color: #335179; }

/* ---- 목차 ---- */
#tocpage { page-break-before: always; }
#tocpage h1 { font-size: 15pt; color: #13171f; border-bottom: 1.2px solid #b8bdc7; padding-bottom: 2mm; margin: 0 0 6mm; }
.toc-row { display: flex; align-items: baseline; font-family: "Apple SD Gothic Neo", sans-serif; }
.toc-row.lv2 { font-weight: 700; font-size: 10pt; margin-top: 3.2mm; color: #1e3a5e; }
.toc-row.lv3 { font-weight: 400; font-size: 9pt; color: #3a404a; margin: 1.1mm 0 0 5mm; }
.toc-dots { flex: 1; border-bottom: 1px dotted #b8bdc7; margin: 0 4px 3px; }
.toc-pg { font-weight: 400; font-size: 9pt; color: #5a616f; }
#tocpage + * { page-break-before: always; }
.bodymark { color: #fff; font-size: 1pt; line-height: 0; margin: 0; }

/* ---- 표 ---- */
table { display: table !important; border-collapse: collapse; margin: 3mm auto 4mm; width: 100%;
        font-family: "Apple SD Gothic Neo", sans-serif; font-size: 8.5pt; line-height: 1.55;
        overflow-x: visible !important; }
tr { page-break-inside: avoid; }
thead, tbody, tr, th, td { display: revert !important; }
th { background: #f0f3f7; color: #1e3a5e; font-weight: 700; text-align: left; }
th, td { border: 1px solid #d7dce5; padding: 1.4mm 2.4mm; text-align: left; vertical-align: top; }
td:last-child[align="right"], th[align="right"], td[align="right"] { text-align: right; }
.tbl-cap { font-family: "Apple SD Gothic Neo", sans-serif; font-size: 8.5pt; font-weight: 700;
           color: #3a404a; margin: 4mm 0 1mm; page-break-after: avoid; text-align: left; }
.tbl-cap + table { margin-top: 1mm; }

/* ---- 수식 (LaTeX → MathML) ---- */
p:has(> math[display="block"]:only-child), .math.display
  { background: #f6f7f9; border: 1px solid #e3e7eb; border-radius: 4px;
    padding: 3mm 4mm; text-align: center; page-break-inside: avoid; margin: 3mm 0; }
math { font-size: 10.5pt; }

/* ---- 코드 블록 (부록 B) ---- */
pre { background: #f6f7f9; border: 1px solid #e3e7eb; border-radius: 4px; padding: 3mm 4mm;
      font-size: 8pt; line-height: 1.6; overflow-x: auto; page-break-inside: avoid; }
code { font-family: "SF Mono", Menlo, monospace; font-size: .92em; }
p code, li code, td code { background: #eef1f6; padding: 0 .25em; border-radius: 3px; }

/* ---- 그림 ---- */
p:has(> img:only-child) { text-align: center; margin: 4mm 0 1mm; page-break-inside: avoid;
                          page-break-after: avoid; }
img { max-width: 92%; }
.fig-cap { text-align: center; font-size: 8.5pt; color: #5a616f; margin: 1mm 0 4mm;
           font-family: "Apple SD Gothic Neo", sans-serif; }

/* ---- 본문 잡요소 ---- */
strong { font-family: "Apple SD Gothic Neo", sans-serif; font-size: .96em; }
ul, ol { padding-left: 5mm; }
li { margin-bottom: .8mm; }
blockquote { margin: 2mm 0 2mm 4mm; padding-left: 3mm; border-left: 2px solid #b8bdc7; color: #3a404a; }
p:has(+ table) { page-break-after: avoid; }
"""

OVERLAY_CSS = """
@page { size: A4; margin: 0; }
body { margin: 0; }
.pg { width: 210mm; height: 296.5mm; position: relative; page-break-after: always; overflow: hidden;
      font-family: "Apple SD Gothic Neo", sans-serif; }
.hdr { position: absolute; top: 9mm; left: 20mm; right: 20mm; font-size: 7.5pt; color: #5a616f;
       display: flex; justify-content: space-between; border-bottom: .6px solid #b8bdc7;
       padding-bottom: 1.6mm; }
.ftr { position: absolute; bottom: 9mm; left: 0; right: 0; text-align: center; font-size: 9pt;
       color: #3a404a; font-family: "Times New Roman", serif; }
"""


def sh(*args):
    subprocess.run(args, check=True, capture_output=True)


def preprocess(md: str) -> tuple[str, list[tuple[int, str]]]:
    """제목 블록 → 표지 HTML, 목차 절 제거. 반환: (본문 md, 목차 항목 [(level, text)])."""
    lines = md.split("\n")
    # 제목 블록(H1~첫 '---')을 표지 HTML로 교체
    first_hr = lines.index("---")
    body = lines[first_hr + 1:]
    # 기존 '## 목차' ~ 다음 '## ' 직전 제거
    toc_i = next(i for i, l in enumerate(body) if l.startswith("## 목차"))
    after = next(i for i in range(toc_i + 1, len(body)) if body[i].startswith("## "))
    body = (body[:toc_i]
            + ["<div id='tocpage'><h1>목차</h1>", "TOC_SLOT", "</div>", "",
               "<div class='bodymark'>BODYSTART977MARK</div>", ""]
            + body[after:])
    # 캡션(단독 문단의 **표 …** / *그림 …*)을 클래스 명시 HTML로 변환 — 여러 줄·\* 이스케이프 포함
    text = "\n".join(body)

    def clean(t: str) -> str:
        t = t.replace("\\*", "*")
        return re.sub(r"`([^`]*)`", r"\1", t).replace("\n", " ")

    text = re.sub(r"(?m)^\*\*(표 .+?)\*\*$",
                  lambda m: f"<p class='tbl-cap'>{clean(m.group(1))}</p>", text)
    text = re.sub(r"(?ms)^\*(그림 .+?)\*$",
                  lambda m: f"<p class='fig-cap'>{clean(m.group(1))}</p>", text)
    body = text.split("\n")
    headings = [(2, "요약")]
    for l in body:
        if l.startswith("### "):
            headings.append((3, l[4:].strip()))
        elif l.startswith("## ") and not l.startswith("## 목차"):
            t = l[3:].strip()
            if t != "요약":
                headings.append((2, t))
    return TITLE_HTML + "\n" + "\n".join(body), headings


def toc_html(headings, pages: dict[str, int] | None) -> str:
    rows = []
    for lv, t in headings:
        disp = re.sub(r"\\\*", "*", t).replace("\\", "")
        disp = re.sub(r"`([^`]*)`", r"\1", disp)
        pg = str(pages.get(t, "")) if pages else "0"
        rows.append(f"<div class='toc-row lv{lv}'><span>{disp}</span>"
                    f"<span class='toc-dots'></span><span class='toc-pg'>{pg}</span></div>")
    return "\n".join(rows)


def render(body_md: str, headings, pages, tag: str) -> Path:
    md = body_md.replace("TOC_SLOT", toc_html(headings, pages))
    md_f, css_f = SP / f"_build_{tag}.md", SP / "_build.css"
    html_f, pdf_f = SP / f"_build_{tag}.html", SP / f"_build_{tag}.pdf"
    md_f.write_text(md)
    css_f.write_text(CSS)
    head = SP / "_build_head.html"
    head.write_text("<style>\n" + CSS + "\n</style>")
    sh("pandoc", str(md_f), "-f", "gfm+tex_math_dollars", "--mathml", "-t", "html5",
       "--standalone", "--embed-resources", "--resource-path", str(REPO / "outputs/reports"),
       "--include-in-header", str(head),
       "--metadata", "title=Lending Club 신용평가모형과 Sharpe Ratio 극대화 승인 전략 — 팀 프로젝트 최종 보고서",
       "-o", str(html_f))
    sh(CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
       f"--print-to-pdf={pdf_f}", "--virtual-time-budget=15000", f"file://{html_f}")
    return pdf_f


def page_map(pdf: Path, headings) -> dict[str, int]:
    r = PdfReader(str(pdf))
    texts = ["".join((p.extract_text() or "").split()) for p in r.pages]
    pages = {"요약": 1}
    ptr = next(i for i, t in enumerate(texts) if "BODYSTART977MARK" in t)
    for lv, t in headings:
        if t == "요약":
            continue
        needle = "".join(re.sub(r"`([^`]*)`", r"\1", t.replace("\\*", "*").replace("\\", "")).split())
        for i in range(ptr, len(texts)):
            if needle in texts[i]:
                pages[t] = i + 1
                ptr = i
                break
        else:
            print(f"  ⚠️ 목차 쪽번호 미해결: {t}")
    return pages


def overlay(content: Path, out: Path):
    n = len(PdfReader(str(content)).pages)
    pgs = "\n".join(
        f"<div class='pg'><div class='hdr'><span>{HEADER_L}</span><span>{HEADER_R}</span></div>"
        f"<div class='ftr'>{i}</div></div>" for i in range(2, n + 1))
    ov_html = SP / "_overlay.html"
    ov_html.write_text(f"<!doctype html><html><head><meta charset='utf-8'>"
                       f"<style>{OVERLAY_CSS}</style></head><body>{pgs}</body></html>")
    ov_pdf = SP / "_overlay.pdf"
    sh(CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
       f"--print-to-pdf={ov_pdf}", f"file://{ov_html}")
    ov = PdfReader(str(ov_pdf))
    src = PdfReader(str(content))
    w = PdfWriter()
    for i, page in enumerate(src.pages):
        if i >= 1 and i - 1 < len(ov.pages):
            page.merge_page(ov.pages[i - 1])
        w.add_page(page)
    w.add_metadata({
        "/Title": "Lending Club 신용평가모형과 Sharpe Ratio 극대화 승인 전략 — 팀 프로젝트 최종 보고서",
        "/Author": "강권재·류성환·안예환·유명곤·이지희·최지원 (서울대학교 핀테크 전문가 과정)",
        "/Subject": "통계, 데이터 사이언스 팀 프로젝트 최종 보고서 (2026-07-31)",
        "/Creator": "final report build pipeline (v3)",
    })
    with open(out, "wb") as f:
        w.write(f)


def main():
    body, headings = preprocess(SRC.read_text())
    p1 = render(body, headings, None, "pass1")
    pages = page_map(p1, headings)
    p2 = render(body, headings, pages, "pass2")
    # 쪽수가 pass 간 달라졌으면 한 번 더 (목차 행 수는 동일하므로 보통 수렴)
    pages2 = page_map(p2, headings)
    if pages2 != pages:
        p2 = render(body, headings, pages2, "pass2")
    OUT.parent.mkdir(exist_ok=True)
    overlay(p2, OUT)
    print(f"✅ {OUT} ({len(PdfReader(str(OUT)).pages)} pages)")


if __name__ == "__main__":
    main()
