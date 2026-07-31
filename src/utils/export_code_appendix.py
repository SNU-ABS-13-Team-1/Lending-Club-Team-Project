"""**전체 구현 코드 별책 생성** — `src/`의 모든 파이썬 파일을 한 파일로 묶는다.

`final_report.md` 「부록 C. 전체 구현 코드」가 가리키는 별책(『최종보고서 별책 — 전체 구현 코드』)을
만든다. 심사자가 저장소를 열지 않고도 구현 전문을 읽을 수 있게 하는 것이 목적이다.

## 왜 손으로 붙이지 않고 스크립트로 만드는가

코드는 계속 바뀐다. 한 번 손으로 이어붙이면 그 순간부터 `src/`와 어긋나기 시작하고, 어느 쪽이
맞는지 알 수 없게 된다. 이 스크립트를 돌리면 **항상 현재 `src/`와 일치하는 별책**이 나온다.

⚠️ **산출물은 `outputs/reports/`에 커밋한다** — 팀원이 저장소에서 바로 읽을 수 있어야 하기
때문이다(`share/`는 `.gitignore` 대상이라 공유되지 않는다). 대신 코드 8,200줄이 저장소에
**중복되므로 갱신을 놓치면 조용히 어긋난다.** `--check`로 어긋났는지 확인할 수 있고,
`src/`를 고친 PR에서는 이 스크립트를 다시 돌려 별책을 함께 커밋한다.

## 무엇을 보장하는가

- **누락 없음**: `src/**/*.py`를 전부 훑고, 아래 분류표에 없는 파일은 「기타」로 모아 반드시 싣는다.
  분류표를 갱신하지 않아도 새 파일이 조용히 빠지지 않는다(끝에 검증 assert).
- **순서**: 보고서 부록 C 표와 같은 순서(공통 유틸 → 전처리 → 본 파이프라인 → 재현·검증 → 시각화).
  읽는 사람이 파이프라인 흐름대로 따라갈 수 있다.
- **역할 설명**: 각 파일의 모듈 docstring 첫 문장을 소제목 아래 붙인다. 별도 관리가 필요 없다.

실행:
    /opt/anaconda3/bin/python src/utils/export_code_appendix.py [--out 경로]
    → outputs/reports/final_report_code_appendix.md  (pandoc+Chrome으로 PDF 변환)

    /opt/anaconda3/bin/python src/utils/export_code_appendix.py --check
    → 커밋된 별책이 현재 `src/`와 일치하는지만 확인한다(고치지 않는다). 다르면 exit 1.
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
DEFAULT_OUT = REPO / "outputs" / "reports" / "final_report_code_appendix.md"

#: 별책에 싣지 않는 파일 — 이 생성기 자체는 분석 구현이 아니라 문서 도구다.
EXCLUDE = {"utils/export_code_appendix.py"}

#: 부록 C 표와 같은 순서. 여기 없는 파일은 「기타」로 자동 수집된다.
SECTIONS: list[tuple[str, str, list[str]]] = [
    ("공통 유틸", "경로·설정·로깅 — 모든 스크립트가 여기를 거친다.", [
        "utils/config.py",
        "utils/logger.py",
    ]),
    ("전처리", "원본 로딩부터 분할·공유 데이터셋 생성까지.", [
        "preprocessing/loader.py",
        "preprocessing/preprocessor.py",
        "preprocessing/export_split_manifest.py",
        "preprocessing/export_shared_dataset.py",
        "preprocessing/label_pre_post_by_rule.py",
        "preprocessing/fetch_treasury_gs1m.py",
        "preprocessing/fetch_macro_cpi.py",
        "preprocessing/fetch_macro_initial_claims.py",
        "preprocessing/fetch_macro_unemployment_rate.py",
        "preprocessing/fetch_macro_yield_spread.py",
    ]),
    ("분석 — 본 파이프라인", "부도확률 모형부터 최종 평가까지, 보고서 결과를 직접 만드는 코드.", [
        "analysis/model.py",
        "analysis/realized_return.py",
        "analysis/realized_return_cashflow.py",
        "analysis/oof_diagnostics.py",
        "analysis/sharpe_optimizer.py",
        "analysis/final_evaluation.py",
        "analysis/second_test_evaluation.py",
    ]),
    ("분석 — 재현·검증", "보고서 근거 표를 다시 만드는 스크립트. 수치를 의심할 때 돌린다.", [
        "analysis/preprocessing_validation.py",
        "analysis/t2_contribution_reassessment.py",
        "analysis/term_split_comparison.py",
        "analysis/missing_scheme_comparison.py",
        "analysis/macro_indicator_screening.py",
        "analysis/auc_sample_filter_comparison.py",
        "analysis/model_comparison.py",
        "analysis/realized_return_spec_check.py",
        "analysis/realized_return_sensitivity.py",
        "analysis/hpr_realized_return.py",
        "analysis/excluded_audit.py",
    ]),
    ("시각화", "보고서 그림 6종. 산출 CSV만 읽고 재계산하지 않는다.", [
        "viz/plots.py",
    ]),
]


def fence_for(body: str) -> str:
    """본문과 충돌하지 않는 코드 펜스를 만든다.

    ⚠️ 이 프로젝트의 docstring에는 **백틱 펜스가 들어 있다**(`model.py`의 구조식,
    `realized_return.py`의 공식 블록 등 4개 파일). 3중 백틱으로 감싸면 그 지점에서
    코드블록이 **조기 종료**되어 나머지 코드가 본문으로 새어 나온다. 그래서 본문에 나오는
    가장 긴 백틱 런보다 하나 더 긴 펜스를 쓴다(CommonMark 규칙).
    """
    longest = max((len(m) for m in re.findall(r"^(`{3,})", body, re.M)), default=0)
    return "`" * max(3, longest + 1)


def first_sentence(path: Path) -> str:
    """모듈 docstring의 첫 문장. 없으면 빈 문자열."""
    try:
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
    except SyntaxError:
        return ""
    line = " ".join(doc.strip().splitlines()[:2]).strip()
    for end in ("—", ". ", "다.", "。"):
        if end in line:
            head = line.split(end)[0] + (end if end in ("다.", "。") else "")
            if len(head) > 10:
                return head.strip().rstrip("—").strip()
    return line[:120].strip()


def build() -> tuple[str, int, int]:
    """별책 본문을 만든다 — `(문서, 파일수, 줄수)`. 파일에 쓰지 않는다(`--check`가 재사용)."""
    found = {str(p.relative_to(SRC)) for p in SRC.rglob("*.py")
             if "__pycache__" not in p.parts and p.name != "__init__.py"} - EXCLUDE
    listed = {rel for _, _, rels in SECTIONS for rel in rels}

    missing_file = sorted(listed - found)          # 분류표에는 있으나 사라진 파일
    extra = sorted(found - listed)                 # 새로 생겼는데 분류표에 없는 파일
    sections = list(SECTIONS)
    if extra:
        sections.append(("기타", "분류표에 없어 자동 수집된 파일 — 별책 갱신 시 분류표에 넣는다.", extra))
        print(f"⚠️  분류표에 없는 파일 {len(extra)}개를 「기타」로 실었다: {', '.join(extra)}")
    if missing_file:
        print(f"⚠️  분류표에 있으나 존재하지 않는 파일: {', '.join(missing_file)}")

    total_lines = sum(len((SRC / r).read_text(encoding="utf-8").splitlines())
                      for _, _, rels in sections for r in rels if (SRC / r).exists())
    n_files = sum(1 for _, _, rels in sections for r in rels if (SRC / r).exists())

    out: list[str] = [
        "# 최종보고서 별책 — 전체 구현 코드",
        "",
        "서울대학교 핀테크 전문가 과정 「통계, 데이터 사이언스」 팀 프로젝트 — "
        "Lending Club 신용평가 / Sharpe Ratio 최적화",
        "",
        f"- **범위**: 저장소 `src/` 전체 — {n_files}개 파일, {total_lines:,}줄",
        "- **본책**: `outputs/reports/final_report.md` (부록 C가 이 별책을 가리킨다)",
        "- 이 문서는 `src/utils/export_code_appendix.py`가 생성한다. "
        "손으로 고치지 말고 코드를 고친 뒤 다시 생성한다.",
        "",
        "## 목차",
        "",
        "| 구분 | 파일 | 줄 |",
        "| --- | --- | ---: |",
    ]
    for title, _, rels in sections:
        for rel in rels:
            p = SRC / rel
            if p.exists():
                n = len(p.read_text(encoding="utf-8").splitlines())
                out.append(f"| {title} | `src/{rel}` | {n:,} |")
    out.append(f"| **합계** | **{n_files}개 파일** | **{total_lines:,}** |")
    out.append("")

    for title, blurb, rels in sections:
        out += [f"# {title}", "", blurb, ""]
        for rel in rels:
            p = SRC / rel
            if not p.exists():
                continue
            body = p.read_text(encoding="utf-8").rstrip("\n")
            desc = first_sentence(p)
            out += [f"## `src/{rel}`", ""]
            if desc:
                out += [desc, ""]
            f = fence_for(body)
            out += [f"{f}python\n{body}\n{f}", ""]

    # 누락 검증 — 분류되지 않은 채 빠진 파일이 하나도 없어야 한다
    written = {r for _, _, rels in sections for r in rels if (SRC / r).exists()}
    assert written == found, f"누락: {sorted(found - written)}"

    return "\n".join(out) + "\n", n_files, total_lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true",
                    help="커밋된 별책이 현재 src/와 일치하는지만 확인한다 (쓰지 않음, 다르면 exit 1)")
    args = ap.parse_args()

    doc, n_files, total_lines = build()

    if args.check:
        if not args.out.exists():
            raise SystemExit(f"❌ 별책이 없다: {args.out}  →  --check 없이 실행해 생성하라")
        current = args.out.read_text(encoding="utf-8")
        if current == doc:
            print(f"✅ 별책이 src/와 일치한다 ({n_files}개 파일 · {total_lines:,}줄)")
            return
        raise SystemExit(
            f"❌ 별책이 src/와 어긋났다: {args.out}\n"
            f"   src/ 기준 {n_files}개 파일 · {total_lines:,}줄\n"
            f"   → /opt/anaconda3/bin/python src/utils/export_code_appendix.py 로 다시 생성하고 함께 커밋하라")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(doc, encoding="utf-8")
    print(f"✅ {args.out}")
    print(f"   {n_files}개 파일 · {total_lines:,}줄 · {args.out.stat().st_size/1024:,.0f}KB")


if __name__ == "__main__":
    main()
