# 아키텍처 개요

## 1. 데이터 파이프라인

```mermaid
graph LR
    subgraph RAW["data/raw (원본, 수정 금지)"]
        A1["원본 대출 데이터 CSV<br/>(git 미추적)"]
        A2["공식 데이터 사전(Data Dictionary)"]
    end

    subgraph EXT["외부 공개 통계"]
        X1["FRED<br/>(거시경제지표, 국채수익률)"]
    end

    subgraph PRE["src/preprocessing"]
        P1["행 필터링 + 만기 표본 필터<br/>loader.py → 723,563건<br/>(Current/Late 제외, 정책미달 재분류,<br/>issue_d + term ≤ 2020-04)"]
        P2["피처 컬럼 선택 + 타겟 라벨링<br/>loader.py<br/>(사전 변수 + PRE_APPROVAL_OVERRIDES)"]
        P3["dtype 정리<br/>preprocessor.py<br/>(결측 NaN 유지 · 범주형 category)"]
        P4["랜덤 6:2:2 층화분할<br/>preprocessor.py"]
        P5["비율 파생변수 · 컬럼명 표준화<br/>(미확정 — 보류)"]
        B1["변수 사전 검증 스크립트<br/>label_pre_post_by_rule.py"]
        B3["거시지표 수집<br/>fetch_macro_*.py<br/>(다운로드+검증+저장)"]
    end

    subgraph VAL["src/analysis — 재현·검증 (본 파이프라인과 별개)"]
        V1["preprocessing_validation.py<br/>t2_contribution_reassessment.py<br/>term_split_comparison.py<br/>missing_scheme_comparison.py<br/>macro_indicator_screening.py<br/>auc_sample_filter_comparison.py<br/>realized_return_spec_check.py<br/>realized_return_sensitivity.py"]
    end

    subgraph PROC["data/processed"]
        C1["변수 사전/라벨 산출물<br/>variable_dictionary_byGJ.xlsx"]
        C2["모델링용 처리 데이터<br/>(파일 저장 미구현 — 현재 in-memory)"]
        C3["거시경제지표 시계열<br/>macro_*.csv + .source.md<br/>#15 최종 미사용"]
        C4["무위험수익률<br/>us_treasury_GS3_GS5_*.csv"]
    end

    subgraph AN["src/analysis"]
        D1["Train 60%<br/>XGBoost PD 모형 · K-fold OOF<br/>model.py"]
        D2["실현수익률 · 칸별 통계표<br/>realized_return.py<br/>(구조 A′, 잠정 가정)"]
        D3["Validation 20%<br/>Sharpe 최대화 threshold 확정<br/>sharpe_optimizer.py"]
        D4["Test 20%<br/>확정 모형·threshold 그대로 검증"]
        D5["설계 진단 C-1/C-2/C-3<br/>oof_diagnostics.py"]
    end

    subgraph VIZ["src/viz"]
        E1["차트 생성<br/>plots.py"]
    end

    subgraph OUT["outputs"]
        F1["reports/"]
        F2["figures/"]
    end

    A1 --> P1 --> P2 --> P3 --> P4 --> D1
    P4 -.저장 미구현.-> C2
    P4 -.보류.-> P5
    A2 --> B1 --> C1
    X1 --> B3
    B3 --> C3
    C1 --> P2
    C4 --> D2
    A1 -."사후 컬럼 (현금흐름 전용)".-> D2
    D1 --> D2 --> D3 --> D4
    D2 -.-> D5
    D5 -.진단 결과.-> F1
    D4 --> E1 --> F2
    D4 --> F1

    A1 -.-> V1
    C1 -.-> V1
    C3 -.-> V1
    V1 -.검증 결과.-> F1

    classDef planned stroke-dasharray: 4 3;
    class P5,C2,C3,D3,D4,E1 planned;
```

> `VAL` 그룹은 **본 파이프라인이 아니다** — 문서에 실린 표를 다시 만드는 재현·검증 스크립트이며,
> 원본/산출물을 읽어 `outputs/`에 근거를 남기는 곁가지다. `config.yaml`의 6:2:2를 따르지 않고
> 자체 분할을 쓴다 (`AGENTS.md`의 코드 색인 ② 참고).
>
> 실선 노드 = 이미 구현/확정된 단계, 점선 노드 = 아직 구현되지 않았거나 팀 미확정인 단계.
> `P1`~`P4`는 **구현 완료**다(커밋 `7e1cb9d`) — 함수 목록은 `src/preprocessing/AGENTS.md`의
> 「구현 — 어느 함수를 부르는가」. `P5`(파생변수·컬럼명)는 미확정이라 보류 상태다.
>
> ⚠️ **거시지표는 본 파이프라인에 결합하지 않는다** — #15에서 **최종 미사용으로 확정**됐다.
> **수집**(`B3`)은 완료돼 `C3`에 남아 있으나, 이제 재현·검증(`V1`)과 과거 기록 용도뿐이다.
> lag·level/YoY·다중공선성 검토는 **종결된 논의**다(`macro_indicator_selection.md`는 배경 자료).
>
> ⚠️ 결측은 **NaN 그대로 투입**하고 범주형은 **`category` dtype**으로 넘긴다 — 결측 더미도,
> 원-핫 더미도 만들지 않는다(#13 ③·#17 ①). 표준화·로그변환·구간화·캡핑도 하지 않는다.
>
> 무위험수익률(`C4`)은 독립변수가 아니라 실현수익률·Sharpe 계산(`D2`)에만 쓴다 —
> 발행시점 × 만기매칭이라 고정 상수가 아니다(#18).
> `A1 -.-> D2` 점선은 **사후(post-approval) 컬럼**(`total_pymnt`·`recoveries` 등)이 실현수익률
> 계산에만 들어간다는 뜻이다 — **피처 테이블에는 절대 넣지 않는다**(누수, `src/preprocessing/AGENTS.md`).

## 2. 승인/거절 의사결정 로직 (Sharpe Ratio 최적화)

```mermaid
graph TD
    L["대출 신청 1건"] --> M{"신용평가모형<br/>부도확률 p 예측"}
    M -->|"p < threshold"| APP["승인"]
    M -->|"p ≥ threshold"| REJ["거절"]

    APP --> R1{"실제 상환 결과"}
    R1 -->|"정상상환"| RET1["R_계약 (건별)<br/>= (W/P)^(12/T) − 1<br/>W = Σ CFm·F(m,T)"]
    R1 -->|"부도"| RET2["r̄_부도 (PD분위 × term 그룹평균)<br/>같은 공식, 회수 현금흐름<br/>구조는 A′로 확정 · 세부 3건 미확정"]

    REJ --> RET3["실현수익률 = 무위험수익률 Rf<br/>(국채 투자 가정)"]

    RET1 --> PORT["포트폴리오 수익률 분포"]
    RET2 --> PORT
    RET3 --> PORT

    PORT --> SR["Sharpe Ratio<br/>= (평균 − Rf) / 표준편차(ddof=1)"]
    SR --> TH["Validation에서<br/>Sharpe Ratio 최대화하는<br/>threshold 탐색"]
    TH -.피드백.-> M
```

> Test set은 위 피드백 루프(threshold 재탐색)에 참여하지 않는다 — 확정된 모형·threshold를 그대로 적용해 검증만 한다.
>
> **재투자 가정(확정, `decision_log.md` #18)**: `RET1`·`RET2` 모두 **매달 받는 상환액을 잔존기간에
> 맞춘 국채에 재투자**한다고 보고 만기 `T` 시점 종가로 평가한다. `F(m,T)`는 `m`월 수령액을 `T`까지
> 굴린 성장배수다. 정상상환·부도 양쪽에 같은 관례를 쓰므로 **"정상상환 = `int_rate`"는 폐기**됐다.
>
> **실현수익률 구조 — A′ 확정** (`decision_log.md` #20, 2026-07-30):
> `RET2`(부도)는 **PD 분위(× term)별 그룹 평균**, `RET1`(정상상환)은 **건별 계약 현금흐름**으로 계산한다.
> ```
> E[XR] = (1 − p̂)·XR_계약  +  p̂·r̄_부도,d
> ```
> **구조 B(2단계 hurdle 건별 회귀)는 기각됐다 — 위 그림에 부도 서브셋 회귀 모델은 없다.** 모형은
> `M`(PD 분류기) 하나뿐이다.
>
> ⚠️ **계산 세부 3건은 미확정**이며 B팀 담당이다(#20) — **조기상환 보정 방식**(1순위: 계약 현금흐름만
> 쓰면 `RET1`이 과대추정된다), 국채 금리 기준(ⓒ발행시점 고정 권고), 서비스수수료. 잠정값으로 진행
> 가능하나, 확정 시 `SR`과 `TH`의 값이 함께 달라진다.
>
> ⚠️ `TH`의 **랭킹 기준은 미확정**이다 — `p̂` 단독 / `E[XR]` / `q_score`(= `E[XR]/√Var[XR]`).
> Validation에서만 비교해 사전 확정한 뒤 Test 1회(#5·#20).
>
> ⚠️ `SR`은 **절대값을 성과로 보고하지 않는다.** 재투자 가정이 모형 전략과 approve-all 대조군(#17 ②)을
> 똑같이 밀어올리므로, 헤드라인 지표는 **Δ Sharpe = (모형 − approve-all)** 이다.

## 참고

이 다이어그램은 단순화된 개요이며, 아래 세부 방법론은 의도적으로 생략했다 — 최신 내용은 각 문서를 참고:
- Threshold 확정 후 Train 전체 재학습, 안정성 검증(30-seed 반복), "전부 승인" 베이스라인 비교: `src/analysis/AGENTS.md`
- 개별 대출 IRR 심화 계산, 대출 간 독립성 문제, 수익률 분포 히스토그램, 사전적·사후적 Sharpe Ratio 구분: `README.md`
