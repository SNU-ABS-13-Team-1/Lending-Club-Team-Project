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
        P1["행 필터링<br/>(Current/Late 제외,<br/>정책미달 라벨 재분류)"]
        P2["결측치 처리<br/>(더미변수화)"]
        P3["범주형 변수 더미화"]
        P4["비율 파생변수 생성<br/>(단위 표준화)"]
        P5["컬럼명 표준화"]
        P6["타겟(종속변수) 라벨링<br/>(정의 미확정)"]
        P7["거시지표 결합<br/>(issue_d 기준, lag 미확정)"]
        B1["변수 사전 검증 스크립트<br/>label_pre_post_by_rule.py"]
        B3["거시지표 수집<br/>fetch_macro_*.py<br/>(다운로드+검증+저장)"]
    end

    subgraph PROC["data/processed"]
        C1["변수 사전/라벨 산출물<br/>variable_dictionary_byGJ.xlsx"]
        C2["모델링용 처리 데이터"]
        C3["거시경제지표 시계열<br/>macro_*.csv + .source.md"]
        C4["무위험수익률<br/>us_treasury_GS3_GS5_*.csv"]
    end

    subgraph AN["src/analysis"]
        D1["Train 60%<br/>부도확률 예측모형 학습"]
        D2["Validation 20%<br/>Sharpe Ratio 기준 threshold 확정"]
        D3["Test 20%<br/>확정 모형·threshold 그대로 검증"]
    end

    subgraph VIZ["src/viz"]
        E1["차트 생성"]
    end

    subgraph OUT["outputs"]
        F1["reports/"]
        F2["figures/"]
    end

    A1 --> P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7 --> C2
    A2 --> B1 --> C1
    X1 --> B3
    B3 --> C3
    C3 --> P7
    C1 --> D1
    C2 --> D1
    C4 --> D2
    D1 --> D2 --> D3
    D3 --> E1 --> F2
    D3 --> F1

    classDef planned stroke-dasharray: 4 3;
    class P2,P3,P4,P5,P6,P7,C2 planned;
```

> 실선 노드 = 이미 구현/확정된 단계, 점선 노드 = 아직 팀에서 확정하지 않은 계획 단계.
> `P2`~`P7`의 순서는 예시이며 실제 처리 순서·세부 방식은 팀 확정 필요 (`src/preprocessing/AGENTS.md` 참고).
> 거시지표 **수집**(`B3`)은 완료됐으나 대출 데이터와의 **결합**(`P7`)은 미확정이다 —
> lag 개월 수·level/YoY 선택·`issue_d` 더미와의 다중공선성 처리가 남았다
> (`outputs/reports/macro_indicator_selection.md` 6절).
> 무위험수익률(`C4`)은 독립변수가 아니라 Sharpe Ratio 계산(`D2`)에만 쓴다.

## 2. 승인/거절 의사결정 로직 (Sharpe Ratio 최적화)

```mermaid
graph TD
    L["대출 신청 1건"] --> M{"신용평가모형<br/>부도확률 p 예측"}
    M -->|"p < threshold"| APP["승인"]
    M -->|"p ≥ threshold"| REJ["거절"]

    APP --> R1{"실제 상환 결과"}
    R1 -->|"정상상환"| RET1["실현수익률 = int_rate"]
    R1 -->|"부도"| RET2["실현수익률 = 0"]

    REJ --> RET3["실현수익률 = 무위험수익률 Rf<br/>(국채 투자 가정)"]

    RET1 --> PORT["포트폴리오 수익률 분포"]
    RET2 --> PORT
    RET3 --> PORT

    PORT --> SR["Sharpe Ratio<br/>= (평균 − Rf) / 표준편차(ddof=1)"]
    SR --> TH["Validation에서<br/>Sharpe Ratio 최대화하는<br/>threshold 탐색"]
    TH -.피드백.-> M
```

> Test set은 위 피드백 루프(threshold 재탐색)에 참여하지 않는다 — 확정된 모형·threshold를 그대로 적용해 검증만 한다.

## 참고

이 다이어그램은 단순화된 개요이며, 아래 세부 방법론은 의도적으로 생략했다 — 최신 내용은 각 문서를 참고:
- Threshold 확정 후 Train 전체 재학습, 안정성 검증(30-seed 반복), "전부 승인" 베이스라인 비교: `src/analysis/AGENTS.md`
- 개별 대출 IRR 심화 계산, 대출 간 독립성 문제, 수익률 분포 히스토그램, 사전적·사후적 Sharpe Ratio 구분: `README.md`
