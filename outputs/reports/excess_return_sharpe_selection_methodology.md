---
title: 조건부 초과수익 기반 대출 선별 계산 명세
status: resolved (2026-07-31 회의 — 가중은 병기 채택, Test 재추출은 기각. decision_log #23)
updated: 2026-07-31
scope: 실현 초과수익률, Train 조건부 모멘트, 대출 선별점수, Validation 목적함수
related:
  - outputs/reports/realized_return_treasury_reinvestment_methodology.md
  - outputs/reports/methodology_23_review_kgj.md
  - JDQS 2026 참고논문 (jdqs-02-2026-0013en)
---

# 조건부 초과수익 기반 대출 선별 계산 명세 (이슈 #23 — 회의 처리 완료)

> ✅ **2026-07-31 회의에서 처리됐다** (`decision_log.md` #23 안건 B·C).
> **가중 방식(안건 B)**: 등가중 헤드라인 유지 + 금액가중(`funded_amnt`) **민감도 병기** —
> 등가중은 본문 공식의 특수해이므로 상충하지 않는다.
> **Test 재추출(안건 C)**: **기각** — 같은 표결로 8:2 + 2nd Test 체계가 정식 확정됐고(K=50 유지),
> 재추출이 풀려던 표본 운 문제는 2nd Test(481,833건)가 더 강하게 해소한다.
>
> ⚠️ 본문 §8.1 「현재 확정」은 **작성자 기준의 확정**이며 팀 확정과 다르다. 특히 정상상환
> 조건부 평균을 칸 평균 `m₀,g`로 두는 부분은 `decision_log.md` #20이 **구조 A′(정상상환 =
> 건별 계약 현금흐름)로 확정하며 기각한 방향**이다. 본문은 제출 원안 그대로 보존한다 —
> 대조표는 `methodology_23_review_kgj.md`에 있다.

## 1. 이번 단계의 결론

현재 채택하는 계산 구조는 다음과 같다.

\[
\text{OOF PD}
\longrightarrow
(\text{계약기간},\text{PD 구간})
\longrightarrow
\text{완납·부도별 초과수익 분포}
\longrightarrow
(\widehat\mu_i,\widehat\sigma_i)
\longrightarrow
q_i=\frac{\widehat\mu_i}{\widehat\sigma_i}
\longrightarrow
a_i(\tau)=\mathbf 1(q_i>\tau)
\]

Validation의 목적함수는 모든 후보 대출에 미리 배정된 자금을 기준으로 계산한 **금액가중 횡단면 위험조정 초과수익률 지표**다.

이 지표는 일반적인 월별·분기별 포트폴리오 Sharpe ratio가 아니다. 대출별 실현 초과수익률의 횡단면 분산을 위험 대용치로 사용하는 Sharpe-like index다.

### 1.1 용어

- `term`: 계약기간. 이 데이터에서는 36개월 또는 60개월이다.
- `FP`: `Fully Paid`. 정상적으로 완납된 대출이다.
- `CO`: `Charged Off`. 상각·부도 처리된 대출이다.
- `PD`: `Probability of Default`. 모형이 추정한 부도확률이다.
- `PD bin`: 비슷한 PD를 가진 대출을 묶은 구간이다. 계약기간별 Train OOF PD를 기준으로 각각 10분위 경계를 정한다.
- `OOF PD`: 해당 행이 포함되지 않은 fold로 학습한 모형이 그 행에 부여한 PD다.

---

## 2. 분석대상

관측기준월을 \(M_{\mathrm{obs}}\), 대출 \(i\)의 발행월을 \(M_i\), 계약기간을 \(T_i\in\{36,60\}\)개월이라고 한다.

분석대상 포함 조건은 다음과 같다.

\[
M_i+T_i+6\le M_{\mathrm{obs}}
\]

현재 원본 데이터 표를 재현하는 관측기준월은 2020년 10월이다.

- 만기 + 6개월 조건 충족: 724,404건
- 그중 `Current`: 451건, 0.0623%
- `Fully Paid` 또는 `Charged Off`: 723,563건
- 36개월: 623,125건
- 60개월: 100,438건

최종 수익률 분석은 `Fully Paid`와 `Charged Off`만 사용한다.

6개월은 표본 성숙도 버퍼다. LendingClub의 [2016 Form 10-K](https://www.sec.gov/Archives/edgar/data/1409970/000140997017000255/a201610-k.htm)가 공시한 120일 연체 시 non-accrual, 늦어도 150일 연체 시 charge-off 기준을 월납 구조에 적용하면 마지막 정상 납입부터 charge-off까지 약 5~6개월이 걸릴 수 있다. 따라서 계약만기 뒤 6개월을 추가해 만기 무렵 미납된 대출의 상태가 확정될 시간을 확보한다.

이는 결과 라벨의 성숙도·청정도를 위한 버퍼이며, 모든 추심과 회수가 종료되었다는 보장은 아니다. 또한 `realized_return_treasury_reinvestment_methodology.md`의 recovery 수령시점 6개월 근사와도 구분한다.

---

## 3. 대출별 실현 초과수익률

### 3.1 실현수익률

대출 \(i\)의 연율화 실현수익률 \(R_i\)는
`realized_return_treasury_reinvestment_methodology.md`의 확정식을 사용한다.

### 3.2 만기 대응 무위험수익률

- 36개월 대출: 발행월의 월별 GS3
- 60개월 대출: 발행월의 월별 GS5

GS3·GS5의 bond-equivalent yield \(y_i\)를 실현수익률과 비교 가능한 유효연율로 변환한다.

\[
r_{f,i}
=
\left(1+\frac{y_i}{200}\right)^2-1
\]

### 3.3 실현 초과수익률

\[
XR_i=R_i-r_{f,i}
\]

이후 조건부 평균, 조건부 분산 및 최종 횡단면 지표는 모두 \(R_i\)가 아니라 \(XR_i\)를 기준으로 계산한다. 이렇게 해야 발행시점과 계약기간에 따라 다른 국채금리의 횡단면 차이가 대출위험으로 분류되는 것을 줄일 수 있다.

---

## 4. Train의 조건부 모멘트 추정

### 4.1 OOF 부도확률

전체 분석대상은 먼저 무작위로 Train 60%, Validation 20%, Test 20%로 분할한다. 각 분할에서 36개월·60개월 및 완납·부도 구성비가 크게 달라지지 않도록 `(term, FP/CO)` 조합을 층화 기준으로 사용한다. 재현을 위해 분할 seed와 대출별 `split`을 저장한다.

Train 내부 OOF 예측은 5-fold로 생성한다. 이때도 `(term, FP/CO)` 조합을 층화하여 각 fold의 계약기간·상태 구성을 유지한다.

Train의 같은 행을 학습한 모형이 그 행의 부도확률을 다시 예측하면 다음 경로가 생긴다.

\[
y_i
\longrightarrow
\text{모형 학습}
\longrightarrow
\widehat p_i
\longrightarrow
\text{PD 구간}
\longrightarrow
(\widehat m_{y,g},\widehat s_{y,g})
\]

이는 Validation/Test 누수는 아니지만, 2단계 조건부 수익률 추정에서 같은 결과변수를 재사용하는 in-sample double dipping이다.

따라서 Train의 구간과 조건부 모멘트는 반드시 OOF 부도확률로 추정한다.

\[
\widehat p_i^{\mathrm{OOF}}
=
f^{(-k(i))}(X_i)
\]

여기서 \(f^{(-k(i))}\)는 대출 \(i\)가 속한 fold를 제외하고 학습한 모형이다. 모든 Train 행은 자신을 학습하지 않은 모형의 예측값을 받아야 한다.

반복 무작위 Train/Validation/Test 분할은 이 OOF 절차를 대신하지 못한다. 각 반복의 Train 내부에서 별도의 cross-fitting이 필요하다.

모형 탐색 단계에서 Train 전체로 학습한 뒤 같은 Train에 예측한 in-sample PD는 모형 구조와 오류를 점검하는 용도로만 사용할 수 있다. 그 확률로 최종 PD 구간 경계나 조건부 모멘트를 확정하지 않는다.

OOF PD는 각 Train 행의 PD가 자기 자신의 부도라벨을 학습한 모형에서 나오지 않도록 한다. 이후 전체 Train의 `XR`을 이용해 그룹별 조건부 모멘트를 추정하고 이를 Validation/Test에 적용하는 것은 데이터 누수가 아니다. 다만 이 그룹 통계량에는 해당 Train 행 자신의 `XR`도 포함되므로, Train 행에 부여한 \(\widehat\mu_i\), \(\widehat\sigma_i\), \(q_i\) 자체를 완전히 out-of-sample 성과로 해석할 수는 없다.

현재 절차에서는 Validation에서 임계값을 선택하고 Test에서 최종 평가하므로 Train 조건부 모멘트를 전체 Train에서 추정해도 된다. 향후 Train 자체의 선별성과까지 편향 없이 보고해야 한다면, 조건부 모멘트도 해당 행을 제외한 fold에서 추정하는 추가 cross-fitting을 적용한다.

### 4.2 계약기간 × PD 구간

Train 대출 \(i\)의 조건부 그룹은 다음과 같이 정의한다.

\[
g_i
=
\left(
T_i,\,
B_{T_i}\!\left(\widehat p_i^{\mathrm{OOF}}\right)
\right)
\]

36개월과 60개월의 PD 분포가 다를 수 있으므로 PD 구간 경계는 계약기간별로 따로 추정한다.

구간 수는 계약기간별 10개로 정한다. 즉, 36개월 Train과 60개월 Train을 각각 OOF PD가 낮은 순서로 정렬한 뒤 10분위로 나눈다. 이에 따라 조건부 모멘트 추정의 기본 그룹 수는 다음과 같다.

\[
2\text{개 계약기간}
\times
10\text{개 PD 분위}
\times
2\text{개 상태(FP/CO)}
=40\text{개 셀}
\]

10분위는 전체 표본을 한꺼번에 나누는 것이 아니라 각 계약기간 안에서 따로 만든다. 따라서 동일한 분위 번호라도 36개월과 60개월의 실제 PD 경계값은 서로 다를 수 있다.

Train에서 정한 수치 경계는 Validation과 Test에 그대로 적용한다. 범위 밖의 값은 양 끝 구간에 포함한다.

동일한 PD 값이 분위 경계에 집중되는 경우의 경계 처리와 `(term, PD bin, FP/CO)` 셀의 최소 건수·유효표본크기 기준은 구현 전에 별도로 확정한다. 표본이 부족한 셀이 확인되면 Test 결과를 보지 않고 Train에서만 인접 PD 구간 병합 여부를 결정한다.

### 4.3 완납·부도별 구간 내 금액가중치

대출 \(i\)의 투자 가능 금액을 \(A_i=funded\_amnt_i\)라고 한다.

대출상태를 다음과 같이 정의한다.

\[
Y_i=
\begin{cases}
0,&\text{Fully Paid}\\
1,&\text{Charged Off}
\end{cases}
\]

그룹 \(g\)와 상태 \(y\in\{0,1\}\) 안의 금액가중치는 다음과 같다.

\[
v_{i\mid g,y}
=
\frac{A_i}
{\sum_{j\in g:Y_j=y}A_j}
\]

### 4.4 완납·부도별 조건부 초과수익 모멘트

그룹 \(g\)와 상태 \(y\)의 조건부 평균은 다음과 같다.

\[
\widehat m_{y,g}
=
\sum_{i\in g:Y_i=y}
v_{i\mid g,y}XR_i
\]

조건부 가중 표본분산은 다음과 같다.

\[
\widehat s_{y,g}^2
=
\frac{
\sum_{i\in g:Y_i=y}
v_{i\mid g,y}
(XR_i-\widehat m_{y,g})^2
}{
1-\sum_{i\in g:Y_i=y}v_{i\mid g,y}^2
}
\]

따라서 각 PD 구간은 다음 네 값을 갖는다.

- 완납 평균 \(\widehat m_{0,g}\)
- 완납 분산 \(\widehat s_{0,g}^2\)
- 부도 평균 \(\widehat m_{1,g}\)
- 부도 분산 \(\widehat s_{1,g}^2\)

이 값들은 임의의 대출 한 건이 아니라 임의의 투자 1달러가 경험하는 상태별 초과수익 분포를 추정한다.

### 4.5 대출별 조건부 기대 초과수익률

대출 \(i\)에 적용할 부도확률을 \(\widehat p_i\)라고 한다.

- Train: \(\widehat p_i^{\mathrm{OOF}}\)
- Validation/Test: Train만으로 학습한 모형 또는 fold 모형 앙상블의 예측확률

대출 \(i\)의 조건부 기대 초과수익률은 완납·부도 분포를 \(\widehat p_i\)로 혼합하여 계산한다.

\[
\widehat\mu_i
=
(1-\widehat p_i)\widehat m_{0,g_i}
+
\widehat p_i\widehat m_{1,g_i}
\]

따라서 PD는 구간을 정하는 데만 쓰이는 것이 아니라, 각 대출의 완납·부도 수익률을 혼합하는 확률로 직접 사용된다. 같은 PD 구간 안에서도 \(\widehat p_i\)가 다르면 \(\widehat\mu_i\)가 달라진다.

이 계산은 \(\widehat p_i\)가 실제 부도확률로 해석될 수 있어야 하므로 확률보정 상태를 반드시 점검해야 한다.

확률보정 방법의 선택과 검증은 부도확률 모형을 담당하는 팀이 수행한다. 조건부 초과수익 계산 단계에서는 보정법을 다시 선택하거나 PD를 재보정하지 않고, 모델팀이 확정한 동일한 보정 절차로 생성한 Train OOF PD와 Validation/Test PD를 입력으로 사용한다. 보정 과정도 각 OOF 학습자료 안에서만 적합되어야 하며, 해당 OOF 검증행이나 Test 라벨을 사용하면 안 된다.

### 4.6 대출별 조건부 초과수익률 분산

전체분산의 법칙을 적용하면 대출 \(i\)의 조건부 분산은 다음과 같다.

\[
\widehat\sigma_i^2
=
(1-\widehat p_i)
\left[
\widehat s_{0,g_i}^2+
(\widehat m_{0,g_i}-\widehat\mu_i)^2
\right]
+
\widehat p_i
\left[
\widehat s_{1,g_i}^2+
(\widehat m_{1,g_i}-\widehat\mu_i)^2
\right]
\]

\[
\widehat\sigma_i
=
\sqrt{\widehat\sigma_i^2}
\]

첫 번째와 두 번째 상태분산 항은 각각 완납·부도 상태 안의 수익 변동을 반영한다. 평균 차이의 제곱 항은 완납과 부도라는 두 상태 사이의 수익 격차를 반영한다.

이 값은 \(X_i\)의 모든 특성을 고정한 진정한 개별 조건부 분산
\(\operatorname{Var}(XR_i\mid X_i=x_i)\)을 직접 추정한 것은 아니다. 같은 계약기간·PD 10분위 안에서는 상태별 평균과 분산을 공유하고, 개별 대출의 \(\widehat p_i\)만 달리 적용한 근사적 조건부 위험 추정치다.

### 4.7 조건부 투자점수

대출 \(i\)의 투자점수는 다음과 같다.

\[
q_i
=
\frac{\widehat\mu_i}{\widehat\sigma_i}
\]

이 값의 정확한 해석은 **conditional excess-return Sharpe-like score**다. 개별 대출의 미래 실현수익률이나 전통적인 포트폴리오 Sharpe ratio를 직접 추정한 값은 아니다.

---

## 5. Validation의 투자 결정

### 5.1 승인 규칙

\[
a_i(\tau)
=
\mathbf 1(q_i>\tau)
\]

- \(a_i(\tau)=1\): 대출에 투자
- \(a_i(\tau)=0\): 해당 자금을 국채에 투자

하나의 공통 임계값 \(\tau\)를 36개월과 60개월 대출에 함께 적용한다. \(q_i\)는 무차원 비율이므로 두 계약기간 사이에서 비교할 수 있다.

### 5.2 전체 후보자금 기준의 고정 가중치

Validation의 전체 후보집합을 \(\mathcal V\)라고 한다.

\[
w_i
=
\frac{A_i}
{\sum_{j\in\mathcal V}A_j}
\]

이 가중치는 \(\tau\)와 무관하게 고정한다. 선택된 대출만으로 가중치를 다시 정규화하지 않는다.

### 5.3 실제 자산수익률

\[
Z_i(\tau)
=
a_i(\tau)R_i
+
\left[1-a_i(\tau)\right]r_{f,i}
\]

전체 후보자금의 실제 자산수익률은 다음과 같다.

\[
R_P(\tau)
=
\sum_{i\in\mathcal V}w_iZ_i(\tau)
\]

대응하는 무위험 벤치마크는 다음과 같다.

\[
r_{f,P}
=
\sum_{i\in\mathcal V}w_ir_{f,i}
\]

전체 후보와 가중치가 고정되어 있으므로 \(r_{f,P}\)도 \(\tau\)와 무관하다.

### 5.4 자본슬롯별 초과수익률

\[
X_i(\tau)
=
Z_i(\tau)-r_{f,i}
=
a_i(\tau)(R_i-r_{f,i})
=
a_i(\tau)XR_i
\]

거절된 대출의 배정자금은 사라지지 않는다. 국채에 투자되므로 해당 자본슬롯의 초과수익률이 0이 된다.

포트폴리오의 금액가중 초과수익률은 다음과 같다.

\[
\overline X_P(\tau)
=
\sum_{i\in\mathcal V}w_iX_i(\tau)
=
R_P(\tau)-r_{f,P}
\]

### 5.5 금액가중 횡단면 표준편차

\[
s_{X,P}^2(\tau)
=
\frac{
\sum_{i\in\mathcal V}
w_i
\left[X_i(\tau)-\overline X_P(\tau)\right]^2
}{
1-\sum_{i\in\mathcal V}w_i^2
}
\]

\[
s_{X,P}(\tau)
=
\sqrt{s_{X,P}^2(\tau)}
\]

이 표준편차의 반복 관측 단위는 시간 월이 아니라 전체 후보 대출에 배정된 자본슬롯이다. 여러 발행시점의 대출을 합친 pooled cross section의 분산이다.

### 5.6 목적함수

금액가중 횡단면 위험조정 초과수익률 지표는 다음과 같다.

\[
S_{\mathrm{XS}}(\tau)
=
\frac{
\overline X_P(\tau)
}{
s_{X,P}(\tau)
}
\]

Validation에서 다음 임계값을 선택한다.

\[
\tau^*
=
\arg\max_{\tau}S_{\mathrm{XS}}(\tau)
\]

보고서의 정확한 영문 명칭은 다음과 같이 사용한다.

> **funded-amount-weighted cross-sectional risk-adjusted excess return index**

필요할 경우 `cross-sectional Sharpe-like index`를 괄호 안의 약칭으로 사용할 수 있다.

### 5.7 경계 사례

- 전부 거절: 모든 \(X_i=0\)이므로 \(0/0\)이다. 경제적 기준선으로 \(S_{\mathrm{XS}}=0\)을 부여한다.
- \(s_{X,P}=0\)이지만 \(\overline X_P\ne0\): 계산 이상 또는 퇴화 표본으로 표시하고 임계값 후보에서 제외한다.
- 대출의 \(\widehat\sigma_i=0\): 해당 \(q_i\)를 계산할 수 없으므로 계산 이상 또는 퇴화 그룹으로 표시한다.
- 완납 또는 부도 상태의 표본이 없거나 상태분산을 계산할 수 없는 그룹: 인접 그룹 병합 등 별도 규칙을 적용한다.
- 임계값 후보: 관측된 \(q_i\) 범위의 grid와 전부 승인·전부 거절 경계를 사용한다.

---

## 6. Test 평가와 반복 분할 안정성 검증

### 6.1 각 분할의 Test 평가

Validation에서 \(\tau^*\)를 선택한 뒤 다음 항목을 모두 동결한다.

- 부도확률 모형 또는 fold 앙상블
- 확률 보정 방법
- 계약기간별 PD 구간 경계
- 그룹별 완납·부도 조건부 모멘트
- 임계값 \(\tau^*\)

각 분할의 Test에는 동결된 규칙을 한 번만 적용한다. 해당 Test 결과를 보고 계산식, 구간, 모형 또는 임계값을 다시 조정하지 않는다.

### 6.2 반복 무작위 분할

단일 무작위 분할의 우연성에 의존하지 않도록 §4.1~§6.1의 전체 절차를 서로 다른 seed로 반복한다.

- 필수 반복 수: 30회, seed 0~29
- 목표 반복 수: 실행시간이 허용되면 100회, seed 0~99
- 각 반복: 무작위 층화 60/20/20 분할부터 Train 5-fold OOF, PD 10분위·조건부 모멘트 추정, Validation 임계값 선택 및 Test 평가까지 모두 새로 수행

반복 실행 전에 모형 구조와 하이퍼파라미터, 확률보정 절차, PD 경계 동률 처리, 표본 부족 병합 규칙, 임계값 탐색 절차와 성과지표 계산식을 동결한다. 각 반복의 Train 자료가 달라지므로 적합된 모형, PD 경계, 조건부 모멘트와 Validation 최적 임계값 자체가 달라지는 것은 정상이다. 반복 Test 결과를 보고 동결한 절차를 수정하지 않는다.

반복 \(b\)의 Test에서 모델 전략과 전부승인 전략의 차이를 다음과 같이 저장한다.

\[
\Delta S_{\mathrm{XS}}^{(b)}
=
S_{\mathrm{XS,model}}^{(b)}
-
S_{\mathrm{XS,approve\text{-}all}}^{(b)}
\]

최종 보고에는 다음을 포함한다.

- \(\Delta S_{\mathrm{XS}}\) 분포 히스토그램
- 승률:
  \(\frac{1}{B}\sum_{b=1}^{B}\mathbf 1(\Delta S_{\mathrm{XS}}^{(b)}>0)\)
- \(\Delta S_{\mathrm{XS}}\) 중앙값과 25·75분위
- 가능하면 5·95분위
- 승인율 분포

반복 분할은 표본이 서로 겹치므로 각 반복 결과를 독립적인 실험으로 간주하지 않는다. 이 분포는 분할에 따른 성과 안정성을 기술하는 용도로 사용하며, 독립표본을 전제로 한 유의확률로 해석하지 않는다.

---

## 7. 논문 방식에 대한 계산 검토

검토 대상: `jdqs-02-2026-0013en.pdf`

### 7.1 Train 예측값 재사용

논문 4.1절과 4.5절은 Train의 예측 PD로 10분위를 만들고, 같은 Train의 실현수익률로 조건부 수익과 구간 분산을 계산한다. OOF 예측 절차는 제시하지 않는다.

논문이 시행하는 반복 무작위 6:2:2 분할은 평가표본의 재사용을 줄이지만, 각 반복의 Train 내부에서 발생하는 in-sample double dipping을 제거하지는 않는다.

따라서 우리 계산에서는 OOF PD를 필수로 사용한다.

### 7.2 가중치 재정규화

논문은 거절자금을 국채에 투자한다고 명시한다. 이 논리를 일관되게 적용하면 전체 후보자금을 분모로 한 \(w_i\)가 고정되어야 한다.

PDF만으로 실제 구현 코드의 가중치 분모를 확인할 수 없으므로 논문이 선택된 대출만으로 재정규화했다고 단정하지는 않는다. 우리 계산에서는 전체 후보자금 분모를 명시적으로 고정한다.

### 7.3 표준편차의 관측 단위

논문 4.4절은 대출별 실현수익률의 가중 횡단면 표준편차를 시간계열 변동성의 대용치로 사용한다고 명시한다. 따라서 논문의 관측 단위가 불명확한 것은 아니다.

다만 이 값은 일반적인 월별·분기별 포트폴리오 수익률의 표준편차와 다르다. 우리 계산도 같은 한계를 인정하되, 실제 자산수익률이 아니라 만기 대응 국채 대비 **초과수익률**의 횡단면 표준편차를 사용하고 명칭을 Sharpe-like index로 제한한다.

### 7.4 조건부 분산 추정 방식

논문은 PD 10분위에서 상태별 평균수익률을 구해 대출별 기대수익률을 계산한 뒤, 그 기대수익률을 다시 구간화하여 구간 내 실현수익률의 표본분산을 위험 대용치로 부여한다. 따라서 같은 기대수익률 구간에 속한 대출은 같은 위험값을 받는다.

우리 계산은 이 두 번째 기대수익률 구간화를 사용하지 않는다. 계약기간별 PD 10분위에서 완납·부도 각각의 평균과 분산을 추정하고, 개별 PD를 이용해 §4.6의 전체분산 공식으로 대출별 \(\widehat\sigma_i^2\)를 직접 계산한다. 따라서 **10분위는 PD 조건부 모멘트를 추정하는 첫 번째 구간에만 적용**한다.

---

## 8. 현재 확정된 계산과 추후 논의사항

### 8.1 현재 확정

- 대출별 실현 초과수익률: \(XR_i=R_i-r_{f,i}\)
- 36개월 GS3, 60개월 GS5
- 무작위 층화 Train/Validation/Test 분할: 60%/20%/20%
- Train 내부 OOF: `(term, FP/CO)` 층화 5-fold
- Train 조건부 모멘트 추정에는 모델팀이 생성한 보정 완료 OOF PD 사용
- 확률보정 방법의 선택·검증은 모델팀 담당이며 조건부 초과수익 단계에서 재보정하지 않음
- 36개월·60개월별 Train OOF PD를 각각 10분위로 나누어 구간 경계 추정
- 각 그룹에서 완납·부도별 금액가중 평균과 분산 계산
- 기대수익률을 다시 구간화하지 않고 전체분산의 법칙으로 대출별 분산 계산
- 대출별 기대 초과수익률:
  \(\widehat\mu_i=(1-\widehat p_i)\widehat m_{0,g_i}
  +\widehat p_i\widehat m_{1,g_i}\)
- 대출별 분산: 완납·부도 혼합분포에 전체분산의 법칙 적용
- 투자점수: \(q_i=\widehat\mu_i/\widehat\sigma_i\)
- 승인규칙: \(a_i(\tau)=\mathbf 1(q_i>\tau)\)
- 전체 후보 `funded_amnt`를 분모로 가중치 고정
- 거절자금은 국채, 자본슬롯 초과수익률은 0
- Validation 목적함수: \(S_{\mathrm{XS}}(\tau)\)
- 각 무작위 분할의 Test는 동결 규칙으로 한 번만 평가
- 전체 파이프라인 반복 무작위 분할: 최소 30회, 실행시간 허용 시 목표 100회
- 반복 Test의 모델 대비 전부승인 \(\Delta S_{\mathrm{XS}}\) 분포·승률·중앙값 보고

### 8.2 추후 논의 필요

다음 항목은 계산식 자체를 바꾸지 않는 구현·분포 구성 선택이므로 현재 확정하지 않는다.

- 동일 PD가 분위 경계에 집중될 때의 세부 경계 처리
- Validation/Test PD에 Train 전체 재학습 모형을 사용할지 fold 모형 앙상블을 사용할지
- 그룹의 최소 건수·유효표본크기와 병합 기준

확률보정 방법과 calibration 검증 기준은 모델팀이 확정해 전달해야 하는 외부 입력사항이다.

다음은 현재 채택한 횡단면 지표와 다른 연구문제이므로 별도 논의한다.

- 대출 간 공분산을 포함한 포트폴리오 분산
- 발행월 코호트별 성과분포
- 월별 포트폴리오 가치 시계열을 복원한 전통적 Sharpe ratio

---

## 9. 다음 작업을 위한 OOF 구현 절차

### 9.1 팀원의 탐색 학습과 최종 OOF 실행 구분

팀원은 먼저 모델을 시험 학습해 변수, 전처리 및 모형 구조를 점검할 수 있다. 이 단계의 in-sample PD는 최종 계산에 사용하지 않는다.

모형 파이프라인의 기본 구조를 확인한 뒤 다음 순서로 함께 최종 OOF 예측을 생성한다.

1. 전체 데이터를 `(term, FP/CO)` 조합으로 층화하여 Train 60%, Validation 20%, Test 20%로 무작위 분할하고 각 행의 `split`을 고정한다.
2. Train 내부에서 동일한 층화 기준으로 5-fold를 수행한다.
3. 각 fold마다 나머지 fold로 전처리와 모형을 학습하고, 제외한 fold의 PD를 예측한다.
4. 모델팀이 선택한 확률보정도 fold 학습자료 안에서만 적합하고, 모든 Train 행이 자신을 학습하거나 보정에 사용하지 않은 모형으로부터 정확히 한 번 보정된 `pd_oof`를 받았는지 검증한다.
5. Train의 `pd_oof`로만 36개월·60개월 각각의 PD 10분위 경계를 정한다.
6. 각 `(term, PD bin, FP/CO)` 그룹에서 `XR`의 금액가중 평균·분산과 표본 수를 계산한다.
7. Train 전체로 재학습한 모형 또는 사전에 정한 fold 앙상블로 Validation PD를 예측한다.
8. Train에서 저장한 PD 구간 경계와 조건부 모멘트를 Validation에 그대로 적용한다.
9. Validation에서 \(S_{\mathrm{XS}}(\tau)\)를 최대화하는 임계값 \(\tau^*\)를 선택한다.
10. 해당 반복에서 모형, 보정법, 구간 경계, 조건부 모멘트와 임계값을 동결한 뒤 Test에 한 번만 적용한다.
11. 위 전체 절차를 seed 0~29로 최소 30회 반복하고, 실행시간이 허용되면 seed 0~99의 100회까지 확장한다.
12. 각 반복의 Test에서 모델 전략과 전부승인 전략의 \(\Delta S_{\mathrm{XS}}\), 승인율 및 임계값을 저장하고 분포를 요약한다.

단순히 Train 행에 fold 번호를 나중에 붙이는 것은 OOF가 아니다. 각 행의 `pd_oof`는 실제로 그 행을 제외하고 학습한 모형에서 생성되어야 한다. 결측치 대체, 스케일링, 변수선택 및 확률보정을 사용한다면 이 과정도 fold의 학습 부분에서만 적합해야 한다.

### 9.2 필요한 중간 데이터

모형 결과와 실현수익률 데이터를 안정적으로 결합하려면 최소한 다음 컬럼을 보존한다.

| 컬럼 | 의미 |
| --- | --- |
| `loan_id` | 대출별 고유 결합키 |
| `split` | `train`, `validation`, `test` 구분 |
| `fold_id` | Train OOF fold 번호 |
| `actual_label` | `FP/CO` 또는 동일 의미의 `0/1` 라벨 |
| `term` | 36개월 또는 60개월 |
| `funded_amnt` | 금액가중치 계산 기준 |
| `R` | 계약만기 등가 연율화 실현수익률 |
| `rf` | 발행월·계약기간 대응 무위험수익률 |
| `XR` | 실현 초과수익률 |
| `pd_oof` | 모델팀의 확률보정 절차까지 적용된 Train OOF PD |
| `pd_pred` | 동일한 보정 절차가 적용된 Validation/Test 예측 PD |

조건부 모멘트 계산 후에는 다음 파생 컬럼을 추가한다.

| 컬럼 | 의미 |
| --- | --- |
| `pd_bin` | Train OOF PD로 정한 계약기간별 구간 |
| `m_fp`, `var_fp` | 해당 그룹의 완납 XR 평균·분산 |
| `m_co`, `var_co` | 해당 그룹의 부도 XR 평균·분산 |
| `expected_xr` | \(\widehat\mu_i\) |
| `expected_xr_var` | \(\widehat\sigma_i^2\) |
| `q_score` | \(\widehat\mu_i/\widehat\sigma_i\) |

### 9.3 실행 전·후 점검표

- [ ] Train/Validation/Test 분리가 모델 학습 전에 고정되어 있는가
- [ ] 무작위 60/20/20 분할과 Train 5-fold가 `(term, FP/CO)` 조합으로 층화되었는가
- [ ] 모든 Train 행에 `pd_oof`가 정확히 하나씩 존재하는가
- [ ] 각 OOF 예측에서 해당 행과 fold가 학습 데이터에서 제외되었는가
- [ ] 전처리와 확률보정도 fold 내부에서만 학습되었는가
- [ ] Validation/Test에서 PD 구간을 다시 `qcut`하지 않았는가
- [ ] Train에서 만든 수치 경계를 Validation/Test에 그대로 적용했는가
- [ ] `(term, PD bin, FP/CO)`별 건수와 유효표본크기를 확인했는가
- [ ] 완납 또는 부도 표본이 부족한 그룹의 병합·fallback 규칙을 사전에 정했는가
- [ ] PD calibration을 별도로 확인했는가
- [ ] 실현수익률 테이블과 결합 전후 행 수 및 고유 `loan_id` 수가 동일한가
- [ ] Validation에서 선택한 규칙을 Test 결과를 보고 수정하지 않았는가
- [ ] 반복 실행 전에 모형·보정·구간 처리·병합·임계값 탐색 절차가 동결되었는가
- [ ] seed 0~29의 최소 30회 전체 파이프라인을 완료했는가
- [ ] \(\Delta S_{\mathrm{XS}}\) 히스토그램·승률·중앙값·분위수와 승인율 분포를 저장했는가
