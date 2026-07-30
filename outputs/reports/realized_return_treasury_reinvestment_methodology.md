---
title: Lending Club 실현수익률 계산 확정 명세
tags:
  - lending-club
  - realized-return
  - treasury-reinvestment
status: finalized
updated: 2026-07-30
---

# Lending Club 실현수익률 계산 확정 명세

> 대출 한 건 전체를 하나의 상품으로 보고, 계약만기 전 현금흐름은 만기까지 재투자하고 계약만기 후 현금흐름은 만기 시점으로 할인한 뒤 계약기간 기준으로 연율화한다.

- 원금: `funded_amnt`
- 평가기간: 각 대출의 `term` 36개월 또는 60개월
- 포함: `Fully Paid`, `Charged Off`
- Recovery: 원칙적으로 `last_pymnt_d` 6개월 후 일시 회수 가정
- 정상 납입 없이 Recovery만 있는 `Charged Off`: 발행월 6개월 후 일시 회수 가정
- 월별 재투자수익률: FRED 월별 `GS1M`
- `GS1M` 월수익률 변환: $\left(1+\texttt{GS1M}/200\right)^{1/6}-1$
- 일반 서비스수수료: 원자료에서 직접 관측할 수 없어 기준 계산은 $f_{svc}=0$
- 미사용: `funded_amnt_inv`, `total_pymnt_inv`, `int_rate`
- 영문 명칭: **Contract-Maturity-Equivalent Annualized Realized Return**
- 국문 명칭: **계약만기 등가 연율화 실현수익률**
- 모든 실현 현금흐름을 계약만기 시점의 등가가치로 환산한 뒤 계약기간으로 연율화한 수익률이다.

### 기준 방법

월별 실제 납입내역이 없는 한계를 인정하되, 현금흐름의 시점 차이를 무시하지 않기 위해 **마지막 납입액 보존 + 이전 정규 수령액 월별 균등배분**을 기준 방법으로 사용한다.

- 마지막 납입액 `last_pymnt_amnt`는 관측된 금액과 월을 그대로 보존한다.
- `total_pymnt - recoveries - last_pymnt_amnt`만 발행 후 1개월부터 마지막 납입 전월까지 균등 배분한다.
- 계약만기 전에 수령한 각 월의 현금흐름은 해당 수령월부터 계약만기까지 국채로 재투자한다.
- 계약만기 후 수령한 현금흐름은 재투자하지 않고 같은 국채수익률 기준으로 계약만기 시점까지 역할인한다.
- Recovery는 순회수액을 `last_pymnt_d` 6개월 후에 수령한 것으로 고정한다.
- 정상 납입이 한 번도 없고 Recovery만 있는 `Charged Off`는 `last_pymnt_d`가 구조적으로 존재하지 않으므로, 별도 규칙으로 순Recovery를 발행월 6개월 후에 배치한다.
- 재투자·역할인에는 FRED의 월별 1개월 만기 미 국채 CMT 수익률 `GS1M`을 사용한다.

---

## 1. 입력과 계산 대상

| 용도 | 필수 컬럼 |
|---|---|
| 상태 | `loan_status` |
| 원금·기간 | `funded_amnt`, `term` |
| 발행·마지막 납입월 | `issue_d`, `last_pymnt_d` |
| 누적·월별 수령액 근사 | `total_pymnt`, `installment`, `last_pymnt_amnt` |
| 회수·추심비용 | `recoveries`, `collection_recovery_fee` |
| 계약만기 환산 | 필요한 달력월의 FRED 월별 `GS1M` |

| `loan_status` | 실현수익률 처리 |
|---|---|
| `Fully Paid` | 포함 |
| `Charged Off` | 포함 |
| `Default` | PD 부도 라벨에는 포함 가능, 현금흐름 미종료로 수익률 계산 제외 |
| `Current`, `Late`, `In Grace Period`, `Issued` | 결과 미확정으로 제외 |

`Does not meet the credit policy. Status:` 접두사는 제거한 뒤 상태를 판정한다. 포함 대상은 `Fully Paid` 899,745건과 `Charged Off` 217,826건, 총 1,117,571건이다.

---

## 2. 변수와 현금흐름

대출 $i$에 대해:

| 기호 | 정의 |
|---|---|
| $P_i$ | 전체 대출원금 |
| $T_i$ | 계약기간(개월) |
| $K_i$ | 발행월부터 마지막 납입월까지의 개월 수 |
| $K_i^{rec}$ | 발행월부터 recovery 가정 수령월까지의 개월 수 |
| $L_i$ | 관측된 마지막 납입액 |
| $CF_{i,t}^{regular,net}$ | 발행 후 $t$개월의 근사 순정규 현금유입 |

$$
P_i=\texttt{funded\_amnt}_i,\qquad
T_i=\texttt{term}_i\text{에서 숫자를 추출한 값}\in\{36,60\}
$$

`term` 원자료에는 선행 공백이 있으므로 문자열을 먼저 정리한 뒤 숫자를 추출한다.

`issue_d`와 `last_pymnt_d`의 연·월을 각각 $(Y_i^{issue},M_i^{issue})$, $(Y_i^{last},M_i^{last})$라고 하면:

$$
\widetilde K_i
=12(Y_i^{last}-Y_i^{issue})+(M_i^{last}-M_i^{issue}),
\qquad
K_i=\max(\widetilde K_i,0)
$$

Recovery는 마지막 납입월 6개월 후에 일시 회수된 것으로 고정한다. 계약만기 이후에 회수되더라도 회수월을 계약만기로 앞당기지 않고, 이후 정의하는 환산계수를 이용해 계약만기 시점으로 할인한다.

$$
K_i^{rec}=K_i+6
$$

다만 다음 조건을 모두 만족하는 `Charged Off`는 정상 납입 자체가 없는 Recovery-only 대출로 구분한다.

- `last_pymnt_d` 결측
- $C_i^{regular}=0$
- `recoveries > 0`
- `last_pymnt_amnt = 0`

이 경우 결측인 `last_pymnt_d`를 실제 납입월처럼 0으로 대체하지 않는다. 별도 현금흐름 규칙으로 정규 현금유입을 0으로 두고 Recovery 가정 수령월만 다음과 같이 정의한다.

$$
K_i^{rec}=6
$$

### Recovery 시점 6개월 근사의 성격

LendingClub의 [2016 Form 10-K](https://www.sec.gov/Archives/edgar/data/1409970/000140997017000255/a201610-k.htm)는 대출을 120일 연체 시 non-accrual로 분류하고, 늦어도 150일 연체 시 charge-off한다고 공시했다. 이 연체일수는 최초 미납금의 납부기일부터 세므로 월납 대출에서는 마지막 정상 납입 후 약 1개월 뒤 최초 미납이 발생하고, 그로부터 120~150일 뒤 charge-off되어 **마지막 납입 후 약 5~6개월**이 된다.

LendingClub의 [2009 Form 10-Q](https://www.sec.gov/Archives/edgar/data/1409970/000095012309058349/c91962e10vq.htm)는 charge-off 이후 받은 금액을 recovery로 설명하고, [2020 Form 10-K](https://www.sec.gov/Archives/edgar/data/1409970/000140997021000016/lc-20201231.htm)는 회수금과 charged-off 대출 매각대금을 일반 원리금 납입과 구분해 제시한다. 그러나 이 공시들은 recovery가 언제 발생하는지는 제시하지 않으며, **Recovery 수령시점 자체를 6개월로 정하는 근거는 아니다.**

원자료에도 실제 recovery 날짜가 없으므로, 본 분석은 예상 charge-off 시점을 대리변수로 삼아 recovery 전액을 **마지막 납입월 6개월 후에 일시 회수한 것으로 배치**한다. 정상 납입이 전혀 없는 Recovery-only 대출은 마지막 납입월이 존재하지 않으므로 발행월을 계산상 기준점으로 사용해 **발행월 6개월 후**에 배치한다.

두 규칙 모두 계산을 위한 편의적 기준 가정이다. 공시는 charge-off까지 걸릴 수 있는 기간의 배경을 제공할 뿐 실제 Recovery 수령월을 6개월로 확정하는 근거는 아니다. 실제 Recovery는 더 빠르거나 늦게 여러 차례 발생할 수 있으므로 수령시점은 별도 민감도 분석으로 점검한다.

원자료에서는 다음 항등식이 전 행에서 최대 1센트 오차로 성립한다.

$$
\texttt{total\_pymnt}
=\texttt{total\_rec\_prncp}
+\texttt{total\_rec\_int}
+\texttt{total\_rec\_late\_fee}
+\texttt{recoveries}
$$

따라서 recovery를 이중계산하지 않도록 현금유입을 분리한다.

$$
C_i^{regular}
=\texttt{total\_pymnt}_i-\texttt{recoveries}_i
$$

$$
C_i^{recovery}
=\texttt{recoveries}_i-\texttt{collection\_recovery\_fee}_i
$$

일반 서비스수수료율 $f_{svc}$를 적용하면:

$$
C_i^{regular,net}=(1-f_{svc})C_i^{regular}
$$

일반 서비스수수료를 직접 관측할 수 없으므로 기준 계산은 $f_{svc}=0$으로 확정한다. 추후 수수료율을 별도로 가정하는 분석은 기준 계산이 아니라 강건성 분석으로 구분한다.

### 월별 정규 현금흐름 근사

월별 실제 납입내역은 없으므로 `total_pymnt - recoveries`를 원자료에서 관측되는 정규 현금유입 총액으로 고정한다. 관측된 마지막 납입액 `last_pymnt_amnt`는 변형하지 않고 마지막 납입월에 그대로 배치하며, 나머지 정규 수령액만 그 이전 월들에 균등 배분한다. 이 원칙을 `Fully Paid`와 `Charged Off`에 동일하게 적용한다.

$$
L_i=\texttt{last\_pymnt\_amnt}_i
$$

기준 계산에서는 $0\le L_i\le C_i^{regular}$인 대출을 사용한다. $K_i\ge2$이면:

$$
\boxed{
CF_{i,t}^{regular,net}
=
\begin{cases}
\displaystyle
(1-f_{svc})\frac{C_i^{regular}-L_i}{K_i-1},
&1\le t<K_i,\\[10pt]
(1-f_{svc})L_i,
&t=K_i.
\end{cases}
}
$$

따라서 마지막 납입액과 정규 수령액 총액을 모두 보존한다.

$$
\sum_{t=1}^{K_i}CF_{i,t}^{regular,net}
=C_i^{regular,net}
$$

`installment`는 월별 금액을 강제로 결정하는 데 사용하지 않고, 이전 월에 균등 배분한 금액이 계약상 할부금과 얼마나 다른지 확인하는 진단 기준으로 사용한다.

$$
D_i=
\frac{(C_i^{regular}-L_i)/(K_i-1)}
{\texttt{installment}_i}
$$

$D_i$가 1에 가까우면 이전 월 균등배분액이 계약상 할부금과 유사하다는 뜻이다. 전체 원본의 $K_i\ge2$이면서 $0\le L_i\le C_i^{regular}$인 대출에서 $0.9\le D_i\le1.1$인 비율은 `Fully Paid` 83.739%, `Charged Off` 82.005%였다.

`Fully Paid`에서 $D_i>1.5$인 경우는 69,106건이며, 이 중 94.56%가 $K_i\le24$에 집중된다. 이는 월별 균등배분 방식의 오류라기보다 조기상환액 또는 소액 잔여지급 때문에 이전 월의 수령액이 계약상 할부금보다 커지는 현상으로 해석한다. 따라서 $D_i$는 제외 기준이 아니라 진단값으로만 사용한다.

$K_i=0$이면 정규 수령액 전부를 발행월 현금유입으로 처리한다. $K_i=1$이면 월 단위 자료로 같은 달의 여러 거래를 구분할 수 없으므로 정규 수령액 전부를 발행 후 1개월 시점의 현금유입으로 처리한다.

---

## 3. GS1M 기반 계약만기 환산계수 \(F\)

$m_{i,u}$를 대출 $i$의 발행월로부터 $u$개월 후의 달력월이라고 둔다.

$$
m_{i,u}=\texttt{issue\_d}_i+u\text{ months}
$$

### 재투자 국채 시계열과 월수익률 변환

재투자수익률은 FRED의 월별 `GS1M`, 즉 **1개월 만기 미 국채 Constant Maturity 수익률**을 사용한다.

- FRED 시리즈: [`GS1M`](https://fred.stlouisfed.org/series/GS1M)
- 원출처: Board of Governors of the Federal Reserve System, H.15 Selected Interest Rates
- 단위: 연율 %, 계절조정 없음
- 월 관측값: 해당 월 영업일 관측치의 평균
- 필요 달력 범위: 2007-07부터 2025-09까지

월 $m$의 `GS1M` 값(단위: %)을 $y_{1m,m}$이라고 둔다. 미 재무부의 CMT는 유효연이율이 아니라 반기복리 기준의 bond-equivalent yield이므로, 월 등가 재투자수익률은 다음과 같이 변환한다.

$$
\boxed{
r_f(m)
=
\left(1+\frac{y_{1m,m}}{200}\right)^{1/6}-1
}
$$

이 변환은 6개월 동안 같은 월수익률을 적용했을 때 반기수익률 $y_{1m,m}/200$과 같아지도록 한다.

$$
\left[1+r_f(m)\right]^6
=1+\frac{y_{1m,m}}{200}
$$

예를 들어 `GS1M = 4%`이면:

$$
r_f(m)
=(1+0.04/2)^{1/6}-1
\approx0.003306
=0.3306\%
$$

단순히 $4\%/12$로 나누지 않는다. 미 재무부는 CMT가 반기 이자지급 채권과 일관된 단순 연율 bond-equivalent yield이며 APY가 아니라고 설명한다. [미 재무부 CMT FAQ](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/interest-rates-frequently-asked-questions)

대출 현금흐름을 월 $m_{i,a}$에 수령하면 그 달에는 재투자수익률을 적용하지 않고, 다음 달 $m_{i,a+1}$부터 적용한다. 월 단위로 복원한 사후 실현수익률이므로 각 달의 실제 사후 `GS1M` 월평균을 사용한다.

발행 후 $a$개월에 수령한 현금흐름을 계약만기 $T_i$의 등가가치로 환산하는 계수는:

$$
\boxed{
F_i(a,T_i)=
\begin{cases}
\displaystyle\prod_{u=a+1}^{T_i}\left[1+r_f(m_{i,u})\right],&a<T_i,\\[10pt]
1,&a=T_i,\\[8pt]
\displaystyle\left\{\prod_{u=T_i+1}^{a}\left[1+r_f(m_{i,u})\right]\right\}^{-1},&a>T_i.
\end{cases}
}
$$

즉 계약만기 전에 받은 돈은 수령 다음 달부터 계약만기까지 `GS1M`으로 재투자하고, 계약만기 이후에 받은 돈은 재투자하는 것이 아니라 같은 `GS1M` 기준으로 계약만기까지 역할인한다. 월수익률이 0.3%로 일정한 12개월의 예시는:

$$
F_i(T_i-12,T_i)=(1+0.003)^{12}\approx1.0366
$$

$$
F_i(T_i+12,T_i)=(1+0.003)^{-12}\approx0.9647
$$

`GS1M` 결측월은 자동으로 전월값 대체·보간하지 않는다. 공식 FRED 원자료와 수집 과정을 점검한 뒤에도 값이 없으면 해당 대출을 기준 계산에서 제외하고 건수를 보고한다.

---

## 4. 대출별 실현수익률

월별 정규 수령액은 §2의 총액·마지막 납입액 보존 방식으로 근사하고, 순회수액은 §2에서 정한 $K_i^{rec}$에 일시 회수된 것으로 배치한다.

$$
W_i(T_i)
=\sum_{t=1}^{K_i}
CF_{i,t}^{regular,net}F_i(t,T_i)
+C_i^{recovery}F_i(K_i^{rec},T_i)
$$

$K_i=0$이면 합계항 대신 $C_i^{regular,net}F_i(0,T_i)$를 사용하고, $K_i=1$이면 $C_i^{regular,net}F_i(1,T_i)$를 사용한다.

$K_i\ge2$인 대출의 기준식 $(f_{svc}=0)$:

$$
\boxed{
\begin{aligned}
W_i(T_i)
=\;&
\frac{
\texttt{total\_pymnt}_i-\texttt{recoveries}_i
-\texttt{last\_pymnt\_amnt}_i
}{K_i-1}
\sum_{t=1}^{K_i-1}F_i(t,T_i)\\
&+
\texttt{last\_pymnt\_amnt}_iF_i(K_i,T_i)\\
&+
(\texttt{recoveries}_i-\texttt{collection\_recovery\_fee}_i)
F_i(K_i^{rec},T_i)
\end{aligned}
}
$$

계약기간 기준 연율화 수익률:

$$
\boxed{
R_i
=\left(
\frac{W_i(T_i)}{\texttt{funded\_amnt}_i}
\right)^{12/T_i}-1
}
$$

$K_i\ge2$인 대출을 상태별로 쓰면:

$$
R_i^{FP}
=
\left[
\frac{
\displaystyle
\frac{\texttt{total\_pymnt}_i-\texttt{last\_pymnt\_amnt}_i}{K_i-1}
\sum_{t=1}^{K_i-1}F_i(t,T_i)
+
\texttt{last\_pymnt\_amnt}_iF_i(K_i,T_i)
}
{\texttt{funded\_amnt}_i}
\right]^{12/T_i}-1
$$

$$
R_i^{CO}
=
\left[
\frac{
\displaystyle
\frac{
\texttt{total\_pymnt}_i-\texttt{recoveries}_i
-\texttt{last\_pymnt\_amnt}_i
}{K_i-1}
\sum_{t=1}^{K_i-1}F_i(t,T_i)
+
\texttt{last\_pymnt\_amnt}_iF_i(K_i,T_i)
+
(\texttt{recoveries}_i-\texttt{collection\_recovery\_fee}_i)F_i(K_i^{rec},T_i)
}{
\texttt{funded\_amnt}_i
}
\right]^{12/T_i}-1
$$

`recoveries = 0`은 결측이 아니라 무회수 관측값이다.

`last_pymnt_d`가 결측이더라도 `total_pymnt = 0`이고 `recoveries = 0`이면 배치할 현금유입 자체가 없으므로 다음과 같이 계산에 포함한다.

$$
W_i(T_i)=0,\qquad R_i=-1
$$

`last_pymnt_d`가 결측이지만 정상 납입 없이 Recovery만 있는 경우에는 해당 Recovery의 순회수액을 발행 후 6개월에 배치한다.

$$
W_i(T_i)
=
C_i^{recovery}F_i(K_i^{rec},T_i)
$$

$$
R_i
=
\left[
\frac{C_i^{recovery}F_i(K_i^{rec},T_i)}{P_i}
\right]^{12/T_i}-1
$$

현재 최종 대상 544건에는 $K_i^{rec}=6$이므로 위 식의 $F_i(K_i^{rec},T_i)$는 $F_i(6,T_i)$와 같다.

### 최종 계산 순서

1. `loan_status`를 정규화하고 `Fully Paid`, `Charged Off`만 선택한다.
2. `issue_d`, `last_pymnt_d`, `term`으로 $K_i$, $T_i$를 계산한다. 일반 대출은 $K_i^{rec}=K_i+6$, Recovery-only 대출은 $K_i^{rec}=6$으로 정한다. Recovery-only의 결측 $K_i$는 임의 대체하지 않는다.
3. `total_pymnt - recoveries`를 정규 수령액, `recoveries - collection_recovery_fee`를 순Recovery로 분리한다.
4. $K_i\ge2$이면 마지막 지급액은 $K_i$월에 그대로 두고 나머지 정규 수령액을 $1,\ldots,K_i-1$월에 균등배분한다. $K_i\le1$이면 §2의 단일 월 규칙을 적용한다. Recovery-only 대출은 정규 현금흐름을 배치하지 않는다.
5. 각 달의 FRED `GS1M`을 $r_f(m)=\left(1+\texttt{GS1M}(m)/200\right)^{1/6}-1$로 변환한다.
6. 각 정규 현금흐름과 순Recovery를 $F_i(a,T_i)$로 계약만기 시점의 등가가치로 환산한다. Recovery가 만기 전이면 만기까지 재투자하고, 만기 후이면 실제 가정 회수월부터 만기까지 역할인해 $W_i(T_i)$를 계산한다.
7. $R_i=\left[W_i(T_i)/P_i\right]^{12/T_i}-1$로 계약기간 기준 연율화한다.
8. §6의 예외 규칙에 해당하는 행은 제외 또는 별도 규칙으로 포함하고 상태·발행연도별 건수를 함께 보고한다.

---

## 5. 전체 원본 적용성 점검

점검 파일은 `data/lending_club_2020_train.csv`이며, 2026-07-29에 전체 1,755,295행을 전수 검사했다. 상태 정규화 후 계산 대상은 `Fully Paid` 899,745건과 `Charged Off` 217,826건, 총 1,117,571건이다.

| 상태 | 계산 대상 | 기준 방법 적용 가능 | 적용률 |
|---|---:|---:|---:|
| `Fully Paid` | 899,745 | 897,540 | 99.755% |
| `Charged Off` | 217,826 | 217,060 | 99.648% |
| **합계** | **1,117,571** | **1,114,600** | **99.734%** |

`Charged Off` 적용 가능 건수에는 `last_pymnt_d`가 없지만 현금유입이 전혀 없어 $R_i=-1$로 직접 계산할 수 있는 832건과, 정상 납입 없이 Recovery만 있어 발행 후 6개월 배치 규칙을 적용하는 1,210건을 포함한다.

현재 관측기준월 2020년 10월의 `만기 + 6개월` 성숙도 조건과 상태 조건을 통과한 723,563건에 같은 규칙을 적용하면 다음과 같다.

| 구분 | 건수 |
|---|---:|
| 기존 현금흐름 계산 가능 | 721,809 |
| 추가 포함 Recovery-only | 544 |
| **수정 후 계산 가능** | **722,353** |
| 기준 계산 제외 | 1,210 |

수정 후 계산 가능 비율은 99.833%다. 2026-07-30에 노트북을 전체 재실행하고 결과 CSV를 독립 점검해 722,353건을 최종 확인했다.

현재 성숙 모집단에서 남는 제외 1,210건은 $K_i\ge2$인데 $L_i>C_i^{regular}$인 1,209건과 `last_pymnt_amnt < 0`인 1건이다. 1,209건의 성격은 다음과 같이 구분한다.

- `Fully Paid` 930건과 `Charged Off` 35건: `last_pymnt_amnt > total_pymnt`까지 성립하므로 누적 총액과 마지막 지급액의 데이터 불일치로 본다.
- `Charged Off` 244건: $L_i>C_i^{regular}$이지만 $L_i\le total_pymnt_i$이고 Recovery가 존재한다. 원자료 전체 누적액 안에서는 모순이 없으나 마지막 지급액을 정규상환과 Recovery로 분해할 수 없으므로 기준 계산에서 제외한다.

따라서 1,209건 전체를 동일한 원자료 오류로 단정하지 않고, 965건은 데이터 불일치, 244건은 현금흐름 분해 불가 사례로 기록한다.

### 전수 점검 결과

- `id` 중복은 없으며, 계산 대상의 `funded_amnt`, `term`, `installment`, `issue_d`, `total_pymnt`, `recoveries`, `collection_recovery_fee`, `last_pymnt_amnt`는 결측이 없다.
- 현금유입 항등식은 계산 대상 전체에서 최대 1센트 오차로 성립한다.
- `collection_recovery_fee > recoveries`인 행은 없다.
- $K_i<0$인 행은 없다.
- $K_i\ge2$인데 $L_i>C_i^{regular}$인 행은 `Fully Paid` 2,204건, `Charged Off` 761건이다.
- `last_pymnt_amnt < 0`인 행은 `Fully Paid` 1건, `Charged Off` 4건이다.
- `Charged Off`의 `last_pymnt_d` 결측은 2,043건이다. 이 중 832건은 현금유입이 전혀 없어 포함하고, 1,210건은 정상 납입 없이 Recovery만 있어 발행 후 6개월에 Recovery를 배치한다. 정규 수령액이 있는 나머지 1건만 현금흐름 시점 미확정으로 제외한다. 최종 성숙 모집단에 포함되는 Recovery-only는 544건이다.
- 최종 대상 722,353건에서 순Recovery가 양수인 행은 93,271건이며, 이 중 가정 회수월이 계약만기 후인 경우는 5,760건(6.176%)이다. 이 중 계약만기보다 6개월을 초과해 늦는 경우는 450건이며 최대 차이는 32개월이다.
- 마지막 정규 지급월 자체가 계약만기 후인 경우도 `Fully Paid` 46,260건, `Charged Off` 624건이므로 환산계수의 만기 후 역할인 분기가 필요하다.
- 계약만기까지 재투자·역할인하려면 `GS1M` 달력 범위가 최소 2007-07부터 2025-09까지 필요하다.

재현 가능한 전수 점검 코드는 `notebooks/realized_return_full_data_validation.ipynb`에 기록한다.

---

## 6. 예외·한계·강건성 분석

### 예외 처리

- `funded_amnt` 결측·0 이하: 0건
- `total_pymnt`, `recoveries`, `collection_recovery_fee` 결측: 0건
- `Charged Off`의 `last_pymnt_d` 결측 2,043건 중 `total_pymnt = recoveries = 0`인 832건: $W_i(T_i)=0$, $R_i=-1$로 포함
- 정상 납입 없이 Recovery만 있는 1,210건: 정규 현금흐름은 0, 순Recovery를 발행 후 6개월에 배치하여 포함. 최종 성숙 모집단에는 544건이 포함
- 나머지 `last_pymnt_d` 결측 1건: 정규 수령액의 시점을 정할 수 없으므로 기준 계산에서 제외
- $K_i\ge2$인데 `installment` 또는 `last_pymnt_amnt`가 결측인 경우: 임의 대체 없이 제외하고 상태·발행연도별 건수 보고
- $K_i\ge2$인데 $L_i>C_i^{regular}$인 2,965건: 관측 총액과 마지막 납입액이 충돌하므로 기준 계산에서 제외하고 상태·발행연도별 건수 보고
- `last_pymnt_amnt < 0`인 5건: 지급액 정의와 맞지 않으므로 기준 계산에서 제외하고 별도 점검
- 기타 필수값 결측: 임의 대체 없이 제외하고 상태·발행연도별 건수 보고
- 국채 월수익률 결측: 자동 전월대체하지 않고 원자료 확인 후 처리

### 한계

- 월별 납입내역이 없어 마지막 납입액을 제외한 나머지 정규 수령액을 이전 월들에 균등 배분한다. `installment`는 배분액의 타당성 진단에만 사용하며, 이는 실제 월별 납입액의 복원이 아니다.
- Recovery 날짜가 없어 원칙적으로 마지막 납입월 6개월 후 일시 회수로 고정한다. 정상 납입이 없는 Recovery-only 대출은 발행월 6개월 후로 고정한다.
- 계약만기 이후 Recovery는 계약만기로 앞당기지 않고 국채수익률로 가정 회수월부터 계약만기 시점까지 할인한다.
- 마지막 지급월 자체가 계약만기 후일 수 있으므로 Recovery의 가정 회수월과 계약만기의 차이는 6개월을 초과할 수 있다.
- `recoveries`는 관측기준월까지의 누적액이므로 최근 `Charged Off`에는 데이터 종료 이후 발생할 추가 recovery가 포함되지 않는다. `만기+6개월` 상태 성숙도 버퍼가 lifetime recovery의 완결성을 보장하지는 않는다.
- Debt settlement의 실제 지급 일정은 알 수 없다.

### 추후 강건성 분석

다음 항목은 기준 실현수익률의 계산 규칙을 변경하지 않는 별도 민감도 분석이다.

1. 일반 서비스수수료 $f_{svc}>0$ 가정
2. Recovery 수령시점을 `last_pymnt_d` 이후 3개월·9개월 등으로 변경
3. 정상 납입 없는 Recovery-only 대출의 Recovery를 발행 후 9개월·12개월 등에 배치하거나 제외하는 시나리오
4. $L_i>C_i^{regular}$인 행에서 $L_i$를 $C_i^{regular}$로 제한하는 대체 시나리오
5. 월별 균등배분 대신 모든 정규 현금흐름을 마지막 지급월에 배치하는 보수적 하한 비교
