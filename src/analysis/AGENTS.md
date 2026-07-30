# 분석 규칙

## 입력/출력
- 입력: `data/processed/` 전처리 완료 데이터
- 출력: `outputs/reports/` (모델·통계 결과), 필요 시 중간 산출물

## 규칙
- **모형은 XGBoost 하나다** (`decision_log.md` #17 ①, 2026-07-29 회의 확정). 로지스틱 회귀는 폐기됐다.
- **대조군은 전부 승인(approve-all) 전략**이다 (#17 ②). 성과 보고의 **헤드라인은
  Δ Sharpe = (모형 전략) − (approve-all)** 이다 — 재투자 가정이 양쪽을 밀어올리므로 절대값은
  가정 효과를 포함한다(#18).
  - ⚠️ **그러나 절대 Sharpe도 반드시 병기한다.** "절대 Sharpe를 보고하지 않는다"가 아니라
    **"절대 Sharpe를 모형의 기여분으로 해석하지 않는다"** 가 정확한 규칙이다. 수업 기준선이
    절대 수준이기 때문이다(`README.md`: "샤프비율은 0.2~0.4 정도 나오면 옳게 한 것").
    실측 **모형 0.279 · approve-all 0.220**으로 그 범위에 들어온다(`sharpe_threshold_kgj.md`).
  - Δ를 헤드라인으로 쓰는 것이 **최적화 목표를 바꾸지는 않는다** — approve-all Sharpe는 `τ`에
    대해 상수이므로 `argmax_τ [Sharpe(τ) − 상수] = argmax_τ Sharpe(τ)` 다.
  - ✅ **#18의 "가정이 양쪽을 똑같이 밀어올린다"는 방향은 맞고 정확히는 틀리다**(실측):
    재투자 가정을 바꿀 때 절대값은 20~30%, Δ는 **12~18%** 움직인다. Δ가 더 안정적이지만 상쇄가
    완전하지 않다. **기준 간 순위는 두 가정에서 동일**하고 **`τ*`는 크게 움직인다**(0.226 → 0.158).
- 승인/거절 threshold는 **오직 Sharpe Ratio 극대화**로 결정한다. accuracy, AUC 등 일반 분류 지표로 결정하지 않는다.
  - 단, **탐색·검증 목적의 상대 비교**(전처리 방식 A vs B 중 무엇이 나은지)에는 AUC를 써도 된다. 금지되는 것은 AUC로 **승인/거절 기준을 정하는 것**이다. 기존 비교 스크립트 4종이 이 예외에 해당한다.
- **표본은 만기 + 버퍼 6개월 필터를 적용한 723,563건**이다 (#16). `issue_d + term ≤ 2020-04`.
- Sharpe Ratio 정의: `(포트폴리오 수익률 평균 − 무위험수익률) / 포트폴리오 수익률의 표본표준편차(ddof=1)`.
  - **재투자 가정(확정, #18)**: 매달 받는 상환액을 **잔존기간에 맞춘 국채**에 재투자한다.
    `R = (W/P)^(12/T) − 1`, `W = Σ CFₘ·F(m,T)`. `m`월 수령액은 잔존기간 `T−m` 만기 금리로 굴린다
    (대출 만기 금리를 일괄 적용하면 짧은 잔존기간에 긴 만기 금리를 주게 되어 무위험이 아니다).
    **정상상환·부도 양쪽에 동일 적용**하므로 정상상환에 `int_rate`를 그대로 쓰지 않는다.
  - **실현수익률 구조 — A′ 확정** (#20, 2026-07-30):

    ```
    E[XR_i] = (1 − p̂_i) · XR_계약,i  +  p̂_i · r̄_부도,d(i)
                      ↑ 건별 계산          ↑ 그룹 평균 (PD 분위 × term)
    ```

    - **부도분은 PD 분위별 그룹 평균**을 쓴다. **정상상환분은 그룹 평균을 쓰지 않고**
      `int_rate`·`term`·`installment`·발행시점 국채곡선으로 **건별 계약 현금흐름을 직접 계산**한다.
    - **2단계 hurdle 회귀는 기각됐다 — 부도 서브셋 회귀 모델을 만들지 않는다.** 모형은 PD 분류기
      하나뿐이다(#17 ①과 정합).
    - PD 분위 경계는 **Train OOF PD**로 만들고 Validation·Test에 **그대로 적용**한다. 재분위 금지 —
      재분위하면 threshold가 "PD 얼마 이하"가 아니라 "그 표본의 상위 몇 %"가 되어 이전할 수 없다.
    - **분산은 총분산의 법칙으로 계산한다 — 교차항을 빠뜨리지 않는다.**
      ```
      Var[XR_i] = (1−p)·var_정상 + p·var_부도 + p(1−p)·(mu_정상 − mu_부도)²
      ```
      마지막 항이 지배적이다(예시 `p=0.10`, `mu_정상=+5%`/sd 3%, `mu_부도=−40%`/sd 20% → 총분산의 **79%**).
      빠뜨리면 sd가 15.2% → 6.9%로 축소되고 `p(1−p)`에 비례해 편향이 걸려 **랭킹 순서가 바뀐다.**
    - `var_부도`는 표본이 작은 칸에서 불안정하다(상대오차 ≈ `√(2/(n−1))` — n=300에서 8.2%,
      n=50에서 20.2%). **분산에는 평균보다 엄한 최소 표본수를 요구하고 pooled 분산으로 축소추정한다.**
      분산이 과소추정된 칸이 체계적으로 승인되는 선택편향을 막기 위한 것이다.
  - ⚠️ **계산 세부 3건은 아직 미확정** (B팀, #20): **조기상환 보정 방식**(1순위 — 계약 현금흐름만 쓰면
    `R` 과대추정), 국채 금리 기준(ⓒ발행시점 고정 권고), 서비스수수료(~1%).
    - 따라서 **부도 손실값·금리 관례를 코드에 상수로 박지 않는다.** 실현수익률 계산은 정의를
      **주입받는 형태**(파라미터/전략 함수)로 짜서, 안을 갈아 끼우며 Sharpe를 비교할 수 있게 한다.
    - **잠정값(ⓒ·수수료 0%)으로 진행하고 산출물에 `잠정(provisional)` 표기를 남긴다.** 확정 시
      **칸별 통계표와 threshold만 재계산**하면 되고 모형 재학습은 불필요하다(#19·#20).
    - ⚠️ **재투자 가정 스위치(국채 / 0%)는 칸별 통계표까지 다시 만든다** — `mu`·`var`가 `XR`에서
      산출되므로 #18이 말한 "계산 스위치 하나"로 끝나지 않는다. 파일명·컬럼에 가정을 남긴다.
    - **분류 모델은 이 세부를 기다리지 않고 만들 수 있다** — 타깃이 이진 `loan_status`라 손실 정의가
      학습에 개입하지 않는다.
  - ✅ **칸별 통계표 설계는 실측으로 검증됐다** (`oof_diagnostics_kgj.md`, 2026-07-30).
    - `ρ(pd_oof, q_score) = −0.797` · `ρ(pd_oof, E[XR]) = −0.445` → **PD 랭킹과 실질적으로 다르다.**
      `|ρ|`≈0.99였다면 칸별 통계표를 버리고 단순 PD threshold로 돌아가야 했으나 그 경우가 아니다.
    - **`E[XR]`이 PD에 대해 비단조다** — 정점이 36m는 4분위, 60m는 2분위다. 최우량 분위는 금리도
      낮아 기대 초과수익이 오히려 낮다. **"PD가 낮을수록 좋다"는 직관이 이 문제에서 틀린다.**
    - 칸별 `int_rate` sd 중앙값 **2.88%p**(범위 최대 25%p) → **A′의 건별 계산이 실효 있다.**
      순수 A였다면 이 산포가 전부 뭉개졌다.
    - Validation 분위 인원 이탈 **최대 0.52%p**(±2%p 이내) → **Train 경계를 그대로 이전할 수 있고
      fold는 5로 충분하다.**
    - ⚠️ 부수 발견: **`mu_부도`는 PD 분위와 거의 무관하다**(폭 36m 1.7%p / 60m 0.8%p). 반면
      term 간 차이는 **6.3%p**다. 구조 B(2단계 회귀) 기각이 옳았음을 뒷받침한다 — 부도 손실은
      PD가 담은 정보로 설명되지 않는다. 그룹 축에서 실제로 일하는 것은 `term`이다.
  - ⚠️ **확률보정(isotonic) — PD의 두 역할을 나눈다** (진단 C-4, 2026-07-30 실측):
    | 쓰임 | 어느 PD |
    | --- | --- |
    | PD 분위 **경계·배정**, 승인선 **점수**로서의 `pd` | **보정 전** |
    | `E[XR]`·`Var[XR]`의 `p̂` (**확률 값**) | **보정 후** |

    `E[XR] = (1−p̂)·XR_정상 + p̂·mu_부도`가 `p̂`를 확률 값으로 쓰므로 보정 오차에
    `(mu_정상−mu_부도)`≈20%p가 곱해져 `E[XR]`에 직접 들어간다. **AUC로는 안 잡힌다.**
    보정은 Validation ECE를 **0.710 → 0.265%p**, 최악 구간(MCE)을 **2.900 → 0.675%p**로 줄이고
    AUC는 −0.00004만 움직인다. Train OOF PD가 곧 학습 데이터라 **추가 학습 0회**다.
    - ⚠️ **"isotonic은 단조변환이라 분위 경계가 바뀌지 않는다"는 틀렸다.** 계단함수라 고유값이
      42.8만 → 125개로 뭉치고, 보정된 PD로 자르면 **칸 인원이 최대 3.13%p 기운다**(보정 전은
      0.001%p). 그래서 위 역할 분리가 필요하다. `decision_log.md` 「부수 결정」 정정 대상.
    - 구현: `model.py`의 `fit_calibrator()`·`apply_calibrator()`·`calibration_metrics()`·
      `calibration_noise_floor()`. ECE 판정은 **바닥값**(완벽 보정 시 나오는 ECE, 여기서는
      0.126%p)과 대조해서 한다 — 절대 기준 0.5%p는 표본이 작으면 바닥값과 구분되지 않는다.
  - ⏳ **승인선 랭킹 기준 — 근거 확보, 팀 확정 대기** (`pd` 단독 / `E[XR]` / `q_score`).
    세 기준은 **같은 중간 테이블에서 정렬만 바꾸면 나오므로 추가 학습 없이 비교**할 수 있다.
    **Validation에서만 비교해 승자를 사전 확정한 뒤 Test 1회** — 세 기준을 Test에서 비교하면
    아래 "Test set으로 모형을 재조정하지 않는다" 규칙 위반이다.
    - 실측(`sharpe_threshold_kgj.md`): **`q_score` 0.27865 > `pd` 0.27385 > `E[XR]` 0.26907**
      (Δ +0.0587 / +0.0539 / +0.0491, 승인율 60.4% / 68.0% / 73.6%). **재투자 가정 2종에서
      순위가 동일**하다. `E[XR]`이 지는 이유는 분산을 안 봐서 sd가 7.75%까지 올라가기 때문이다.
    - ⚠️ `var_정상 = 0`인 잠정 구현 기준이다. 조기상환 보정이 들어가면 `q_score` 분모가 커지므로
      **이 우열은 다시 확인해야 한다**(#20 B팀 1순위).
    - 어느 기준이든 threshold는 **점수의 이론값이 아니라 Validation 실현 XR로 계산한 실제 Sharpe**
      그리드서치로 정한다. 개별 대출 `q_score` 최대화는 포트폴리오 Sharpe 최대화와 같은 문제가 아니다.
    - threshold 곡선에는 **Sharpe·Δ Sharpe와 함께 승인율을 반드시 병기**한다. 위 Sharpe 정의는
      횡단면 표준편차라 분산 감소 효과가 없어, Sharpe만 보고 조이면 승인율이 비현실적으로 낮아질 수 있다.
    - ⚠️ "부도 시 0"이라는 옛 표기를 만나면 **회수액 0(= 수익률 -100%)** 로 읽는다. 수익률 0%(원금 전액 회수)가 아니다 — 이 혼동이 문서 4곳에 퍼져 있었고 2026-07-29에 정정했다(#4 정정 기록).
  - 거절한 대출의 실현수익률: 무위험수익률로 대체.
- 데이터 분할: Train 60% / Validation 20% / Test 20% (`train_test_split`을 두 번 적용 — 전체→80/20으로 Test 분리 후, 남은 80%를 75/25로 Train/Validation 분리).
  - 비율·seed는 **`config/config.yaml`이 단일 출처**다. 값을 스크립트에 직접 써 넣지 말고 `src/utils/config.py`의 `load_config()`로 읽는다.
  - 랜덤 분할을 유지한다 — 시간순 분할은 `decision_log.md` #13 ④에서 실측으로 기각됐다(Train의 T2 관측률이 0%가 됨). 필요하면 보조 진단으로만 돌린다.
- Train으로 부도확률 예측 모형을 학습하고, Validation으로 threshold를 확정한다.
- **Test set으로 모형을 재조정하지 않는다.** Train으로 학습한 모형과 Validation으로 확정한 threshold를 그대로 적용해 검증만 한다.
- Threshold 확정 후에는 Train 전체(60%)로 모형을 재학습한 뒤 Test에 적용한다.
- **사전/사후(pre/post-approval) 변수 구분을 피처 선택에 적용한다** (`decision_log.md` #1 확정). 투자자 관점이므로 `grade`·`sub_grade`·`int_rate`·`installment`·`funded_amnt`·`funded_amnt_inv`·`issue_d`·`initial_list_status`는 **사전 변수**다. 라벨 원본은 `data/processed/variable_dictionary_byGJ.xlsx`의 `is_pre_approval`.
  - **LC 조건변수(`grade`·`sub_grade`·`int_rate`)는 피처로 투입한다** (#17 ③, 2026-07-29 회의 확정).
    ⚠️ 기존 문서의 0.68대는 조건변수를 뺀 Lean 스펙 값이므로 섞어 인용하지 않는다.
    ⚠️ **"조건변수 포함 시 0.71대"는 #16 만기필터 확정 이전 값이다** — 확정 표본에서 재측정하면
    **0.70대**다(`oof_diagnostics_kgj.md`, 2026-07-30 실측 0.7024). 같은 피처·모델로 필터만 빼면
    0.7283이 나와 **−2.1%p가 오롯이 필터 효과**임이 확인됐다(만기 미도래분의 조기부도 과대표집).
    Lean 0.68은 이미 확정 표본 기준이므로(#13 ⑤), **확정 표본에서 조건변수 효과는 0.68 → 0.70**이다.
    `decision_log.md` 정정은 팀 확인 후 한다.
- **무위험수익률(Rf) — 참고논문 방식 확정** (#18): 거절한 대출의 자본은 **발행시점(`issue_d`)에 대출 만기와
  만기를 맞춘 미국채**(3년물/5년물)에 투자했다고 가정한다. 데이터는
  `data/processed/us_treasury_GS3_GS5_monthly_2007-06_to_2020-09.csv`.
  - 고정 상수가 아니므로 `config/config.yaml`의 `risk_free_rate.value`는 **`null`을 유지**하고,
    `issue_d` × `term` 매칭 로직으로 처리한다. `value`를 읽어 쓰는 코드를 작성하지 않는다.
- **Threshold 확정 절차**: 랜덤 6:2:2 분할을 **K=50회 반복**하고 `τ*`의 분포(평균·표준편차)를 본다 (#18).
- 안정성 검증 시 `random_state` 0~29로 30회 반복해 모형 Sharpe가 "전부 승인" 베이스라인을 이기는 비율을 확인한다.
- 모형 구성은 **통합 모형 + `term` 피처 투입**이다 — 36m/60m 분리 모형은 `decision_log.md` #13 ②에서 실측 기각(60m에서 AUC −0.00564, 5/5 seed). 단 성능표는 term별로 나눠 보고한다.
  - ⚠️ 이는 **PD 모형**을 나누지 않는다는 뜻이며, IRR·Sharpe 계산식에는 term별 현금흐름 기간이 반드시 들어간다(#4).
  - XGBoost 단일화가 확정됐으므로(#17 ①) 이 결론은 **조건부가 아니라 확정**이다.
- **결측 처리는 한 줄로 확정됐다** (#13 ③·#17): **모든 결측을 NaN 그대로 투입한다. 대체하지 않고 더미도 만들지 않는다.**
  표준화·로그변환·구간화·캡핑도 하지 않는다 — 트리는 값의 순서만 쓴다.

## 이 폴더 스크립트의 성격 구분
`src/analysis/`에는 성격이 다른 두 종류가 섞여 있다. 혼동하면 규칙을 잘못 적용한다.

| 종류 | 파일 | 성격 |
| --- | --- | --- |
| **탐색·검증** | `preprocessing_validation.py`, `missing_scheme_comparison.py`, `term_split_comparison.py`, `t2_contribution_reassessment.py`, `macro_indicator_screening.py`, `auc_sample_filter_comparison.py` | 문서의 표를 재생성하는 재현 스크립트. 자체 2분할·자체 seed 루프를 쓰며 `config.yaml`을 따르지 않는다(의도된 차이). AUC는 상대 비교용. |
| **실현수익률 검증** | `realized_return_spec_check.py`, `realized_return_sensitivity.py` | 모형 학습이 없다(pandas/numpy만). `config.yaml`을 따르지 않으며 AUC도 쓰지 않는다. 위 "규칙" 절의 **재투자 가정·연율화 식**이 적용되는 대상. |
| **본 파이프라인** | `model.py` (동작) · `realized_return.py` (동작, **잠정 가정**) · `oof_diagnostics.py` (동작) · `sharpe_optimizer.py` (동작, **잠정 가정**) | 위 "규칙" 절이 그대로 적용되는 대상. `config.yaml`을 반드시 경유한다. |

**본 파이프라인 모듈이 무엇을 담당하는가** (2026-07-30 구현, 커밋 `c18a40b`)

| 모듈 | 담당 | 주요 함수 |
| --- | --- | --- |
| `model.py` | XGBoost PD 모형 · K-fold OOF · **isotonic 보정** · PD 분위 경계 | `train_model()`, `compute_oof()`, `fit_calibrator()`, `apply_calibrator()`, `calibration_metrics()`, `calibration_table()`, `calibration_noise_floor()`, `make_quantile_edges()`, `assign_pd_quantile()`, `quantile_edges_by_term()`, `assign_quantile_by_term()` |
| `realized_return.py` | 구조 A′ 실현수익률 · `E[XR]` · `Var[XR]` · `q_score` | `contract_return()`(정상상환 건별), `realized_return_defaulted()`, `build_excess_returns()`(`xr_normal`·`xr_default`·**`xr_realized`**), `default_cell_stats()`, `expected_excess_return()`, `variance_excess_return()`, `q_score()` |
| `oof_diagnostics.py` | 설계 선택 실측 검증 C-1/C-2/C-3/**C-4** | `main()` — 산출물 5종은 루트 `AGENTS.md` 코드 색인 ③ 참고 |
| `sharpe_optimizer.py` | threshold 탐색 · 랭킹 기준 비교 | `sharpe_ratio()`, `sharpe_curve()`(누적합으로 **모든 컷** 평가), `find_optimal_threshold()`, `approve_all_sharpe()`, `compare_ranking_criteria()`, `repeat_threshold_search()`(K=50, **미실행**), `build_validation_scores()`, `scores_for_assumptions()` |

> `sharpe_optimizer.py`가 지키는 것들 — 고칠 때 깨뜨리지 말 것:
> - **Sharpe는 기대값이 아니라 Validation 실현 `XR`(`xr_realized`)로** 계산한다. 점수는 승인선을
>   그을 때만 쓴다 — 기대값으로 재면 모형이 자기 예측으로 자기를 채점한다.
> - **거절 건은 표본에서 빼지 않고 `XR = 0`으로 남긴다.** 실측으로 빼면 Sharpe가 **+0.24** 뛴다.
> - **격자를 찍지 않는다.** 정렬 후 `k=1..n`을 누적합으로 전부 평가한다(brute force와 5e-16 일치).
> - 세 기준을 **같은 유효 표본**에서 비교한다(`q_score` NaN 51건을 세 기준 모두에서 제외).
> - 칸별 통계표·분위 경계는 **Train에서만** 만들어 Validation에 적용한다(재분위 금지).

> `realized_return.py`의 가정은 전부 `ReturnAssumptions`(frozen dataclass)로 **주입받는다** —
> `reinvest`("treasury"/"cash") · `treasury_basis` · `servicing_fee_annual` · `prepayment_adjustment`.
> 손실값을 상수로 박지 않는다는 위 규칙의 구현이며, 산출물 파일명에 `label()`이 붙는다.

> 인용 주의: 비교 스크립트가 내는 AUC(0.68대)는 LC 조건변수를 뺀 Lean 스펙 값이다.
> **최종 모형 성능으로 인용하면 안 된다** (`decision_log.md` #13 ⑤).
> 회의에서 조건변수 투입이 확정됐으므로(#17 ③) 최종 모형 기준은 그보다 높다 —
> **확정 표본 실측 0.70대**다(`oof_diagnostics_kgj.md`). 문서에 남아 있는 "0.71대"는
> #16 만기필터 이전 값이므로 인용하지 않는다.
>
> `macro_indicator_screening.py`는 거시지표 미사용 확정(#15)으로 **과거 산출물 재현 전용**이 됐다.
>
> `realized_return_sensitivity.py`의 `realized_return()`은 **실측 현금흐름 기준으로 #18 재투자 가정을
> 검증한** 구현이다(`allocation` × `use_rates` 스위치, 민감도 표 재현 전용).
> 구조 A′의 **계약 `R`은 `realized_return.py`의 `contract_return()`에 구현돼 있다**(연금 종가 공식).
> 남은 미결은 그 계약 `R`과 **실현 `R`의 차이인 조기상환 보정항**이며, 현재 `prepayment_adjustment = 0`
> 으로 고정돼 있어 정상상환분 `R`이 과대추정되고 `var_정상 = 0`이 된다 — B팀 1순위(#20).
> 인계 내용은 `outputs/reports/handoff_teamb_realized_return.md`.

## 참고
- **확정 사항의 원본은 `outputs/reports/decision_log.md`다.** 이 문서와 어긋나면 decision_log가 우선이고, 이 문서를 고친다.
- 배경/근거: `docs/references/legacy-분석에이전트-CLAUDE.md` (⚠️ 과거 폴더 구조 기준이라 문서 내 경로 상당수가 현존하지 않는다 — 방법론만 참고할 것)
- 파이프라인 **코드 구조** 템플릿: `notebooks/LendingClub_실습_v2.ipynb`
  - ⚠️ **교육용 baseline이다.** rf 고정 5% · 정상상환=`int_rate` · 부도=0을 쓰는데 **셋 다 현행 방법론이 아니다**(#18).
    셀 구성·함수 분리 같은 **코드 구조만** 참고하고, 수익률·rf 정의는 절대 가져오지 않는다.
- 실행 환경: 저장소 루트 `requirements.txt` (scikit-learn·xgboost 필요 — 시스템 python에는 없다)
