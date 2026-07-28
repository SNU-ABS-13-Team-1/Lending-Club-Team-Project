# Git / GitHub 협업 컨벤션

## 1. 브랜치 관리 전략
- `main`: 배포/실행 가능한 안정 브랜치. **직접 push 금지 — PR을 통해서만 반영한다.**
- `feature/#이슈번호-기능명-이니셜`: 새 기능/전처리/모델 개발 (예: `feature/#12-data-loader-jm`)
- `fix/#이슈번호-버그명-이니셜`: 버그 수정 (예: `fix/#45-sharpe-bug-jm`)
- `docs/#이슈번호-문서명-이니셜`: 문서 작성/수정
- `refactor/#이슈번호-리팩토링명-이니셜`: 코드 구조 개선 (기능 변경 없음)
- `experiment/#이슈번호-실험명-이니셜`: 같은 이슈에 대해 서로 다른 접근법(모델, threshold, 피처셋 등)을
  비교할 때 쓰는 후보 브랜치. 사용법은 6번 참고.
- 개인 브랜치는 본인이 자유롭게 push 가능 (커밋 단위·횟수 제약 없음). `main`만 보호 대상이다.
- **새 feature 브랜치를 만들기 전에는 반드시 `main`을 최신으로 pull 받는다**
  (`git checkout main && git pull origin main` 후 새 브랜치 생성). 오래된 `main` 기준으로
  작업하면 나중에 PR에서 충돌이 발생하거나, 이미 반영된 남의 작업을 놓칠 수 있다.
- 브랜치가 아니라 **파일 이름**에 이니셜을 붙일지는 7번을 따른다 (전부 붙이는 것이 아니다).

## 2. 커밋 메시지 규칙
형식: `Type: Description (#IssueNumber)`

| Type | 용도 |
| --- | --- |
| feat | 새로운 기능 구현 (전처리, 피처 생성, 모델 학습 등) |
| fix | 버그 및 오류 수정 |
| docs | 문서 작성/수정 |
| refactor | 코드 구조 개선 (기능 변경 없음) |
| test | 단위 테스트 코드 작성/수정 |
| chore | `.gitignore`, 의존성(requirements.txt) 등 설정 수정 |
| data | 데이터셋 관련 수정 |

예: `feat: preprocessor 6:2:2 split 로직 구현 (#12)`

## 3. PR & 리뷰 프로세스
1. 작업 시작 전 GitHub Issue를 등록한다 (담당 파트·목표 명시).
2. `main`을 pull 받아 최신 상태로 맞춘 뒤, 이슈 번호 기반으로 개인 브랜치를 만들어 자유롭게 커밋·push한다.
3. PR을 올리기 전, 본인 브랜치에서 최신 `main`을 받아 미리 병합해 충돌을 로컬에서
   해결한다 (`git fetch origin && git merge origin/main`, 또는 팀 합의 시 rebase).
   충돌은 PR을 올린 뒤가 아니라 올리기 전에 해소하는 것이 원칙이다.
4. 작업이 끝나면 PR을 생성한다 — 본문에 변경 요약과 `Closes #이슈번호`를 명시하고, 리뷰어로 **권재**를 지정한다.
5. **PR 리뷰·승인·merge는 권재가 담당한다.** 수정 요청이 있으면 같은 PR에 커밋을 추가해 반영한다 (새 PR을 다시 만들 필요 없음).
6. 승인 후 권재가 **Squash and Merge**로 `main`에 반영하고, 이슈가 자동 종결됐는지 확인한다.
7. merge된 개인 브랜치는 정리(삭제)한다.

> 리뷰 대기 기준: PR 등록 후 원칙적으로 1일 이내 확인을 목표로 한다. 권재가 바로 확인하기 어려운 경우 팀 채팅방에 미리 공지한다.

> **이 규칙은 GitHub 설정으로 강제되지 않고 팀 합의로 지켜진다.** (아래 4번 참고 — private 저장소 + 무료 플랜이라 GitHub의 branch protection/ruleset 기능 자체가 적용되지 않는다.) `main`에 직접 push하지 않는 것, PR로만 반영하는 것, 권재만 merge하는 것 모두 GitHub이 막아주는 게 아니라 팀원 각자가 지켜야 하는 부분이다.

## 4. Branch Protection — 기술적 강제는 현재 불가 (문서 규칙으로 대체)
GitHub의 Branch protection rule/Ruleset 기능(`main`에 직접 push 금지, 특정 인원만 merge 허용 등)은 **private 저장소 + 무료 플랜에서는 적용되지 않는다** (public 저장소이거나 GitHub Pro/Team 이상 유료 플랜이어야 실제로 강제됨). 이 저장소는 private + 무료 플랜이므로 Settings에서 rule을 만들어도 실제로 동작하지 않는다.

따라서 위 3번 프로세스(개인 브랜치 push → PR → 권재 리뷰/승인/merge)는 **기술적 강제가 아니라 팀 합의 규칙**이다. 나중에 저장소를 public으로 전환하거나 GitHub Pro로 업그레이드하면 아래 설정으로 기술적 강제를 켤 수 있다 (참고용, 현재는 적용 안 됨):

- **Require a pull request before merging**: 체크 — 직접 push 차단.
- **Require approvals**: 1건.
- **Restrict who can push to matching branches** (Ruleset에서는 `Restrict updates` + Bypass list에 권재 추가): 이 옵션이 있어야 승인 후에도 권재만 merge 버튼을 누를 수 있다 — 없으면 다른 팀원도 승인 후 merge할 수 있어 "권재만 merge" 규칙이 강제되지 않는다.
- 이 규칙은 `main`에만 적용한다 — 개인 feature 브랜치는 보호 대상이 아니므로 위 1번 규칙대로 자유롭게 push 가능하다.

## 5. Pull 관련 주의사항

- **PR이 승인·merge되어도 다른 팀원의 로컬 저장소에 자동으로 반영되지 않는다.**
  GitHub은 원격 저장소(`main`)만 갱신할 뿐, 각자의 로컬 컴퓨터까지 push해주지 않는다.
  merge 소식을 팀 채팅방에 공지하고, 각자 다음 작업 시작 전 `git pull origin main`을
  직접 실행해야 한다.
- `main`에 직접 push하는 것은 원칙적으로 금지되지만, 이 저장소는 branch protection이
  기술적으로 걸려 있지 않다(4번 참고). 다만 Git 자체의 기본 동작으로, 원격 `main`이
  로컬보다 앞서 있는 상태(즉 pull을 받지 않은 상태)에서 push하면 Git이
  **non-fast-forward 에러로 자동 거부**한다. 이때 절대 `git push --force`로 덮어쓰지
  말고, 반드시 `git pull`로 먼저 받아 병합한 뒤 다시 push한다.

## 6. 실험 브랜치 비교 & Merge 전략

지금까지처럼 `main` 위에 커밋을 계속 이어붙이면 여러 접근법을 동시에 갖고 비교할 수 없다.
같은 이슈에 대해 서로 다른 접근법(예: 다른 threshold, 다른 모델, 다른 피처셋)을 시도해보고
**가장 결과가 좋은 것 하나만 `main`에 반영**하고 싶을 때는 아래 절차를 따른다.

1. `main`을 최신으로 pull한 뒤, **같은 지점에서** 후보 브랜치를 여러 개 분기한다.
   ```bash
   git checkout main && git pull origin main
   git checkout -b experiment/#12-threshold-optA-kgj
   # 커밋...
   git checkout main
   git checkout -b experiment/#12-threshold-optB-kgj
   ```
2. 여러 브랜치를 동시에 켜두고 비교하고 싶으면 `git worktree`로 폴더를 분리해서 각 브랜치를
   독립적으로 체크아웃할 수 있다.
   ```bash
   git worktree add ../team-project-optA experiment/#12-threshold-optA-kgj
   git worktree add ../team-project-optB experiment/#12-threshold-optB-kgj
   ```
3. 각 브랜치는 PR로 올리되, PR 본문에 Sharpe Ratio 등 핵심 지표를 비교 표로 남긴다.
   `outputs/reports/decision_log.md`에도 동일한 비교 내용을 기록해 근거를 남긴다.
4. 지표를 비교해 가장 나은 브랜치 하나만 권재가 승인·**Squash and Merge**한다 (3번 프로세스와 동일).
5. 나머지 브랜치의 PR은 merge하지 않고 close하며, 브랜치는 삭제한다 (참고 가치가 있으면 팀 합의
   하에 남겨둔다).

## 7. 파일명 규칙 — 이니셜을 언제 붙이나

이니셜의 목적은 "누가 썼는지" 표시가 아니다. 그건 `git log`가 이미 알려준다.
**여러 명이 같은 주제를 각자 브랜치에서 동시에 쓸 때, 파일명이 겹쳐 머지 충돌이 나는 것을 막는 것**이
목적이다. 따라서 겹칠 일이 없는 파일에는 붙이지 않는다.

| 파일 종류 | 이니셜 | 예 |
| --- | --- | --- |
| 개인 분석·검증·리뷰 문서 (여러 명이 같은 주제로 각자 작성) | **붙인다** | `preprocessing_validation_kgj.md`, `preprocessing_review_jh.md` |
| 팀 단일 원본 문서 (하나만 존재해야 하는 문서) | 붙이지 않는다 | `decision_log.md`, `AGENTS.md`, `macro_cpi.md` |
| 코드 (`src/**`) | 붙이지 않는다 | `preprocessing_validation.py` |
| 데이터 파일 (`data/**`) | 붙이지 않는다 | `macro_cpi_monthly_*.csv` — 수집자는 짝이 되는 `.source.md`에 기록 |

- 형식은 **접미사 + 소문자**로 통일한다: `{주제}_{이니셜}.md`. 브랜치에 쓰는 이니셜과 같은 값을 쓴다.
  접미사여야 알파벳 정렬 시 같은 주제끼리 묶인다.
- **팀 단일 원본에는 붙이지 않는다.** 붙이면 "누구 버전이 진짜냐"가 생긴다. 단일 원본을 고칠 때는
  자기 이름의 파일을 새로 만들지 말고, 기존 파일을 수정하는 PR을 올린다.
- **코드에는 붙이지 않는다.** 스크립트는 개인 견해가 아니라 재현 도구다. 같은 기능의 스크립트가
  이니셜별로 여러 개 생기면 어느 것이 정본인지 알 수 없다.
- 개인 문서라도 팀 논의를 거쳐 결론이 확정되면, 그 결론은 `decision_log.md`(단일 원본)에 옮겨 적는다.
  개인 문서는 근거·과정 기록으로 남는다.

> 예외: `data/processed/variable_dictionary_byGJ.xlsx`는 단일 원본인데도 이니셜이 붙어 있다.
> 이미 여러 문서·코드가 이 이름을 참조하고 있어 그대로 두지만, 새로 만드는 파일은 위 규칙을 따른다.
