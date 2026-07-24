# Git / GitHub 협업 컨벤션

## 1. 브랜치 관리 전략
- `main`: 배포/실행 가능한 안정 브랜치. **직접 push 금지 — PR을 통해서만 반영한다.**
- `feature/#이슈번호-기능명-이니셜`: 새 기능/전처리/모델 개발 (예: `feature/#12-data-loader-jm`)
- `fix/#이슈번호-버그명-이니셜`: 버그 수정 (예: `fix/#45-sharpe-bug-jm`)
- `docs/#이슈번호-문서명-이니셜`: 문서 작성/수정
- `refactor/#이슈번호-리팩토링명-이니셜`: 코드 구조 개선 (기능 변경 없음)
- 개인 브랜치는 본인이 자유롭게 push 가능 (커밋 단위·횟수 제약 없음). `main`만 보호 대상이다.

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
2. 이슈 번호 기반으로 개인 브랜치를 만들어 자유롭게 커밋·push한다.
3. 작업이 끝나면 PR을 생성한다 — 본문에 변경 요약과 `Closes #이슈번호`를 명시하고, 리뷰어로 **권재**를 지정한다.
4. **PR 리뷰·승인·merge는 권재가 담당한다.** 수정 요청이 있으면 같은 PR에 커밋을 추가해 반영한다 (새 PR을 다시 만들 필요 없음).
5. 승인 후 권재가 **Squash and Merge**로 `main`에 반영하고, 이슈가 자동 종결됐는지 확인한다.
6. merge된 개인 브랜치는 정리(삭제)한다.

> 리뷰 대기 기준: PR 등록 후 원칙적으로 1일 이내 확인을 목표로 한다. 권재가 바로 확인하기 어려운 경우 팀 채팅방에 미리 공지한다.

## 4. Branch Protection Rule (GitHub 저장소 설정)
저장소 **Settings → Branches**에서 `main`에 대해 아래를 설정한다 (Owner/Admin 권한 필요 — 문서화만으로는 강제되지 않는다):

- **Require a pull request before merging**: 체크 — 직접 push 차단.
- **Require approvals**: 1건.
- **Restrict who can push to matching branches**: 체크, 허용 대상에 **권재만** 추가.
  - GitHub은 PR merge도 내부적으로 `main`에 대한 push로 처리한다. 이 옵션이 없으면 다른 팀원도 승인 후 merge 버튼을 누를 수 있어 "PR은 권재만 merge한다"는 규칙이 실제로 강제되지 않는다.
- 이 규칙은 `main`에만 적용된다 — 개인 feature 브랜치는 보호 대상이 아니므로 위 1번 규칙대로 자유롭게 push 가능하다.
