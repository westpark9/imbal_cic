# 연구 작업 원칙

모델을 개발하면서 발생한 문제/병목 지점에 대해 오류나 낮은 퍼포먼스를 인정하는 것이 아니라 개선하고 해결하기 위한 방안을 제시합니다.

# HTML 보고서와 온라인 아티팩트 동시 갱신

사용자 요청(2026-09-10): 로컬 HTML 보고서를 수정하면 대응하는 온라인 아티팩트도 같은 내용으로 갱신합니다. 사용자가 매번 따로 요청하게 하지 않습니다.

- 개별 보고서 원본: `lablog/html_report/<report_id>.html`.
- TabPFN 도입 이후 통합본: `lablog/html_report/post_tabpfn.html`.
  대응 아티팩트: https://claude.ai/code/artifact/688578fa-0974-4310-a8a4-200b6c9bad28
- TabPFN 도입 이전 통합본: `lablog/html_report/pre_tabpfn.html`.
  대응 아티팩트: https://claude.ai/code/artifact/2d550cef-c026-4638-957d-84bdb7889b31

보고서 수정 시 다음 순서를 따릅니다.

1. 원본 HTML을 수정하고 새 보고서라면 통합본의 목차·개요·`REPORTS` 항목도 추가합니다.
2. `python scripts/sync_html_reports.py`로 개별 HTML의 내용을 통합본의 내장 사본에 반영합니다. `python scripts/sync_html_reports.py --check`로 동기화를 확인합니다.
3. 연결된 아티팩트 편집 도구로 해당 온라인 아티팩트의 내용을 최신 통합 HTML로 갱신하고, 0909 같은 최신 항목과 본문이 실제로 반영되었는지 확인합니다. 기존 아티팩트 주소를 유지합니다.
4. 로컬 파일·Git push·온라인 아티팩트 갱신의 완료 여부를 구분해서 보고합니다. Git push만으로 Claude 아티팩트가 갱신된 것으로 간주하지 않습니다.

아티팩트 편집 도구나 접근 권한이 없으면 로컬 동기화와 검증까지 마친 뒤, 원격 갱신이 되지 않았다는 사실과 교체할 최신 HTML 경로를 명시합니다. 온라인 갱신이 완료됐다고 쓰거나 별도 서비스를 임의로 만들어 대신하지 않습니다.
