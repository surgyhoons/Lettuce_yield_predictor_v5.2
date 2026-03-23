# Streamlit 실행 방법

## 1) 설치
```bash
pip install -r requirements.txt
```

## 2) secrets 설정
`secrets.toml.example`를 참고해서 아래 위치에 저장하세요.

- 로컬: `.streamlit/secrets.toml`
- Streamlit Community Cloud: 앱 설정의 Secrets

## 3) Google Sheets 공유
서비스 계정 이메일을 Google Sheets 공유 대상에 **편집자**로 추가해야 합니다.

## 4) 실행
```bash
streamlit run lettuce_streamlit_app.py
```

## 사용 시트
- `DB_배치데이터` : 메인 DB
- `예측결과_log` : D+3~4 스냅샷 로그

## 현재 구현된 화면
- 3일 대시보드
- 달별 전체 예측
- 배치 DB 관리
- 실적 입력
- 노션 마크다운 출력
- 고정 재배대 용량표

## 참고
- `st.secrets`가 없으면 앱은 공개 CSV를 읽는 **읽기 전용 모드**로만 동작할 수 있습니다.
- 쓰기 기능(배치 저장, 삭제, 실적 저장, 로그 저장)은 서비스 계정 연결이 필요합니다.
