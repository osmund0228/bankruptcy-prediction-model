# 다중 지표 융합 기업 부도예측모델

정형 재무지표와 회계 부정징후 지표, 사업보고서 비정형 텍스트를 융합해 재무제표 이면에 숨겨진 부도 리스크를 예측하고, 그 근거를 설명하는 모델입니다.

## 개요

- **소속**: 서강대학교 경영데이터사이언스 학회 INSIGHT
- **대회/기간**: 2025년 2학기 2차 인사이콘(학회 내부 경진대회), 2025년 12월 진행
- **목표**: 재무제표·시장지표 중심의 기존 부도예측모델이 갖는 한계를 넘어, 부정적 이슈를 숨기려는 경영진의 의도까지 포착해 신용평가·회계감사 실무에 활용 가능한 부도 예측 모델을 개발하는 것

## 담당 역할

프로젝트 리더 (PM 및 모델링)

## 데모 및 발표자료

- 데모 링크: https://bankruptcy-prediction-model-gy5akqsutrspnkemarvypb.streamlit.app/
- 데모 시연 영상: https://youtu.be/zYJQLuzcxbk
- 발표 자료: `docs/2차_인콘_발표자료.pdf`

## 주요 내용 및 성과

- **다각적 리스크 지표 통합**: Altman Z-score, KMV 등 기존 부도예측모델의 핵심 지표에, 경영자의 재무 조작 의도를 포착하는 분식회계 징후 지표(Beneish M-score)를 결합해 평가 기준을 다각화
- **사업보고서 텍스트 피처화**: 사업보고서 내 핵심 영역(이사의 경영진단, 감사보고서, 기타 투자자 보호를 위한 사항)의 텍스트를 추출하고, 긍·부정 감성 점수와 Ko-BigBird 기반 텍스트 부도 예측 확률을 각각 독립적인 피처로 도출
- **XGBoost 기반 부도 확률 산출**: 재무 지표와 텍스트 기반 피처를 융합한 총 49개 지표를 XGBoost에 학습시켜 부도 예측 확률 산출. 재무 지표에 회계 부정징후·텍스트 지표를 단계적으로 추가할수록 예측 성능이 유의미하게 향상됨을 검증
- **설명 가능한 AI(XAI) 기반 대시보드 배포**: SHAP 시각화와 LLM을 도입해 예측 근거를 직관적으로 설명하는 웹 서비스를 구현하고 배포까지 완료

정형 데이터와 비정형 텍스트를 함께 다뤄 재무제표 이면의 숨겨진 리스크를 찾아내고, 이를 실무자가 직관적으로 납득할 수 있는 데이터 솔루션으로 기획·구현하는 데 집중한 프로젝트입니다.

## 파이프라인

프로젝트는 아래 순서로 진행됩니다. 폴더명이 곧 진행 순서입니다.

1. **Data Collection** — DART Open API로 정형(재무) 데이터를, 사업보고서 크롤링으로 비정형(텍스트) 데이터를 수집
   - `1-1_numerical_data_collection.ipynb` : DART API 기반 재무제표 수집
   - `1-2_text_data_collection_bankrupt.py`, `1-3_text_data_collection_normal.py` : 부도/정상 기업 텍스트 데이터 크롤링
2. **Data Preprocessing** — 정형/비정형 데이터 각각 전처리 및 피처 엔지니어링 후 통합
   - `2-1` : 재무 데이터 전처리 및 피처 엔지니어링
   - `2-2` : 텍스트 데이터 전처리
   - `2-3` : 감성분석 기반 텍스트 피처 모델링
   - `2-4` : Ko-BigBird 기반 텍스트 피처 모델링
   - `2-5` : 정형+비정형 데이터 통합
3. **Data Modeling** — XGBoost 기반 부도 예측 모델 학습 및 시각화
   - `3-1` : 하이퍼파라미터 튜닝
   - `3-2` : 모델 결과 시각화 (feature importance 등)
4. **Streamlit 대시보드** (`streamlit-dashboard/`) — 학습된 모델로 예측 결과를 조회하고 SHAP·LLM 기반 설명을 제공

## 데이터

DART 공시 재무제표(원본 xlsx), 사업보고서 크롤링 텍스트, 상장폐지 기업 리스트 등 원본 데이터는
용량이 매우 크고(수 GB) 재배포 이슈가 있어 이 레포에는 포함하지 않았습니다.
데이터 수집 방법은 위 `1. Data Collection` 코드에 그대로 구현되어 있습니다.

2025년 12월 진행 시점 기준 2025년 사업보고서가 아직 공시되지 않아, 2024년까지의 데이터로 학습·검증했습니다.

## 결과

2025년 2차 인사이콘(학회 내부 경진대회) 3위 · 우수상 수상

## 기술 스택

Python, Pandas, XGBoost, Ko-BigBird(KoBigBird), SHAP, DART Open API, Streamlit, Gemini API

## Getting Started

### 1. Data Collection (DART Open API 키 필요)

1. `1. Data Collection/.env.example`을 같은 폴더에 `.env`로 복사
2. [DART Open API](https://opendart.fss.or.kr)에서 발급받은 키를 `.env`의 `DART_API_KEY`에 입력
3. `pip install python-dotenv opendart-reader` 후 실행

### 2. Streamlit 데모 (`streamlit-dashboard/`)

Gemini API 연동과 참고용 구글시트 연결이 필요합니다.

```bash
cd streamlit-dashboard
pip install -r requirements.txt
```

1. `.streamlit/secrets.toml.example`을 같은 폴더에 `.streamlit/secrets.toml`로 복사
2. [Google AI Studio](https://aistudio.google.com/apikey)에서 발급받은 키를 `GEMINI_API_KEY`에 입력
3. `connections.gsheets.spreadsheet`에 사용할 구글시트 URL 입력
4. `streamlit run app.py`
