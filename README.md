# 기업 부도 예측 모델 (Bankruptcy Prediction Model)

DART 공시 재무 데이터와 기업 뉴스/공시 텍스트 데이터를 결합해 기업의 부도 여부를 예측하는 모델입니다.
대학생 학회 인사이트(Insight) 14기 2차 인사이콘(경진대회) 프로젝트입니다.

## 파이프라인

프로젝트는 아래 순서로 진행됩니다. 폴더명이 곧 진행 순서입니다.

1. **Data Collection** — DART Open API로 정형(재무) 데이터를, 뉴스/공시 크롤링으로 비정형(텍스트) 데이터를 수집
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

`docs/` 폴더에는 최종 발표 자료(PPT)가 있습니다.

## 데이터에 대해

DART 공시 재무제표(원본 xlsx), 뉴스/공시 크롤링 텍스트, 상장폐지 기업 리스트 등 원본 데이터는
용량이 매우 크고(수 GB) 재배포 이슈가 있어 이 레포에는 포함하지 않았습니다.
데이터 수집 방법은 위 `1. Data Collection` 코드에 그대로 구현되어 있습니다.

## 실행 전 준비

`1. Data Collection` 코드는 DART Open API 키가 필요합니다.
1. `1. Data Collection/.env.example`을 같은 폴더에 `.env`로 복사
2. [DART Open API](https://opendart.fss.or.kr)에서 발급받은 키를 `.env`의 `DART_API_KEY`에 입력
3. `pip install python-dotenv opendart-reader` 후 실행

## 기술 스택

Python, Pandas, XGBoost, Ko-BigBird(KoBigBird), DART Open API

## TODO

- [ ] 예측 결과를 보여주는 Streamlit 데모 코드 추가 예정
