# dashboard.py
import os
from pykrx import stock
import pandas as pd
import numpy as np
from google import genai
import streamlit as st
import joblib
import shap
from streamlit_gsheets import GSheetsConnection

# === 설정 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

FEATURE_NAMES = [
'F1_Equity_Growth', 'F1_Retained_Earnings_Ratio', 'F1_ROA', 'F1_Debt_Ratio', 'F1_Current_Ratio', 'F1_ROE', 'F1_Interest_Coverage',
'F2_KMV_DD', 'F3_Z_Score','F4_M_Score', 'M_Short_Term_Rate', 'M_Long_Term_Rate', 'M_Rate_Spread', 'M_Nominal_GDP_Growth',
'M_Real_GDP_Growth', 'M_Inflation', 'M_Exchange_Rate', 'F1_Equity_Growth_change', 'F1_Equity_Growth_pct_change',
'F1_Equity_Growth_improving', 'F1_Retained_Earnings_Ratio_change', 'F1_Retained_Earnings_Ratio_pct_change', 'F1_Retained_Earnings_Ratio_improving',
'F1_ROA_change', 'F1_ROA_pct_change', 'F1_ROA_improving', 'F1_Debt_Ratio_change', 'F1_Debt_Ratio_pct_change', 'F1_Debt_Ratio_improving',
'F1_Current_Ratio_change', 'F1_Current_Ratio_pct_change', 'F1_Current_Ratio_improving', 'F1_ROE_change', 'F1_ROE_pct_change', 'F1_ROE_improving',
'F1_Interest_Coverage_change', 'F1_Interest_Coverage_pct_change', 'F1_Interest_Coverage_improving',
'audit_prob', 'etc_prob', 'mda_prob', 'lex_sent_mean', 'lex_sent_sum', 'lex_pos_tf', 'lex_neg_tf', 'lex_pos_cnt', 'lex_neg_cnt', 'lex_abs_mean', 'lex_covered_tf' ]

FEATURE_MAP = {
'F1_Equity_Growth':'자기자본 증가율 / 자기자본 대비 순이익의 비율로 높을수록 주주 자본을 효율적으로 활용하는 것',
'F1_Retained_Earnings_Ratio':'이익잉여금 비율 / 자본 중 이익잉여금이 차지하는 비중',
'F1_ROA':'총자산이익률 / 기업이 보유한 자산으로 얼마나 효율적으로 이익을 창출하는지',
'F1_Debt_Ratio':'부채비율 / 자기자본 대비 부채 수준으로 직관적인 파산 위험의 지표',
'F1_Current_Ratio':'유동비율 / 단기적인 채무 상환의 능력을 의미',
'F1_ROE':'자기자본이익률 / 자기자본 대비 순이익의 비율로 높을수록 주주 자본을 효율적으로 활용하는 것',
'F1_Interest_Coverage':'이자보상배율 / 영업이익이 이자비용의 몇 배인지를 나타내면 1 미만이면 이자조차 감당할 수 없음을 의미',
'F2_KMV_DD':'Distance to Default / 자산가치가 부채 임계치로부터 얼마나 떨어져 있는지를 나타내며 거리가 낮을수록 부도 가능성이 증가',
'F3_Z_Score':'Altman Z-score / 여러 재무 비율을 종합한 파산 예측 점수로 낮을수록 파산 가능성이 증가',
'F4_M_Score':'Beneish M-score / 회계 조작 가능성을 나타내는 점수로 높을 수록 이익을 조정했을 가능성이 증가',
'M_Short_Term_Rate':'단기금리 / 단기적인 차입 비용으로 단기금리가 상승하면 재무적으로 취약한 기업에 부담으로 가중',
'M_Long_Term_Rate':'장기금리 / 장기적인 자본 조달을 위한 비용으로 상승하면 투자가 위축되고, 재무구조가 약한 기업에 불리',
'M_Rate_Spread':'금리 스프레드 / 장기, 단기 금리의 차이로 경기가 침체되는 신호를 나타냄',
'M_Nominal_GDP_Growth':'명목 GDP 성장률 / 경기 규모의 성장을 나타내며 낮을수록 매출 성장이 둔화',
'M_Real_GDP_Growth':'실질 GDP 성장률 / 물가 효과를 제거한 실질적인 경기 성장을 반영',
'M_Inflation':'물가상승률 / 전반적인 물가 수준의 변화를 의미하며 급등하면 비용으로 압박이 작용',
'M_Exchange_Rate':'환율 / 원화 대비 외화의 가치를 의미하며 수입 및 외화부채가 많은 기업에 위험 신호로 작용',
'F1_Equity_Growth_change':'자기자본 증가율 변동폭',
'F1_Equity_Growth_pct_change':'자기자본 증가율 증감률(%)',
'F1_Equity_Growth_improving':'자기자본 증가율 개선 여부',
'F1_Retained_Earnings_Ratio_change':'이익잉여금 비율 변동폭',
'F1_Retained_Earnings_Ratio_pct_change':'이익잉여금 비율 증감률(%)',
'F1_Retained_Earnings_Ratio_improving':'이익잉여금 비율 개선 여부',
'F1_ROA_change':'ROA 변동폭',
'F1_ROA_pct_change':'ROA 증감률(%)',
'F1_ROA_improving':'ROA 개선 여부',
'F1_Debt_Ratio_change':'부채비율 변동폭',
'F1_Debt_Ratio_pct_change':'부채비율 증감률(%)',
'F1_Debt_Ratio_improving':'부채비율 개선 여부',
'F1_Current_Ratio_change':'유동비율 변동폭',
'F1_Current_Ratio_pct_change':'유동비율 증감률(%)',
'F1_Current_Ratio_improving':'유동비율 개선 여부',
'F1_ROE_change':'ROE 변동폭',
'F1_ROE_pct_change':'ROE 증감률(%)',
'F1_ROE_improving':'ROE 개선 여부',
'F1_Interest_Coverage_change':'이자보상배율 변동폭',
'F1_Interest_Coverage_pct_change':'이자보상배율 증감률(%)',
'F1_Interest_Coverage_improving':'이자보상배율 개선 여부',
'audit_prob':'감사의견 텍스트 기반 부도 위험도',
'etc_prob':'기타 공시 텍스트 기반 부도 위험도',
'mda_prob':'MD&A 텍스트 기반 부도 위험도',
'lex_sent_mean':'문서 전반의 평균적인 감성 점수로 낮을수록 부정적인 톤이 증가',
'lex_sent_sum':'전체 문성의 감성 누적 정도로 부정 감성의 누적은 리스크가 커지는 것을 의미',
'lex_pos_tf':'긍정 단어의 빈도를 의미',
'lex_neg_tf':'부정 단어의 빈도를 의미',
'lex_pos_cnt':'긍정 단어가 등장하는 문장의 수를 의미',
'lex_neg_cnt':'부정 단어가 등장하는 문장의 수를 의미',
'lex_abs_mean':'감성 강도의 절댓값의 평균으로 높을수록 표현의 강도가 크며 불확실성이 증가한다는 것을 의미',
'lex_covered_tf':'감성 사전이 커버한 단어의 수를 의미하며 텍스트 분석 신뢰도 지표'
}

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "model_xgb_new_23.pkl"))

def load_data_and_model(ticker):
    code = ticker.strip() 
    
    # 1. 주가 데이터 (기존 유지)
    try:
        today = pd.Timestamp.now().strftime("%Y%m%d")
        start_date = (pd.Timestamp.now() - pd.DateOffset(years=1)).strftime("%Y%m%d")
        hist = stock.get_market_ohlcv(start_date, today, code)
        current_price = hist['종가'].iloc[-1] if not hist.empty else 0
    except:
        current_price = 0

    # ==========================================================
    # [수정] secrets.toml 없이 '직접 링크'로 불러오기 (가장 확실함)
    # ==========================================================
    
    # 본인의 구글 시트 ID (주소 중간에 있는 긴 문자열)
    SHEET_ID = "16OBBXMXJpw8DYFVdzyM5f1AIYyYlHIyMxn1-ZB2TXNk"
    
    # 각 시트의 GID를 정확히 적어주세요! (브라우저 주소창 확인 필수)
    GID_SHEET1 = "1720662044"  # Sheet1의 gid
    GID_SHEET2 = "1526907458"  # Sheet2의 gid
    GID_SHEET3 = "1075256900"  # Sheet3의 gid
    
    # CSV 변환 URL 생성 함수
    def get_csv_url(sheet_id, gid):
        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

    try:
        df_company = pd.read_csv(get_csv_url(SHEET_ID, GID_SHEET1))
        df_ind_avg = pd.read_csv(get_csv_url(SHEET_ID, GID_SHEET2))
        df_stat_avg = pd.read_csv(get_csv_url(SHEET_ID, GID_SHEET3))
    except Exception as e:
        st.error(f"구글 시트 로드 중 에러 발생: {e}")
        return None

    # 3. 49개 피처 누락 방지 (0으로 채우기)
    for col in FEATURE_NAMES:
        if col not in df_company.columns: df_company[col] = 0.0

    # 4. 데이터 필터링 (6자리 코드로 변환하여 비교)
    # 문자열로 변환하고, 6자리를 맞춥니다 (예: 5930 -> 005930)
    company_row = df_company[df_company['stock_code'].astype(str).str.zfill(6) == code]
    
    if company_row.empty:
        return None
    
    company_row = company_row.iloc[0]
    
    # 1. 내 기업의 산업군(섹터) 이름 가져오기
    # 컬럼명이 '섹터'일 수도 있고 '산업군'일 수도 있어서 둘 다 확인
    if '섹터' in company_row:
        my_sector = str(company_row['섹터']).strip()
    elif '산업군' in company_row:
        my_sector = str(company_row['산업군']).strip()
    else:
        my_sector = "Unknown"

    # 2. 산업군 평균 (ind_row) 찾기
    try:
        # Sheet2(산업평균)의 '섹터' 컬럼도 공백 제거하여 비교 준비
        # (만약 Sheet2의 컬럼명이 '산업군'이라면 아래 '섹터'를 '산업군'으로 바꿔주세요)
        if '섹터' in df_ind_avg.columns:
            target_col = '섹터'
        elif '산업군' in df_ind_avg.columns:
            target_col = '산업군'
        else:
            raise ValueError("Sheet2에 '섹터' 또는 '산업군' 컬럼이 없습니다.")

        # 비교를 위해 문자열 변환 및 공백 제거
        df_ind_avg[target_col] = df_ind_avg[target_col].astype(str).str.strip()
        
        # 매칭 시도
        matched_rows = df_ind_avg[df_ind_avg[target_col] == my_sector]
        
        if not matched_rows.empty:
            # 매칭 성공! 해당 산업군 평균 사용
            ind_row = matched_rows.iloc[0]
            # (디버깅용: 필요시 주석 해제)
            print(f"✅ 산업군 매칭 성공: {my_sector}")
        else:
            # 매칭 실패 -> 전체 정상기업 평균 사용 (Fallback)
            print(f"⚠️ 산업군 매칭 실패: '{my_sector}' (Sheet2 목록에 없음)")
            ind_row = df_stat_avg[df_stat_avg['Target'] == 0].iloc[0]

    except Exception as e:
        # 에러 발생 시 -> 전체 정상기업 평균 사용
        # print(f"❌ 산업군 로직 에러: {e}")
        ind_row = df_stat_avg[df_stat_avg['Target'] == 0].iloc[0]

    # 2. 정상기업 평균 (norm_row) 가져오기
    # 비교 그래프를 그리기 위해 'Target 0(정상)' 데이터를 가져옵니다.
    try:
        norm_row = df_stat_avg[df_stat_avg['Target'] == 0].iloc[0]
    except IndexError:
        # 혹시 Target 0인 데이터가 하나도 없다면, 그냥 첫 번째 줄을 가져옵니다.
        norm_row = df_stat_avg.iloc[0]

    # 5. XGBoost 모델 예측 (순서 강제 정렬)
    model = load_model()
    
    # 49개 컬럼 순서대로 데이터프레임 생성
    X_input = pd.DataFrame([company_row[FEATURE_NAMES].values], columns=FEATURE_NAMES)
    
    # 숫자로 변환 (에러 방지)
    X_input = X_input.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    prob = model.predict_proba(X_input)[0][1]
    
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_input)[0]

    LOWER_IS_BETTER = [
        # 1. 재무 비율 (부채는 적을수록 좋음)
        "F1_Debt_Ratio", 
        
        # 2. 재무 변화량 (부채비율이 늘어나는 건 나쁨)
        "F1_Debt_Ratio_change", 
        "F1_Debt_Ratio_pct_change",

        # 3. 리스크 모델 (M-Score는 높으면 회계부정 의심 -> 낮아야 좋음)
        "F4_M_Score", 
        
        # 4. 거시경제 (금리/물가/환율은 오르면 기업 부담 -> 낮아야 좋음)
        "M_Short_Term_Rate", 
        "M_Long_Term_Rate",   # [추가됨] 설명: "상승하면... 불리"
        "M_Inflation",        # 설명: "급등하면 비용 압박"
        "M_Exchange_Rate",    # 설명: "위험 신호로 작용"

        # 5. AI 부도 확률 예측 (당연히 확률이 낮아야 안전)
        "audit_prob", 
        "etc_prob", 
        "mda_prob",
        
        # 6. 텍스트 감성 분석 (부정적 단어/불확실성은 적을수록 좋음)
        "lex_neg_cnt",       # 부정 문장 수
        "lex_neg_tf",        # 부정 단어 빈도
        "lex_abs_mean"       # [추가됨] 설명: "높을수록... 불확실성 증가"
    ]

    # [수정] 이상치(Outlier)에 강한 '백분위(Rank) 기반' 스코어링 함수
    def calculate_score(val, col_name):
        try:
            # 1. 결측치 방어
            val = float(val)
            if pd.isna(val): return 50

            # 2. 해당 컬럼의 유효한 데이터 전체 가져오기 (NaN 제외)
            all_values = pd.to_numeric(df_company[col_name], errors='coerce').dropna()
            
            if all_values.empty: return 50
            
            # 3. 내 값이 전체에서 상위 몇 %인지 계산 (0.0 ~ 1.0)
            # (scipy 없이 순수 pandas/numpy로 구현)
            # 내 값보다 작은 데이터의 비율을 구함
            percentile = (all_values < val).mean()
            
            # 4. 점수 변환 (0~100점)
            score = percentile * 100
            
            # 5. 낮을수록 좋은 지표(LOWER_IS_BETTER)는 점수 뒤집기
            # (예: 부채비율은 상위 90%(=값이 큼)일수록 나쁜 거니까 100 - 90 = 10점)
            if col_name in LOWER_IS_BETTER:
                score = 100 - score
                
            return np.clip(score, 0, 100)
            
        except Exception as e:
            # print(f"Score Error {col_name}: {e}") # 디버깅용
            return 50

    shap_data = []
    
    for i, name in enumerate(FEATURE_NAMES):
        # 카테고리 자동 분류
        if name.startswith("F1"): category = "financial"
        elif name.startswith("M_"): category = "macro"
        elif "lex" in name or "prob" in name: category = "text"
        else: category = "risk_model"

        shap_data.append({
            "name": name,
            "category": category,
            "shap": float(shap_vals[i]),
            "score": calculate_score(company_row[name], name),
            "industry_avg": calculate_score(ind_row[name], name),
            "normal_avg": calculate_score(norm_row[name], name),
            "val": str(company_row[name]),
            "desc": FEATURE_MAP.get(name, name)
        })

    # SHAP 절대값 기준 내림차순 정렬
    shap_data = sorted(shap_data, key=lambda x: abs(x['shap']), reverse=True)
    
    # 신호등 로직 (함수 존재 시 실행)
    try:
        indicators = determine_traffic_lights_by_group(shap_data)
    except:
        indicators = {}

    return {
        "ticker": code,
        "company_name": company_row.get('Company_Name', code),
        "price": current_price,
        "risk_score": int(prob * 100),
        "indicators": indicators,
        "shap_data": shap_data
    }
    

def determine_traffic_lights_by_group(shap_data):
    # 1. 5개 그룹별 SHAP 값 수집
    vals_f1 = []      # 재무비율
    vals_macro = []   # 시장지표
    vals_model = []   # 부도모델
    vals_fraud = []   # 부정징후
    vals_text = []    # 텍스트

    for item in shap_data:
        name = item['name']
        val = item['shap']
        
        if name.startswith('F1'): vals_f1.append(val)
        elif name.startswith('M_'): vals_macro.append(val)
        elif name.startswith('F2') or name.startswith('F3'): vals_model.append(val)
        elif name.startswith('F4'): vals_fraud.append(val)
        elif 'prob' in name or 'lex' in name: vals_text.append(val)
        else: vals_text.append(val) # 기타

    # 2. 위험도 합계 계산
    def calculate_risk_impact(values):
        if not values: return 0.0
        return np.nansum(values)

    score_f1 = calculate_risk_impact(vals_f1)
    score_macro = calculate_risk_impact(vals_macro)
    score_model = calculate_risk_impact(vals_model)
    score_fraud = calculate_risk_impact(vals_fraud)
    score_text = calculate_risk_impact(vals_text)

    # =========================================================================
    # [핵심 수정] 섹터별 임계값(Threshold) 차별화 설정
    # =========================================================================
    # red: 이 점수를 넘으면 '위험(빨강)'
    # yellow: 이 점수를 넘으면 '주의(노랑)'
    THRESHOLDS = {
        # 1. 재무비율 (현대차 등 대기업 부채 고려하여 0.3으로 넉넉하게)
        "f1":    {"red": 0.40, "yellow": 0.10},
        
        # 2. 거시경제 (점수 변동폭이 작으므로)
        "macro": {"red": 0.08, "yellow": 0.03},
        
        # 3. 부도모델 (가장 결정적이나 수치가 크게 튀므로 높게 설정)
        "model": {"red": 3.00, "yellow": 2.00},
        
        # 4. 부정징후 (M-score는 0.2 정도면 꽤 높은 편)
        "fraud": {"red": 0.20, "yellow": 0.10},
        
        # 5. 텍스트 (노이즈가 많으므로 재무 수준인 0.3 적용)
        "text":  {"red": 0.30, "yellow": 0.10}
    }

    def get_color(score, category):
        # 해당 카테고리의 기준 가져오기 (없으면 기본값 f1 기준 사용)
        t = THRESHOLDS.get(category, THRESHOLDS["f1"])
        
        if score > t["red"]: return "red"
        elif score > t["yellow"]: return "yellow"
        else: return "green"

    return {
        "f1": get_color(score_f1, "f1"),
        "macro": get_color(score_macro, "macro"),
        "model": get_color(score_model, "model"),
        "fraud": get_color(score_fraud, "fraud"),
        "text": get_color(score_text, "text")
    }

def get_gemini_rag_analysis(data_summary, shap_data):
    # 1. API 키 확인
    if not GEMINI_API_KEY: 
        return "⚠️ API 키가 설정되지 않았습니다."

    # 2. 기본 정보 추출
    ticker = data_summary.get('ticker', 'Unknown')
    risk_score = data_summary.get('risk_score', 0)
    company_name = data_summary.get('company_name', ticker)
    
    # =====================================================================
    # [핵심 수정] 절대값 기준 Top 5가 아니라, '위험'과 '안전'을 각각 추출
    # =====================================================================
    
    # 1. 위험 요인 (SHAP > 0): 부도 확률을 높이는 요소
    # 값이 큰 순서대로 정렬 (가장 위험한 것부터)
    risks = sorted([x for x in shap_data if x['shap'] > 0], key=lambda x: x['shap'], reverse=True)
    top_risks = risks[:5] # 상위 5개 추출

    # 2. 안전 요인 (SHAP < 0): 부도 확률을 낮추는(방어하는) 요소
    # 절대값이 큰 순서대로 정렬 (가장 안전하게 만드는 것부터)
    safes = sorted([x for x in shap_data if x['shap'] < 0], key=lambda x: abs(x['shap']), reverse=True)
    top_safes = safes[:5] # 상위 5개 추출

    # 3. 텍스트로 변환
    risk_text = ""
    if top_risks:
        for item in top_risks:
            risk_text += f"- {item['name']} ({item['desc']}): SHAP={item['shap']:.4f} [🚨위험요인], 실제값={item['val']}\n"
    else:
        risk_text = "(특이할 만한 위험 요인이 발견되지 않음 - 재무적으로 매우 안정적임)"

    safe_text = ""
    if top_safes:
        for item in top_safes:
            safe_text += f"- {item['name']} ({item['desc']}): SHAP={item['shap']:.4f} [✅안전요인], 실제값={item['val']}\n"
    else:
        safe_text = "(뚜렷한 방어 기제가 부족함)"

    # 4. Gemini 프롬프트 구성
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        prompt = f"""
        당신은 기업 구조조정 및 부도 예측 전문가 AI입니다. 
        사용자가 제공한 데이터를 바탕으로 투자자를 위한 정밀 분석 보고서를 작성하세요.

        [분석 대상 기업]
        - 기업명: {company_name} ({ticker})
        - AI 종합 부도 위험 점수: {risk_score}점 (0점: 매우 안전 ~ 100점: 부도 위험 심각)

        [데이터 분석 결과]
        
        1. 🚨 주요 위험 요인 (Risk Factors) - 부도 가능성을 높이는 요인들:
        {risk_text}
        
        2. ✅ 주요 안전 요인 (Strength Factors) - 부도 가능성을 낮추는 방어 기제:
        {safe_text}

        [작성 가이드]
        1. **종합 의견**: 위험 점수와 위 요인들을 종합하여 이 기업의 현재 상황을 2~3문장으로 요약하세요.
        2. **위험 요인 분석**: 위 '주요 위험 요인' 목록에 있는 항목들이 왜 위험한지, 이것이 기업에 어떤 악영향을 줄 수 있는지 구체적으로 설명하세요. (목록이 없다면 안전하다고 칭찬하세요.)
        3. **긍정 요인 분석**: 위 '주요 안전 요인' 목록을 바탕으로 이 기업의 재무적 강점이 무엇인지 설명하세요.
        4. **제언**: 투자 관점에서 유의해야 할 점이나 모니터링해야 할 지표를 제시하세요.

        (주의: SHAP 값이 양수(+)면 위험, 음수(-)면 안전입니다. 이 규칙을 절대 혼동하지 마세요.)
        """

        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text

    except Exception as e:
        return f"분석 생성 실패: {str(e)}"