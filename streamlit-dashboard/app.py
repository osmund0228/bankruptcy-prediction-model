import streamlit as st
st.set_page_config(layout="wide", page_title="잡았다 요놈! Risk Dashboard")

import dashboard as db
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# 1. CSS 스타일 (사용자님 원본 유지)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .shap-row { display: flex; align-items: center; margin-bottom: 6px; padding: 5px; background-color: #ffffff; border-radius: 4px; font-size: 14px; border-bottom: 1px solid #eee; }
    .feature-name { flex: 2; font-weight: 600; color: #333; }
    .bar-container { flex: 3; display: flex; align-items: center; }
    .shap-bar { height: 10px; border-radius: 5px; }
    .shap-value { width: 50px; text-align: right; margin-left: 8px; font-size: 12px; color: #666; font-family: monospace;}
    .actual-val { flex: 2; text-align: right; font-size: 13px; font-weight: bold; color: #444; }
    .desc-text { flex: 2; text-align: right; color: #888; font-size: 12px; margin-left: 10px; }
    .metric-box { text-align: center; padding: 15px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. 사이드바 및 데이터 로드
# -----------------------------------------------------------------------------
st.sidebar.title("🔎 기업 검색")
ticker_input = st.sidebar.text_input("종목 코드", value="005930") # 입력값 유지 위해 value 추가

if st.sidebar.button("진단 시작"):
    st.session_state['run'] = True
    st.session_state['current_ticker'] = ticker_input

if st.session_state.get('run'):
    with st.spinner("데이터 분석 중..."):
        # ticker_input 대신 세션의 ticker 사용 (새로고침 방지)
        target_ticker = st.session_state.get('current_ticker', ticker_input)
        data = db.load_data_and_model(target_ticker)
        
        if data is None:
            st.error("⚠️ 해당 종목 코드를 찾을 수 없습니다. 다시 확인해주세요.")
            st.stop()

    # 변수 정의 (에러 방지용으로 최상단 배치)
    shap_data = data['shap_data']     
    df_all = pd.DataFrame(shap_data)  
    risk = data['risk_score']         
    
    # -------------------------------------------------------------------------
    # 3. 메인 UI 헤더 (원본 디자인 유지)
    # -------------------------------------------------------------------------
    st.title(f"📊 {data['company_name']}({data['ticker']}) 통합 부도 리스크 분석")
    
    col_h1, col_h2 = st.columns([1, 2])
    with col_h1: 
        st.metric("현재 주가", f"{data['price']:,.0f}원")
    with col_h2:
        st.subheader(f"🚨 부도 위험 스코어: {risk}%")
        st.progress(risk/100)
    
    st.divider()
    
    st.subheader("🚦 5대 핵심 리스크 감지")
    
    # 5개 컬럼 생성
    c1, c2, c3, c4, c5 = st.columns(5)
    
    ind = data['indicators']
    
    # 신호등 그리는 함수 (디자인 유지)
    def draw_light(col, title, subtitle, status, icon):
        colors = {"red": "#FFEBEE", "yellow": "#FFFDE7", "green": "#E8F5E9"}
        emoji = {"red": "🔴 위험", "yellow": "🟡 주의", "green": "🟢 양호"}
        status = status if status else "green"
        
        with col:
            st.markdown(f"""
                <div class='metric-box' style='background-color: {colors.get(status, "#fff")}; padding: 10px;'>
                    <div style='font-size:24px; margin-bottom:4px;'>{icon}</div>
                    <div style='font-size:14px; font-weight:bold; color:#333;'>{title}</div>
                    <div style='font-size:11px; color:#666; margin-bottom:8px;'>{subtitle}</div>
                    <div style='font-size:13px;'>{emoji.get(status, "🟢 양호")}</div>
                </div>
            """, unsafe_allow_html=True)

    # 5개 항목 배치
    # 1. 재무비율 (F1)
    draw_light(c1, "재무 건전성", "기초 재무비율(F1)", ind.get('f1'), "💰")
    
    # 2. 시장지표 (Macro)
    draw_light(c2, "시장 환경", "거시경제 지표(M)", ind.get('macro'), "🌍")
    
    # 3. 부도 위험 (F2, F3)
    draw_light(c3, "부도 예측", "KMV & Altman Z-score", ind.get('model'), "📉")
    
    # 4. 부정 징후 (F4)
    draw_light(c4, "회계 부정 징후", "Beneish M-score", ind.get('fraud'), "🕵️")
    
    # 5. 텍스트 분석
    draw_light(c5, "텍스트 분석", "공시 보고서 내 텍스트 분석", ind.get('text'), "📝")

    # --------------------------------------------------------------------------------
    # 4. 7대 핵심 건전성 분석 (수정된 로직 적용)
    # --------------------------------------------------------------------------------
    st.divider()
    st.subheader("📊 7대 핵심 건전성 분석")
    st.caption("※ 49개 세부 지표를 7가지 핵심 역량으로 그룹화하여 분석한 결과입니다. (점수가 높을수록 우량/안전)")

    # (1) 매핑 로직 (요청하신 네이밍 적용)
    def get_category(name):
        name = name.lower()
        if any(x in name for x in ['roa', 'roe', 'interest_coverage']): return '💰 수익성'
        if any(x in name for x in ['debt', 'current_ratio', 'retained']): return '🛡️ 재무안정성'
        if any(x in name for x in ['equity_growth']): return '📈 성장성'
        if any(x in name for x in ['kmv', 'z_score', 'm_score']): return '🔎 탐지모델'
        if name.startswith('m_'): return '🌍 거시환경'
        if 'prob' in name: return '📝 NLP분석'
        if 'lex' in name: return '❤️ 감성분석'
        return '기타'

    # (2) 데이터 그룹화
    radar_data = {} 
    target_categories = ['💰 수익성', '🛡️ 재무안정성', '📈 성장성', '🔎 탐지모델', '🌍 거시환경', '📝 NLP분석', '❤️ 감성분석']
    
    for cat in target_categories:
        radar_data[cat] = {'company': [], 'industry': [], 'normal': []}

    for item in shap_data:
        cat = get_category(item['name'])
        if cat in radar_data:
            radar_data[cat]['company'].append(item['score'])
            radar_data[cat]['industry'].append(item['industry_avg'])
            radar_data[cat]['normal'].append(item['normal_avg'])

    # (3) 평균 계산 함수 (결측치 제외 로직)
    def get_valid_mean(scores):
        # 유효한 숫자만 필터링
        valid_scores = [s for s in scores if pd.notna(s) and isinstance(s, (int, float))]
        
        # 유효한 데이터가 하나라도 있으면 평균 계산
        if len(valid_scores) > 0:
            return sum(valid_scores) / len(valid_scores)
        
        # 유효한 데이터가 아예 없으면 50점(중립) 반환
        return 50.0

    # 최종 점수 계산
    final_cats = []
    c_scores, i_scores, n_scores = [], [], []

    for cat in target_categories:
        final_cats.append(cat)
        # 있는 데이터끼리만 평균 내기
        c_scores.append(get_valid_mean(radar_data[cat]['company']))
        i_scores.append(get_valid_mean(radar_data[cat]['industry']))
        n_scores.append(get_valid_mean(radar_data[cat]['normal']))

    # (4) 차트 그리기
    col_bar, col_radar = st.columns(2)

    # [왼쪽] 바 차트
    with col_bar:
        fig_bar = go.Figure()
        
        # 내 기업
        fig_bar.add_trace(go.Bar(
            x=final_cats, y=c_scores, 
            name='대상 기업', marker_color='#2962ff',
            text=[f"{s:.0f}" for s in c_scores], textposition='auto',
            hovertemplate="<b>%{x}</b><br>건전성: %{y:.1f}점<extra></extra>"
        ))
        # 정상 평균
        fig_bar.add_trace(go.Bar(x=final_cats, y=n_scores, name='정상 평균', marker_color='green', opacity=0.5))
        # 산업 평균
        fig_bar.add_trace(go.Bar(x=final_cats, y=i_scores, name='산업 평균', marker_color='orange', opacity=0.5))
        
        fig_bar.update_layout(
            title="분야별 건전성 점수 비교", barmode='group',
            yaxis=dict(title="점수 (100점 만점)", range=[0, 100]),
            height=400, legend=dict(orientation="h", y=-0.2)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # [오른쪽] 레이더 차트
    with col_radar:
        def wrap(l): return l + [l[0]] 
        
        fig_radar = go.Figure()
        
        # 1. 정상/산업 (배경)
        # [수정 3] showlegend=True로 변경 (기본값이 True이므로 False 옵션 삭제)
        fig_radar.add_trace(go.Scatterpolar(
            r=wrap(n_scores), theta=wrap(final_cats), 
            name='정상 평균', 
            line=dict(color='green', dash='solid'),       # 진한 녹색 선 (두께 2)
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=wrap(i_scores), theta=wrap(final_cats), 
            name='산업 평균', 
            line=dict(color='orange', dash='dash')
        ))
        
        # 2. 내 기업 (메인)
        fig_radar.add_trace(go.Scatterpolar(
            r=wrap(c_scores), theta=wrap(final_cats), 
            name='분석 대상', 
            fill='toself', 
            line=dict(color='#2962ff', width=3), 
            opacity=0.4,
            hovertemplate="<b>%{theta}</b><br>건전성: %{r:.1f}점<extra></extra>"
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], ticksuffix="점", gridcolor='#eee'),
                angularaxis=dict(gridcolor='#eee', tickfont=dict(size=12, color='black')),
                bgcolor='white'
            ),
            title="다차원 건전성 균형도",
            height=400,
            margin=dict(t=40, b=40, l=40, r=40),
            legend=dict(orientation="h", y=-0.15) # 범례 표시
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # --------------------------------------------------------------------------------
    # 5. SHAP 전체 출력 (토글 적용 + 잘림 방지)
    # --------------------------------------------------------------------------------
    st.divider()
    st.subheader("📉 전체 요인별 상세 분석")
    st.caption("※ 클릭하면 모든 49개 지표의 기여도를 볼 수 있습니다.")

    with st.expander("🔍 전체 지표 기여도 보기 (Click to Open)", expanded=False):
        # 데이터 개수에 따라 높이 자동 조절 (항목당 30px)
        dynamic_height = max(500, len(df_all) * 30)
        
        fig_shap_all = go.Figure(go.Bar(
            y=df_all['name'], 
            x=df_all['shap'], 
            orientation='h',
            marker_color=['#ff5252' if x > 0 else '#2962ff' for x in df_all['shap']], 
            customdata=[db.FEATURE_MAP.get(n, n) for n in df_all['name']],
            hovertemplate="<b>%{customdata}</b> (%{y})<br>기여도: %{x:+.4f}<extra></extra>"
        ))
        
        fig_shap_all.update_layout(
            height=dynamic_height,  
            yaxis=dict(
                dtick=1, 
                categoryorder='total ascending', 
                automargin=True 
            ),
            xaxis_title="부도 위험 기여도 (SHAP Value)",
            margin=dict(l=10, r=10, t=30, b=50)
        )
        st.plotly_chart(fig_shap_all, use_container_width=True)

    # --------------------------------------------------------------------------------
    # 6. Gemini 리포트
    # --------------------------------------------------------------------------------
    st.divider()
    st.subheader("✨ Generative AI 리포트")
    st.info(db.get_gemini_rag_analysis(data, shap_data))