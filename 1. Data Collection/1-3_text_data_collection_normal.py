import OpenDartReader
import pandas as pd
import requests
import zipfile
import io
import re
import time
import os
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
from dotenv import load_dotenv
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 경고 무시
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ==============================================================================
# 0. API 키 설정
# ==============================================================================
load_dotenv()  # 같은 폴더의 .env 파일을 읽어서 환경변수로 등록

API_KEY = os.environ.get("DART_API_KEY")
if not API_KEY:
    raise RuntimeError("환경변수 DART_API_KEY가 설정되어 있지 않습니다. (.env 파일에 DART_API_KEY=발급받은키 추가)")
dart = OpenDartReader(API_KEY)

# ==============================================================================
# 1. 텍스트 추출 로직 (V5: 표 내용 포함 + 문장형 필터링)
# ==============================================================================
def fetch_html_from_url(url):
    """
    [강화된 버전] 접속 실패 시 3번까지 자동 재시도 + 차단 방지 헤더
    """
    # 1. 세션 생성
    session = requests.Session()
    
    # 2. 재시도 전략 설정 (총 3회, 실패 시 1초, 2초, 4초 대기 후 재시도)
    retry_strategy = Retry(
        total=3, 
        backoff_factor=1, 
        status_forcelist=[429, 500, 502, 503, 504], # 서버 에러 시 재시도
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # 3. 헤더 설정 (봇 탐지 회피)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'
    }

    try:
        # verify=False는 SSL 인증서 에러 무시 (DART 구형 서버 호환성 위해)
        r = session.get(url, headers=headers, timeout=20, verify=False)
        
        if r.status_code != 200:
            print(f"⚠️ [Status {r.status_code}]", end=" ")
            return None
            
        return r.content
        
    except Exception as e:
        # 재시도까지 다 했는데도 실패하면 그냥 넘어가도록 처리
        # (로그가 너무 지저분해지지 않게 짧게 출력)
        print(f"❌ [Network Error] Skipping...", end=" ")
        return None

def extract_sentences_from_html(html_content):
    """
    [V5 Core] 표를 지우지 않고 텍스트를 가져온 뒤, 
    '10자 이상' + '서술형 종결' 문장만 남기는 필터링
    """
    if not html_content: return ""
    
    soup = BeautifulSoup(html_content, 'lxml')
    text = soup.get_text(separator='\n', strip=True)
    
    lines = text.split('\n')
    valid_sentences = []
    
    # 문장 종결 패턴 (다, 오, 함, 음, 임, 니다)
    ending_pattern = re.compile(r'(다|니다|오|시오|함|음|임)[\.]?\s*$')
    
    for line in lines:
        l = line.strip()
        l = re.sub(r'\s+', ' ', l) # 공백 정리
        
        if not l: continue
        
        # [핵심] 10자 이상 & 서술형 어미 -> 유효 문장으로 판단
        if len(l) >= 10 and ending_pattern.search(l):
            valid_sentences.append(l)
            
    return "\n".join(valid_sentences)

def extract_text_soft(html_content):
    """
    [Balanced V9] MD&A 및 기타사항 전용
    - 'soft'의 과도한 쓰레기 수집 방지 + 'general'의 과잉 삭제 방지
    - 전략: 
        1. 긴 문장(30자↑)은 무조건 수집 (표 내용일 확률 낮음)
        2. 불렛포인트(1., -)는 짧아도 수집 (개조식 문장)
        3. 그 외 짧은 줄은 '서술어'가 있어야만 수집 (단순 명사 삭제)
    """
    if not html_content: return ""
    
    soup = BeautifulSoup(html_content, 'lxml')
    text = soup.get_text(separator='\n', strip=True)
    
    lines = text.split('\n')
    valid_sentences = []
    
    # 1. 쓰레기 키워드 (표 헤더, 단위 등)
    trash_keywords = [
        "단위 :", "(단위", "단위:", "원)", "천원)", "백만원)", 
        "참조)", "출처:", "주석", "해당사항", "없음", "평균", "누적",
        "소재지", "설립일", "대표자", "자본금", "주요사업"
    ]
    
    # 2. 불렛포인트 패턴 (개조식 문장 인식)
    bullet_pattern = re.compile(r'^\s*([0-9]+\.|[가-하]\.|[①-⑮]|\(?[\d가-하]+\)|□|■|※|-|o|\*)')
    
    # 3. 문장 종결 패턴 (서술어)
    ending_pattern = re.compile(r'(다|니다|오|시오|함|음|임|됨|것|점)[\.]?\s*$')
    
    # 4. 순수 숫자/기호 패턴
    number_pattern = re.compile(r'^[\d,\.\-%]+$')

    for line in lines:
        l = line.strip()
        if not l: continue
        
        # [Step 1] 기본 필터링 (명백한 쓰레기)
        if len(l) < 2: continue # 1글자는 버림
        if number_pattern.match(l): continue # 숫자만 있는 줄 버림
        if any(tk in l for tk in trash_keywords): continue # 쓰레기 키워드 버림

        # [Step 2] 면제권 부여 (이 조건이면 짧아도/서술어 없어도 수집)
        
        # A. 긴 문장 (30자 이상) -> 표의 셀 내용이 이렇게 길 확률은 낮음
        if len(l) >= 30:
            valid_sentences.append(l)
            continue
            
        # B. 불렛 포인트로 시작하는 줄 -> "1. 시장 점유율 확대" (수집)
        if bullet_pattern.match(l):
            valid_sentences.append(l)
            continue
            
        # C. 서술어로 끝나는 문장 -> "매출이 증가함" (수집)
        if ending_pattern.search(l):
            valid_sentences.append(l)
            continue
            
        # [Step 3] 탈락
        # 위 3가지(긴글, 불렛, 서술어)에 해당 안 되는 '애매한 짧은 명사'는 과감히 버림
        # 예: "매출액", "영업이익", "삼성전자", "서울시 강남구" -> 삭제됨
        pass 

    return "\n".join(valid_sentences)

def extract_text_general(html_content):
    """
    [V8 Final] 과잉 삭제 방지 로직
    1. '서술어(~다)'나 '불렛포인트'가 있으면 무조건 수집 (Immunity)
    2. 그 후에 남은 '짧은 단어' 중에서 쓰레기 키워드 포함 시 삭제
    """
    if not html_content: return ""
    
    soup = BeautifulSoup(html_content, 'lxml')
    text = soup.get_text(separator='\n', strip=True)
    
    lines = text.split('\n')
    valid_sentences = []
    
    # 1. 표 헤더로 의심되는 키워드들
    trash_keywords = [
        "단위 :", "(단위", "단위:", "원)", "천원)", "백만원)", 
        "계정과목", "제 (", "제(", "주석", "구분", "비고", "금액",
        "금융기관", "약정", "대출", "잔액", "이자율", "만기일", "보증", "담보",
        "지분율", "주식수", "취득", "처분", "소재지", "회사명", "대표자",
        "기초", "기말", "증감", "내역", "비율", "평가", "등급", "일자"
    ]
    
    # 2. 불렛포인트 패턴
    bullet_pattern = re.compile(r'^\s*([0-9]+\.|[가-하]\.|[①-⑮]|\(?[\d가-하]+\)|□|■|※|-)')
    
    # 3. 서술형 종결 패턴 (~다, ~음, ~함)
    ending_pattern = re.compile(r'(다|니다|오|시오|함|음|임|됨)[\.]?\s*$')
    
    # 4. 무의미한 기호 라인
    meaningless_pattern = re.compile(r'^[-_ \t\r\n\u3000\.]+$')

    for line in lines:
        l = line.strip()
        if not l: continue
        
        # [Filter 0] 기호만 있는 줄은 즉시 삭제
        if meaningless_pattern.match(l): continue

        # ==================================================================
        # [Priority 1] 면제권 부여 (이 조건이면 쓰레기 단어 있어도 수집)
        # ==================================================================
        
        # A. 서술어로 끝나는 문장 (길이 5자 이상) -> "대출을 받음." (수집 O)
        if ending_pattern.search(l) and len(l) >= 5:
            valid_sentences.append(l)
            continue # 수집했으니 다음 라인으로
            
        # B. 불렛포인트로 시작하는 항목 -> "1. 대출 약정" (수집 O)
        if bullet_pattern.match(l):
            valid_sentences.append(l)
            continue
            
        # C. 길이가 아주 긴 문장 (25자 이상) -> 중간에 잘린 줄글일 확률 높음
        if len(l) >= 25:
            valid_sentences.append(l)
            continue

        # ==================================================================
        # [Priority 2] 남은 찌꺼기 중에서 쓰레기 처리
        # ==================================================================
        
        # 위 면제권을 못 받은 '짧고', '서술어 없고', '불렛 없는' 녀석들 중...
        
        # D. 숫자만 있는 줄 -> 삭제 (표 데이터)
        if re.match(r'^[\d,\.%]+$', l): continue
        
        # E. 쓰레기 키워드가 포함된 줄 -> 삭제 (표 헤더)
        # 예: "금융기관", "약정 금액"
        if any(tk in l for tk in trash_keywords): continue
        
        # F. 너무 짧은 명사 -> 삭제
        if len(l) < 5: continue
        
        # 여기까지 살아남은 건 제목(Header)일 가능성이 있음 -> 수집
        # 예: "경영 방침", "주요 현황"
        valid_sentences.append(l)

    return "\n".join(valid_sentences)

def extract_audit_grade_smart(html_content):
    """
    [Logic]
    1. HTML의 모든 테이블 셀(td)을 순서대로 훑는다. (상단 표 우선)
    2. 셀 내용이 정확히 '적정', '한정', '부적정', '의견거절' 중 하나라면 즉시 반환한다.
    3. 표에서 못 찾으면, 차선책으로 텍스트 패턴을 찾는다.
    """
    if not html_content: return "Missing"
    
    soup = BeautifulSoup(html_content, 'lxml') # 속도를 위해 lxml 추천
    
    # ----------------------------------------------------------
    # 전략 1: 테이블(td) 안의 정확한 단어 매칭 (가장 정확함)
    # 감사의견 요약표는 보통 문서 최상단에 위치하므로, 발견 즉시 리턴하면 됨
    # ----------------------------------------------------------
    target_grades = {"적정", "부적정", "의견거절", "한정", "한정의견"}
    
    for td in soup.find_all(['td', 'th'], limit=50): # 상단 50개 셀만 검사 (속도 UP)
        # 공백 제거 및 정리
        cell_text = td.get_text().strip().replace(" ", "")
        
        # 1. 셀 내용이 딱 감사의견 등급만 있는 경우 (예: <td> 적정 </td>)
        if cell_text in target_grades:
            return "한정" if cell_text == "한정의견" else cell_text
            
        # 2. "감사의견: 적정" 처럼 같이 들어있는 경우
        if "감사의견" in cell_text and len(cell_text) < 15: # 너무 긴 문장은 제외
            if "절" in cell_text: return "의견거절"
            if "부" in cell_text: return "부적정"
            if "한" in cell_text: return "한정"
            if "적" in cell_text: return "적정"

    # ----------------------------------------------------------
    # 전략 2: 표가 깨졌거나 텍스트로만 된 경우 (Fallback)
    # 이때는 "적정"이라는 단어가 "감사의견"이라는 단어 근처에 있는지 봐야 함
    # ----------------------------------------------------------
    text = soup.get_text(separator='\n')
    text = re.sub(r'\s+', ' ', text) # 한 줄로 정리

    # 정규식: "감사의견" 뒤에 20자 이내에 "적정/한정/..."이 오는지 확인
    # (단순히 문서 어딘가에 있는 '적정 가치' 등을 피하기 위함)
    match = re.search(r'감사의견.{0,20}(의견거절|부적정|한정|적정)', text)
    
    if match:
        return match.group(1)

    return "Missing"

# ==============================================================================
# 2-1. [Upgrade] MD&A Fallback 패턴 강화 (K-GAAP 대응)
# ==============================================================================
def find_mdna_in_text(text):
    """
    [Modified] 통파일에서 MD&A 영역을 더 투박하게 뜯어내는 로직
    """
    # 1. 시작점 패턴 (우선순위)
    start_patterns = [
        r"이사의\s*경영진단", 
        r"경영진의\s*분석",
        r"재무상태\s*및\s*영업실적",
        r"영업의\s*개황"
    ]
    
    start_idx = -1
    for pat in start_patterns:
        match = re.search(pat, text)
        if match:
            start_idx = match.start()
            break
            
    if start_idx == -1: return None

    # 2. 끝점 찾기 (다음 큰 목차 찾기)
    # 로마자(IV, V, VI)나 숫자 목차(4. 5.)가 나오면 거기서 끊음
    search_txt = text[start_idx+50:] # 제목 건너뛰고 검색
    
    # 다음 챕터 패턴: 줄바꿈 뒤에 "로마자." 혹은 "숫자." 패턴이 오면 다음 챕터로 간주
    end_match = re.search(r'\n\s*(IV|V|VI|VII|VIII|IX|X|[1-9])\.', search_txt)
    
    if end_match:
        end_idx = start_idx + 50 + end_match.start()
        return text[start_idx:end_idx]
    else:
        # 못 찾으면 그냥 끝까지 (필터가 알아서 표는 지워줌)
        return text[start_idx:]

# ==============================================================================
# 2-2. [Upgrade] 하이브리드 데이터 수집 (사전 대폭 강화)
# ==============================================================================
def get_report_data_hybrid(rcept_no):
    data = {
        '감사의견_등급': 'Missing',
        '감사의견_전문': 'Missing',
        'MD&A': 'Missing',
        '기타_보호사항': 'Missing'
    }

    # --- Phase A: 목차(TOC) 기반 시도 ---
    try:
        toc = dart.sub_docs(rcept_no)
        if toc is not None and not toc.empty:
            
            # [1] 감사의견 (키워드 단순화)
            # "감사의견", "외부감사" 등 핵심 단어 하나만 있어도 OK
            audit_kw = ["감사의견", "외부감사", "회계감사", "감사보고서", "종합의견"]
            
            for _, row in toc.iterrows():
                # [수정 1] 강력한 공백 제거 (특수문자 \xa0 등 모두 제거)
                title_clean = re.sub(r'\s+', '', str(row['title']))
                
                if any(k in title_clean for k in audit_kw):
                    content = fetch_html_from_url(row['url'])
                    grade = extract_audit_grade_smart(content)
                    if grade != "Missing":
                        data['감사의견_등급'] = grade
                        data['감사의견_전문'] = extract_text_general(content)
                        break
            
            # [2] MD&A (키워드 쪼개기 & 범위 확장)
            # 긴 단어 대신 짧은 핵심 단어로 매칭 확률 극대화
            mdna_kw_priority = ["경영진단", "경영진의분석", "분석의견", "영업의개황", "재무상태및영업실적"]
            mdna_kw_secondary = ["사업의내용", "사업의개요", "영업보고서", "영업의현황", "영업실적"]
            
            # 1순위: MD&A 전용 섹션 탐색
            for _, row in toc.iterrows():
                title_clean = re.sub(r'\s+', '', str(row['title']))
                if any(k in title_clean for k in mdna_kw_priority):
                    content = fetch_html_from_url(row['url'])
                    extracted = extract_text_soft(content)
                    if len(extracted) > 100:
                        data['MD&A'] = extracted
                        break
            
            # 2순위: 없으면 '사업의 내용'이나 '영업보고서'라도 긁음 (금융/서비스업 대비)
            if data['MD&A'] == 'Missing':
                for _, row in toc.iterrows():
                    title_clean = re.sub(r'\s+', '', str(row['title']))
                    if any(k in title_clean for k in mdna_kw_secondary):
                        content = fetch_html_from_url(row['url'])
                        extracted = extract_text_soft(content)
                        # 사업의 내용은 워낙 기니까, 너무 짧으면(목차만 있는 경우) 무시
                        if len(extracted) > 200:
                            data['MD&A'] = extracted
                            break

            # [3] 기타 보호사항
            other_kw = ["그밖에투자자", "기타필요한", "제재현황", "기타사항", "우발채무", "소송"]
            for _, row in toc.iterrows():
                title_clean = re.sub(r'\s+', '', str(row['title']))
                if any(k in title_clean for k in other_kw):
                    content = fetch_html_from_url(row['url'])
                    data['기타_보호사항'] = extract_text_soft(content)
                    break
                    
    except: pass

    # --- Phase B: 통파일(ZIP) 전수조사 (목차 실패 시 Backup) ---
    if data['MD&A'] == 'Missing' or data['감사의견_등급'] == 'Missing':
        try:
            url = "https://opendart.fss.or.kr/api/document.xml"
            r = requests.get(url, params={'crtfc_key': API_KEY, 'rcept_no': rcept_no}, timeout=60)
            
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                for fname in zf.namelist():
                    if not fname.lower().endswith(('.xml', '.htm', '.html')): continue
                    
                    raw_bytes = zf.read(fname)
                    try: raw_html = raw_bytes.decode('utf-8')
                    except: raw_html = raw_bytes.decode('euc-kr', errors='ignore')
                    
                    soup_text = BeautifulSoup(raw_html, 'lxml').get_text('\n')

                    # 1. 감사의견
                    if data['감사의견_등급'] == 'Missing':
                        if "감사" in fname or "Audit" in fname or "감사의견" in raw_html[:3000]:
                            grade = extract_audit_grade_smart(raw_html)
                            if grade != "Missing":
                                data['감사의견_등급'] = grade
                                data['감사의견_전문'] = extract_text_general(raw_html)

                    # 2. MD&A (패턴 매칭도 공백 유연하게 대응)
                    if data['MD&A'] == 'Missing':
                        # 정규식 패턴에 \s*를 많이 넣어서 공백/특수문자 무시하도록 설정
                        patterns = [
                            r"이사의\s*경영진단", r"경영진의\s*분석", r"분석\s*의견",
                            r"재무상태\s*및\s*영업실적", r"영업의\s*개황", 
                            r"영업\s*보고서", r"사업의\s*개요", r"사업의\s*내용"
                        ]
                        for pat in patterns:
                            match = re.search(pat, soup_text)
                            if match:
                                start = match.start()
                                # 다음 챕터 찾기 (로마자 or 숫자)
                                end_match = re.search(r'\n\s*(IV|V|VI|VII|VIII|IX|X|[1-9])\.', soup_text[start+100:])
                                end = start + 100 + end_match.start() if end_match else len(soup_text)
                                
                                extracted = extract_text_soft(soup_text[start:end])
                                if len(extracted) > 200:
                                    data['MD&A'] = extracted
                                    break
                        
                    if data['MD&A'] != 'Missing' and data['감사의견_등급'] != 'Missing': break
        except: pass

    return data

# ==============================================================================
# 3. [NEW] 전체 연도 자동 탐색 함수 (정상기업용 핵심)
# ==============================================================================
def get_all_report_years(corp_code):
    """
    [수정됨] 이 기업이 DART에 제출한 사업보고서 중 
    '2015년 이후'의 연도(YYYY) 리스트만 반환
    """
    report_map = {} 
    try:
        # 1. API 요청 범위를 2015년 1월 1일 이후로 설정 (속도 향상)
        lst = dart.list(corp_code, start='20150101', kind='A', final=False)
        if lst is None or lst.empty: return {}
        
        # 사업보고서만 필터링
        hit = lst[lst["report_nm"].str.contains("사업보고서", na=False)]
        
        for _, row in hit.iterrows():
            title = row['report_nm'] # 예: "제 25 기 사업보고서 (2023.12)"
            rcept_no = row['rcept_no']
            
            # 연도 추출 (괄호 안의 숫자 4자리)
            match = re.search(r'\((\d{4})\.', title)
            if match:
                year = int(match.group(1))
                
                # [필터 추가] 2015년 미만인 경우 제외
                if year < 2015: continue
                
                # 같은 연도에 수정보고서가 여러 개면, 가장 최신 것(접수번호 큰 것)만 유지
                if year not in report_map or rcept_no > report_map[year]['rcept_no']:
                    report_map[year] = {'rcept_no': rcept_no, 'title': title}
                    
    except: pass
    return report_map

# ==============================================================================
# 4. Main Execution (정상기업용)
# ==============================================================================
def main():
    # [설정] 파일명
    input_file = "master_table_cleaned(2).csv"
    
    print(f"📂 파일 로딩: {input_file}")
    if not os.path.exists(input_file):
        print("❌ 입력 파일이 없습니다.")
        return

    # 1. 파일 로드
    if input_file.endswith('.xlsx'):
        df_input = pd.read_excel(input_file)
    else:
        try: df_input = pd.read_csv(input_file, encoding='utf-8')
        except: df_input = pd.read_csv(input_file, encoding='cp949')

    print(f"📊 입력 데이터: {len(df_input)}개 기업 로드됨")

    # 2. DART 코드 매핑
    print("⏳ DART 기업코드 매핑 중...")
    try: dart_codes = dart.corp_codes
    except: 
        print("❌ API 호출 실패. 키를 확인하세요.")
        return

    # 종목코드 컬럼명 자동 탐지
    possible_cols = ['Symbol', '종목코드', 'Code', 'Ticker', '단축코드']
    stock_col = next((c for c in possible_cols if c in df_input.columns), None)
    
    if not stock_col:
        print(f"❌ 종목코드 컬럼을 찾을 수 없습니다. (현재 컬럼: {df_input.columns.tolist()})")
        return

    # 종목코드 6자리 문자열로 변환
    df_input['stock_code_clean'] = df_input[stock_col].astype(str).str.zfill(6)
    
    # DART의 'corp_name'을 같이 Merge
    df_merged = pd.merge(df_input, dart_codes[['corp_code', 'stock_code', 'corp_name']], 
                        left_on='stock_code_clean', right_on='stock_code', how='left')
    
    # ==========================================================================
    # [설정] 분할 실행 설정
    # ==========================================================================
    BATCH_SIZE = 449       
    BATCH_NUM = 6          
    # ==========================================================================

    # 1. 매핑 성공한 전체 리스트 확보
    full_targets = df_merged[df_merged['corp_code'].notnull()]
    
    # 2. 슬라이싱 계산
    start_idx = (BATCH_NUM - 1) * BATCH_SIZE
    end_idx = BATCH_NUM * BATCH_SIZE
    
    # 3. 해당 구간만 선택 & 인덱스 리셋
    targets = full_targets.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    
    output_file = f"정상기업_크롤링_Batch_{BATCH_NUM}_new.csv" 
    
    print(f"🚀 분석 시작: Batch {BATCH_NUM}번 ({start_idx} ~ {end_idx} 구간)")
    print(f"   -> 대상 기업 수: {len(targets)}개")

    # ==========================================================================
    # [Logic Fix 1] 이어하기 로직 강화 (타입 불일치 해결)
    # ==========================================================================
    processed_keys = set()
    if os.path.exists(output_file):
        try:
            # 1. 파일 읽기
            existing = pd.read_csv(output_file)
            
            # 2. corp_code가 숫자로 읽혔을 경우를 대비해 문자열 변환 및 0 채우기
            existing['corp_code'] = existing['corp_code'].astype(str).str.zfill(8)
            existing['target_year'] = existing['target_year'].astype(str)
            
            # 3. 키 생성
            processed_keys = set(existing['corp_code'] + "_" + existing['target_year'])
            print(f"🔄 기존 {len(processed_keys)}개 데이터 발견 (Skip)")
        except Exception as e:
            print(f"⚠️ 기존 파일 읽기 실패 (새로 시작): {e}")

    # 4. 크롤링 루프
    temp_rows = []
    
    for idx, row in targets.iterrows():
        code = row['corp_code'] # 이미 문자열(8자리) 상태임
        
        name = row['corp_name']
        if pd.isna(name): name = "Unknown"

        industry = row.get('섹터대분류', row.get('Industry', row.get('업종', row.get('Sector', ''))))
        
        print(f"\n[{idx+1}/{len(targets)}] {name} ({code}) 연도 탐색...", end=" ")
        
        # 전체 연도 리스트 가져오기
        report_map = get_all_report_years(code)
        
        if not report_map:
            print("-> ❌ 없음", end="")
            continue
            
        years = sorted(report_map.keys(), reverse=True)
        print(f"-> 📅 {len(years)}개 ({min(years)}~{max(years)})", end=" ")

        for y in years:
            # [Logic Fix] 정확한 키 매칭
            current_key = f"{code}_{y}"
            
            if current_key in processed_keys:
                print(".", end="")
                continue
                
            info = report_map[y]
            rcept_no = info['rcept_no']
            title = info['title']
            
            row_data = {
                'corp_code': code, 
                'corp_name': name, 
                'stock_code': row['stock_code_clean'],
                'target_year': y, 
                'industry': industry,
                'rcept_no': rcept_no, 
                'report_title': title
            }
            
            # [Logic Fix 3] Sleep 최적화: 요청 직전에 한 번만 (혹은 요청 함수 안에서 처리)
            # 여기서는 API 호출 안정성을 위해 호출 직전에 짧게 쉽니다.
            time.sleep(0.5) 
            
            extracted = get_report_data_hybrid(rcept_no)
            row_data.update(extracted)
            row_data['report_found'] = True
            
            temp_rows.append(row_data)
            print(f"[{y}:O]", end="")
            
            # [Logic Fix 2] 저장 로직 개선: 데이터가 10개 쌓이면 즉시 저장 (안정성 확보)
            if len(temp_rows) >= 10:
                pd.DataFrame(temp_rows).to_csv(output_file, index=False, encoding='utf-8-sig', mode='a', header=not os.path.exists(output_file))
                temp_rows = [] # 버퍼 비우기
                print(" 💾", end="")

    # 남은 데이터 저장
    if temp_rows:
        pd.DataFrame(temp_rows).to_csv(output_file, index=False, encoding='utf-8-sig', mode='a', header=not os.path.exists(output_file))
        print(" 💾 Final Save", end="")

    print("\n🎉 완료!")
    
if __name__ == "__main__":
    main()