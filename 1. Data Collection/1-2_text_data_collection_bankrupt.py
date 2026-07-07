import OpenDartReader
import pandas as pd
import requests
import zipfile
import io
import re
import time
import datetime
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
# 1. 텍스트 추출 로직 (Unified: Soft Filter 적용)
# ==============================================================================
def fetch_html_from_url(url):
    """
    접속 실패 시 3번까지 자동 재시도 + 차단 방지 헤더
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=3, 
        backoff_factor=1, 
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'
    }

    try:
        r = session.get(url, headers=headers, timeout=20, verify=False)
        if r.status_code != 200:
            print(f"⚠️ [Status {r.status_code}]", end=" ")
            return None
        return r.content
    except Exception as e:
        print(f"❌ [Network Error] Skipping...", end=" ")
        return None

def extract_text_soft(html_content):
    """
    [이식됨] 정상기업용 스크립트에서 가져온 '부드러운' 필터링 로직
    - NLP 학습용으로 문맥을 최대한 살리되, 표(Table) 데이터인 숫자 뭉치는 제거
    - 부도기업용의 'general'보다 텍스트 보존율이 높음
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
        "소재지", "설립일", "대표자", "자본금", "주요사업", "지분율"
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
        if len(l) < 2: continue 
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
        pass 

    return "\n".join(valid_sentences)

def extract_text_general(html_content):
    """
    감사의견 등 짧고 명확한 정보 추출용 (MD&A 외 용도)
    """
    if not html_content: return ""
    soup = BeautifulSoup(html_content, 'lxml')
    text = soup.get_text(separator='\n', strip=True)
    lines = text.split('\n')
    valid_sentences = []
    
    trash_keywords = ["단위 :", "금액", "이자율", "만기일", "보증", "담보", "주식수"]
    bullet_pattern = re.compile(r'^\s*([0-9]+\.|[가-하]\.|[①-⑮]|\(?[\d가-하]+\)|□|■|※|-)')
    ending_pattern = re.compile(r'(다|니다|오|시오|함|음|임|됨)[\.]?\s*$')
    meaningless_pattern = re.compile(r'^[-_ \t\r\n\u3000\.]+$')

    for line in lines:
        l = line.strip()
        if not l: continue
        if meaningless_pattern.match(l): continue

        if ending_pattern.search(l) and len(l) >= 5:
            valid_sentences.append(l)
            continue
        if bullet_pattern.match(l):
            valid_sentences.append(l)
            continue
        if len(l) >= 25:
            valid_sentences.append(l)
            continue

        if re.match(r'^[\d,\.%]+$', l): continue
        if any(tk in l for tk in trash_keywords): continue
        if len(l) < 5: continue
        valid_sentences.append(l)

    return "\n".join(valid_sentences)

def extract_audit_grade_smart(html_content):
    """
    감사의견 등급 추출 (표 내부 우선 탐색 -> 텍스트 탐색)
    """
    if not html_content: return "Missing"
    soup = BeautifulSoup(html_content, 'lxml')
    
    target_grades = {"적정", "부적정", "의견거절", "한정", "한정의견"}
    
    # 1. 테이블 셀 검사
    for td in soup.find_all(['td', 'th'], limit=50):
        cell_text = td.get_text().strip().replace(" ", "")
        if cell_text in target_grades:
            return "한정" if cell_text == "한정의견" else cell_text
        if "감사의견" in cell_text and len(cell_text) < 15:
            if "절" in cell_text: return "의견거절"
            if "부" in cell_text: return "부적정"
            if "한" in cell_text: return "한정"
            if "적" in cell_text: return "적정"

    # 2. 텍스트 패턴 검사 (Fallback)
    text = soup.get_text(separator='\n')
    text = re.sub(r'\s+', ' ', text)
    match = re.search(r'감사의견.{0,20}(의견거절|부적정|한정|적정)', text)
    if match: return match.group(1)

    return "Missing"

# ==============================================================================
# 2-1. [Upgrade] MD&A Fallback 패턴 (순서 중요: MD&A 우선 -> 개요 후순위)
# ==============================================================================
def find_mdna_in_text(text):
    """
    [Priority Logic Applied]
    통파일 검색 시에도 '사업의 개요'가 먼저 걸리지 않도록
    진짜 MD&A 패턴을 먼저 돌리고, 없으면 사업의 개요를 찾습니다.
    """
    # 1순위: 진짜 MD&A 및 상세 분석
    priority_patterns = [
        r"이사의\s*경영진단", 
        r"경영진의\s*분석", 
        r"재무상태\s*및\s*영업실적",
        r"분석\s*의견",
        r"영업\s*실적\s*분석"
    ]
    
    # 2순위: 차선책 (MD&A가 없을 때만)
    secondary_patterns = [
        r"영업의\s*개황",
        r"사업의\s*개요", 
        r"회사의\s*개황"
    ]

    # 1. Priority 패턴 먼저 검색
    for pat in priority_patterns:
        match = re.search(pat, text)
        if match:
            start_idx = match.start()
            # 다음 챕터(IV, V, 숫자 등) 찾기
            search_txt = text[start_idx+100:]
            end_match = re.search(r'\n\s*(IV|V|VI|VII|VIII|IX|X|[1-9])\.\s', search_txt)
            
            if end_match:
                end_idx = start_idx + 100 + end_match.start()
                return text[start_idx:end_idx]
            else:
                return text[start_idx:] 

    # 2. 없으면 Secondary 패턴 검색
    for pat in secondary_patterns:
        match = re.search(pat, text)
        if match:
            start_idx = match.start()
            search_txt = text[start_idx+100:]
            end_match = re.search(r'\n\s*(IV|V|VI|VII|VIII|IX|X|[1-9])\.\s', search_txt)
            
            if end_match:
                end_idx = start_idx + 100 + end_match.start()
                return text[start_idx:end_idx]
            else:
                return text[start_idx:]

    return None

# ==============================================================================
# 2-2. [Hybrid] 데이터 수집 Orchestrator (2단계 검색 적용)
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
            
            # [1] 감사의견 (변동 없음)
            audit_kw = ["감사의견", "외부감사", "회계감사", "감사보고서", "종합의견", "감사결과"]
            for _, row in toc.iterrows():
                title_clean = row['title'].replace(" ", "")
                if any(k in title_clean for k in audit_kw):
                    content = fetch_html_from_url(row['url'])
                    data['감사의견_등급'] = extract_audit_grade_smart(content)
                    data['감사의견_전문'] = extract_text_general(content)
                    break
            
            # [2] MD&A (수정됨: 2단계 검색)
            
            # (1단계) 진짜 MD&A 키워드
            mdna_priority_kw = ["이사의경영진단", "경영진의분석", "분석의견", "재무상태및영업실적"]
            
            # (2단계) 차선책 키워드
            mdna_fallback_kw = ["영업의개황", "경영실적", "사업의개요", "사업의내용"]

            # Step 1: Priority Loop
            for _, row in toc.iterrows():
                title_clean = row['title'].replace(" ", "")
                # 여기서 '사업의 개요'는 걸러지지 않음 (리스트에 없으므로)
                if any(k in title_clean for k in mdna_priority_kw):
                    content = fetch_html_from_url(row['url'])
                    extracted = extract_text_soft(content)
                    if len(extracted) > 100:
                        data['MD&A'] = extracted
                        break # 진짜를 찾았으니 루프 종료
            
            # Step 2: Fallback Loop (진짜를 못 찾았을 때만 실행)
            if data['MD&A'] == 'Missing':
                for _, row in toc.iterrows():
                    title_clean = row['title'].replace(" ", "")
                    if any(k in title_clean for k in mdna_fallback_kw):
                        content = fetch_html_from_url(row['url'])
                        extracted = extract_text_soft(content)
                        if len(extracted) > 100:
                            data['MD&A'] = extracted
                            break

            # [3] 기타 보호사항 (변동 없음)
            other_kw = ["그밖에투자자", "기타필요한", "기타사항", "우발채무", "약정사항", "소송사건", "제재현황"]
            for _, row in toc.iterrows():
                title_clean = row['title'].replace(" ", "")
                if any(k in title_clean for k in other_kw):
                    content = fetch_html_from_url(row['url'])
                    data['기타_보호사항'] = extract_text_soft(content)
                    break
                    
    except: pass

    # --- Phase B: 실패 시 ZIP 파일 전수조사 ---
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
                            data['감사의견_등급'] = extract_audit_grade_smart(raw_html)
                            data['감사의견_전문'] = extract_text_general(raw_html)

                    # 2. MD&A (수정됨: 함수 호출로 Priority 로직 적용)
                    if data['MD&A'] == 'Missing':
                        # 위에서 수정한 find_mdna_in_text 함수를 사용
                        # 이 함수 안에 Priority -> Fallback 순서가 이미 내장되어 있음
                        found_mdna = find_mdna_in_text(soup_text)
                        
                        if found_mdna:
                            data['MD&A'] = extract_text_soft(found_mdna)
                        
                    if data['MD&A'] != 'Missing' and data['감사의견_등급'] != 'Missing': break
        except: pass

    return data

# ==============================================================================
# 3. 유틸리티 (연도 검색 - 부도기업용 로직 유지)
# ==============================================================================
def get_all_report_years(corp_code):
    """
    [Rescue Mode] 2015년부터 현재까지 연도별 보고서 수집
    - 로직: 사업보고서 우선 -> 없으면 감사보고서(반기/분기 제외)
    - 특징: 거래정지/상장폐지 위기 기업의 '감사보고서'만 제출된 케이스도 포착
    """
    report_map = {}
    current_year = datetime.datetime.now().year
    
    for year in range(2015, current_year):
        try:
            start_date = str(year) + "0101"
            end_date = str(year + 1) + "0731"
            
            # kind='A' 미사용 -> 수시공시 포함 전체 검색
            lst = dart.list(corp_code, start=start_date, end=end_date, final=False)
            if lst is None or lst.empty: continue
            
            found_rcept_no = None
            found_title = None
            
            # Step 1: 사업보고서
            business = lst[lst["report_nm"].str.contains("사업보고서", na=False)]
            if not business.empty:
                latest = business.sort_values(by='rcept_dt', ascending=False).iloc[0]
                found_rcept_no = latest["rcept_no"]
                found_title = latest["report_nm"]
            
            # Step 2: 감사보고서 Fallback (부도기업용 핵심)
            else:
                audit = lst[lst["report_nm"].str.contains("감사보고서", na=False)]
                audit = audit[~audit["report_nm"].str.contains("반기|분기|중간", na=False)]
                
                if not audit.empty:
                    latest = audit.sort_values(by='rcept_dt', ascending=False).iloc[0]
                    found_rcept_no = latest["rcept_no"]
                    found_title = latest["report_nm"]
            
            if found_rcept_no:
                report_map[year] = {'rcept_no': found_rcept_no, 'title': found_title}
                
        except: continue
            
    return report_map

# ==============================================================================
# 4. Main Execution
# ==============================================================================
def main():
    # 파일 설정
    input_file = "관리종목_상장폐지_업종포함.csv" 
    output_file = "부도기업_진짜다시.csv" # 파일명 변경
    
    print(f"📂 파일 로딩: {input_file}")
    if not os.path.exists(input_file):
        print("❌ 입력 파일이 없습니다.")
        return

    try: df_input = pd.read_csv(input_file, encoding='utf-8')
    except: 
        try: df_input = pd.read_csv(input_file, encoding='cp949')
        except: return

    print(f"📊 입력 데이터: {len(df_input)}개 기업 로드됨")

    try: dart_codes = dart.corp_codes
    except: 
        print("❌ DART API 키 확인 필요")
        return

    df_input['stock_code_clean'] = df_input['종목코드'].astype(str).str.zfill(6)
    df_merged = pd.merge(df_input, dart_codes[['corp_code', 'stock_code']], 
                        left_on='stock_code_clean', right_on='stock_code', how='left')
    
    targets = df_merged[df_merged['corp_code'].notnull()].copy()
    print(f"✅ 매핑 완료: {len(targets)}개 기업 분석 시작")

    processed_keys = set()
    if os.path.exists(output_file):
        try:
            existing = pd.read_csv(output_file)
            processed_keys = set(existing['corp_code'].astype(str).str.zfill(8) + "_" + existing['target_year'].astype(str))
            print(f"🔄 기존 {len(processed_keys)}개 데이터 발견 (Skip)")
        except: pass

    temp_rows = []
    
    for idx, row in targets.iterrows():
        code = row['corp_code']
        name = row['회사명']
        event_date = str(row['발생일자'])
        
        print(f"\n[{idx+1}/{len(targets)}] {name} ({code})", end=" ")

        report_map = get_all_report_years(code) 
        available_years = sorted(report_map.keys(), reverse=True)
        
        if not available_years:
            print("-> ❌ 없음", end="")
            continue

        print(f"-> 📅 {len(available_years)}개", end=" ")

        for y in available_years:
            if f"{code}_{y}" in processed_keys:
                print(".", end="")
                continue
            
            rcept_no = report_map[y]['rcept_no']
            report_title = report_map[y]['title']

            row_data = {
                'corp_code': code, 
                'corp_name': name, 
                'stock_code': row['stock_code_clean'],
                'target_year': y, 
                'event_date': event_date,
                'category': row['구분'], 
                'reason': row['reason'] if 'reason' in row else row.get('사유', ''),
                'industry': row['업종'], 
                'rcept_no': rcept_no, 
                'report_title': report_title
            }
            
            print(f"[{y}:O]", end="")
            extracted = get_report_data_hybrid(rcept_no)
            row_data.update(extracted)
            row_data['report_found'] = True
            
            temp_rows.append(row_data)
            time.sleep(0.5) 

        if len(temp_rows) > 0 and (idx + 1) % 5 == 0:
            pd.DataFrame(temp_rows).to_csv(output_file, index=False, encoding='utf-8-sig', mode='a', header=not os.path.exists(output_file))
            temp_rows = []
            print(" 💾", end="")

    if temp_rows:
        pd.DataFrame(temp_rows).to_csv(output_file, index=False, encoding='utf-8-sig', mode='a', header=not os.path.exists(output_file))

    print("\n🎉 완료!")

if __name__ == "__main__":
    main()