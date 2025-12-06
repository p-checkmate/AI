import os
import requests
from dotenv import load_dotenv
import pandas as pd
import json
import math

load_dotenv()

ttb_key = os.getenv("TTB_KEY")
API_URL = "http://www.aladin.co.kr/ttb/api/ItemList.aspx"

# 수집할 쿼리 타입 목록
QUERY_TYPES = ["BestSeller", "ItemNewAll", "ItemNewSpecial", "ItemEditorChoice"]

# ItemEditorChoice에 사용할 Category ID 목록
EDITOR_CHOICE_CATEGORY_IDS = [1, 170, 172, 336, 222, 987, 798, 1196, 74, 517, 656]

# 한 페이지당 결과 수
MAX_RESULTS = 50

def extract_and_format_item(item: dict) -> dict:
    """API 응답 항목에서 필요한 필드를 추출하고 포맷팅"""
    
    categories = ", ".join(item.get('categoryNames', [])) 
    
    return {
        "itemId": item.get("itemId"),
        "title": item.get("title"),
        "author": item.get("author"),
        "publisher": item.get("publisher"),
        "pubDate": item.get("pubDate"),
        "description": item.get("description"),
        "categoryNames": categories
    }


def fetch_data_page(query_type: str, page: int, category_id: int = None) -> tuple[list, int, int]:
    """단일 페이지를 호출하고 데이터를 추출하며 totalResults를 반환"""
    
    params = {
        "TTBKey": ttb_key, 
        "QueryType": query_type,
        "MaxResults": MAX_RESULTS,
        "Start": page,
        "SearchTarget": "Book",
        "Output": "JS",
        "Version": "20131101"
    }
    
    if category_id is not None:
        params["CategoryId"] = category_id
    
    try:
        response = requests.get(API_URL, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        items = data.get('item', [])
        
        total_results = data.get('totalResults', 0)
        
        extracted_items = [extract_and_format_item(item) for item in items]
        
        return extracted_items, len(extracted_items), total_results

    except requests.exceptions.RequestException as e:
        print(f"API 요청 오류 발생: {e}")
        return [], 0, 0
    except json.JSONDecodeError:
        print("응답 JSON 파싱 오류. TTBKey 및 파라미터를 다시 확인하세요.")
        return [], 0, 0
    except Exception as e:
        print(f"예기치 않은 오류 발생: {e}")
        return [], 0, 0


def collect_data_by_query_type(query_type: str) -> list:
    """주어진 쿼리 타입에 따라 데이터를 수집 (totalResults 기반 전체 수집)"""
    
    collected_books = []
    
    # ItemEditorChoice는 카테고리 순회, 나머지는 단일 쿼리
    if query_type == "ItemEditorChoice":
        categories_to_fetch = EDITOR_CHOICE_CATEGORY_IDS
        if not categories_to_fetch:
             print("ItemEditorChoice 수집을 위한 Category ID 목록이 비어 있습니다.")
             return []
    else:
        categories_to_fetch = [None]

    for category_id in categories_to_fetch:
        
        current_category_count = 0
        total_results = 0
        
        # 첫 페이지 요청 (전체 결과 수 확인)
        items, current_count, total_results = fetch_data_page(query_type, 1, category_id)
        
        if not items:
            continue
            
        collected_books.extend(items)
        current_category_count += current_count
        
        # 필요한 총 페이지 수 계산
        if total_results > current_count:
            remaining_results = total_results - current_count
            total_pages_needed = math.ceil(remaining_results / MAX_RESULTS)
            
            # 2 페이지부터 필요한 페이지 수만큼 반복 요청
            for page in range(2, total_pages_needed + 2):
                items, count_on_page, _ = fetch_data_page(query_type, page, category_id)
                
                if not items:
                    break
                    
                collected_books.extend(items)
                current_category_count += count_on_page
             
    return collected_books


if __name__ == "__main__":
    
    if not ttb_key:
        print("TTB_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
        exit()

    collected_data = []

    for query_type in QUERY_TYPES:
        books = collect_data_by_query_type(query_type)
        collected_data.extend(books)
    
    print("\n" + "=" * 50)
    print(f"총 수집된 책 데이터 (중복 포함): {len(collected_data)}권")

    # DataFrame 생성 및 중복 제거
    df = pd.DataFrame(collected_data)

    df_unique = df.drop_duplicates(subset=['itemId'], keep='first')
    
    file_name = "aladin_dataset.csv"
    df_unique.to_csv(file_name, index=False, encoding='utf-8-sig')

    print("=" * 50)
    print(f"최종 수집된 고유한 책 데이터: {len(df_unique)}권")
    print(f"CSV 파일 저장 완료: {file_name}")