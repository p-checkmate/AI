from fastapi import FastAPI, Depends
from pydantic import BaseModel
from .recommendation_service import AIService

class ActivityBookInput(BaseModel):
    itemId: int
    title: str
    author: str
    publisher: str
    pubDate: str
    description: str
    categoryNames: str

class UserProfileInput(BaseModel):
    genres: list[str]
    # 쿼리 생성에 사용할 최근 활동 기록 (최대 4개)
    recent_activities: list[ActivityBookInput] = [] 
    # 제외 필터링에만 사용할 나머지 과거 기록 ID
    excluded_item_ids: list[int] = []

app = FastAPI()

# 서버 시작 시 AIService 초기화 (단 1회 로드)
ai_service = AIService() 

def get_ai_service():
    """의존성 주입: AIService 객체 반환"""
    return ai_service

@app.post("/api/v1/recommend")
async def recommend_books(
    user_input: UserProfileInput,
    service: AIService = Depends(get_ai_service)
):
    # 쿼리 문장 생성
    query_text = service.create_user_profile_query(user_input.dict())
    
    # 쿼리 임베딩 생성
    query_vector = service.model.encode([query_text], convert_to_tensor=False)[0]
    
    # 유사도 검색
    results = service.find_similar_items(query_vector, k=10, exclude_ids=user_input.excluded_item_ids)
    
    return {
        "recommendations": results
    }