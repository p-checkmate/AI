import os
import numpy as np
import pandas as pd
import boto3
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()

BUCKET_NAME = os.getenv("BUCKET_NAME")
S3_MODEL_PREFIX = 'models/book_sbert_finetuned/'
S3_EMBEDDING_KEY = 'embeddings/book_embeddings.npy'
S3_ID_KEY = 'embeddings/book_ids.csv'
LOCAL_MODEL_PATH = '/tmp/models/book_sbert_finetuned' # 모델 다운로드 위치

class AIService:
    def __init__(self):
        self.model = None
        self.embeddings = None
        self.item_ids = None
        self.tokenizer = None
        self._load_assets()

    def _load_assets(self):
        """API 서버 시작 시 모델과 임베딩 파일을 메모리에 로드"""
        s3 = boto3.client('s3')
        
        # 모델 로드
        try:
            self.model = SentenceTransformer(LOCAL_MODEL_PATH)
            self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
            print("SBERT 모델 메모리 로드 완료.")
        except Exception as e:
            print(f"오류: 모델 로드 실패. {e}")
            raise
            
        # 임베딩 벡터 및 ID 파일 로드
        try:
            response_emb = s3.get_object(Bucket=BUCKET_NAME, Key=S3_EMBEDDING_KEY)
            self.embeddings = np.load(BytesIO(response_emb['Body'].read()))
            
            response_id = s3.get_object(Bucket=BUCKET_NAME, Key=S3_ID_KEY)
            self.item_ids = pd.read_csv(BytesIO(response_id['Body'].read()))['itemId'].tolist()
            
            print(f"임베딩 데이터 메모리 로드 완료. (총 {len(self.item_ids)}개 도서)")
        except Exception as e:
            print(f"오류: 임베딩 데이터 로드 실패. {e}")
            raise


    def get_book_combined_text(self, book_data: Dict[str, Any]) -> str:
        """책 데이터(딕셔너리)를 결합된 텍스트로 변환"""
        return (
            f"[제목]: {book_data.get('title', '')} "
            f"[저자]: {book_data.get('author', '')} "
            f"[출판사]: {book_data.get('publisher', '')} "
            f"[장르]: {book_data.get('categoryNames', '')} "
            f"[설명]: {book_data.get('description', '')}"
        )

    def create_user_profile_query(self, user_data: Dict[str, Any]) -> str:
        """
        사용자 데이터를 종합하여 쿼리 문장 생성 및 장르 가중치 부여.
        최근에 상호작용한 책 정보(recent_activities)를 사용하여 쿼리 벡터를 생성.
        """
        
        recent_books = user_data.get('recent_activities', [])
        
        # 텍스트 결합 및 활동 레이블링
        combined_texts = []
        for book in recent_books:
            text = self.get_book_combined_text(book)
            combined_texts.append(f"[최근활동]: {text}")

        # 장르 가중치 부여 (2배 반복)
        genres = user_data.get('genres', [])
        genre_str = ", ".join(genres)
        weighted_genre_segment = f"[선호장르]: {genre_str} " * 2
        
        # 최종 쿼리 문장 생성
        book_segment = " ".join([f"({i+1}) {text}" for i, text in enumerate(combined_texts)])
        
        final_query = f"[사용자 프로필]: {weighted_genre_segment} [활동]: {book_segment}"
        return final_query

    def find_similar_items(self, query_vector: np.ndarray, k: int = 10, exclude_ids: List[int] = None) -> List[Dict[str, Any]]:
        """
        쿼리 벡터와 메모리에 로드된 전체 임베딩 벡터 간의 코사인 유사도를 계산하여
        가장 유사한 상위 K개의 itemId를 반환하고, exclude_ids 목록은 제외.
        """
        if self.embeddings is None or self.item_ids is None:
            raise Exception("임베딩 데이터가 로드되지 않았습니다.")
            
        # 쿼리 벡터를 2차원 배열로 변환
        query_vector = query_vector.reshape(1, -1)
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(query_vector, self.embeddings)[0]
        
        # 제외할 항목 필터링
        if exclude_ids:
            # 제외할 ID 목록을 인덱스로 변환
            exclude_indices = [self.item_ids.index(i) for i in exclude_ids if i in self.item_ids]
            # 유사도 배열에서 제외할 항목의 점수를 최솟값(-1.0)으로 설정하여 추천에서 제외
            similarities[exclude_indices] = -1.0 
        
        # 유사도 점수 기준 상위 K개 인덱스 찾기
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for index in top_k_indices:
            item_id = self.item_ids[index]
            score = similarities[index]
            results.append({
                "itemId": item_id,
                "similarity_score": float(score)
            })
            
        return results