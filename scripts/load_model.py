import os
import sys
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'aladin_dataset.csv')
MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'book_sbert_finetuned')
EMBEDDING_PATH = os.path.join(PROJECT_ROOT, 'embeddings', 'book_embeddings.npy')
ID_PATH = os.path.join(PROJECT_ROOT, 'embeddings', 'book_ids.csv')


def data_preparation(file_path: str) -> tuple[pd.DataFrame, list]:
    """데이터 전처리"""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다: {file_path}")
        return None, None
    
    # 결측치 처리
    df = df.fillna('')
    
    # 텍스트 결합 로직
    def combine_text(row):
        return (
            f"[제목]: {row['title']} "
            f"[저자]: {row['author']} "
            f"[출판사]: {row['publisher']} "
            f"[카테고리]: {row['categoryNames']} "
            f"[설명]: {row['description']}"
        )

    df['combined_text'] = df.apply(combine_text, axis=1)
    texts = df['combined_text'].tolist()
    
    return df, texts


def generate_and_save_embeddings(df_data: pd.DataFrame, combined_texts: list):
    """SBERT 모델을 로드하고 임베딩을 생성하여 관련 파일을 저장"""
    
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"오류: 모델 로드 실패. {e}")
        sys.exit(1)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_DIR)

    # 전체 데이터 임베딩 벡터 생성
    embeddings = model.encode(
        combined_texts, 
        batch_size=64,
        show_progress_bar=True, 
        convert_to_tensor=False 
    )

    # 임베딩 벡터 저장 (API 검색 인덱스 데이터로 사용)
    os.makedirs(os.path.dirname(EMBEDDING_PATH), exist_ok=True)
    np.save(EMBEDDING_PATH, embeddings)

    # 도서 ID 저장 (임베딩 인덱스 매핑 테이블로 사용)
    df_ids = df_data[['itemId']].copy()
    df_ids.to_csv(ID_PATH, index=False)


if __name__ == "__main__":
    # 데이터 로드 및 전처리
    df_data, combined_texts = data_preparation(file_path=DATA_PATH)

    if df_data is None:
        print("데이터 로드 실패로 인해 스크립트를 종료합니다.")
        sys.exit(1)

    # 모델 로드, 임베딩 생성 및 파일 저장
    generate_and_save_embeddings(df_data, combined_texts)