import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset

class BookDataset(Dataset):
    """
    토크나이징된 데이터를 담는 PyTorch Dataset 클래스
    """
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        # 텐서를 반환
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}

    def __len__(self):
        # 데이터셋의 총 항목 수를 반환
        return len(self.encodings.input_ids)


def data_preparation(file_path: str, model_name: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS", max_length: int = 512):
    """
    도서 데이터를 로드하고 전처리하여 PyTorch Dataset 객체를 반환

    Args:
        file_path (str): CSV 파일 경로
        model_name (str): 사용할 SBERT 모델 이름
        max_length (int): 토크나이징 시 최대 시퀀스 길이

    Returns:
        BookDataset: 모델 훈련에 사용될 PyTorch Dataset 객체
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해 주세요: {file_path}")
        return None
    
    # 결측치 처리
    df = df.fillna('')
    
    # 텍스트 결합
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
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 토크나이징 (인코딩)
    encodings = tokenizer(
        texts,
        padding=True,          # 배치의 모든 문장을 최대 길이로 패딩
        truncation=True,       # 최대 길이 초과 시 자르기
        max_length=max_length, # 최대 길이 설정
        return_tensors="pt"    # PyTorch 텐서 형식으로 반환
    )
    
    # Dataset 객체 생성
    dataset = BookDataset(encodings)
    
    return dataset