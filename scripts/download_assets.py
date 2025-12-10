import os
import sys
import boto3

BUCKET_NAME = os.getenv("BUCKET_NAME")
S3_MODEL_PREFIX = 'models/book_sbert_finetuned/'
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "/tmp/models/book_sbert_finetuned")

if not all([BUCKET_NAME, S3_MODEL_PREFIX, LOCAL_MODEL_PATH]):
    print("환경 변수가 부족합니다.")
    sys.exit(1) 

s3 = boto3.client('s3')

# 로컬 디렉토리 생성
os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

try:
    # S3에서 객체 목록 가져오기
    objects = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=S3_MODEL_PREFIX)
    
    # 모델 파일이 이미 존재하는지 확인
    test_file = os.path.join(LOCAL_MODEL_PATH, 'model.safetensors')
    if os.path.exists(test_file):
        print(f"모델 파일이 이미 존재하므로 다운로드를 건너뜁니다.")
        sys.exit(0)

    # 객체 목록 순회하며 다운로드
    for content in objects.get('Contents', []):
        s3_key = content['Key']
        
        # S3의 가상 폴더 객체 (슬래시로 끝나는 객체) 건너뛰기
        if s3_key.endswith('/'):
            continue
            
        # 로컬 저장 경로 설정
        relative_key = os.path.relpath(s3_key, S3_MODEL_PREFIX)
        local_path = os.path.join(LOCAL_MODEL_PATH, relative_key)
        
        # 로컬 폴더 생성
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # 파일 다운로드
        s3.download_file(BUCKET_NAME, s3_key, local_path)

except Exception as e:
    print(f"모델 다운로드 실패: {e}")
    sys.exit(1)