# 1. Build Stage
FROM python:3.10-slim AS builder

# 1.1. 시스템 및 환경 변수 설정
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 8000
# 모델 다운로드 경로 (Docker Layer에 캐싱될 경로)
ENV LOCAL_DOWNLOAD_PATH "/app/.local_cache/models/book_sbert_finetuned"

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG BUCKET_NAME
ARG S3_MODEL_PREFIX="models/book_sbert_finetuned"

ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV BUCKET_NAME=${BUCKET_NAME}
ENV S3_MODEL_PREFIX=${S3_MODEL_PREFIX}

WORKDIR /app

# 1.2. Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 1.3. 모델 다운로드 스크립트 복사 및 실행 (캐싱 레이어)
COPY scripts/download_assets.py .
RUN python download_assets.py

# 1.4. 나머지 소스 코드 복사
COPY . .


# 2. Production Stage
FROM python:3.10-slim AS production

# 2.1. 환경 설정 및 작업 디렉토리
WORKDIR /app

# 2.2. Python 의존성 복사
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# 2.3. 소스 코드 및 설정 파일 복사
COPY --from=builder /app /app

# 2.4. 모델 파일 복사 (Docker Layer 캐싱 활용)
COPY --from=builder ${LOCAL_DOWNLOAD_PATH} ${LOCAL_DOWNLOAD_PATH}

# 2.5. 포트 노출
EXPOSE 8000

# 2.6. 애플리케이션 시작
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]