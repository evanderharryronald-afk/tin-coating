# tin-coating 生产模型容器
#
# 构建:
#   docker build -t tin-coating:latest .
#
# 或者直接 docker compose:
#   docker compose build && docker compose run --rm app pytest tests/test_smoke.py -v

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TIN_COATING_ROOT=/app

WORKDIR /app

# 系统依赖（openpyxl 需要 libxml2，sklearn 需要编译工具链）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
        gfortran \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖（docker cache 友好）
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install -e . pytest && \
    pip install optuna matplotlib seaborn pyyaml openpyxl joblib scikit-learn pandas numpy

# 再装代码
COPY . .

# CLI 和测试都从 Task1/ 根目录触发
WORKDIR /app/Task1

# 默认跑 smoke test
CMD ["python", "-m", "pytest", "tests/test_smoke.py", "-v"]
