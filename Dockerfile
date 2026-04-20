FROM python:3.9-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY sfp-2/requirements_docker.txt /tmp/requirements_docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements_docker.txt

# 复制 sfp-2 代码
COPY sfp-2/src/          /app/sfp-2/src/
COPY sfp-2/scripts/      /app/sfp-2/scripts/
COPY sfp-2/app.py        /app/sfp-2/app.py

# 复制 traindata 下的元数据（构建 NPY 时需要，不会变化）
COPY traindata/station_mapping_dataset.csv  /app/traindata/
COPY traindata/*.xlsx  /app/traindata/

# 创建挂载点目录
RUN mkdir -p /app/sfp-2/datasets/npy \
             /app/sfp-2/results \
             /app/sfp-2/checkpoints \
             /app/traindata

# 环境变量
ENV SFP_TRAINDATA_ROOT=/app/traindata
ENV SFP_NPY_ROOT=/app/sfp-2/datasets/npy
ENV PYTHONUNBUFFERED=1

EXPOSE 9527

WORKDIR /app/sfp-2

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9527"]
