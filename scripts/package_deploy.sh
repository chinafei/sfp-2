#!/bin/bash
# ============================================================
# SFP-2 部署包打包脚本
# 在本地（开发机）运行，生成 sfp2-deploy.tar.gz
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SFP2_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$SFP2_DIR")"
DEPLOY_DIR="/tmp/sfp2-deploy"
OUTPUT="$PROJECT_DIR/sfp2-deploy.tar.gz"

echo "============================================================"
echo " SFP-2 部署包打包"
echo "============================================================"
echo "  SFP2 目录:   $SFP2_DIR"
echo "  项目根目录:  $PROJECT_DIR"
echo "  输出文件:    $OUTPUT"
echo ""

# 清理
rm -rf "$DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"

# ── 1. Dockerfile + docker-compose
echo "[1/6] 复制 Docker 配置..."
cp "$SFP2_DIR/Dockerfile.deploy"            "$DEPLOY_DIR/Dockerfile"
cp "$SFP2_DIR/docker-compose.deploy.yml"    "$DEPLOY_DIR/docker-compose.yml"

# ── 2. sfp-2 代码
echo "[2/6] 复制 sfp-2 代码..."
mkdir -p "$DEPLOY_DIR/sfp-2/src"
mkdir -p "$DEPLOY_DIR/sfp-2/scripts"
cp "$SFP2_DIR/app.py"                       "$DEPLOY_DIR/sfp-2/"
cp "$SFP2_DIR/requirements_docker.txt"      "$DEPLOY_DIR/sfp-2/"
cp -r "$SFP2_DIR/src/"*                     "$DEPLOY_DIR/sfp-2/src/"
cp -r "$SFP2_DIR/scripts/"*                 "$DEPLOY_DIR/sfp-2/scripts/"

# ── 3. 模型权重
echo "[3/6] 复制模型权重..."
mkdir -p "$DEPLOY_DIR/sfp-2/checkpoints"
cp "$SFP2_DIR/checkpoints/best_dual_tower_model.pt" "$DEPLOY_DIR/sfp-2/checkpoints/"
cp "$SFP2_DIR/checkpoints/params.json"              "$DEPLOY_DIR/sfp-2/checkpoints/"

# ── 4. NPY 数据集
echo "[4/6] 复制 NPY 数据集 (预构建)..."
mkdir -p "$DEPLOY_DIR/sfp-2/datasets"
cp -r "$SFP2_DIR/datasets/npy"              "$DEPLOY_DIR/sfp-2/datasets/"

# ── 5. traindata
echo "[5/6] 复制 traindata..."
cp -r "$PROJECT_DIR/traindata"              "$DEPLOY_DIR/traindata"

# ── 6. results 目录（空）
mkdir -p "$DEPLOY_DIR/sfp-2/results"

# ── .dockerignore
cat > "$DEPLOY_DIR/.dockerignore" << 'EOIGNORE'
**/__pycache__
**/*.pyc
sfp-2/datasets/
sfp-2/results/
sfp-2/checkpoints/
sfp-2/versions/
*.pt
*.pkl
EOIGNORE

# ── deploy.sh
cat > "$DEPLOY_DIR/deploy.sh" << 'EODEPLOY'
#!/bin/bash
set -e
echo "============================================================"
echo " SFP-2 二次调频预测服务 — 部署"
echo "============================================================"

if ! command -v docker &>/dev/null; then
    echo "[!] 未找到 docker，请先安装 Docker"
    exit 1
fi

COMPOSE_CMD=""
if command -v docker-compose &>/dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &>/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    echo "[!] 未找到 docker-compose，请先安装"
    exit 1
fi

echo ""
echo "[1/3] 构建 Docker 镜像..."
docker build -t sfp-2:latest .

echo ""
echo "[2/3] 启动服务..."
$COMPOSE_CMD up -d

echo ""
echo "[3/3] 等待服务就绪..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:9527/health > /dev/null 2>&1; then
        echo ""
        echo "============================================================"
        echo " 部署成功！"
        echo ""
        echo " 服务地址:  http://<本机IP>:9527"
        echo " 健康检查:  curl http://localhost:9527/health"
        echo " 预测接口:  curl -X POST 'http://localhost:9527/predict?date=2026-03-05'"
        echo " NPY 重建:  curl -X POST 'http://localhost:9527/datasets/rebuild'"
        echo " 数据集状态: curl http://localhost:9527/datasets/status"
        echo " 结果列表:  curl http://localhost:9527/results"
        echo ""
        echo " 日志查看:  docker logs -f sfp2-predict"
        echo " 停止服务:  $COMPOSE_CMD down"
        echo " 重启服务:  $COMPOSE_CMD restart"
        echo "============================================================"
        exit 0
    fi
    printf "."
    sleep 2
done
echo ""
echo "[!] 服务启动超时，请检查日志: docker logs sfp2-predict"
exit 1
EODEPLOY
chmod +x "$DEPLOY_DIR/deploy.sh"

# ── README
cat > "$DEPLOY_DIR/README.txt" << 'EOREADME'
SFP-2 二次调频预测服务 — 部署包
================================

目录结构:
  sfp2-deploy/
  ├── Dockerfile
  ├── docker-compose.yml
  ├── deploy.sh               一键部署脚本
  ├── traindata/              原始 CSV 数据（每日更新时替换内容）
  └── sfp-2/
      ├── app.py              FastAPI 服务
      ├── src/                模型代码 + 数据加载器
      ├── scripts/            NPY 构建脚本
      ├── checkpoints/        模型权重
      ├── datasets/npy/       预构建 NPY 数据集
      └── results/            预测结果输出

部署:
  1. tar xzf sfp2-deploy.tar.gz
  2. cd sfp2-deploy
  3. bash deploy.sh

API (端口 9527):
  GET  /health                      健康检查
  POST /predict?date=YYYY-MM-DD     预测指定日期
  POST /datasets/rebuild            新数据到了以后重建 NPY
  GET  /datasets/status             数据集状态
  GET  /results                     已生成的结果列表
EOREADME

# ── 打包
echo "[6/6] 打包中..."
cd /tmp
tar czf "$OUTPUT" sfp2-deploy/

SIZE=$(du -sh "$OUTPUT" | cut -f1)
echo ""
echo "============================================================"
echo " 打包完成!"
echo "  输出: $OUTPUT"
echo "  大小: $SIZE"
echo ""
echo " 部署: scp $OUTPUT user@target:/path/"
echo "       ssh user@target 'tar xzf sfp2-deploy.tar.gz && cd sfp2-deploy && bash deploy.sh'"
echo "============================================================"

rm -rf "$DEPLOY_DIR"
