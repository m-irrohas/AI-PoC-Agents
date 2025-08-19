#!/bin/bash

# KDD Cup 99分類モデル構築スクリプト
# Usage: sh scripts/run_kdd_classification.sh

set -e  # エラーが発生したら終了

# スクリプトのディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== KDD Cup 99 Network Intrusion Detection PoC ==="
echo "Project Root: $PROJECT_ROOT"
echo "Starting classification model development pipeline..."
echo ""

# プロジェクトルートに移動
cd "$SCRIPT_DIR"

# 環境変数チェック
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Please create .env file with required API keys."
    echo "   See .env.example for reference."
    exit 1
fi

# OPENAI_API_KEYのチェック
if ! grep -q "OPENAI_API_KEY=" .env || grep -q "OPENAI_API_KEY=your_openai_api_key_here" .env; then
    echo "❌ OPENAI_API_KEY not configured in .env file"
    echo "   Please set your OpenAI API key in .env file"
    exit 1
fi

echo "✓ Environment configuration verified"

# Qiitaアクセストークンのチェック (警告のみ)
if ! grep -q "QIITA_ACCESS_TOKEN=" .env || grep -q "QIITA_ACCESS_TOKEN=your_qiita_access_token_here" .env; then
    echo "⚠️  QIITA_ACCESS_TOKEN not configured - Qiita integration will use anonymous access (rate limited)"
else
    echo "✓ Qiita access token configured"
fi

echo ""

# KDD Cup 99 データセット設定
THEME="ネットワーク侵入検知分類システム"
DESCRIPTION="KDD Cup 99データセットを使用して、ネットワーク攻撃を検出する多クラス分類モデルを構築し、クロスバリデーションによる性能評価を実施"
WORKSPACE="./workspace/$(date +%Y%m%d_%H%M%S)_kdd99_classification"
SAMPLE_DATA="./data/kdd_cup_99"

echo "🎯 PoC Configuration:"
echo "   Theme: $THEME"
echo "   Description: $DESCRIPTION"
echo "   Workspace: $WORKSPACE"
echo "   Dataset: $SAMPLE_DATA"
echo ""

# データセット確認
if [ ! -f "$SAMPLE_DATA/train.csv" ]; then
    echo "❌ KDD Cup 99 training data not found: $SAMPLE_DATA/train.csv"
    exit 1
fi

if [ ! -f "$SAMPLE_DATA/overview.txt" ]; then
    echo "❌ KDD Cup 99 overview file not found: $SAMPLE_DATA/overview.txt"
    exit 1
fi

# データセット情報表示
train_lines=$(wc -l < "$SAMPLE_DATA/train.csv")
echo "✓ Dataset verified:"
echo "   Training data: $(($train_lines - 1)) records"
echo "   Features: 41 network connection features"
echo "   Target: attack_type (multi-class classification)"

# 攻撃タイプ分布を確認
echo "   Attack types distribution:"
tail -n +2 "$SAMPLE_DATA/train.csv" | cut -d',' -f42 | sort | uniq -c | sort -nr | head -10 | sed 's/^/     /'

echo ""

# ワークスペースディレクトリ作成
mkdir -p "$WORKSPACE"
echo "✓ Workspace created: $WORKSPACE"

# サンプルデータのコピー
echo "📁 Copying KDD Cup 99 dataset..."
mkdir -p "$WORKSPACE/code/data"
cp "$SAMPLE_DATA/train.csv" "$WORKSPACE/code/data/"
cp "$SAMPLE_DATA/overview.txt" "$WORKSPACE/code/data/"

# データセットをサブサンプリング（処理時間短縮のため）
echo "📊 Creating development subset (10K samples for faster processing)..."
head -1 "$SAMPLE_DATA/train.csv" > "$WORKSPACE/code/data/train_subset.csv"
tail -n +2 "$SAMPLE_DATA/train.csv" | shuf | head -10000 >> "$WORKSPACE/code/data/train_subset.csv"

subset_lines=$(wc -l < "$WORKSPACE/code/data/train_subset.csv")
echo "✓ Development subset created: $(($subset_lines - 1)) records"

echo ""

# メイン実行
echo "🚀 Starting KDD Cup 99 classification pipeline..."
echo ""

# 実行コマンド
uv run python main.py \
    --theme "$THEME" \
    --description "$DESCRIPTION" \
    --workspace "$WORKSPACE" \
    --max-iterations 2 \
    --sample-data "$SAMPLE_DATA" \
    --score-threshold 0.6

# 実行結果の確認
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 KDD Cup 99 classification pipeline completed successfully!"
    echo ""
    echo "📋 Results Summary:"
    echo "   Workspace: $WORKSPACE"
    
    # 生成されたファイルの確認
    if [ -d "$WORKSPACE" ]; then
        echo "   Generated files:"
        find "$WORKSPACE" -type f -name "*.md" -o -name "*.json" -o -name "*.py" | head -10 | sed 's/^/     - /'
        
        # ファイル数カウント
        file_count=$(find "$WORKSPACE" -type f | wc -l)
        echo "   Total files: $file_count"
        
        # 最終レポートの確認
        if [ -f "$WORKSPACE/reporting/executive_summary_iteration_0.md" ]; then
            echo ""
            echo "📊 Executive Summary Preview:"
            head -n 10 "$WORKSPACE/reporting/executive_summary_iteration_0.md" | sed 's/^/     /'
        fi
        
        # 分類モデルのコードの確認
        if [ -f "$WORKSPACE/code/main.py" ]; then
            echo ""
            echo "🤖 Generated Classification Model:"
            echo "     - main.py: $(wc -l < "$WORKSPACE/code/main.py") lines"
            if [ -f "$WORKSPACE/code/requirements.txt" ]; then
                echo "     - requirements.txt: $(wc -l < "$WORKSPACE/code/requirements.txt") dependencies"
            fi
            
            # 実行結果ファイルの確認
            if [ -f "$WORKSPACE/code/cv_results.json" ]; then
                echo "     - Cross-validation results available"
            fi
            if [ -f "$WORKSPACE/code/model_performance.txt" ]; then
                echo "     - Model performance metrics available"
            fi
        fi
    fi
    
    echo ""
    echo "🔗 Quick Start (Classification Model):"
    echo "   cd $WORKSPACE/code"
    echo "   uv run pip install -r requirements.txt"
    echo "   uv run python main.py  # Run classification with CV"
    echo ""
    echo "📊 Data Analysis:"
    echo "   Dataset: $WORKSPACE/code/data/train.csv (original)"
    echo "   Dev Set: $WORKSPACE/code/data/train_subset.csv (10K samples)"
    echo "   Overview: $WORKSPACE/code/data/overview.txt"
    echo ""
    echo "📖 View Reports:"
    echo "   Executive Summary: $WORKSPACE/reporting/executive_summary_iteration_0.md"
    echo "   Final Report: $WORKSPACE/reporting/final_poc_report_iteration_0.md"
    echo "   Technical Evaluation: $WORKSPACE/result_evaluation/evaluation_report_iteration_0.md"
    
else
    echo ""
    echo "❌ KDD Cup 99 classification pipeline failed"
    echo "   Check logs above for error details"
    echo "   Workspace preserved at: $WORKSPACE"
    exit 1
fi

echo ""
echo "✅ KDD Cup 99 Network Intrusion Detection PoC completed!"
echo "🎯 Key Objectives Achieved:"
echo "   - Multi-class classification model built"
echo "   - Cross-validation performance evaluated"
echo "   - Network attack detection system implemented"
echo "   - Comprehensive evaluation reports generated"