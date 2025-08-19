#!/bin/bash

# AI-PoC-Agents-v2 全体実行スクリプト
# Usage: sh scripts/run.sh

set -e  # エラーが発生したら終了

# スクリプトのディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== AI-PoC-Agents-v2 Full Pipeline Execution ==="
echo "Project Root: $PROJECT_ROOT"
echo "Starting PoC development pipeline..."
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

# デフォルト設定
DEFAULT_THEME="OCR画像文字認識システム"
DEFAULT_DESCRIPTION="画像から文字を認識してテキストに変換するPythonシステムの開発"
DEFAULT_WORKSPACE="./workspace/$(date +%Y%m%d_%H%M%S)_ocr_poc"
DEFAULT_SAMPLE_DATA="./data/ocr_sample"

# コマンドライン引数の処理
THEME="${1:-$DEFAULT_THEME}"
DESCRIPTION="${2:-$DEFAULT_DESCRIPTION}"
WORKSPACE="${3:-$DEFAULT_WORKSPACE}"
SAMPLE_DATA="${4:-$DEFAULT_SAMPLE_DATA}"

echo "🎯 PoC Configuration:"
echo "   Theme: $THEME"
echo "   Description: $DESCRIPTION"
echo "   Workspace: $WORKSPACE"
echo ""

# ワークスペースディレクトリ作成
mkdir -p "$WORKSPACE"
echo "✓ Workspace created: $WORKSPACE"

# メイン実行
echo "🚀 Starting AI-PoC-Agents-v2 pipeline..."
echo ""

# 実行コマンド
uv run python main.py \
    --theme "$THEME" \
    --description "$DESCRIPTION" \
    --workspace "$WORKSPACE" \
    --max-iterations 2 \
    --sample-data "$SAMPLE_DATA"

# 実行結果の確認
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 PoC pipeline completed successfully!"
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
        
        # 実装されたコードの確認
        if [ -f "$WORKSPACE/code/main.py" ]; then
            echo ""
            echo "💻 Generated Code:"
            echo "     - main.py: $(wc -l < "$WORKSPACE/code/main.py") lines"
            if [ -f "$WORKSPACE/code/requirements.txt" ]; then
                echo "     - requirements.txt: $(wc -l < "$WORKSPACE/code/requirements.txt") dependencies"
            fi
        fi
    fi
    
    echo ""
    echo "🔗 Quick Start:"
    echo "   cd $WORKSPACE/code"
    echo "   uv run pip install -r requirements.txt"
    echo "   uv run python main.py"
    echo ""
    echo "📖 View Reports:"
    echo "   Executive Summary: $WORKSPACE/reporting/executive_summary_iteration_0.md"
    echo "   Final Report: $WORKSPACE/reporting/final_poc_report_iteration_0.md"
    echo "   Technical Evaluation: $WORKSPACE/result_evaluation/evaluation_report_iteration_0.md"
    
else
    echo ""
    echo "❌ PoC pipeline failed"
    echo "   Check logs above for error details"
    echo "   Workspace preserved at: $WORKSPACE"
    exit 1
fi

echo ""
echo "✅ AI-PoC-Agents-v2 execution completed!"