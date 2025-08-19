# AI-PoC-Agents-v2 設計書

## 概要

AI-PoC-Agents-v2は、ユーザーが与えたテーマについて自動的にProof of Concept (PoC)を実施するマルチエージェントフレームワークです。既存のAI-PoC-Agentsをベースに、より汎用的で自動化されたPoC実行システムを構築します。

## アーキテクチャ

### 全体フロー
```
[ユーザーのテーマ入力]
　　↓
1️⃣ 課題明確化・アイデア生成エージェント
　　↓
2️⃣ PoC設計・実施エージェント  
　　↓
3️⃣ 結果評価・リフレクションエージェント
　　↓
[ユーザーへ結果を提示・次のステップ提案]
```

## エージェント設計

### 1. 課題明確化・アイデア生成エージェント (Problem Identification & Ideation Agent)

**責任範囲：**
- ユーザーのテーマから具体的な課題を特定
- 実現可能なアイデアを複数提示
- 最適なPoC候補を選定

**サブモジュール：**
- **Planner（計画立案モジュール）**
  - ユーザーのテーマを分析し、重要課題を洗い出し
  - 明確化のための質問生成
  - 課題の優先順位付け

- **Ideator（アイデア生成モジュール）**
  - 課題に対する実行可能な解決策を多様な観点で提示
  - アイデアの重要度・実現性・技術的難易度を評価
  - アイデアの絞り込みと最適化

- **Selector（選定モジュール）**
  - 提示したアイデアから最適なPoC候補を自動選定
  - 選定理由と期待される成果の明文化
  - ユーザーフィードバックの統合

**出力：**
- 明確化された課題定義
- 複数のアイデア案と評価スコア
- 選定されたPoC案と実施計画
- 成功指標の定義

### 2. PoC設計・実施エージェント (PoC Design & Execution Agent)

**責任範囲：**
- PoCの詳細設計
- 実施環境の構築
- コード実装とテスト
- 実施状況の監視と修正

**サブモジュール：**
- **Designer（設計モジュール）**
  - PoCの目的・スコープ・評価基準の定義
  - 技術スタック選定
  - アーキテクチャ設計
  - 実施手順の詳細化

- **Executor（実行モジュール）**
  - 開発環境の自動構築
  - コード生成と実装
  - テストケース作成と実行
  - Docker環境での再現可能な実行

- **Monitor（監視モジュール）**
  - 実施状況のリアルタイム監視
  - エラー検出と自動修復
  - 進捗レポートの生成
  - パフォーマンス指標の収集

**出力：**
- 詳細な設計書
- 実装されたコード
- テスト結果
- 実行可能な環境（Docker等）
- 実施レポート

### 3. 結果評価・リフレクションエージェント (Evaluation & Reflection Agent)

**責任範囲：**
- PoCの結果評価
- 成功・失敗要因の分析
- 改善提案の生成
- 次ステップの提示

**サブモジュール：**
- **Evaluator（評価モジュール）**
  - 事前定義した評価基準に基づく定量評価
  - 定性的な結果分析
  - データの可視化
  - 統計的分析

- **Reflector（リフレクションモジュール）**
  - 成功要因と失敗要因の特定
  - 技術的課題と解決策の分析
  - ビジネス価値の評価
  - 学習ポイントの抽出

- **Reporter（報告モジュール）**
  - 結果の構造化されたレポート作成
  - ビジュアライゼーションの生成
  - エグゼクティブサマリーの作成
  - 次のステップ提案

**出力：**
- 評価レポート
- 可視化されたデータ
- 改善提案書
- 次段階のアクションプラン

## 技術実装

### コア技術スタック
- **ワークフロー管理**: LangGraph (既存から継承)
- **LLM**: GPT-4o, GPT-4o-mini (モデル選択可能)
- **状態管理**: カスタムState管理システム
- **コード実行**: Docker containerized environments
- **データ永続化**: JSON-based state persistence

### 状態管理
```python
@dataclass
class PoCProject:
    theme: str
    description: str  
    domain: str
    requirements: List[str]
    constraints: List[str]

@dataclass
class PoCState:
    project: PoCProject
    current_phase: PoCPhase
    ideas: List[PoCIdea]
    selected_idea: Optional[PoCIdea]
    implementation: Optional[PoCImplementation]
    evaluation_results: Optional[EvaluationResult]
    artifacts: List[str]
    next_steps: List[str]
```

### フェーズ管理
```python
PoCPhase = Literal[
    "problem_identification",
    "idea_generation", 
    "idea_selection",
    "poc_design",
    "poc_implementation",
    "poc_execution",
    "result_evaluation",
    "reflection",
    "reporting"
]
```

## OCR実装例

### テーマ入力
「画像内の日本語テキストを自動で読み取るOCRを作りたい」

### フェーズ1: 課題明確化・アイデア生成
**Planner**:
- OCRの用途特定（文書デジタル化、名刺読み取り、看板認識等）
- 日本語特有の課題（ひらがな・カタカナ・漢字・縦書き）
- 精度要件とパフォーマンス要件の明確化

**Ideator**:
- Tesseract + 日本語学習データ
- Google Cloud Vision API活用
- 深層学習ベースの自作モデル（TrOCR, EasyOCR等）
- ハイブリッドアプローチ

**Selector**:
- EasyOCRを選択（実装容易性・精度・日本語サポートのバランス）

### フェーズ2: PoC設計・実施
**Designer**:
- 入力：画像ファイル（JPEG, PNG）
- 出力：認識されたテキスト（JSON形式）
- 評価指標：文字認識精度、処理時間、サポート形式

**Executor**:
```python
# EasyOCRベースのOCR実装
# Docker環境での実行
# 複数画像での一括処理機能
```

**Monitor**:
- 処理時間計測
- メモリ使用量監視
- エラーハンドリング

### フェーズ3: 結果評価・リフレクション
**Evaluator**:
- 文字認識精度: 85%
- 平均処理時間: 2.3秒/画像
- サポート形式: JPEG, PNG, PDF

**Reflector**:
- 成功要因：EasyOCRの日本語サポートが優秀
- 課題：縦書きテキストの認識精度が低い
- 改善案：前処理での画像回転、OCRエンジンの組み合わせ

**Reporter**:
- 概念実証成功
- 次のステップ：本格的なデータセット収集、精度改善

## 拡張性

### 新ドメイン対応
- ドメイン固有のエージェントモジュールを追加可能
- プラグイン形式でのカスタムエージェント実装
- 業界特化型のテンプレート提供

### スケーラビリティ
- 分散実行環境への対応
- クラウドネイティブデプロイ
- リソース自動スケーリング

### 学習・改善
- 過去のPoCから学習するメモリシステム
- 成功パターンの自動抽出
- ユーザーフィードバック統合

## まとめ

AI-PoC-Agents-v2は、既存のデータサイエンス特化型システムから、汎用的なPoC自動化フレームワークへの進化を目指します。3つの専門エージェントによる段階的なアプローチにより、ユーザーの曖昧なテーマから実用的なPoCまでを自動的に実現し、その結果を客観的に評価・改善提案することで、イノベーション創出を支援します。