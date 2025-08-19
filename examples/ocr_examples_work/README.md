# Examples

このディレクトリには、AI-PoC-Agents-v2で生成されたPoCの実行例が保存されます。

## ディレクトリ構造

各PoCプロジェクトは以下のような構造で保存されます：

```
examples/
└── {プロジェクト名}/
    ├── problem_identification/
    │   └── problem_analysis_iteration_0.json
    ├── idea_generation/
    │   └── generated_ideas_iteration_0.json
    ├── idea_selection/
    │   └── idea_selection_iteration_0.json
    ├── poc_design/
    │   ├── poc_design_iteration_0.json
    │   └── poc_design_document_iteration_0.md
    ├── poc_implementation/
    │   ├── code/
    │   │   ├── main.py
    │   │   ├── requirements.txt
    │   │   └── README.md
    │   └── execution_plan_iteration_0.md
    ├── poc_execution/
    │   └── execution_plan_iteration_0.md
    ├── result_evaluation/
    │   ├── poc_evaluation_iteration_0.json
    │   └── evaluation_report_iteration_0.md
    ├── reflection/
    │   ├── reflection_analysis_iteration_0.json
    │   └── reflection_report_iteration_0.md
    ├── reporting/
    │   ├── final_poc_report_iteration_0.md
    │   ├── executive_summary_iteration_0.md
    │   └── poc_summary_iteration_0.json
    └── final_state.json
```

## ファイル説明

### 各フェーズの出力

- **problem_identification/**: 課題分析結果
- **idea_generation/**: 生成されたアイデア一覧
- **idea_selection/**: 選択されたアイデアと評価
- **poc_design/**: 技術設計書と仕様
- **poc_implementation/**: 実装コードとドキュメント
- **poc_execution/**: 実行計画と結果
- **result_evaluation/**: 評価レポートと分析
- **reflection/**: リフレクション分析
- **reporting/**: 最終レポートと要約

### 主要ファイル

- **final_state.json**: ワークフロー全体の状態と結果
- **executive_summary.md**: エグゼクティブ向け要約
- **final_poc_report.md**: 包括的な最終レポート

## 使用方法

生成されたファイルを参照して、PoCの結果を理解し、次のステップを計画してください。コードは`poc_implementation/code/`ディレクトリから実際に実行可能です。