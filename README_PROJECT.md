# Kaggle: Predicting Irrigation Need

AIを活用したコンペ参加プロジェクト（csiro-biomass-agentic-solution-main を参考）

## 📁 ディレクトリ構成

```
Kaggle_Predicting_Irrigation_Need/
│
├── notebooks/                           # 対話的分析・実験
│   ├── 01_EDA.ipynb                    # 基本統計・分布・相関
│   ├── 02_Feature_Engineering.ipynb    # 特徴生成
│   ├── 03_Feature_Selection.ipynb      # 特徴選択
│   └── 04_Baseline_Model.ipynb         # 複数モデル比較
│
├── data/
│   ├── train.csv                       # 元のトレーニングデータ
│   ├── test.csv                        # テストデータ
│   ├── sample_submission.csv           # サンプル提出ファイル
│   ├── processed/                      # 前処理・加工済みデータ
│   │   ├── train_engineered.csv
│   │   ├── test_engineered.csv
│   │   ├── train_engineered_selected.csv
│   │   ├── test_engineered_selected.csv
│   │   ├── train_engineered_scaled.csv
│   │   └── test_engineered_scaled.csv
│   └── analysis/                       # 分析中間結果（JSON形式）
│       ├── feature_stats.json
│       ├── correlation_matrix.json
│       ├── null_analysis.json
│       ├── feature_importance.json
│       └── baseline_cv_results.json
│
├── docs/
│   ├── DATASET.md                      # データセット説明
│   ├── APPROACH.md                     # 全体戦略
│   └── Idea_Research/                  # 発見・仮説・軌歡（日付ごと）
│       ├── 20260406_EDA_Summary.md
│       ├── 20260406_Feature_Analysis.md
│       └── ...
│
├── EXP/                                # 実験ディレクトリ
│   ├── EXP001/
│   │   ├── train.py                    # 訓練スクリプト
│   │   ├── infer.py                    # 推論スクリプト
│   │   ├── config/
│   │   │   ├── child-exp000.yaml
│   │   │   ├── child-exp001.yaml
│   │   │   └── ...
│   │   └── outputs/
│   │       ├── child-exp000/
│   │       │   ├── config.yaml
│   │       │   ├── oof_predictions.csv
│   │       │   └── results.json
│   │       └── ...
│   └── EXP_SUMMARY.md                  # 実験記録
│
├── execute_train.ipynb                 # メイン訓練パイプライン
├── _CLAUDE.md                          # Claude Code用指示
├── _AGENTS.md                          # Codex用指示
└── README.md                           # このファイル
```

---

## 🔄 ワークフロー

### Phase 1: データ理解（EDA）
```
notebooks/01_EDA.ipynb
  ├── データ形状・型の確認
  ├── 欠損値・外れ値分析
  ├── クラス分布の可視化
  ├── 特徴量相関分析
  └── → data/analysis/ に結果保存
```

### Phase 2: 特徴工学（Feature Engineering）
```
notebooks/02_Feature_Engineering.ipynb
  ├── ドメイン知識に基づく特徴生成
  │   ├── 土壌-天候相互作用
  │   ├── 水分バランス
  │   └── 環境ストレス指標
  ├── カテゴリ変数エンコーディング
  ├── 欠損値処理
  └── → data/processed/train_engineered.csv 保存
```

### Phase 3: 特徴選択（Feature Selection）
```
notebooks/03_Feature_Selection.ipynb
  ├── 相互情報量スコア計算
  ├── Random Forest 重要度計算
  ├── 複合スコア統合
  ├── 上位50%の特徴選択
  └── → data/processed/train_engineered_selected.csv 保存
```

### Phase 4: ベースラインモデル
```
notebooks/04_Baseline_Model.ipynb
  ├── Logistic Regression
  ├── LightGBM
  ├── (XGBoost)
  ├── (CatBoost)
  └── → CV結果比較・最良モデル決定
```

### Phase 5: 本実験
```
EXP001/train.py
  ├── 選定されたモデルで訓練
  ├── StratifiedKFold (5-fold)
  ├── OOF予測 + 結果保存
  └── → EXP_SUMMARY.md に記録
```

---

## 📊 データフロー

```
raw data
  ↓
[01_EDA] → analysis results
  ↓
[02_FE] → engineered features
  ↓
[03_FS] → selected features ← statistics
  ↓
[04_Baseline] → model selection
  ↓
EXP001/train.py → experiments
  ↓
results & submission
```

---

## 🚀 使い方

### 1. EDA実行
```bash
# notebooks/ フォルダで Jupyter を起動
jupyter notebook

# 01_EDA.ipynb を実行（全セル実行）
# → data/analysis/ に中間結果が保存される
```

### 2. 特徴工学
```bash
# 02_Feature_Engineering.ipynb を実行
# → data/processed/ にエンジニアー済みデータが保存される
```

### 3. 特徴選択
```bash
# 03_Feature_Selection.ipynb を実行
# → 選択された特徴のみのデータセット作成
```

### 4. ベースライン評価
```bash
# 04_Baseline_Model.ipynb を実行
# → CV結果を確認して最良モデルを決定
```

### 5. 本実験実行
```bash
python EXP001/train.py --config EXP001/config/child-exp000.yaml
```

---

## 📝 設定ファイル（YAML）

`config/child-exp{N}.yaml` の例：

```yaml
# child-exp000.yaml
exp_name: "EXP001"
child_exp_name: "child-exp000"
description: "LightGBM baseline"

model_type: "lgbm"
model_params:
  num_leaves: 31
  learning_rate: 0.05
  n_estimators: 1000
  random_state: 42

feature_params:
  use_engineered: true
  use_selected: true
  feature_set: "selected"  # "all" / "engineered" / "selected"

cv_params:
  n_splits: 5
  random_state: 42
  stratify: true
```

---

## 📋 関連ドキュメント

- [DATASET.md](docs/DATASET.md) - データセット詳細説明
- [APPROACH.md](docs/APPROACH.md) - 全体的なアプローチ説明
- [EXP_SUMMARY.md](EXP/EXP_SUMMARY.md) - 実験記録
- [_CLAUDE.md](_CLAUDE.md) - Claude Code用詳細指示
- [_AGENTS.md](_AGENTS.md) - Codex用簡略指示

---

## 🎯 Success Criteria

- ✓ EDA完了：データ理解、外れ値・欠損具体化
- ✓ FE完了：10+の新特徴、妥当性確認
- ✓ FS完了：特徴選択、重要度ランキング
- ✓ Baseline完了：複数モデル比較、最良モデル選定
- ✓ CV結果：Baseline AUC > 0.75
- ✓ Submission：テスト予測・Kaggle提出

---

## 💡 Tips

1. **再現性**: すべてコードに `random_state` を設定する
2. **Data Leakage**: テスト統計を学習時に使わない（train medianのみ使用）
3. **ログ記録**: 各実験結果を `docs/Idea_Research/YYYYMMDD_*.md` に記録
4. **継続改善**: 失敗したアイデアは `EXP_SUMMARY.md` に記録して繰り返さない

---

## 参考リポジトリ

本プロジェクトは以下を参考にしています：
- [CSIRO - Image2Biomass Prediction (5th place solution)](https://github.com/user/csiro-biomass-agentic-solution-main)

---

**作成日**: 2026-01-06
**最終更新**: 2026-04-06
