# v6 骨架（`paper_v2/v6_aistats/`，由 `scripts/integrity/v6_build.py` 從 v5_aistats 組裝，數字不動）

## 結構（理論＋診斷第一）
1 Intro（global-vs-local 開場 → 雙邊張力 → EBB → 兩側驗證 → 貢獻 C1 理論診斷／C2 方法／C3 評估警告）
2 Related Work（原文＋新引 Januschowski 2020、Hewamalage 2022）
3 EBB（3.1 模型、3.2 折扣、3.3 候選結構縮成一段、3.4 選擇、3.5 計算縮短）
4 Resolution under Forgetting（Prop 1 縮成一句、Prop 2＋推論＋機制圖）
5 Experiments：5.1 設定；5.2 Where pooling pays（cold-start 圖＋曲面圖＋room 兩段＋forgetting 量級段）；
  5.3 Accuracy and cost（新合併主表 tab:main＋單欄前沿圖）；5.4 Two evaluation cautions
6 Limitations（三項）；7 Conclusion
附錄新增 `app:demoted`（sections/demoted.tex）：fig:problem、Prop 1 全文、tab:pool-candidates＋成本段、
tab:prob＋5.2 原文、tab:wfprob＋tab:regret＋雙面板前沿圖(fig:frontier-full)＋5.3 原文、
tab:sanity＋模擬節原文、tab:leverage＋fig10＋ablation＋5.6 其餘段落。其他附錄不變。

## 編譯狀態
36 頁、0 錯誤、0 未解引用；正文到第 9 頁（References 第 10 頁）→ 需再裁約 1 頁到 8 頁。

## 建議裁減（作者）與估計
- Intro 貢獻段精簡（−0.2 頁）；Related「Learning cross-series structure」段縮到三句（−0.2）
- §3.4「Resolution diagnostics」段移附錄留一句（−0.25，已可由 builder 執行）
- §5.1 設定縮到半欄（−0.25）；§5.2 room 兩段縮 30%（−0.15）；Limitations 縮到 0.4 頁（−0.2）
合計約 −1.25 頁。

## 注意
- 標題改為問句式（runningtitle 同步）；EBB 命名 footnote 保留。
- 所有被移走的段落原封在 app:demoted，可剪貼回正文。
- builder 可重跑（覆寫 main.tex、sections/experiments_v6.tex、sections/demoted.tex）。
