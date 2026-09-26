# SUMMARY — integrity_2026-08-07（定稿戰役：階段 1 重跑 + v5 稿完成）

> 執行：2026-08-07。輸入規則遵守：REMIX 數字全部來自修正配置重跑；
> baseline 僅使用存檔 quantile **預測**（1e/1f 指令明示），評分一律現行實作。
> `outputs/aistats2027/` 未寫入。v5 稿在 `paper_v2/v5/`（main.tex + sections/）。

## 任務狀態

| 項 | 狀態 | 輸出 |
|---|---|---|
| 1a WF 重跑（★） | ✅ | `wf_remix_corrected.csv`、`wf_remix_quantiles_*.csv`、`or_point_wf_remix_corrected/`、`f1_run_meta.json` |
| 1b leverage 兩格 | ✅ OR −2.9%、Auto −3.6% | `f234_summary.csv`、`f234_run_meta.json` |
| 1c OR ablation auto 三列 @0.95 | ✅ | 同上（prob+point 六 run） |
| 1d M5 seed variance (mixture,0.95) | ✅ 1.7501±0.084 | `m5_prob_seed4{3,4}_mixture_095/` |
| 1e SPL 顯著性（含 TweedieGP） | ✅ 43/48 顯著較好、0 較差 | `spl_significance_corrected.csv`、`f57_run_meta.json` |
| 1f tab:prob 統一評分 | ✅ 唯一變動 baseline 格＝OR Zero 0.9224→0.9161 | `rescored_tab_prob.csv` |
| 1g Table 8 | ✅ 刪除（連同附錄舊 leverage Table 9、E.9、E.11、E.13） | — |
| 1h 計時實測 | ✅ 選模 51–243s（不含載入）；預測 6.2/38.0/124.0s | `f8_timing.csv`、`f8_run_meta.json` |
| 追加：Carparts 遠尾 @ (global,0.90) | ✅ 0.365/0.251/0.144（三格全升第二） | `carparts_prob_fartail_corrected/` |
| 階段 2–5（稿件） | ✅（編譯待 TeX 引擎，見下） | `paper_v2/v5/` |

## ★ 1a：WF 新舊對照（headline 變動，全部對 REMIX 有利）

| 量 | 舊（稿中） | **新** |
|---|---|---|
| OR wf mean SPL | 0.8676 | **0.8540** |
| OR vs ACI-ADIDA | 4.7% | **6.2%** |
| OR q90 | 1.4598（vs AutoTheta +1.6%） | **1.4002（+5.6%）** |
| OR Cov⁺@80 / AIW | 0.611 / 10.41 | 0.621 / 10.31 |
| OR point MAE/RMSE/RMSSE | 5.462/17.193/4.710 | **5.421/17.188/4.696** |
| Carparts wf mean SPL | 0.3257 | **0.3145** |
| Carparts vs CP-IMAPA | 6.7% | **9.9%** |
| Carparts 分位勝場 | 四/五（q75 輸 CP-TSB） | **五/五** |
| Carparts q50 | 與 Zero 完全同值 | **0.3724 < Zero 0.3736**（僅 q10/q25 仍飽和） |

**連動更新位置（已全部執行）**：abstract、Introduction 末段、§5.3 兩處
margin 句與三特徵段（「four times」→「more than three times」、中央分位句
改 q10/q25）、tab:wfprob、附錄 Table 11（列名改 selected global,.95）/12/13、
walk-forward 後記段。Conclusion 的 v5 文字無 wf 數字，無需改。

## 主表新值（v5 稿已填）

- tab:prob REMIX 列：0.8849/1.5268、1.6707/2.4758、0.2937/0.2599、
  0.3229/0.4817、0.2680/0.4856；margin 句 = 10.1%/3.0%/1.6%/1.2%、
  Carparts 落後 TweedieGP 0.5%（0.3214 vs 0.3229）。
- tab:point REMIX 列（重數：**best 6/10、top two 7/10**）：OR 5.5325/4.7901
  （OR MAE 轉為全場最佳）、M5 1.2032/2.2938（MAE 讓位 TSB-tuned 1.1932）、
  Auto 3.2145/1.1374、Carparts 0.5726/1.0120、RAF 2.3874/**1.6245**
  （RAF MAE 掉到第 7；RMSSE 以 1.62445 對 ADIDA 1.62452 反成最佳）。
- tab:coverage：0.892/9.25、0.846/2.81、0.907/9.21、0.931/1.47、0.921/0.94。
- tab:leverage：OR/Auto 兩格補齊 −2.9%/−3.6%；五格全負
  （−1.2/−2.3/−2.9/−3.6/−31.5）。
- tab:ablation（auto 列 @0.95）：global 0.8849/1.5268/9.25、taxonomy
  0.8846/1.5254/9.43、learned 0.8869/1.5343/10.69；外部最佳落在 taxonomy
  auto（0.8846，比 global 好 0.03%＝噪音）——與「結構軸不解析」一致，
  ablation 段措辭已同步（<0.1% off、<0.3% auto；區間縮 11–12%；MAE 改善
  3.5–4.0%）。
- M5 多 seed：REMIX 1.6707/1.7424/1.8371（1.7501±0.084）vs CP-Croston
  1.6902/1.7543/1.8195（1.7547±0.065）。
- 遠尾 Carparts REMIX：0.365/0.251/0.144（原 0.378/0.256/0.146 為舊配置）。

## 1e/1f：顯著性與統一評分

- 統一評分（現行實作、存檔預測）之下，**唯一變動的 baseline 格＝OR Zero
  （0.9224→0.9161，legacy 評分殘留）**；其餘 baseline 逐位不變——caption
  拿掉揭露句的前提現在實際成立。
- SPL 顯著性（新表 tab:spl-significance）：48 對比較，**REMIX 顯著較好
  43、顯著較差 0**；未定 5＝M5 四個 CP wrapper（均值有利 REMIX 但 paired t
  Holm 後不顯著；Wilcoxon 強烈拒斥對稱）＋ Carparts 對 TweedieGP
  （雙向皆不顯著，p≥0.50）。

## 對 REMIX 不利的發現（照實列，未淡化）

1. **M5 seed 44 輸 CP-Croston**：1.8371 vs 1.8195（−1.0%）。舊敘述「seed 44
   實質平手」不再成立，稿中已改「ahead on two samples, behind by 1.0% on
   the third」。跨 seed 平均領先僅 0.3%，遠小於 1 sd。
2. **M5 MAE 讓位 TSB-tuned**（1.2032 vs 1.1932）、**RAF MAE 掉到第 7**
   （2.3874；TSB 2.1660 最佳）——tab:point 由 5/10 best 變 6/10 是靠 OR MAE
   與 RAF RMSSE 補回，RAF MAE 的惡化已如實入表。
3. **M5 對四個 CP wrapper 的 paired t 不顯著**（顯著性表明列 Undecided）。
4. Carparts 對 TweedieGP 統計上不可分（互不顯著），aggregate 仍落後 0.5%。
5. tab:ablation 外部最佳為 taxonomy@0.95 而非選中的 global@0.95（0.03%
   噪音級，但如實入表，佐證結構軸不解析而非 REMIX 選對）。

## 殘留事項

1. **編譯**：本機無任何 TeX 引擎（pdflatex/MiKTeX/texlive/tectonic/WSL 皆
   無）。v5 稿已通過自建靜態檢查（環境配對、label/ref、bibitem、圖檔存在）。
   待作者裁決：裝 MiKTeX（需下載授權）或上 Overleaf 編譯。
2. app:mcmc 的 REMIX 子樣本 context 列仍為舊選模產物（該表明示僅為
   context、非 inference 對比；未重跑）。
3. run_point 的 M5 分支忽略 forced 配置的 CLI 缺陷仍未修（遵守不改程式；
   v5 稿 CONSEQUENTIAL 註記 D 已建議投稿前修）。
4. scripts/integrity/ 與 paper_v2/v5/ 均未 commit。
