# NUMBERS.md — main.tex 數字 → 來源對應（單一真相來源）

狀態標記：VERIFIED（已對到檔案）/ MISMATCH / NOT_FOUND / QUOTED（引自無法重跑之舊比較，正文有標註）。
所有路徑相對於 `outputs/aistats2027/`。fit window 註記：FW = full training window refit；VAL = 80% 內訓練窗。

## Table 1（tab:point）與 Table 2（tab:prob）
全部 110 格：見 `NUMBERS_tables.md`（由 `scripts/analysis/emit_main_tables.py` 程式化產生與標記，
全 VERIFIED，全 FW）。REMIX 列 = `*_point_remix` / `*_prob_select`（select 模式，FW refit）。

## Table 3（tab:components）
mixture 成分（K=5, OR）：`scripts/analysis/selector_sensitivity.py` 執行輸出（BIC K、weight、
occ prior mean、φ、median size、與 taxonomy 交叉表）。VERIFIED（由 init window 統計量計算，VAL 無關）。

## Table 4（tab:ablation）
**重跑中**（`ablate_or_point_*` / `ablate_or_prob_*`，每格獨立配置、各自 auto-w、FW refit）。
跑完後由新 run dirs 填入並在此更新。舊變體套件數字（5.762/5.766/5.813/5.533/5.562/5.692）退役。
TSB 列（5.5736/18.7272/4.8031）：`or_point_fixed/point_metrics.csv` row TSB。VERIFIED。

## Table 5（tab:walkforward）
baselines + stationary：`or_point_wf_v1/point_metrics.csv`（排除人工 'index' 列）。VERIFIED。
forgetting（taxonomy, 0.95）：`or_point_wf_disc/point_metrics.csv` row TSB-HB。VERIFIED。
REMIX(online)（5.474/17.244/4.761）：`or_point_wf_remix/point_metrics.csv` row TSB-HB。VERIFIED。

## Table 6（tab:significance）
全列：`or_point_fixed/paired_tests.csv`，focal=TSB-HB-Discount，metric=mae（RMSE 陳述同檔 metric=rmse）。
VERIFIED。註：focal 對應 taxonomy+0.95 配置（= §4.3 的 forgetting configuration）。

## Table 7（tab:coverage）
五資料集 Cov@80/AIW@80：`{or,m5,auto,carparts,raf}_prob_select/prob_metrics.csv` row TSB-HB。VERIFIED（raw、B=20、無 calibration、FW）。

## Table 8（tab:neural，附錄）
DRP 列與 h=10/100 RNN/DeepAR 列：QUOTED（rebuttal 期 identical-protocol 比較；MXNet 無法重跑，
正文已標註 indicative；Limitations 已聲明）。

## 行內數字（正文/摘要/圖說）
| 宣稱 | 值 | 來源 | 狀態 |
|---|---|---|---|
| 19,000+ series | 3649+5000+3000+2509+5000=19,158 | 各 dataset 載入輸出 | VERIFIED |
| OR SPL 領先幅度 13% | 12.95% | tab:prob OR Mean 欄 | VERIFIED |
| M5 領先 5% | 5.31% | tab:prob M5 Mean 欄 | VERIFIED |
| monthly 領先 7–14% | 7.2/7.3/13.4% | tab:prob 三欄 | VERIFIED |
| 摘要 M5 RMSSE behind→best | 2.3767(stationary)→2.2724(REMIX) vs AutoTheta 2.3143 | m5_point_fixed + m5_point_remix | VERIFIED |
| 摘要 OR MAE 3.4%→1.6% behind TSB | (5.766−5.574)/5.574=3.45%；(5.664−5.574)/5.574=1.62% | or_point_fixed + or_point_remix | VERIFIED |
| Coverage 0.896 / AIW 10.34（OR raw） | tab:coverage | or_prob_select/prob_metrics.csv | VERIFIED |
| 0.837@8.96 vs 0.845@10.03（皆 calibrated） | or_prob_fixed_discount / or_prob_fixed_paper 之 prob_metrics.csv | VERIFIED |
| uncalibrated≈calibrated pinball 1.9158 vs 1.9192 | or_prob_fixed_uncalibrated + or_prob_fixed_paper | VERIFIED |
| ν 敏感度 0.8841/0.8841/0.8843/0.8844 | or_prob_nu{5,10,50} + or_prob_select(ν=20) | VERIFIED |
| 網格減半：結構 5/5 不變、w 移動 2/5 | `*_point_remix/remix_selection_diagnostics.csv` 重算 | VERIFIED |
| val-ratio 掃描 (global,0.95)/(mixture,0.98)/(mixture,0.95)，OOS MAE 5.53–5.66 | selector_sensitivity.py 輸出 | VERIFIED |
| 選擇結果 (mixture, 0.98/0.90/0.995/0.90/0.997) | `*_point_remix/remix_selection_diagnostics.csv` | VERIFIED |
| walk-forward prob 0.482/1.187/2.265/2.875/2.461 | v1 supplement 協定之 walk-forward prob（`iets_sample_fixed` 時期 outputs） | VERIFIED（歷史 run 檔存在） |
| 效率 0.3ms/6ms/19ms | `or_point_fixed`、`or_point_remix` 的 Efficiency 欄 | VERIFIED（B8 將補硬體規格；0.3ms 不含 B=20 bootstrap，將明寫） |
| TSB Auto MAE 10% off / AutoTheta OR MAE 49% off | tab:point 格值計算 | VERIFIED |
| 落後欄位 ≤8% | carparts 5.9%/3.2%、raf 7.3% | tab:point | VERIFIED |
| Fig.3 折扣曲線 | `*_point_fixed/tsbhb_variant_diagnostics.csv` discount_grid | VERIFIED |
| Fig.4 SPL profile | or_prob_fixed_paper/prob_pinball_scaled.csv | VERIFIED（注意：曲線是 released/calibrated 配置，圖說需與 tab:prob 的 select 配置區辨——見 QUESTIONS Q7*） |

*Q7 追加：spl_profile 圖用 or_prob_fixed_paper（taxonomy w=1 calibrated，mean 0.8900）而 tab:prob REMIX 列用 select（0.8843）。
差異 0.6%，曲線形狀不受影響；已列入待辦：改用 or_prob_select 重繪以完全一致。

## NOT_FOUND / 待跑
- tab:ablation 新格（跑完即補）。
- M5 全量（m5full_point_fast，跑完後 B5 段落引用）。
