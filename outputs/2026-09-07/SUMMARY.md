# SUMMARY — 2026-09-07（強化項 1–2–4：ACI-EBB、wf 顯著性、前沿圖）

> 作者裁決：依序做「ACI 套 EBB → wf 顯著性 → 短歷史縮放 → 前沿圖 → occurrence 折扣」，
> 每項跑完紀錄分析再往下；若值得採用就加入並重跑數據。本檔記錄前三項（1、2、4）。

## 項 1：ACI 套在 EBB 上（`w8_wf_aci_ebb.py`，五面板 wf）

處理與 ACI-ADIDA 完全相同（γ=0.05、per-series×per-quantile 自適應層級、裁切 [1e-3, 0.999]、
每 block 更新一次）；唯一差別是基底分位數函數：讀 EBB 的解析預測分位數在自適應層級的值
（細網格 203 個層級、逐序列插值）。plain EBB 對照列由同一狀態重出，**與已發布數字逐格相同**。

| 面板 | EBB | ACI-EBB | Δ | 最強對手 | 顯著性（ACI-EBB vs EBB） |
|---|---|---|---|---|---|
| Carparts | 0.3145 | **0.3128** | −0.5% | TG 0.3121 | 未定 |
| Auto | 0.2910 | **0.2901** | −0.3% | TG 0.2924 | 顯著較好 |
| RAF | **0.2681** | 0.2743 | +2.3% | TG 0.2801 | 顯著較差 |
| Online Retail | **0.8540** | 0.9082 | +6.3% | TG 0.8526 | 顯著較差 |
| M5 | 1.4851 | **1.3785** | −7.2% | ACI-ADIDA 1.4192 | 顯著較好 |

**判定：ACI-EBB 作為新列（與 ACI-ADIDA 平行對照）加入，EBB 列不變、不取代。**
理由：校準層的效益依面板而定，不是模型屬性；OR/RAF 上 per-series 覆蓋訊號太吵，
反而破壞已校準的預測。若要「何時套 ACI」的規則，需在初始窗內以 leakage-safe 方式選擇，
本輪未做（列為後續）。

兩個可寫進去的事實：
- **同一校準層下 EBB 五面板全勝 ADIDA**（ACI-EBB vs ACI-ADIDA：Carparts/Auto/RAF/M5 顯著、OR 未定 0.9082 vs 0.9108）。
- **M5 的排名由校準層決定**：ACI-EBB 1.3785 > ACI-ADIDA 1.4192 > EBB 1.4851（皆顯著）。EBB 對 ACI-ADIDA 落後 4.4% 不是 forecaster 問題。

成本：ACI 層每面板加幾秒（`wf_cost.csv` ACI-EBB 列：OR 12.0s、Carparts 1.6s、Auto 2.0s、RAF 7.5s、M5 ~50s）。

## 項 2：wf 顯著性（`w9_wf_significance.py` → `spl_significance_wf.csv`、附錄 `tab:wf-significance`）

與 fixed-origin 家族同構（逐序列 mean scaled pinball、配對 t + Wilcoxon、Holm、兩者 <0.05 才算顯著）。

| 面板 | EBB vs TweedieGP | EBB vs ACI-ADIDA |
|---|---|---|
| Online Retail | **未定**（t p=0.87） | 顯著較好 |
| Carparts | **未定** | 顯著較好 |
| Auto | **未定**（t p=0.09） | 顯著較好 |
| RAF | 顯著較好 | 顯著較好 |
| M5 | — | 顯著較差 |

結論：wf 下 EBB 與 TweedieGP 在 OR/Carparts/Auto 三面板統計不可分、RAF EBB 顯著較好。
「兩者不可分、成本差兩個量級」的敘事成立。EBB 家族：OR 6/7 顯著較好、Carparts 9/11、
Auto 9/11、RAF 11/11、M5 5/7（2 較差：ACI-ADIDA、ACI-EBB）。

## Regret 表更新（`regret_wf.csv`，含 ACI-EBB）

加入 ACI-EBB 後「對所有方法」的 regret：EBB wf 最大 7.73%（M5，對自家 ACI-EBB）；
ACI-EBB 最大 6.52%（OR）。**對競爭方法**（排除自家列）：EBB fixed 2.95%、wf 4.64%，
TG 3.10% / 4.48%——「只有這兩者處處在 5% 內」的主張改以「best competing method」口徑陳述，
正文、abstract、Intro、Conclusion 已同步改口。

## 項 4：前沿圖（`w11_fig_frontier.py` → `figs/fig12_frontier.pdf`，正文 fig:frontier）

左：fixed-origin 五面板最大 regret 對 Carparts 每序列牆鐘；右：wf 三月頻面板最大 regret 對
三面板 wf 牆鐘總和（每個方法三面板都有實測）。只有 EBB/ACI-EBB/TweedieGP 在 5% 線下，
EBB 在 TG 左邊約三個量級（wf）。

## 稿件連動（兩目錄，`w12_aci_ebb_tex.py` + `w12b`）

- §5.1 comparators 句加 ACI-EBB 定義；tab:wfprob 加 ACI-EBB 列、caption 改 M5 排名句；
  §5.3 重寫為三段（EBB vs TG 與顯著性 / ACI-EBB 的混合結果與兩個事實 / regret+成本，含 7.7% 自家 regret 的誠實句）；
  tab:regret 含 ACI-EBB 列；fig:frontier 插在 tab:regret 後。
- abstract / Intro / 貢獻(3) / Conclusion：「best method」→「best competing method」，加 ACI-EBB 一句。
- 附錄：E.3 段加 ACI-EBB 逐分位敘述；新表 tab:wf-significance；tab:wfcost 加 ACI-EBB 列；五張全表加 ACI-EBB 列。
- 編譯：v5_aistats 31 頁、0 錯誤、0 未解引用、overfull 1 個 5.1pt（既有）；v5 6 個既有。
- 原文保留：`% [ORIG 2026-09-07]` + `% ORIG|`。

## 檔案（本目錄）
`wf_aci_ebb_<panel>.csv`、`wf_aci_ebb_quantiles_<panel>.csv`、`w8_run_meta_<panel>.json`、
`spl_significance_wf.csv`、`wf_rescored_check.csv`、`w9_run_meta.json`、
`wf_all_panels*.csv`、`tab_*.tex`、`regret_*.csv`、`wf_cost.csv`、`ebb_wf_timing.csv`、`w4/w6_run_meta.json`。

## 下一項
項 3（短歷史／cold-start 縮放實驗）：預註冊設計寫在 `docs/DESIGN_coldstart.md` 後執行。
項 5（occurrence 獨立折扣）之後。
