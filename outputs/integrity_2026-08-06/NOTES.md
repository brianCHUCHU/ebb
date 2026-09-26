# NOTES — integrity_2026-08-06

## 任務 1 實作備註

- 標籤範圍：`_split_init_head_tail(init, val_ratio=0.2)` 的 head（與
  `select_pooling_and_discount` 內部重切的 head 逐位相同——同一函式、
  同一決定性排序），taxonomy 與 mixture 都只看 head。
- 評分：`select_pooling_and_discount` / `_score_discount_grid` 原封 import，
  quantiles=(0.5, 0.75, 0.9)、per-series 分母 = fitting head 的 mean
  absolute demand、item_variance_mode=conjugate、shrink=20.0
  （run_prob CLI 預設）。
- 驗證（global 列不涉及標籤，應與 legacy 逐位一致）：OR corrected global 列
  與 `or_prob_select` legacy global 列**逐位相同**（0.753199 / 0.744335 等）
  ✓ 證明 audit 與歷史 run 用同一評分路徑，差異全部來自標籤範圍。
- mixture 標籤依 repo 現行邏輯**只建一次**（`mixture_group_labels(head, k=0)`，
  fit_discount=1.0），24 個候選共用。v4 sec:select 寫「六次 BIC fit per w」
  ——repo 從未實作 per-w mixture 重學；此差異可能是本輪與前一 session
  audit（若存在）結果不同的原因之一。未經指示不實作該變體。

## 協定變體對照（用既有曲面推算，非新 run）

若 taxonomy 用全窗標籤（unsupervised 特徵規則，可辯不算洩漏）而 mixture 用
head-only：OR → taxonomy（0.74403 < global 0.74434）、Carparts → taxonomy
（0.91191 < 0.91345）、**Auto → taxonomy（0.33025 < 0.33379）≠ v4 的 global**、
RAF → taxonomy 0.98294 vs head-mixture 0.98293（1e-5 差，擲硬幣）。
結論：混合協定同樣無法複現 v4 的五元組（Auto 反例）。v4 的組合
（三 taxonomy + Auto global + M5 mixture）在「全 head-only」與「混合」兩種
一致協定下都不成立。

## 任務 4：forgetting gain 十倍差異——結論

**問題**：稿中「refinement 0.54% / forgetting 1.4%」 vs 本 repo 稍早計算
「refinement 4.6% / forgetting 12.3%」。

**方法**：`t4_gain_scan.csv` 對全部 12 份曲面（5 legacy select、5 份 M5
correct/thirdsplit 各 seed、本輪 5 份 corrected 中含 M5）× 每折扣 × 每種
定義（global→best、global→mixture、taxonomy→mixture、global→taxonomy、
各結構 w=1→w 的 forgetting）做窮舉比對，容差 ±0.03pp。

**發現**：
1. 作者手算引用的 0.75320 → 0.74434（給出 1.18%）**逐位命中 Online Retail
   legacy 曲面的 global w=1 與 global w=0.95**，不存在於任何 M5 曲面。
   手算標註「M5 seed 44」是張冠李戴：那是 OR 的 forgetting gain。
2. 稿中 (0.54%, 1.4%) **無法在任何單一曲面、單一折扣、一致定義下同時重現**。
   1.4% 在全掃描中零命中；0.54% 的近似命中散落在互不相關的
   （曲面, 折扣, 定義）組合。
3. 最接近稿值的單一一致來源是**本輪 corrected M5 曲面 @ 其 w*=0.95**：
   refinement（global→mixture）= **0.579%** ≈ 稿中 0.54%。同曲面同折扣的
   forgetting = global 8.95% / mixture 9.46%——與稿中 1.4% 差一個量級。
   推測：前一 session 的 audit 得到過 ~0.5% 的 refinement（與本輪相容），
   但 1.4% 的 forgetting 來自別的曲面或別的定義（未能定位；OR legacy 的
   global w=1→0.95 為 1.18%，是最可能被誤植的鄰居）。
4. 分母定義檢查：所有 selection 曲面共用同一實作（`_score_discount_grid`），
   分母一律是 fitting head 的 per-series mean absolute demand——**分母定義
   沒有跨 run 差異**。曲面數值量級差異（M5 legacy ~1.06 vs correct ~1.75）
   來自 init/validation 窗不同（1/3 vs 2/3 協定），不是 scale 定義。

**對正文的含義**（稿中比較句的正確寫法）：
- 「refinement vs forgetting」必須在**單一曲面內**計算。以本輪 corrected
  M5 @ w*=0.95 為準：refinement +0.58%、forgetting（同結構 w=1→w*）9.5%
  ——forgetting 仍比 refinement 大一個量級，稿中的定性結論
  （refinement 是二階）**成立且更強**，但 0.54%/1.4% 這對數字本身不可引用。
- 先前（本 session 稍早）算的 4.6%/12.3% 來自 m5_prob_correct_split 曲面
  @ w=0.90——該曲面的 mixture 標籤是全窗學的（帶洩漏），其 4.6% refinement
  同樣不可引用；12.3% 的 forgetting 屬同曲面內一致計算，但基準曲面已廢棄。

## 第二輪執行備註（裁決後）

- **run_point 的 M5 分支忽略 forced 配置**：`--dataset m5` 走
  `_run_m5_point`，該函式只認 `--hb-grouping select` 與
  `--m5-hierarchy-mode`；`--hb-grouping mixture --hb-fit-discount 0.95`
  被靜默忽略，跑成 global @ w=1（P0 首輪 M5 point 因此作廢）。
  以 `p0b_m5_point_fix.py` 依 generic 路徑語義補跑（mixture 標籤在全窗
  @w*、`evaluate_point_models` 評分）。**這是 CLI 缺陷，值得修但本輪
  遵守「不改程式」規則未修。**
- **標籤重學折扣的 code-vs-paper 差異**：v4 sec:select 寫外部 fit 的
  partition 「relearned on the complete training window at ŵ」。
  `run_prob --hb-grouping mixture` 與 run_point generic 路徑確實在 w*
  重學（本輪採用，與稿一致）；但 **select 分支重用選模前在 w=1 建的
  標籤**——若日後用 select 跑外部數字，會與 forced-config 數字有
  細微差異（M5 上可觀察）。寫作時協定描述應與所用路徑對齊。
- P0 各 run wall-clock（`outputs/logs/integrity_p0.log`）：OR prob 217s、
  M5 prob 591s、Auto prob 19s、Carparts prob 14s、RAF prob ~90s；
  point 全部 <120s。t1b：seed 43 246s、seed 44 303s。

## 雜項

- `outputs/aistats2027/` 依指引視為修復前資料，本輪未寫入任何檔案；
  本 session 稍早寫入該目錄的 `selection_surfaces.csv`、
  `refinement_leverage.csv` 與 `paper_v2/figs_v3/fig10_refinement_leverage.*`
  基於帶洩漏的 legacy 曲面，**應視為作廢**（保留原地待作者確認後清理）。
- M5 head-only mixture 的 K=6 與全窗版相同；RAF 的 K 從 6 變 4、Auto 從
  6 變 5（head 資訊較少，BIC 選了較小的 K）。
