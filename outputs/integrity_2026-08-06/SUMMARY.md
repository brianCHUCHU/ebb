# SUMMARY — integrity_2026-08-06（corrected selection artifact 重建）

> 執行：2026-08-06，兩輪。第一輪任務 1 驗收未過（2/5）→ 停下回報；
> 作者裁決**接受 leakage-safe 結果為新權威**後執行第二輪
> （外部評估重跑、M5 seed 43/44、λ 表、fig10a/b、作廢清理）。
> 規則遵守：不修改模型/超參/選模邏輯、評分全部 import 既有函式、
> 未改動任何 .tex、不利發現照實列。

## 任務狀態（最終）

| 任務 | 狀態 | 輸出 |
|---|---|---|
| 1. 五面板 leakage-safe 聯合選模 | ✅（原驗收預期被否證，作者已改裁決採用實際結果） | `selection_surfaces.csv`、`selected_pairs.csv`、`t1_run_meta.json` |
| 1b. M5 seed 43/44 head-only 選模 | ✅ 三 seed 全部 (mixture, 0.95, K=6) | `selection_surfaces_m5_seeds.csv`、`selected_pairs_m5_seeds.csv`、`t1b_run_meta.json` |
| P0. REMIX 外部評估重跑（修正配置） | ✅（M5 point 經 p0b 補跑，見 NOTES CLI 缺陷） | `remix_corrected_rows.csv`、各 `<panel>_{prob,point}_remix_corrected/`、`p0_run_meta.json`、`p0b_run_meta.json` |
| 2. corrected w* 重算 λ 表 | ✅ | `leverage_dual_lambda_corrected.csv`、`t2_run_meta.json` |
| 3. fig10 雙 panel 重畫 | ✅（模擬點在正確軸上**不再**預測真實點——照實報，見下） | `fig10a_refinement_leverage_size.{pdf,png}`、`fig10b_refinement_leverage_occ.{pdf,png}`、`refinement_leverage_corrected.csv`、`t3_run_meta.json` |
| 4. forgetting gain 十倍差異 | ✅ 已定位來源 | `t4_gain_scan.csv`、NOTES.md §任務4 |
| 清理作廢輸出 | ✅ DEPRECATED_leaky_labels 重命名 + MANIFEST 記錄 | `outputs/aistats2027/MANIFEST.md` 末節 |

## 第二輪關鍵數字（新權威 REMIX 列，`remix_corrected_rows.csv`）

| 面板（配置） | SPL mean | q90 | Cov@80 | AIW@80 | MAE | RMSSE |
|---|---|---|---|---|---|---|
| OR (global, 0.95) | **0.8849** | 1.5268 | 0.892 | 9.25 | 5.5325 | 4.7901 |
| M5 (mixture, 0.95) | **1.6707** | 2.4758 | 0.846 | 2.81 | 1.2032 | 2.2938 |
| Auto (global, 0.99) | **0.2937** | 0.2599 | 0.907 | 9.21 | 3.2145 | 1.1374 |
| Carparts (global, 0.90) | **0.3229** | 0.4817 | 0.931 | 1.47 | 0.5726 | 1.0120 |
| RAF (mixture, 0.997) | **0.2680** | 0.4856 | 0.921 | 0.94 | 2.3874 | 1.6245 |

對 v4 稿現值的方向：OR 0.8910→0.8849（更好）、Auto 0.2957→0.2937（更好）、
Carparts 0.3305→0.3229（更好，但仍輸 TweedieGP 0.3214，差距 2.8%→0.5%）、
RAF 0.2681→0.2680（持平）、**M5 1.6540→1.6707（變差）**——1.6540 是帶洩漏
mixture@0.90 的產物；修正後對 CP-Croston 1.6902 的領先由 2.1% 縮為 **1.2%**。
M5 q90 2.4758（vs CP-Croston 2.1551，Limitation (1) 不變）。tab:coverage 的
M5 缺格可用 0.846 / 2.81 填。Point 端：OR MAE 5.5325（勝過 v4 的 5.6078，
為非退化方法最佳）、M5 MAE 1.2032/RMSSE 2.2938（v4 為 1.1873/2.2885，
變差——1.1873 是舊 select 的產物；RMSSE 仍為全場最佳級距，MAE 需重排名）、
RAF MAE 2.3874（v4 2.2534，變差）。

## M5 多 seed（修正協定）

三個 seed 的 head-only 選擇**完全相同：(mixture, w=0.95, K=6)**。
多 seed 敘述由「折扣在 {0.90,0.95} 平坦帶內移動」升級為「結構、折扣、K
三者跨 seed 全同」——這是修正協定下更強的穩定性陳述（注意：外部 SPL 的
跨 seed 變異需另行重評，舊 `m5_seed_variance_correct.csv` 基於帶洩漏
select@0.90 的預測，不再適用）。

## λ 表（corrected w*，`leverage_dual_lambda_corrected.csv`）

| 面板 | w* | λ⁽⁺⁾ w=1 | λ⁽⁺⁾ w* | λ⁽⁺⁾<0.1 | λ⁽ᵒ⁾ w=1 | λ⁽ᵒ⁾ w* |
|---|---|---|---|---|---|---|
| OR | 0.95 | 0.897 | **0.553** | 14.6% | 0.969 | 0.895 |
| M5 | 0.95 | 0.996 | **0.841** | 12.9% | 0.998 | 0.931 |
| Auto | 0.99 | 0.944 | 0.940 | 0.0% | 0.525 | 0.422 |
| Carparts | 0.90 | 0.802 | **0.224** | 22.5% | 0.882 | 0.628 |
| RAF | 0.997 | 0.926 | 0.917 | 0.0% | 0.085 | 0.106 |

與 v4 表的重大差異：**OR 列整列變動**（w* 0.98→0.95：λ⁽⁺⁾ 0.749→0.553、
<0.1 share 3.2%→14.6%）；M5 列（w* 0.90→0.95：λ⁽⁺⁾ 0.712→0.841、share
18.4%→12.9%）；Auto（w* 0.997→0.99：λ⁽ᵒ⁾ 0.537→0.422）。v4 正文引用的
「four panels retain median λ⁽⁺⁾ ≥ 0.75」在新表下變成 **3/5**（OR 0.553
掉出）；「four of five ≥ 0.71」也不再成立。相關句子需連動改寫。

## fig10a/b 的誠實結論（不利發現，照實列）

在一致的 λ⁽⁺⁾ 軸上（fig10a），**模擬點不再「預測」真實點**：真實面板的
internal refinement gain 全部貼近 0（−0.31% 到 +0.03%），唯一例外 M5
+0.58%；而已知真分割的模擬在 λ≈0.5 可達 +2.2%。OR 修正後移動到
λ⁽⁺⁾=0.553——正處模擬顯示「真分割可獲利」的槓桿區——但 learned 結構
的增益是 −0.05%。正確的圖語義因此從「機制的量化預測」變為：**模擬曲線
是正確結構的上包絡；真實面板全部遠低於包絡 → 槓桿存在但 learned
partition 拿不到，learnability（而非 leverage）是約束**。這與論文的
負面結果主軸一致，但比先前（作廢版 fig10）的「Auto 落在模擬線上」
弱得多，寫作時不得宣稱模擬-真實吻合。fig10b（λ⁽ᵒ⁾ 軸）顯示 RAF/Auto
佔據低 occ-λ 端而增益仍為 0/負——「高槓桿不保證增益」在 occurrence 端
同樣成立。

## 遺留給寫作端的連動修改清單（.tex 未動，需作者確認後執行）

1. tab:prob REMIX 列五格 + M5 領先幅度 2.1%→1.2% + Carparts 對 TweedieGP
   差距 2.8%→0.5%；tab:coverage M5 格 = 0.846/2.81。
2. tab:point REMIX 列（OR 5.5325/4.7901、M5 1.2032/2.2938、RAF 2.3874
   ——M5 MAE 與 RAF MAE 的粗體/排名需重校）。
3. selected pairs：OR (global,0.95)、M5 (mixture,0.95)、Auto (global,0.99)、
   Carparts (global,0.90)、RAF (mixture,0.997)；「only M5 selects the
   learned mixture」→「refined 結構僅在 M5（實質）與 RAF（0.017%，噪音級）
   勝出；OR/Auto/Carparts 選 global」。taxonomy 敘事取消。
4. tab:leverage 整表換 `leverage_dual_lambda_corrected.csv`；
   「four panels ≥ 0.75」→ 3/5；M5 seed 敘述改「三 seed 配置全同
   (mixture, 0.95, K=6)」。
5. fig10 改用 fig10a/b（雙 panel、單一軸語義），caption 按上節誠實結論寫。
6. 「refinement 0.54% vs forgetting 1.4%」→ 單一曲面內正確值：
   M5 corrected @0.95：refinement +0.58% vs forgetting ~9%（global 8.95%）。
7. M5 seed variance 表（外部 SPL ± sd）需用修正配置重跑後才能引用。

## 任務 1：實際 vs 預期（驗收對照，逐面板）

Leakage-safe 協定：taxonomy 與 mixture 標籤只在 selection fitting head
（init 前 80%）上學習；`select_pooling_and_discount` 為既有函式，未改動。

| 面板 | **實際選中** | 預期（v4 稿） | 符合 | legacy（全窗標籤） |
|---|---|---|---|---|
| Online Retail | **(global, 0.95)** | (taxonomy, 0.98) | ❌ | (mixture, 0.98) |
| M5（2/3, seed 42） | **(mixture, 0.95)** | (mixture, 0.90) | 結構✅／**w✗** | (mixture, 0.90) |
| Auto | **(global, 0.99)** | (global, 0.997) | 結構✅／w✗ | (mixture, 0.995) |
| Carparts | **(global, 0.90)** | (taxonomy, 0.90) | ❌ | (mixture, 0.90) |
| RAF | **(mixture, 0.997)** | (taxonomy, 0.997) | ❌ | (mixture, 0.997) |

Wall-clock：OR 84s、M5 259s、Auto 72s、Carparts 38s、RAF 130s（總 ~10 分，
含資料載入；v4 稿寫「275–711 seconds」，量級相近但非同一機器/實作，見 NOTES）。

## 裁決前提的部分證實與部分否證

**證實的部分（方向）**：「全選 mixture 是 leakage 指紋」成立。標籤改為
head-only 後，mixture 的全面優勢消失——Auto 上 mixture 從領先 global 2.2%
（legacy）反轉為**落後 1.7%**；OR/Carparts 的 mixture 同樣掉到 global 之後。
洩漏被移除的效果清楚可見。

**否證的部分（具體配置）**：修正後的選擇不是 v4 稿宣稱的
「OR/Carparts/RAF = taxonomy」：
- OR 與 Carparts 選 **global**（taxonomy 分別只差 +0.051% / +0.082%，在噪音內）；
- RAF 選 **mixture**（taxonomy 差 +0.017%，同樣在噪音內）；
- M5 的 w* 移到 **0.95**（非 0.90；mixture@0.95=1.73362 vs @0.90=1.74542）。

**關鍵量化事實：結構軸的邊際在三個面板上小於 0.1%**（OR 0.05%、Carparts
0.08%、RAF 0.02%），只有 M5（refined 勝 global **+0.58%**）與 Auto（global 勝
refined **+0.31%**）是實質分辨。在 <0.1% 邊際上宣告「哪個結構被選中」
不具統計意義——這既適用於本次的 global/mixture 判定，也適用於 v4 稿的
taxonomy 判定。

## 對 REMIX 不利的發現（照實列）

1. **修正後選擇器在 3/5 面板偏好單一 global pool**（OR、Auto、Carparts）。
   「選擇器做結構分流（structural triage）」的敘事進一步弱化：refined 結構
   實質勝出的只剩 M5（+0.58%）；RAF 的 mixture 勝出（+0.017%）是噪音。
2. **v4 稿宣稱的選中配置在任何一致協定下都無法複現**（本輪 head-only 全套
   不行；「taxonomy 用全窗、mixture 用 head」的混合協定也不行——那會讓 Auto
   選 taxonomy 而非 global，見 NOTES §協定變體）。
3. **M5 的 w\*=0.90 敘述受影響**：修正協定下 seed 42 選 0.95。v4 稿
   「w=0.90 for seeds 42 and 43」與 leverage 表 M5 列 w*=0.90 需重審
   （seed 43/44 的 head-only 重跑未執行）。
4. 十倍差異的來源＝**跨曲面混用**（詳 NOTES 任務 4）：稿中 0.54%/1.4% 無法
   在任何單一曲面以一致定義重現；作者手算引用的 0.75320/0.74434 逐位命中
   **OR legacy 曲面 global 列**（非 M5 seed 44）。在單一修正曲面內的正確值：
   M5 @ w*=0.95 refinement **+0.58%**、forgetting（global, w=1→0.95）**8.95%**
   （mixture 9.46%）。

## 與 v4 稿現有數字的差異點清單

| v4 稿位置 | 稿中值 | 本輪 artifact | 
|---|---|---|
| selected pairs（appendix） | OR (taxonomy,.98) / Carparts (taxonomy,.90) / RAF (taxonomy,.997) / Auto (global,.997) / M5 (mixture,.90) | OR (global,.95) / Carparts (global,.90) / RAF (mixture,.997) / Auto (global,.99) / M5 (mixture,.95) |
| tab:leverage w* 欄 | M5 0.90、Auto 0.997（本 session 稍早依廢棄資料改 0.995，兩者皆與本輪不符） | M5 0.95、Auto 0.99 |
| 「refinement 0.54% vs forgetting 1.4%」 | 無法溯源 | 單一曲面內：0.58% vs 8.95%（global）/9.46%（mixture） |
| 「only M5 selects the learned mixture」 | — | 修正後：M5 與 RAF（後者噪音級）選 mixture，其餘 global |
| tab:prob REMIX OR = 0.8910（taxonomy,.98 外部值） | — | 修正選擇為 (global, 0.95)，對應外部值需重跑才有（未跑） |
| 「Corrected joint selection takes 275–711 seconds」 | 無對應檔案 | 本機實測 38–259s/面板（`t1_run_meta.json`） |

## 停止點與待裁決事項

依指引停在任務 1。需要作者裁決的問題：

1. v4 稿的結構敘事（taxonomy 三面板）**沒有任何 artifact 支持**；本輪
   leakage-safe 結果（global 三面板 + M5/RAF mixture）是否成為新的權威？
2. 若接受本輪結果：tab:prob/tab:point 的 REMIX 列需要用修正配置重跑外部
   評估（OR=global@0.95、M5=mixture@0.95 等），「What gets selected」、
   leverage 表 w*、λ 值全部連動——工作量約等於重出主表。
3. 或者：以「結構邊際 <0.1%、不具分辨力」為由，把結構選擇敘事改寫為
   「修正洩漏後選擇器在多數面板回到單一 pool，refined 結構僅在 M5 有實質
   內部增益且不轉移」——這與 Round 7 的證偽結論一致，且不需要指認
   哪個結構「被選中」。
4. v4 sec:select 描述的「learned candidate 每個 w 各做六次 BIC fit」
   在 repo 程式中不存在（mixture 標籤只建一次、fit_discount=1.0）。
   若前一個 session 的 audit 實作了 per-w mixture 重學，結果可能不同；
   是否要實作該變體驗證，等指示（未經指示不做，避免「換協定去湊敘事」）。
