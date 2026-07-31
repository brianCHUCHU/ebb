# QUESTIONS.md — 交回作者裁決

> 這份檔案只管**等作者裁決的事項**。專案狀態、貢獻定調、待辦優先順序見 `STATUS.md`
> （其 §3 是目前尚未解決問題的權威清單；本檔保留逐題的完整脈絡）。

> **2026-07-29 作者裁決**：Q1 接受處置；Q2 描述屬實；Q3 採 (a)（維持 rebuttal 數字 + caveat +
> Limitations 聲明）；Q5 不重跑，改以引用 + 附錄說明處理；Q6 TSB-tuned 暫緩。
> Q4 另行處理中——見下方「Q4 後續」。

## Q1（A3）Table 1 REMIX(OR) 5.6639 vs 舊 ablation `learned+auto` 5.6925 的差異來源 — 已查明，請確認處置
兩者**都是 full-window refit**，差異不是 fit window，而是配置本身：
- Table 1 REMIX = 聯合選擇 (mixture, w=0.98)，mixture labels 在未折扣統計量上學習。
- 舊 ablation `learned+auto` = 變體套件的 Mix-Disc：w=0.95（由 taxonomy 配置選出後共用），labels 在**折扣後**統計量上學習。
處置（已執行）：重跑 ablation 全格——每格獨立用 CLI 配置（w 各自 auto 選擇、labels 各自學習、全窗 refit），caption 明寫協定。舊變體套件數字退役。
**請確認接受此處置**；若你偏好保留舊變體定義，告知我回退。

## Q2（B5）M5 只用 5,000 條的原因
我的理解：v1 時期的算力考量（AutoARIMA 在全量上慢）。已跑全量 30,490 條（快速經典方法 + REMIX；AutoARIMA/AutoTheta/深度模型仍為 5,000 樣本），論文將明寫此設計。**請確認這個描述符合實情。**

## Q3（F2/R4）DRP 的 MXNet 環境
py3.13 無法安裝 mxnet（專案已停止維護）。選項：(a) 維持現狀（rebuttal 數字 + 明確 caveat + Limitations 聲明）；(b) 投入時間用 repo pyproject 的 py3.10 uv 環境重跑；(c) 用 PyTorch 重實作 DRP。目前採 (a)。**要升級到 (b)/(c) 請指示。**

## Q4（C1d）learned pooling 的定位
若 ablation SPL 欄顯示 mixture 在機率指標上也不占優，將誠實改寫 finding (i) 為「partition 選擇無害但非決定性，增益主要來自 forgetting 軸」，並同步下修摘要與貢獻 (2) 的措辭。**預先請示：接受這個較謙虛的定位嗎？**（等 ablation grid 跑完有數字後我會回報實際情況再定稿。）

## Q4 後續（2026-07-29）：mixture 劣勢的機制已查明，定位待重新裁決
上一輪把 finding (i) 定調為「partition 選擇二階」。診斷顯示這個描述掩蓋了一個實際缺陷：
learned partition 下 **size 的群間變異成分 τ² 塌到下限**，使 credibility
λ_i = n_pos/(n_pos + σ²/τ²) 歸零 —— 該群每個 item 的自身資料被完全丟棄、一律預測為群平均。
Online Retail 在 selector 實選的 w=0.98 下，五個 component 有三個塌陷。

機制**不是** post-selection 偏誤（sample splitting 只回收一小部分，見 CHANGELOG C1f），
而是**變異成分的邊界退化**：群內 mean_log 的觀測變異 < 平均抽樣變異 σ²/n_i，動差／REML 估計為負
而被下限截斷。折扣把有效 n_pos 從中位 17 壓到 6.8（w=0.98）與 1.55（w=0.90），抽樣變異因此暴增
—— 也就是說**是 forgetting 觸發了 pooling 的塌陷**，兩個軸在現行參數化下相互破壞。

**這解釋了 ablation 為何顯示兩軸不疊加，且提示正確修法是正則化 τ²**（往全域 τ² 部分收縮，
或給 τ² 一個真正的先驗）。粗鈍的既有旋鈕 `--hb-group-shrink-strength` 已可讓 mixture
在 OR 上 MAE 5.6925 → 5.5220、RMSSE → 4.78723，**同時勝過 global pool（5.5325 / 4.79005）**，
且該點是內點最優（強度→∞ 會退回 global）。註：該強度是以測試集挑的，僅為診斷，不可入文；
入文版本必須在初始窗驗證切分上選，或改用不需調參的先驗。

**作者裁決：(c) 先 (a)，不成落回 (b)。**

**(a) 已執行完畢，判定不成立**（詳見 CHANGELOG Round 7）。τ² 正則化（Bühlmann credibility，
無可調參數）把塌陷完全消除（k_eff 上限 917,811 → 10.8），但 OR 總體指標變動 <0.3%；
加上 sample splitting 也只把 mixture 從 ablation 帶底推到帶中，仍非最佳。第三個假設
「群 hyper 過強」亦證偽：對 μ_g 套同一公式，資料給的權重 ω≈0.999，沒有收縮空間；
先前 `--hb-group-shrink-strength 3000` 的 5.5220 因此判定為測試集過擬合產物，**已退役**。

**→ 進入 (b)**。定位維持謙虛，但論述基礎從「二階」升級為三次獨立證偽 + 機制解釋：
在這些面板的樣本量下 item 自身資料已主導先驗（global pool 的 λ 中位數 0.75–0.90），
learned partition 沒有施力空間；其價值集中在 λ 明顯小於 1 的最稀疏切片（cold-start −4.5%）。
**待確認**：是否加做 cold-start scaling law（截短初始窗至 14/28/56，驗證增益 ∝ 1/n_eff）
把這個負面結果轉成被 credibility 公式預測到的定量規律？成本低，且能同時支撐 Prop 4 的敘事。

## Q5（B1）GP-Tweedie 對照
TweedieGP（Damato et al. 2025）的開源實作需要 GPyTorch 環境與逐序列 GP 訓練（他們報告與 iETS 同量級的訓練時間，5 資料集全跑估計數天）。最低限度方案：引用其論文在相同 last-h holdout 上的原始數字——但其 SPL scaling 定義需逐一核對才能同表。目前論文尚未加入。**請裁決：投入重跑（需時間）或先以引用+附錄說明處理？**

## Q6（B3）TSB-tuned
你說明已自行調參並將自行補附錄。主表是否要加 `TSB-tuned` 列？加的話請提供調參後的預測結果檔或參數設定，我重跑對齊協定。
