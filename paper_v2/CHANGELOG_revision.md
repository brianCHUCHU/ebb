# CHANGELOG_revision.md

依 REMIX_revision_todolist.md（同目錄）執行的修訂紀錄，append-only。
格式：任務 ID｜檔案｜變動（舊 → 新）｜來源。

> 這份檔案只記**做過什麼**。「現在站在哪、下一步做什麼」見 `STATUS.md`；
> 執行輸出的清單見 `outputs/aistats2027/MANIFEST.md`。

## Round 1

- **A0**｜`paper_v2/NUMBERS_tables.md`（新增，90 條）｜Table 1/2 每格 → 來源 CSV 對應，由
  `scripts/analysis/emit_main_tables.py` 程式化產生；標記全部 VERIFIED。完整 NUMBERS.md 見同目錄。
- **A1+A4**｜`main.tex` tab:point、tab:prob｜全表改為 4 位小數、best/secondbest 由程式判定
  （並列自動同標）。修正的錯標：RAF q90 second-best CP-ADIDA 0.491 → iETS 0.4900。
  §4.2 重寫：「eight of ten / no baseline more than two / different leader per panel」（均為假）
  → 「REMIX best×4、top-2 7/10；TSB best×3；落後欄位皆在 8% 內」（每個計數經程式重數）。
- **A2**｜`main.tex` §4.4｜刪除「improves every metric ... however it is configured」（與
  RMSSE 4.725→4.761 矛盾）；腳註併入正文，明寫 REMIX-online MAE 改善/RMSSE 退讓。
- **A6**｜`main.tex` 附錄 C.2｜vs AutoARIMA p_t `<10^{-16}` → `<10^{-15}`
  （來源 or_point_fixed/paired_tests.csv：1.0028e-16）。§4.3 TSB 段改寫：明示
  median diff +0.29（TSB 優）vs mean diff −0.110（我方優）、Wilcoxon 顯著/t 不顯著的正確解讀。
- **A5**｜`main.tex` §4.2、tab:coverage 段落｜正文 0.90/10.3 → 0.896/10.34（引表值）；
  「11% narrower」比較基準明寫為「兩者皆啟用 calibration」；RAF 0.922 過覆蓋點名並指向 Limitations。
- **A7**｜`main.tex` 摘要+貢獻｜「repairs the point-forecast metrics」→ 量化表述
  （M5 RMSSE behind→best；OR MAE 3.4%→1.6% behind TSB）；「sub-millisecond per-item
  inference」→「sub-millisecond base；single-digit ms end-to-end incl. selection」。
- **F1**｜`main.tex`｜`\mathbb{1}`→`\mathbf{1}`；刪 `\bibliographystyle{plainnat}`（與手寫
  thebibliography 衝突）；tab:neural `[h]`→`[t]`；fig1_overview 補 `.pdf`；刪未用
  tikz/arrows.meta/`\newtheorem{remark}`。
- **F2**｜`main.tex`｜「matching the prior-sensitivity findings reported for the stationary
  model」（疑似自指）→ 中性表述；附錄「The v2 additions」→「The forgetting and pooling
  operators」；DRP 數字標註「cannot currently be re-verified, indicative」。
  `ANONYMITY_CHECK.md` 建立。
- **D1**｜`main.tex` Cor.1 + 附錄 A.3｜`(2δ)^{2/3}` 常數不一致修正：變異數極限改正為
  b/(2−b)/4 → b/8 leading order；b* = (16δ²)^{1/3} = 2^{4/3}δ^{2/3}；Corollary 正文改為
  δ^{2/3} 階 + 指向證明中的精確常數。`tests/test_theory_symbolic.py`（sympy 驗證，通過）。
- **B9**｜`tests/test_determinism.py`（新增，通過）：REMIX 全管線兩次執行 bit-identical。

## Round 2（C1 主體完成）

- **C1a**｜tab:ablation 全面重建：每格獨立 run（`ablate_or_{point,prob}_{config}`，各自 auto-w、
  全窗 refit、prob=raw B=20），新增 SPL / SPL_q90 / AIW 欄。關鍵數據：六配置 SPL 帶寬
  0.8826–0.8869（0.5%）；forgetting 軸 AIW 10.51→9.25（12% 銳化）且 coverage 0.901→0.892。
- **C1d**｜定調（todolist 選項二）：mixture 在點與機率指標皆不占優 → §4.3 finding 重寫為
  「partition 選擇二階、pooling 本身與 forgetting 軸一階」；學習型 partition 的價值改述為
  移除手設閾值 + 最稀疏 cold-start 箱 −4.5% + 可解釋性。QUESTIONS Q4 標記為已執行待確認。
- **C1e**｜finding 重排：flat-plane（連結 Prop 4 oracle bound）→ drift-tracking → partition 誠實敘述。
- **C1c**｜cold-start 宣稱以實數入文（CS_0_1 mixture vs global −4.5%），舊的模糊句刪除。
- **B4**｜Zero/Naive 入 tab:point（獨立區塊、不參與標記）+ Zero 入 tab:prob（RAF 加 †）；
  新增「Degenerate baselines as measurement instruments」段：MAE 病理（Zero 三面板最低 MAE，
  含對 TSB MAE 冠軍的再解讀）+ RAF q≤0.9 飽和現象。來源 `trivial_baselines.csv`。
- **A3**｜tab:ablation caption 明寫「每列獨立 run、各自選 w、全窗 refit」——與 Table 1 的
  差異來源（配置不同）就此消解；舊變體套件數字全面退役。
- **C2**｜K∈{2..6} 敏感度（`ablate_or_K*`）：MAE <2.2% 變動、RMSSE <0.1% —— BIC 非關鍵超參數。
- **C3**｜horizon 分解：MAE 分桶誠實檢查（領先方法差 1–2%，對 REMIX 不利）→ 依主指標改用
  RMSE 分桶：REMIX 六桶全部第一；圖+圖說含 MAE 誠實註記。§4.1 加協定正當化與前向指引。
- **B8**｜附錄環境規格 + bootstrap 成本歸屬（0.3ms 不含 B=20；含 B=20 <10ms）。
- **B6**｜CD 圖改用 per-series scaled squared error 排名（避免 MAE zero-collapse 偏差）；
  等 M5 預測檔後出五資料集完整版。DM 檢定輸出 `dm_tests.csv`。
- **B9 追加驗證**｜monthly remix 重跑三資料集數字與 Table 1 完全一致（bit-identical 實證）。

## Round 3 完成

- **B5**｜全量 M5 30,490（快速經典法）：REMIX 三指標全勝（1.1400/2.7072/2.9143），選擇
  (mixture, 0.90) 與樣本一致；正文設計說明 + 附錄表（`m5full_point_fast`）。
- **B6**｜五資料集 CD 圖（16,833 可排名序列；一度誤寫 19,472，經 dm_tests.csv 驗證改正）：
  ADIDA 4.048 / IMAPA 4.127 / REMIX 4.138（與 IMAPA 無統計差異）、TSB 墊底；
  DM 檢定 25/35 顯著優、2/35 顯著劣（Carparts-IMAPA、RAF-ADIDA）。新增小節+圖。
- **B7**｜MASE 附錄小節（Zero 又在三面板最佳——指標病理再證）；`mase_table.csv`。
- **B4 補完**｜M5 的 Zero/Naive 格填入兩大表（M5 上真實方法勝過 Zero，與其他面板對比有意義）。
- **B9 三度驗證**｜M5 remix 重跑 MATCH。

## Round 5 完成

- **H5**｜Limitations 升獨立節、擴為七項（含 B2 無外部階層式比較聲明、RAF 飽和、DRP）。
- **H9**｜over-claim 全掃描：14 處逐條驗證；攔截並實跑驗證了 Limitations 中的 5.508
  （實得 5.5083，`ablate_or_point_global_090`）。
- **H8**｜notation table；**H10**｜術語統一（stationary/forgetting configuration）。
- **F3**｜`SUBMISSION_CHECKLIST.md`；**F4**｜Reproducibility Statement（匿名 repo 連結待建）。

## Round 6（2026-07-29，C1 重啟：mixture 劣勢的機制）

- **C1f**｜`src/models/tsb_hb.py`、`src/experiments/run_{point,prob}.py`｜新增 sample splitting：
  `split_for_hyper_estimation()` + `fit_tsb_hb(hyper_train_df=...)` + CLI `--hb-label-split
  {off,parity,chrono}`（預設 off）。partition 在一半觀測上學、群 hyper 在另一半估、逐 item
  充分統計量仍用全窗。`parity`（交錯）為預設：兩半都橫跨整個窗，折扣輪廓與漂移在兩半中同時保留；
  新增 `LAG_COLUMN` 讓子樣本保留相對原始 forecast origin 的 lag，折扣權重不受切分扭曲。
  selector（`select_fit_discount` / `select_pooling_and_discount`）同步吃 `hyper_split`，
  確保「評分的規格 = 實際擬合的規格」。
  **回歸驗證**：`--hb-label-split off` 與既有 `ablate_or_point_mixture_auto` 五個指標
  逐位元相同（`scratch_regress_off`）。
- **C1g**｜診斷結論（推翻 Round 2 的 C1d 歸因）｜mixture 落後**不是**「二階、無害」，而是
  learned partition 下 τ² 塌到 1e-6 下限 → k_eff = σ²/τ² 達 10⁵–10⁶ → credibility λ→0 →
  群內每個 item 的自身資料被丟棄。OR、w=0.98：3,649 條中 1,182 條（32%）λ 精確為 0；
  λ 中位數 global 0.749 vs mixture 0.356，λ>0.9 的比例 15.1% vs 0.0%。
  跨面板一致：mixture vs global（皆 auto-w）MAE，OR +2.9%、Auto +3.6%、RAF +13.0%、
  Carparts −2.7%（唯一勝出）。
- **C1h**｜機制查明｜**不是** post-selection 偏誤：parity split 只把 OR MAE 從 5.6925 拉到
  5.6264（RMSSE 4.79022 → 4.78590），塌陷群數 3→2，λ 中位數甚至微降（切一半後抽樣變異更大）。
  真正機制是**變異成分的邊界退化**：群內 mean_log 觀測變異 < 平均抽樣變異 σ²/n_i，動差／REML
  估計為負而被截斷。折扣把有效 n_pos 中位數從 17 壓到 6.8（w=0.98）／1.55（w=0.90），
  塌陷群數隨之 1/5 → 3/5 → 4/5。**forgetting 觸發 pooling 塌陷，兩軸在現行參數化下相互破壞。**
- **C1i**｜修法方向驗證（診斷用，不可入文）｜既有 `--hb-group-shrink-strength`（把群 hyper 往
  全域收）即可翻盤：OR mixture+auto-w，強度 0 → 500 → 3000 給出 MAE 5.6925 → 5.5674 → 5.5220、
  RMSSE 4.79022 → 4.78901 → 4.78723，**同時勝過 global pool 的 5.5325 / 4.79005**；
  強度→∞ 退回 global，故 3000 為內點最優而非內插。**但該強度是對測試集挑的**（違反 R1/R2 精神），
  入文版本必須改為驗證切分上選、或改用不需調參的 τ² 先驗。輸出：`scratch_mix_shrink{500,3000}`、
  `scratch_mix_split_parity`、`scratch_regress_off`。

## Round 7（2026-07-29，路線 (a)：兩個修法都實作並量測，結論為不成立）

- **C1j**｜`_credibility_shrink_tau()` + CLI `--hb-hyper-shrink {off,credibility}`（預設 off）｜
  以 Bühlmann credibility 把群變異成分往全域值部分收縮：
  ω_g = V_0/(V_0 + Var(τ̂²_g))，Var(τ̂²_g) 取變異成分估計式的漸近變異
  1/(0.5 Σ_i (τ²+σ²/n_i)^-2)（在穩定點 τ̂²_0 上求值以免落在邊界），
  V_0 為群間離散度扣掉該估計噪音。**無任何可調參數**，兩個量都由同一個初始窗讀出；
  單一 pool 時為恆等映射。
  **機制驗證（OR, mixture）**：k_eff 上限 w=0.98 917,811 → 10.8、w=0.90 1,714,949 → 23.3；
  塌陷群數 2/3 → 0；λ≈0 的序列比例 w=0.90 37.7% → 0.7%。**塌陷完全消除。**
- **C1k**｜**但總體指標幾乎不動**（OR fixed，auto-w，hb_only）：

  | 配置 | MAE | RMSE | RMSSE |
  |---|---|---|---|
  | global（無正則化） | 5.5325 | 17.839 | 4.79005 |
  | global + credibility | 5.5325 | 17.839 | 4.79005 |（no-op 驗證通過）
  | taxonomy | 5.5624 | 17.842 | 4.78971 |
  | taxonomy + credibility | 5.5628 | 17.842 | 4.78971 |
  | mixture | 5.6925 | 17.876 | 4.79022 |
  | mixture + credibility | 5.6949 | 17.872 | 4.79008 |
  | mixture + parity split | 5.6264 | 17.760 | 4.78590 |
  | mixture + parity + credibility | 5.6275 | 17.759 | 4.78584 |

  機率端（OR，raw、B=20、未校準、mixture）：SPL mean 0.8869（off）→ 0.8866（credibility）
  → 0.8848（credibility+split）；q90 1.5343 → 1.5333 → 1.5283。六配置 ablation 帶寬為
  0.8826–0.8869，故兩項修法合計只把 mixture 從帶底推到帶中，**仍非最佳**。
- **C1l**｜`--hb-group-shrink-strength 3000` 的 5.5220 已判定為**測試集過擬合產物，不可用**｜
  該設定實質是把群 hyper 往全域收 71–89%。對群均值 μ_g 套用同一套 credibility 公式，
  資料給出的權重是 **ω ≈ 0.999**（群間 Var(μ)=0.7674，估計誤差僅 0.00059）——
  也就是說 μ_g 是統計上被高度確定的量，任何資料驅動準則都不會丟棄它。
  ω≈0.11–0.29 只可能來自對測試集搜尋。**該數字退役，不入文。**
- **C1m**｜**路線 (a) 判定不成立**（作者裁決 Q4 選 (c)：先 (a) 不成落回 (b)）｜
  三個獨立假設均已證偽：(i) post-selection 依賴（sample splitting，回收 ~1.2% MAE / 0.2% SPL）、
  (ii) 變異成分邊界退化（credibility，塌陷全消但總體 <0.3%）、(iii) 群 hyper 過強
  （資料給的權重 ≈1，無收縮空間）。結論：learned partition 在總體指標上不占優**不是實作缺陷**，
  而是這些面板的樣本量下 item 自身資料已主導先驗（global pool 的 λ 中位數 0.75–0.90），
  partition 沒有施力空間。這比 Round 2 的「二階」定調強得多，且經得起追問。
  → 進 (b)：維持謙虛定位，但把上述機制與三次證偽寫入論文；並建議補 cold-start scaling law
  （截短初始窗，驗證增益 ∝ 1/n_eff）把負面結果轉成被理論預測的定量規律。
  輸出：`scratch_cred_or_{global,taxonomy,mixture}`、`scratch_cred_or_mixture_split`、
  `scratch_prob_mix_{off,cred,credsplit}`。

## Round 8（2026-07-29，Q-A 保留標題、Q-B 做 Prop 5，正文改寫）

- **T9 / Prop 5**｜`main.tex` 新增 §3.x「How much structure is estimable?」｜
  **Proposition 5（雙邊 credibility；塌陷事件）** + **Corollary 5.1（learned partition 的解析度上限）**：
  (i) λ_i ≤ [1+(1−w)κ_g]⁻¹ < 1；(ii) E[S_g²]=τ_g²+s̄_g（不等 n_i 亦精確），
  Pr[τ̂_g²=0] ≃ Φ(−√((m_g−1)/2)·ρ_g)，ρ_g=τ_g²/(τ_g²+s̄_g)，且 n_i≤(1−w)⁻¹ ⇒ s̄_g≥(1−w)σ_g²
  ⇒ **折扣直接壓低可識別比 ρ_g**；(iii) 變異成分部分匯集後 λ_i ≥ n_i(1−ω_g)τ̂_0²/(…) > 0。
  Cor 5.1：可估條件 ρ_g√(m_g−1) ≳ 1；分群越細（τ_g²↓、m_g↓）、遺忘越強（(1−w)σ_g²↑）越快越界；
  越界後正則化估計退回**全域 pool 而非群平均**——這正是 partition 風險面平坦的理由，
  也把 Prop 4 的 oracle 界與經驗觀察接起來。附錄 A 補完整證明。
- **T9 驗證**｜`tests/test_theory_symbolic.py` 新增 5 個檢查（全過）：(i) 上界、
  (ii) E[S²] 精確無偏（不等 n_i，符號驗證）、(iii) 下界與其對 ω 的單調性、
  ρ_g 對 (1−w) 的單調性、以及 (ii) 的常態近似**對 Monte Carlo 模擬**（20,000 抽樣 × 3 組參數，
  機率誤差 <0.05）。
- **T8 / Q-A**｜標題保留 "Learning to Pool and to Forget"。摘要與貢獻三點依 STATUS §1 改寫：
  貢獻 (2) 改為「選擇器的理論 + 可選擇範圍的上限」，貢獻 (3) 改為「分開量測兩軸並報告哪一個有效」，
  明寫三次修復皆未使 learned partition 勝出、且這正是 Cor 5.1 所預測。
- **T10 / Q-C**｜§4.3 新增「The flatness is a resolution limit, not an estimation defect」段落
  （正文），機制與完整格子進附錄新節 `app:resolution`（2 表 + 4 段）。
- **E1 cold-start scaling law**（用既有 slice 資料，無需新跑）｜learned vs global 的 MAE 差
  依正觀測數分箱：OR **+24.5/+19.5/+17.8/+7.1/−4.2%** 單調遞減；Carparts
  −33.2/+7.6/+9.3/+4.2/+1.1%；RAF 全負但幅度遞減 −13.4/−16.7/−13.3/−9.3%。
  **符號取決於群先驗是否優於全域先驗，幅度隨自身樣本數衰減——四個面板皆然**，即 Cor 5.1 的預測。
- **E2 五面板修復格子**（`scratch_e2_{auto,carparts,raf}_{base,reg}`）｜learned + split + credibility：
  RAF MAE 2.3874 → **2.2181（−7.1%，塌陷最嚴重的面板修復最多）**；Auto 3.3293 → 3.3317（RMSSE
  1.14726 → 1.14202）；Carparts 0.5571 → 0.5818（**+4.4% 變差**，該面板是唯一 learned 原本領先者）；
  OR 5.6925 → 5.6275。**沒有任何面板因此超越 single pool。**誠實寫入 `tab:repairs`。
- **識別性統計量表**｜`tab:identifiability`：以**未截斷**的 S_g²−s̄_g 計算 z_g（避免用截斷值
  自證），OR 五個 component 在 w=1.00/0.98/0.95/0.90 下的 z_g，塌陷數 1/5 → 3/5 → 3/5 → 4/5。
- **Limitations**｜新增 (1b)（Cor 5.1 的高斯／χ² 近似邊界、它不說明門檻以下哪個 partition 最好、
  以及「換一個分群目標函數是否能繞過上限」是下一個實驗）；(3) 補上 soft-responsibility
  **是唯一不受 Cor 5.1 同樣約束的擴充**（它改變 predictive 的形狀而非只換收縮目標）。
- 靜態檢查通過（env/refs/bibitems/brace/dollar 全清）；兩個測試檔在改動後皆通過。
- **未解**：正文再增約 1.5 頁，AISTATS 8 頁上限的壓力升高，搬遷計畫見 `SUBMISSION_CHECKLIST.md`。

## Round 2（其餘進行中）

- **A3**｜判定：Table1 vs 舊 ablation 差異 = 配置不同（w 與 label-learning stats），非 fit window。
  處置：ablation 全格以獨立 CLI 配置重跑（`ablate_or_{point,prob}_*`，各格 w 自選、全窗 refit），
  詳見 QUESTIONS.md Q1。
- **B5**｜`run_point.py` 新增 `--point-baselines` 子集參數；M5 全量 30,490（快速經典法+REMIX）
  執行中（`m5full_point_fast`）。
