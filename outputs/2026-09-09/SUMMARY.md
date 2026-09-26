# SUMMARY — 2026-09-09（強化項 3/5 收尾；五項全部完成）

## 項 3 cold-start 縮放（`docs/DESIGN_coldstart.md`、`c1_coldstart.py`、`c2_coldstart_report.py`）
EBB 對 TweedieGP 的 mean-SPL 領先（%），初始窗截短至最後 L 期，評估區間不變：

| 面板 | 最短 L | … | full |
|---|---|---|---|
| RAF | 6: **+11.3** | 12: +4.1 / 24: +4.3 / 48: +3.4 | +3.0 |
| Auto | 6: **+6.6** | 12: +2.3 | +1.6 |
| Carparts | 6: +2.4 | 12: −0.8 / 24: −1.9 | −0.5 |
| Online Retail | 14: +0.3 | 28: +0.2 / 56: +0.1 | −0.05 |
| M5 | 28: −1.8 | 56: +0.8 / 112: −0.8 / 224: −4.0 / 448: −3.8 | −3.0 |

判定：H1（λ 隨 L 下降）成立於月頻面板；H2（領先隨 L 縮短擴大）在 RAF、Auto 成立、
OR 方向對但幅度可忽略、Carparts 只在最短窗、**M5 不成立**——M5 即使 28 天 λ⁽ᵒ⁾/λ⁽⁺⁾
仍 0.91/0.81，沒有先驗施力空間；TweedieGP 隨歷史加長明顯變好（1.66→1.60）。
全長 sanity 五面板皆與已發布數字一致。已入稿：§5 新小節「Short histories」
（sec:coldstart、fig:coldstart）、附錄 tab:coldstart。

## 項 5 two-rate forgetting（`docs/DESIGN_occurrence_discount.md`、`o1_occdisc_eval.py`）
閘門未過（M5 −0.14%，門檻 1%）；只有 Carparts 選 w⁽ᵒ⁾=0.8（−0.77%，0.3204 反超 TG 0.3214）。
已入稿：app:negative 第三個補救＋tab:occdisc、Limitations(1) 一句。模型改動
`occurrence_fit_discount` 保留，預設 None 純量路徑 sha 不變。

## 五項總結（fixed-origin / walk-forward 主張）
- EBB 與 TweedieGP 是唯二在五面板皆落在最佳競爭者 5% 內的方法（fixed 2.95/3.10%，wf 4.64/4.48%）；EBB 成本低 1–3 個量級、可線上更新。
- ACI-EBB：M5 wf 第一（1.3785）、Auto 第一、Carparts 距 TG 0.2%；OR/RAF 惡化 → 作為列，不取代。
- wf 顯著性：EBB vs TG 在 OR/Carparts/Auto 未定、RAF 顯著。
- 三個模型側補救（形狀、per-item w_i、two-rate）皆未收復 TG 差距 → app:negative。
- cold-start：pooling 的增益跟診斷的槓桿走，不跟頻率走；M5 沒有槓桿。

## 稿件
v5_aistats 32 頁、0 錯誤、0 未解引用、overfull 1（5.1pt 既有）；v5 同步。
新增元素（09-06 至 09-09）：tab:regret、fig:frontier、tab:wfsig、app:negative（三表）、
sec:coldstart+fig:coldstart+tab:coldstart、tab:wfcost、tab:regret-full、tab:wf-*-full ×5。

## 加跑：ACI 層「何時套用」的 leakage-safe 規則（`a1_aci_selector.py`，`aci_selector.csv`）
規則：初始窗 80/20 時序切分，head 初始化 EBB（標籤只看 head），tail 上小型 walk-forward
同時跑 plain 與 ACI，tail 的 scaled pinball（q .5/.75/.9）較低者為部署選擇。
判定：M5、Auto 啟用；OR、RAF、Carparts 不啟用。與事後最佳一致 4/5（Carparts 兩者差 0.5%、未定）。
tail 增益與外部對得上（M5 +7.8% vs +7.2%；OR −7.9% vs −6.3%；RAF −2.4% vs −2.3%）。
**EBB (rule)** 列：OR 0.8540 / Carparts 0.3145 / Auto 0.2901 / RAF 0.2681 / M5 1.3785，
對最佳競爭方法的最大 regret **0.77%（Carparts）**。已入稿：tab:wfprob 新列（不參與排名）、
tab:regret、tab:wfcost、fig:frontier、§5.1 定義句、§5.3 段落改寫、abstract/Intro/Conclusion 一句、
附錄 tab:aci-rule。兩稿 33 頁、0 錯誤。暫存（coldstart_tmp ×3、wf_tmp）已刪，清出 5.6 GB。
