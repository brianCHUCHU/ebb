# SUBMISSION_CHECKLIST.md（AISTATS 2027）

> 這份檔案只管**投稿前的機械檢查**。專案狀態、貢獻定調、待辦優先順序見 `STATUS.md`。
> 最後同步：2026-07-29（Round 7 後）。

## 等 CFP 確認（F3）
- [ ] 官方 style file 版本與頁數上限（歷年主文 8 頁 + 參考文獻/附錄不計）
- [ ] 是否兩階段（初稿 + supplementary）；checklist 要求
- [ ] 換官方模板後重測頁數；目前 article 兩欄估 9–10 頁 → 需搬遷候選：
      tab:significance / tab:coverage / tab:walkforward 至附錄；§4.6 walk-forward 壓縮

## 投稿前必辦
- [ ] 移除 `\author`（雙盲）；`ANONYMITY_CHECK.md` 復查
- [ ] 建匿名 repo（anonymous.4open.science），回填 Reproducibility Statement 的連結（F4）
- [ ] `fig1_overview.pdf` 確認向量、字體嵌入、灰階可讀（G4）
- [ ] pdflatex ×2 無 error；overfull box 檢視
- [ ] `NUMBERS.md` / `NUMBERS_tables.md` 最終同步（Round 7 若入文須補登）
- [ ] `outputs/aistats2027/MANIFEST.md` 與最終表格對齊（附錄 C.1 引用了它）

## 內容完成度（對照 REMIX_revision_todolist.md）
- [x] A 組全部（A0–A7）
- [x] B4 trivial baselines／B5 全量 M5 30,490／B6 五資料集 CD 圖 + DM 檢定／B7 MASE／
      B8 環境規格／B9 決定論測試
- [x] C1（含 Round 7 的機制與三次證偽）／C2 K 敏感度／C3 horizon 分解
- [x] D1 常數修正（sympy 驗證）／D2 Prop 4 oracle／D3 credibility／D4 Complexity 獨立
- [x] E1–E3 文獻補充與 Related Work 改寫
- [x] F1 編譯問題／F2 雙盲掃描
- [x] H2–H5、H8–H10
- [ ] H1 摘要與貢獻三點最終定稿 —— **等 STATUS.md Q-A（標題／定位）裁決**
- [ ] Round 7 內容入文 —— **等 STATUS.md Q-C（正文 vs 附錄）裁決**
- [ ] Prop 5 雙邊 credibility 界 —— **等 STATUS.md Q-B 裁決**（做則需重編附錄 A）

## 已決定不做（不要再列為缺口）
- B1 GP-Tweedie 重跑 → 引用 + 附錄說明（QUESTIONS Q5）
- B2 外部階層式 baseline → Limitations 明確聲明
- B3 TSB-tuned → 作者自行處理（QUESTIONS Q6）
- DRP 重跑 → 採 (a)：rebuttal 數字 + caveat + Limitations（QUESTIONS Q3）

## 可選加分（時間允許）
- [ ] G1 庫存模擬（newsvendor / base-stock）
- [ ] G2 lead-time demand 分位數評估
- [ ] cold-start scaling law 圖（STATUS.md E1；建議做，成本低）
