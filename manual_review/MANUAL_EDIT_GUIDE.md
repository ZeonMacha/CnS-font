# CnS-Manual-Master-Bank 人工精修指引

本目录下的 [`CnS-Manual-Master-Bank.otf`](./CnS-Manual-Master-Bank.otf) 是全套 **8,105 《通用规范汉字表》** 的高复用核心母件字库（仅包含下列约 116 个核心字形，打开极快，可直接用 **FontCreator / FontLab / Glyphs / FontForge** 编辑）。

## 修改后如何自动传播到全部 8,105 字？

编辑保存 `manual_review/CnS-Manual-Master-Bank.otf`（**保持每个字的 Unicode 码位不变**）后，只需重新运行一条构建命令：

```bash
python scripts/v2_build_tg8105.py --master_otf manual_review/CnS-Manual-Master-Bank.otf
```

全库合成器会优先读取你修改过的母件，自动把修改传播至所有下游形声字、合体字与异形字！

## 收录字形分组与优先精修建议

1. **第一优先级 — 12 个高频偏旁母件**（单个修改影响 50~400 个下游合体字）：
   - `讠钅纟饣辶亠礻衤忄艹冫阝`
   - 建议重点检查 `讠`（点与折提比例）、`钅`（三横间距与提笔角度）、`纟`（两撇折交错锐度）、`饣`、`辶`（国标一点走之底）。
2. **第二优先级 — 独体与高频简体根字（74 字）**：
   - `车贝见页马鸟鱼风韦齿龙长东专书乐为发乌义乡习买农仓仑佥尧乔夹执肃头丝亚严两丧业丛丢伞丌卫丑丐电飞办丰丽举师军宾责员岂齐卖卢刍寻罗聂婴赞华黾监厌鲁冈奂`
   - 建议重点审阅独体标杆字：`伞 (U+4F1E)`、`丌 (U+4E0C)`、`习 (U+4E60)`、`乡 (U+4E61)`、`卫 (U+536B)`、`专 (U+4E13)`、`书 (U+4E66)`、`东 (U+4E1C)`。
3. **第三优先级 — 中日字形差异母件（30 字）**：
   - `言直令骨微真反刃每衣食鬼角免象曾半平户羽道海神社达迁关异运亏`
   - 含 `言 (U+8A00)`、`直 (U+76F4)`、`令 (U+4EE4)`、`骨 (U+9AA8)`、`微 (U+5FAE)`、`道 (U+9053)` 等高频陆标差异原字。

## 节点 Wireframe 预览图

![Master Bank Wireframe](./master_bank_wireframe.png)

## 各母件生成来源与节点统计

| 字符 | 码位 | 合成路径 | On-Curve 点数 | 直线中段冗余节点 |
| :---: | :---: | :--- | :---: | :---: |
| 伞 | `U+4F1E` | `Tier2_Atomic(全+十+米)` | 57 | 0 |
| 丌 | `U+4E0C` | `Tier2_Atomic(二+川)` | 22 | 0 |
| 言 | `U+8A00` | `Tier0_TopDotSwap(言)` | 111 | 1 |
| 讠 | `U+8BA0` | `Tier3_HessianSDF` | 69 | 0 |
| 钅 | `U+9485` | `Tier3_HessianSDF` | 77 | 0 |
| 纟 | `U+7E9F` | `Tier3_HessianSDF` | 78 | 0 |
| 饣 | `U+9963` | `Tier3_HessianSDF` | 72 | 0 |
| 辶 | `U+8FB6` | `Tier3_HessianSDF` | 62 | 0 |
| 亠 | `U+4EA0` | `Tier3_HessianSDF` | 37 | 0 |
| 礻 | `U+793B` | `Tier3_HessianSDF` | 67 | 0 |
| 衤 | `U+8864` | `Tier3_HessianSDF` | 79 | 0 |
| 忄 | `U+5FC4` | `Tier3_HessianSDF` | 67 | 0 |
| 艹 | `U+8279` | `Tier3_HessianSDF` | 64 | 0 |
| 冫 | `U+51AB` | `Tier3_HessianSDF` | 58 | 0 |
| 阝 | `U+961D` | `Tier3_HessianSDF` | 55 | 0 |
| 车 | `U+8F66` | `Tier3_HessianSDF` | 88 | 0 |
| 贝 | `U+8D1D` | `Tier3_HessianSDF` | 72 | 0 |
| 见 | `U+89C1` | `Tier3_HessianSDF` | 110 | 0 |
| 页 | `U+9875` | `Tier3_HessianSDF` | 101 | 0 |
| 马 | `U+9A6C` | `Tier3_HessianSDF` | 94 | 0 |
| 鸟 | `U+9E1F` | `Tier3_HessianSDF` | 96 | 0 |
| 鱼 | `U+9C7C` | `Tier3_HessianSDF` | 118 | 0 |
| 风 | `U+98CE` | `Tier3_HessianSDF` | 103 | 0 |
| 韦 | `U+97E6` | `Tier3_HessianSDF` | 94 | 0 |
| 齿 | `U+9F7F` | `Tier3_HessianSDF` | 116 | 0 |
| 龙 | `U+9F99` | `Tier3_HessianSDF` | 112 | 0 |
| 长 | `U+957F` | `Tier3_HessianSDF` | 96 | 0 |
| 东 | `U+4E1C` | `Tier3_HessianSDF` | 125 | 0 |
| 专 | `U+4E13` | `Tier3_HessianSDF` | 99 | 0 |
| 书 | `U+4E66` | `Tier3_HessianSDF` | 87 | 0 |
| 乐 | `U+4E50` | `Tier3_HessianSDF` | 110 | 0 |
| 为 | `U+4E3A` | `Tier3_HessianSDF` | 102 | 0 |
| 发 | `U+53D1` | `Tier3_HessianSDF` | 133 | 0 |
| 乌 | `U+4E4C` | `Tier3_HessianSDF` | 93 | 0 |
| 义 | `U+4E49` | `Tier3_HessianSDF` | 83 | 0 |
| 乡 | `U+4E61` | `Tier3_HessianSDF` | 94 | 0 |
| 习 | `U+4E60` | `Tier3_HessianSDF` | 75 | 0 |
| 买 | `U+4E70` | `Tier3_HessianSDF` | 126 | 0 |
| 农 | `U+519C` | `Tier3_HessianSDF` | 120 | 0 |
| 仓 | `U+4ED3` | `Tier3_HessianSDF` | 98 | 0 |
| 仑 | `U+4ED1` | `Tier3_HessianSDF` | 83 | 0 |
| 尧 | `U+5C27` | `Tier3_HessianSDF` | 130 | 0 |
| 乔 | `U+4E54` | `Tier3_HessianSDF` | 98 | 0 |
| 夹 | `U+5939` | `Tier3_HessianSDF` | 104 | 0 |
| 肃 | `U+8083` | `Tier3_HessianSDF` | 110 | 0 |
| 头 | `U+5934` | `Tier3_HessianSDF` | 109 | 0 |
| 丝 | `U+4E1D` | `Tier3_HessianSDF` | 129 | 0 |
| 严 | `U+4E25` | `Tier3_HessianSDF` | 97 | 0 |
| 两 | `U+4E24` | `Tier3_HessianSDF` | 124 | 0 |
| 丧 | `U+4E27` | `Tier3_HessianSDF` | 128 | 0 |
| 业 | `U+4E1A` | `Tier3_HessianSDF` | 94 | 0 |
| 卫 | `U+536B` | `Tier3_HessianSDF` | 81 | 0 |
| 丑 | `U+4E11` | `Tier3_HessianSDF` | 81 | 0 |
| 丐 | `U+4E10` | `Tier3_HessianSDF` | 77 | 0 |
| 电 | `U+7535` | `Tier3_HessianSDF` | 94 | 0 |
| 飞 | `U+98DE` | `Tier3_HessianSDF` | 61 | 0 |
| 办 | `U+529E` | `Tier3_HessianSDF` | 97 | 0 |
| 丰 | `U+4E30` | `Tier3_HessianSDF` | 77 | 0 |
| 丽 | `U+4E3D` | `Tier3_HessianSDF` | 110 | 0 |
| 举 | `U+4E3E` | `Tier3_HessianSDF` | 137 | 0 |
| 师 | `U+5E08` | `Tier3_HessianSDF` | 102 | 0 |
| 齐 | `U+9F50` | `Tier3_HessianSDF` | 103 | 0 |
| 卢 | `U+5362` | `Tier3_HessianSDF` | 72 | 0 |
| 刍 | `U+520D` | `Tier3_HessianSDF` | 82 | 0 |
| 监 | `U+76D1` | `Tier3_HessianSDF` | 122 | 0 |
| 直 | `U+76F4` | `Tier3_HessianSDF` | 94 | 0 |
| 令 | `U+4EE4` | `Tier3_HessianSDF` | 115 | 0 |
| 骨 | `U+9AA8` | `Tier3_HessianSDF` | 132 | 0 |
| 微 | `U+5FAE` | `Tier3_HessianSDF` | 176 | 0 |
| 真 | `U+771F` | `Tier3_HessianSDF` | 137 | 0 |
| 反 | `U+53CD` | `Tier3_HessianSDF` | 113 | 0 |
| 刃 | `U+5203` | `Tier3_HessianSDF` | 70 | 0 |
| 每 | `U+6BCF` | `Tier3_HessianSDF` | 123 | 0 |
| 衣 | `U+8863` | `Tier3_HessianSDF` | 106 | 0 |
| 食 | `U+98DF` | `Tier3_HessianSDF` | 130 | 0 |
| 鬼 | `U+9B3C` | `Tier3_HessianSDF` | 137 | 0 |
| 角 | `U+89D2` | `Tier3_HessianSDF` | 121 | 0 |
| 免 | `U+514D` | `Tier3_HessianSDF` | 129 | 0 |
| 象 | `U+8C61` | `Tier3_HessianSDF` | 165 | 0 |
| 曾 | `U+66FE` | `Tier3_HessianSDF` | 124 | 0 |
| 半 | `U+534A` | `Tier3_HessianSDF` | 89 | 0 |
| 平 | `U+5E73` | `Tier3_HessianSDF` | 94 | 0 |
| 户 | `U+6237` | `Tier3_HessianSDF` | 71 | 0 |
| 执 | `U+6267` | `Tier1[D:挿,D:丸]` | 96 | 3 |
| 责 | `U+8D23` | `Tier1[D:青,M:贝]` | 147 | 2 |
| 员 | `U+5458` | `Tier1[D:虽,M:贝]` | 98 | 0 |
| 寻 | `U+5BFB` | `Tier1[D:彗,D:寺]` | 113 | 3 |
| 赞 | `U+8D5E` | `Tier1[D:兟,M:贝]` | 181 | 0 |
| 冈 | `U+5188` | `Tier1[D:冂,D:㐅]` | 56 | 2 |
| 羽 | `U+7FBD` | `Tier1[M:习,M:习]` | 127 | 0 |
| 道 | `U+9053` | `Tier0_IDS(道)` | 133 | 2 |
| 海 | `U+6D77` | `Tier0_IDS(海)` | 158 | 0 |
| 神 | `U+795E` | `Tier0_IDS(神)` | 125 | 3 |
| 社 | `U+793E` | `Tier0_IDS(社)` | 126 | 2 |
| 达 | `U+8FBE` | `Tier1[M:辶,D:大]` | 105 | 0 |
| 迁 | `U+8FC1` | `Tier1[M:辶,D:千]` | 107 | 0 |
| 异 | `U+5F02` | `Tier1[D:巳,D:舁]` | 89 | 0 |
| 亏 | `U+4E8F` | `Tier1[D:三,D:号]` | 62 | 0 |
| 佥 | `U+4F65` | `Tier3_HessianSDF` | 114 | 0 |
| 亚 | `U+4E9A` | `Tier3_HessianSDF` | 94 | 0 |
| 丛 | `U+4E1B` | `Tier3_HessianSDF` | 82 | 0 |
| 丢 | `U+4E22` | `Tier3_HessianSDF` | 103 | 0 |
| 军 | `U+519B` | `Tier3_HessianSDF` | 114 | 0 |
| 宾 | `U+5BBE` | `Tier3_HessianSDF` | 147 | 0 |
| 岂 | `U+5C82` | `Tier3_HessianSDF` | 103 | 0 |
| 卖 | `U+5356` | `Tier3_HessianSDF` | 158 | 0 |
| 罗 | `U+7F57` | `Tier3_HessianSDF` | 92 | 0 |
| 聂 | `U+8042` | `Tier3_HessianSDF` | 163 | 0 |
| 婴 | `U+5A74` | `Tier3_HessianSDF` | 189 | 0 |
| 华 | `U+534E` | `Tier3_HessianSDF` | 125 | 0 |
| 黾 | `U+9EFE` | `Tier3_HessianSDF` | 111 | 0 |
| 厌 | `U+538C` | `Tier3_HessianSDF` | 105 | 0 |
| 鲁 | `U+9C81` | `Tier3_HessianSDF` | 130 | 0 |
| 奂 | `U+5942` | `Tier3_HessianSDF` | 121 | 0 |
| 关 | `U+5173` | `Tier3_HessianSDF` | 112 | 0 |
| 运 | `U+8FD0` | `Tier3_HessianSDF` | 113 | 0 |
