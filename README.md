# comfyui-prompt-skill

两个 ComfyUI 节点：**提示词优化**（调 LLM，带提示词工程规范）+ **提示词合规自检**（纯本地、免费）。

主要面向 **FLUX.2 [dev]**（32B 开源版，本地 ComfyUI）和 **Qwen-Image 2.1**，改提示词时不用自己背那些规范。

---

## 装什么

| 节点 | 干什么 | 花钱吗 |
|---|---|---|
| **提示词优化 (带 skill 调 LLM)** | 读你的提示词 + 参考图 → 调 OpenAI 兼容的 LLM，按「提示词工程规范」改写 | 调一次花一次（有缓存兜底） |
| **提示词合规自检 (本地)** | 按官方规范检查你写好的提示词，列出问题 | **不花钱**，纯本地规则，不联网 |

自检节点**不需要 API key、不需要任何配置文件**，装上就能用。只有优化节点需要配置接口。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/DarthVanDerLimdr/comfyui-prompt-skill.git
```

或直接把这个文件夹拷进 `ComfyUI/custom_nodes/`，重启 ComfyUI。

依赖：`requests`（ComfyUI 环境里通常已有）。

## 快速开始（只用免费的自检）

1. 重启 ComfyUI，加一个「提示词合规自检 (本地)」节点
2. 「目标模型」选 `flux2-dev` 或 `qwen-image-2-1`
3. 把提示词接到它的「提示词」输入上
4. 运行 —— 报告会列出：篇幅是否合适、有没有填充词、有没有排除式否定、有没有交代光照、色值用了几个、有没有误用 Qwen 的 `<imageN>` 写法

## 用优化节点（需要 API）

### 1. 告诉它用哪家的模型

节点上有「接口地址 / API_KEY / 文字模型 / 视觉模型」四格。**「文字模型」「视觉模型」会按接口地址自动换成那家的候选**：

| 接口地址 | 文字模型候选 | 视觉模型候选 |
|---|---|---|
| `https://api.deepseek.com/v1` | deepseek-chat / deepseek-reasoner | deepseek-flash / deepseek-chat |
| `https://ark.cn-beijing.volces.com/api/v3` | doubao-1-5-pro-32k … | doubao-1-5-vision-pro … |
| `https://dashscope.aliyuncs.com/compatible-mode/v1` | qwen-plus / qwen-max … | qwen-vl-max / qwen-vl-plus |
| `https://open.bigmodel.cn/api/paas/v4` | glm-4-plus / glm-4 … | glm-4v-plus / glm-4v |
| `https://api.moonshot.cn/v1` | moonshot-v1-8k … | moonshot-v1-8k-vision-preview |
| `https://api.openai.com/v1` | gpt-4o-mini / gpt-4o … | gpt-4o-mini / gpt-4o |
| `http://localhost:11434/v1`（Ollama） | qwen2.5:7b … | qwen2.5vl:7b / llava:7b |

**下拉只是方便，不是限制** —— 选不到就直接打字填，填什么发什么。

模型名和接口地址明显不匹配时会**当场拦住并告诉你该填什么**（比如方舟地址配 `deepseek-chat`），不会白跑一趟。

### 2. skill（提示词规范文件）——**已随插件自带，装完即可用**

插件按下面的顺序找 skill 文件夹：

1. **插件自带 `skills/`** ← **本仓库已包含 flux2 那份，排在第一位，无需任何手动步骤**
2. `ComfyUI/output/workflow_llm/skills/`
3. `~/.agents/skills/`
4. `~/Documents/agent-skills/`
5. `~/.claude/skills/`
6. `~/.dsh/skills/`

所以把仓库 clone 进 `custom_nodes/` 之后，**flux2 的规范就自动生效**——不用拷文件，也不用重启。

需要两个文件夹名，**必须一模一样**：

- `flux2-dev-prompt-engineering` — 对应「目标模型 = flux2-dev」→ ✅ **本仓库自带**
- `qwen-image-2-1-prompter` — 对应「目标模型 = qwen-image-2-1」→ 需自备（见下）

每个文件夹里要有 `SKILL.md`；有 `references/` 子目录的话里面的 md 会一起读进去。

**想换成自己那份 flux2 规范？** 放到上面第 2–6 条任一位置，并把仓库里的
`skills/flux2-dev-prompt-engineering/` 删掉即可。

> **qwen 那份是第三方作品，本仓库不分发。** 放到 `~/.agents/skills/qwen-image-2-1-prompter/` 即可。
> 只用 FLUX.2 的话什么都不用做。

skill 是每次运行现读的，改完 `.md` **不用重启 ComfyUI**。

### 3. 目标模型怎么定

节点从**连线**读，不猜提示词文字：

- **推荐**：把 UNetLoader / CheckpointLoader 的「模型」紫线接到本节点的「模型输入」——只读文件名，**不加载模型、不占显存**
- 或直接把「目标模型」下拉从「自动判断」改成 `flux2-dev` / `qwen-image-2-1`

没连线又没选下拉时它**不会瞎猜**（双模型工作流里扫到的加载器不一定是这次要用的），会让你自己指定。

## 省钱：什么时候花 token

「生成后控制」和你采样器上那个同名同义，**自动跟着采样器走**：

- 采样器 `fixed`（不抽卡）→ 跳过优化，给上次那份结果，**一字不差、0 token**
- 切成 `randomize` → **切换那一次**重抽一份新提示词；之后一直 randomize 不再重复花钱
- 再切回 `fixed` → 也换一次，退回固定版

想随时手动换一份：把「优化模式」选成「重新优化」。

改了这些会重新调 LLM：提示词 / 目标模型 / 你的意图 / 优化模式 / 思考强度 / 参考图。
（换文件名标签、换 key、换接口**不**重调。）

另外「优化模式」里有个 **`仅输出将要发送的内容`** —— 干跑模式：不发请求、不花钱，把将要发给模型的完整内容写成 txt，还把参考图导出成 jpg 供核对。

## 输出用什么语言

节点自动判定，不用管：

- **Qwen-Image 纯文生图**（没接任何参考图）→ 强制英文（千问官方 PE-T2I 规范要求 "always in English"）
- **其它情况**（接了参考图的编辑/合成，以及 FLUX.2 全部情况）→ 跟随你的输入语言

## 自动质量兜底

LLM 写完还会过几道程序化处理，写完就生效：

- **否定句式改写**：FLUX 没有负向提示词通道，「不要 X」会换成正向描述（保留条款「保持 X 不变」不动）
- **篇幅压缩**：超上限才压，且保证色值、光照、carry-over 声明、硬约束一个不丢
  - flux2：单图/文生图 300 词（中文 360 字）· 多参考图 420 词（中文 504 字）
  - qwen：文生图 800 词 · 编辑 900 词（官方对编辑篇幅无规定，只当失控兜底）
- **重复短语去重**（含中英文、英美拼写差异）
- **失控重复截断**：输出陷入「元素与元素与元素…」这类退化循环时截断
- **外壳剥离**：规范若要求三段式输出，自动只取正文，保证接到采样器的是纯文本

## 参考图

「参考图」是**动态生长**输入：接上第 1 个，自动出现第 2 个……最多 **10** 张（FLUX.2 [dev] 官方支持到 10）。留空自动跳过，接几张算几张，依次编号成 `<image1>`、`<image2>`……

接了任意一张 → 自动切「视觉模型」，模型先看图再写。

## 示例工作流

`examples/prompt-tool.json` —— 拖进 ComfyUI 即可。包含四个部分：

1. **待优化的提示词**（在这里改）
2. **提示词优化**（②）+ 目标模型加载器（连紫线）
3. **合规自检**（③，本地免费）
4. 两个显示节点，分别看优化结果和自检报告

示例里所有会填人的格子都是空的（没有 API key、没有提示词），接口地址填的是 DeepSeek，按需改。

## 常见问题

**「找不到 skill 目录: xxx」**
按上面第 2 节把 skill 文件夹放到 5 个位置之一。报错里会列出它搜过的完整路径。

**「配置对不上：模型名像「X」家的，但接口地址是「Y」」**
接口地址和模型名不是同一家的。报错里会给出这个接口地址下该用的模型名。

**「调用失败: 配置里没有 api_key」**
节点上「API_KEY」留空时会用配置文件里的 key：`ComfyUI/output/workflow_llm/config.json`。要么在那填，要么在节点上填。

**改了 skill 文件要重启吗？**
不用。skill 是每次运行现读的。只有改插件的 `.py` 才需要重启 ComfyUI。

**优化结果在哪？**
`ComfyUI/output/workflow_llm/` —— 结果 txt、报告 txt、缓存都在那儿。

## 许可

MIT，见 `LICENSE`。

skill 规范文件（`flux2-dev-prompt-engineering`、`qwen-image-2-1-prompter`）**不在**本项目内，各自归属其作者。
