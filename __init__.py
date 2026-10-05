# ComfyUI 提示词优化节点 —— 带 skill 的 LLM 改写
#
# 两个节点：
#   1. 提示词优化 (skill)    —— 读提示词 + 带 skill 调 LLM → 输出优化后的提示词
#   2. 提示词合规自检        —— 不调 LLM，按 skill 规则本地检查（零成本快速自检）
#
# 配置文件：output/workflow_llm/config.json
#   {
#     "base_url": "https://api.deepseek.com/v1",
#     "api_key": "sk-...",
#     "model": "deepseek-chat",
#     "temperature": 0.3,
#     "timeout": 120
#   }
#   base_url 填任意 OpenAI 兼容的 /v1 端点（DeepSeek / 火山方舟 / 通义 / OpenRouter / 本地 Ollama 均可）

import os
import re
import json
import glob
import hashlib

try:
    import folder_paths
    _HAS_FP = True
except Exception:
    _HAS_FP = False


# ------------------------------------------------------------------ 路径与配置

def _out_root():
    if _HAS_FP:
        base = folder_paths.get_output_directory()
    else:
        base = os.path.join(os.getcwd(), "output")
    d = os.path.join(base, "workflow_llm")
    os.makedirs(d, exist_ok=True)
    return d


CONFIG_PATH = os.path.join(_out_root(), "config.json")
CONFIG_TEMPLATE = {
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "",
    "model": "deepseek-chat",
    "vision_model": "deepseek-flash",
    "temperature": 0.3,
    "timeout": 300,
    "max_image_side": 1024,
    "image_quality": 85,
}


def _ts():
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def slug(s):
    s = "".join(c for c in str(s) if c.isalnum() or c in "-_")
    return s or "out"


def _write(name, content):
    p = os.path.join(_out_root(), name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return p


# ------------------------------------------------------------------ 结果缓存
# 目的：同一份输入第二次跑，直接给上次那份优化稿，一字不差、0 token。
# LLM 本身有随机性（temperature 0.3），没有缓存的话同一份提示词重跑会变样。

CACHE_VERSION = 2          # 缓存格式变了就 +1，老缓存自动失效
CACHE_DIR_NAME = "prompt-cache"


def _cache_dir():
    d = os.path.join(_out_root(), CACHE_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def _img_fingerprint(raw):
    """从 JPEG 字节里抠出宽高。改了图 → 尺寸或字节数会变，够用了。
    只读文件头，不解码整张图。"""
    w = h = 0
    try:
        i = 2
        n = len(raw)
        while i < n - 9:
            if raw[i] != 0xFF:
                i += 1
                continue
            m = raw[i + 1]
            if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
                i += 2
                continue
            seg = (raw[i + 2] << 8) | raw[i + 3]
            if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
                h = (raw[i + 5] << 8) | raw[i + 6]
                w = (raw[i + 7] << 8) | raw[i + 8]
                break
            if seg <= 0:
                break
            i += 2 + seg
    except Exception:
        pass
    return f"{w}x{h}"


def _cache_key(parts, img_raws, skill_text):
    """输入指纹。parts 里放会影响优化结果的东西；文件名标签之类刻意不放。"""
    h = hashlib.sha256()
    h.update(f"v{CACHE_VERSION}\0".encode())
    for p in parts:
        h.update(str(p).encode("utf-8", "replace"))
        h.update(b"\0")
    h.update(hashlib.sha256((skill_text or "").encode("utf-8", "replace")).digest())
    for raw in (img_raws or []):
        h.update(hashlib.sha256(raw).digest())
        h.update(_img_fingerprint(raw).encode())
    return h.hexdigest()[:32]


def _cache_read(key):
    p = os.path.join(_cache_dir(), key + ".json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            e = json.load(f)
        if e.get("key") != key or e.get("version") != CACHE_VERSION:
            return None
        if not str(e.get("result") or "").strip():
            return None
        e["_path"] = p
        return e
    except Exception:
        return None          # 缓存坏了就当没有，不影响正常跑


def _cache_write(key, result, meta):
    p = os.path.join(_cache_dir(), key + ".json")
    entry = dict(meta)
    entry.update({"version": CACHE_VERSION, "key": key, "created": _ts(),
                  "result": result})
    tmp = p + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=1)
        os.replace(tmp, p)
    except Exception as e:
        print(f"[提示词优化] 写缓存失败（不影响本次结果）: {e}")
    return p


def encode_images(image, max_side=1024, quality=85, max_count=5):
    """把 ComfyUI 的 IMAGE 张量编码成 OpenAI 视觉格式的 data URL

    ComfyUI 的 IMAGE 形状是 (B,H,W,C)，float 0..1。
    返回 (urls, notes, raws)：urls 是 API 消息片段，raws 是每张图的 JPEG 原始字节
    （预览模式可以直接把 raws 存成文件，省一次解码）。
    任何一步失败都不抛异常，只把原因记进 notes。
    """
    urls, notes, raws = [], [], []
    if image is None:
        return urls, notes, raws
    try:
        import io as _io
        import base64
        import numpy as np
        from PIL import Image as PILImage

        t = image
        if hasattr(t, "ndim") and t.ndim == 3:      # 单张 (H,W,C)
            # torch 有 unsqueeze；纯 numpy（离线脚本/别的节点）没有，退回 expand_dims
            if hasattr(t, "unsqueeze"):
                t = t.unsqueeze(0)
            else:
                t = np.expand_dims(t, 0)
        n = int(t.shape[0]) if hasattr(t, "shape") else 0
        if n == 0:
            return urls, notes, raws
        if n > max_count:
            notes.append(f"批次有 {n} 张，只取前 {max_count} 张")
            n = max_count

        for i in range(n):
            try:
                one = t[i]
                arr = one.detach().cpu().numpy() if hasattr(one, "detach") else np.asarray(one)
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                pil = PILImage.fromarray(arr)

                w, h = pil.size
                long_side = max(w, h)
                if max_side and long_side > max_side:
                    sc = max_side / float(long_side)
                    pil = pil.resize((max(1, int(w * sc)), max(1, int(h * sc))),
                                     PILImage.LANCZOS)
                if pil.mode != "RGB":
                    pil = pil.convert("RGB")

                buf = _io.BytesIO()
                pil.save(buf, format="JPEG", quality=int(quality), optimize=True)
                raw = buf.getvalue()
                b64 = base64.b64encode(raw).decode("ascii")
                urls.append({"type": "image_url",
                             "image_url": {"url": "data:image/jpeg;base64," + b64}})
                raws.append(raw)
                notes.append(f"图{i+1}: {pil.size[0]}x{pil.size[1]}, {len(b64)//1024} KB(base64)")
            except Exception as e:
                notes.append(f"图{i+1} 编码失败: {type(e).__name__}: {e}")
    except Exception as e:
        notes.append(f"图像处理库不可用: {type(e).__name__}: {e}")
    return urls, notes, raws


def load_config():
    cfg = dict(CONFIG_TEMPLATE)
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    else:
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(CONFIG_TEMPLATE, f, ensure_ascii=False, indent=1)
        except Exception:
            pass
    # 环境变量可覆盖（方便临时用）
    if os.environ.get("WORKFLOW_LLM_BASE_URL"):
        cfg["base_url"] = os.environ["WORKFLOW_LLM_BASE_URL"]
    if os.environ.get("WORKFLOW_LLM_API_KEY"):
        cfg["api_key"] = os.environ["WORKFLOW_LLM_API_KEY"]
    if os.environ.get("WORKFLOW_LLM_MODEL"):
        cfg["model"] = os.environ["WORKFLOW_LLM_MODEL"]
    return cfg


# ------------------------------------------------------------------ skill 读取

# 插件自带的 skill：随仓库一起走的 skills/ 目录（和 __init__.py 同级）。
# 装了插件就等于有了这份规范，不用再去别处找文件。
# 放在搜索顺序最前面 —— 用户在 ~/.agents/skills 里自己放的那份优先级更低，
# 但只要自带这份存在，它就先被用上（想换成自己的，删掉自带目录即可）。
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLED_SKILLS_DIR = os.path.join(_PLUGIN_DIR, "skills")

SKILL_ROOTS = [
    BUNDLED_SKILLS_DIR,                                  # 插件自带（随仓库分发）
    os.path.join(_out_root(), "skills"),                 # 放插件的输出目录
    os.path.join(os.path.expanduser("~"), ".agents", "skills"),   # 默认位置
    os.path.expanduser("~/Documents/agent-skills"),
    os.path.expanduser("~/.claude/skills"),
    os.path.expanduser("~/.dsh/skills"),
]

SKILL_MAP = {
    "flux2-dev": "flux2-dev-prompt-engineering",
    "qwen-image-2-1": "qwen-image-2-1-prompter",
}


def _find_skill_dir(name):
    for root in SKILL_ROOTS:
        p = os.path.join(root, name)
        if os.path.isdir(p):
            return p
    return None


def load_skill(name):
    """读取一个 skill 的全部 md 内容（含 references）"""
    d = _find_skill_dir(name)
    if not d:
        hint = ""
        if os.path.isdir(BUNDLED_SKILLS_DIR):
            have = [x for x in os.listdir(BUNDLED_SKILLS_DIR)
                    if os.path.isdir(os.path.join(BUNDLED_SKILLS_DIR, x))]
            hint = f"\n插件自带 skills/ 里有: {have or '（空）'}"
        else:
            hint = f"\n（插件自带的 skills/ 目录不存在: {BUNDLED_SKILLS_DIR}）"
        return None, (f"找不到 skill 目录: {name}\n搜索过: " + " | ".join(SKILL_ROOTS)
                      + hint)
    parts = []
    for f in ["SKILL.md"] + sorted(glob.glob(os.path.join(d, "references", "*.md"))):
        p = f if os.path.isabs(f) else os.path.join(d, f)
        if os.path.isfile(p):
            try:
                with open(p, encoding="utf-8") as fh:
                    parts.append(f"\n\n===== {os.path.relpath(p, d)} =====\n" + fh.read())
            except Exception as e:
                parts.append(f"\n\n(读取失败 {p}: {e})")
    if not parts:
        return None, f"skill 目录里没有 md 文件: {d}"
    return "".join(parts), None


# ------------------------------------------------------------------ LLM 调用

def call_llm(system_prompt, user_prompt, cfg, image_urls=None, thinking="关闭",
             first_reply=None):
    """调 OpenAI 兼容端点。image_urls 非空时走视觉消息格式。

    thinking: 关闭 / 低 / 中 / 高
      实测 DeepSeek 只认 thinking={"type":"disabled"}（其余参数名会被忽略）
      关闭思考能省掉 30–500 个推理 token，写提示词这种任务不需要长思考。

    first_reply: 把上一轮的回答作为 assistant 消息放进上下文，
      用于"你这版太长了，压到 N 词以内"这种追问（超长兜底压缩用）。
    """
    import urllib.request
    import urllib.error

    base = (cfg.get("base_url") or "").rstrip("/")
    key = cfg.get("api_key") or ""
    if not base:
        return None, "配置里没有 base_url。请编辑: " + CONFIG_PATH
    if not key:
        return None, "配置里没有 api_key。请编辑: " + CONFIG_PATH

    url = base + "/chat/completions"

    if image_urls:
        content = [{"type": "text", "text": user_prompt}] + list(image_urls)
        model = cfg.get("vision_model") or cfg.get("model")
    else:
        content = user_prompt
        model = cfg.get("model")

    messages = [{"role": "system", "content": system_prompt}]
    if first_reply:
        messages.append({"role": "user", "content": "【待优化提示词】\n（见下）"})
        messages.append({"role": "assistant", "content": first_reply})
        messages.append({"role": "user", "content": user_prompt})
    else:
        messages.append({"role": "user", "content": content})

    body = {
        "model": model,
        "messages": messages,
        "temperature": float(cfg.get("temperature", 0.3)),
        "stream": False,
    }

    # 思考强度
    eff_map = {"低": "low", "中": "high", "高": "max"}
    if thinking == "关闭":
        body["thinking"] = {"type": "disabled"}
    else:
        lvl = eff_map.get(thinking)
        if lvl:
            # 同时给两种写法，兼容不同端点
            body["reasoning_effort"] = lvl
            body["thinking"] = {"type": "enabled"}

    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=float(cfg.get("timeout", 300))) as r:
            data = json.loads(r.read().decode("utf-8"))
        msg = data["choices"][0]["message"]
        text = msg.get("content") or ""
        # 有些推理模型把内容放在 reasoning_content，正文可能为空
        if not text.strip():
            text = msg.get("reasoning_content") or ""
        return text, None
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")[:400]
        except Exception:
            pass
        return None, f"HTTP {e.code}: {detail}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _extract_prompt_text(value):
    """value 可能是字符串，也可能是 ComfyUI 传进来的任意对象"""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        for v in value:
            if isinstance(v, str):
                return v
    return str(value)


# ------------------------------------------------------------------ 服务商与模型候选
# 为什么需要它：换服务商时模型名要手填，一填错就是 404，很难查。
# 这里按「接口地址」关键字给一份候选（下拉里能选，同时也能手填 —— 见下面的
# VALIDATE_INPUTS，它让这两个输入跳过"必须在列表内"的校验）。
#
# ⚠️ 名单只是"常见候选"，不是权威清单，各家都会随时上新模型。
# 所以：选不到就手填，填什么就发什么，不会被拦住。
PROVIDER_MODELS = [
    # (接口地址里出现的关键字, 文字模型候选, 视觉模型候选)
    ("deepseek", ["deepseek-chat", "deepseek-reasoner"],
     ["deepseek-flash", "deepseek-chat"]),
    ("dashscope", ["qwen-plus", "qwen-max", "qwen-turbo", "qwen-long"],
     ["qwen-vl-max", "qwen-vl-plus"]),
    ("aliyuncs", ["qwen-plus", "qwen-max", "qwen-turbo"],
     ["qwen-vl-max", "qwen-vl-plus"]),
    ("ark.cn-beijing.volces.com", ["doubao-1-5-pro-32k", "doubao-pro-32k", "doubao-lite-32k"],
     ["doubao-1-5-vision-pro", "doubao-vision-pro-32k"]),
    ("bigmodel", ["glm-4-plus", "glm-4", "glm-4-flash"],
     ["glm-4v-plus", "glm-4v"]),
    ("moonshot", ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
     ["moonshot-v1-8k-vision-preview"]),
    ("siliconflow", ["Qwen/Qwen2.5-7B-Instruct", "deepseek-ai/DeepSeek-V3"],
     ["Qwen/Qwen2.5-VL-7B-Instruct"]),
    ("openai.com", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
     ["gpt-4o-mini", "gpt-4o"]),
    ("openrouter", ["openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet"],
     ["openai/gpt-4o-mini", "google/gemini-flash-1.5"]),
    ("ollama", ["qwen2.5:7b", "llama3.1:8b", "deepseek-r1:7b"],
     ["qwen2.5vl:7b", "llava:7b"]),
    ("localhost", ["qwen2.5:7b", "llama3.1:8b"], ["qwen2.5vl:7b", "llava:7b"]),
    ("127.0.0.1", ["qwen2.5:7b", "llama3.1:8b"], ["qwen2.5vl:7b", "llava:7b"]),
]
DEFAULT_MODELS = (["deepseek-chat", "deepseek-reasoner"],
                  ["deepseek-flash", "deepseek-chat"])


def _current_base_url():
    """当前生效的接口地址：config.json 优先，节点上的填写在运行时才覆盖。
    这里只用于「给下拉填候选」，取配置文件的值就够了。"""
    try:
        return str(load_config().get("base_url") or "")
    except Exception:
        return ""


def _config_model(key, kind="text"):
    """从配置里取模型名当默认值；取不到就用该服务商的第一个候选。

    这样节点上显示的就是「真正会用的那个模型」，不是写死的 deepseek-*。
    （实测踩过：配置换成火山方舟后，这格还显示 deepseek-chat，会让人以为没生效。）
    """
    try:
        v = str(load_config().get(key) or "").strip()
    except Exception:
        v = ""
    if v:
        return v
    cands = _model_choices(kind)
    return cands[0] if cands else ""


def _model_choices(kind, base_url=None, prefix_provider=False):
    """kind = "text" | "vision"，返回候选模型名列表。

    prefix_provider=True 时给候选加上「服务商 · 」前缀（仅用于界面显示）：
    选了别家模型会一眼看出来 —— 实测踩过：接口地址是火山方舟、文字模型却还
    留着 deepseek-chat，两个都不报错，跑起来才发现调用失败。
    """
    base = (base_url if base_url is not None else _current_base_url()).lower()
    idx = 0 if kind == "text" else 1
    for key, text_models, vision_models in PROVIDER_MODELS:
        if key in base:
            names = list(text_models if kind == "text" else vision_models)
            if not prefix_provider:
                return names
            return [f"{key} · {n}" for n in names]
    names = list(DEFAULT_MODELS[idx])
    return [f"默认 · {n}" for n in names] if prefix_provider else names


def _provider_of(base_url):
    """接口地址属于哪一家（返回命中关键字），认不出返回 None。"""
    base = (base_url or "").lower()
    for key, _, _ in PROVIDER_MODELS:
        if key in base:
            return key
    return None


def _strip_provider_prefix(model):
    """去掉界面用的「服务商 · 」前缀，取回真正的模型名。"""
    s = str(model or "").strip()
    if " · " in s:
        s = s.split(" · ", 1)[1].strip()
    return s


# 各家的「特征词」：用来判断一个模型名是不是别家的。
# 为什么单独列：服务商关键字和模型名特征往往不是同一个词
# （智谱地址里是 bigmodel、模型叫 glm-4；月之暗面地址是 moonshot、模型叫 kimi/moonshot）。
PROVIDER_SIGNATURES = {
    "deepseek": ["deepseek"],
    "dashscope": ["qwen"],
    "aliyuncs": ["qwen"],
    "ark.cn-beijing.volces.com": ["doubao"],
    "bigmodel": ["glm"],
    "moonshot": ["moonshot", "kimi"],
    "siliconflow": [],          # 聚合平台，什么模型都有，不做特征判断
    "openai.com": ["gpt-", "o1-", "dall-e"],
    "openrouter": [],           # 同上，聚合平台
    "ollama": [],
    "localhost": [],
    "127.0.0.1": [],
}


# 同一家的多个地址关键字要归并成一家，否则会互相误判成"别家"。
# 实测踩过：地址是 dashscope、模型是 qwen-max，被 aliyuncs 那条规则误拦。
PROVIDER_CANON = {
    "aliyuncs": "dashscope",          # 阿里：dashscope / aliyuncs 是同一家
    "localhost": "local",
    "127.0.0.1": "local",
}


def _canon(key):
    return PROVIDER_CANON.get(key, key)


def _models_for_provider_key(key):
    for k, t, v in PROVIDER_MODELS:
        if k == key:
            return list(t), list(v)
    return [], []


def _model_matches_endpoint(model, base_url):
    """模型名跟这个接口地址合不合得来。

    返回 (是否匹配, 说明)。判据宽松，只拦明显不可能的情况：
      · 认不出接口属于谁 → 一律放行（自建服务商，模型名随你填）
      · 聚合平台（siliconflow/openrouter/本地）→ 不做特征判断，一律放行
      · 模型名里出现别家的特征词，而地址不是那家 → 拦住
    这样不会挡住合法的自定义模型名，但拦得住「方舟地址 + deepseek-chat」这类错配。
    """
    key = _provider_of(base_url)
    if not key:
        return True, ""
    name = _strip_provider_prefix(model).lower()
    mine = PROVIDER_SIGNATURES.get(key)
    if mine is not None and not mine:
        return True, ""                      # 明确标记为"不判断"的家
    for other, _, _ in PROVIDER_MODELS:
        if other == key or _canon(other) == _canon(key):
            continue                         # 同一家的别名不算别家
        sigs = PROVIDER_SIGNATURES.get(other, [])
        if any(s and s in name for s in sigs):
            return False, f"模型名像「{other}」家的，但接口地址是「{key}」"
    return True, ""


def _endpoint_models_hint(base_url):
    """这个接口地址下可用的模型名（用于报错时给出正确建议）。"""
    key = _provider_of(base_url)
    if not key:
        return None
    for k, t, v in PROVIDER_MODELS:
        if k == key:
            return k, list(t), list(v)
    return None


# ------------------------------------------------------------------ 节点 1

class PromptOptimizerSkill:
    """读提示词 + 带 skill 调 LLM → 优化后的提示词"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True, "default": ""}),
                "目标模型": (["自动判断", "flux2-dev", "qwen-image-2-1"],),
                "优化模式": (["标准优化", "保守改写", "大幅重构", "重新优化", "仅输出将要发送的内容"],
                        {"default": "标准优化"}),
                # 和采样器上那个「生成后控制」同名同义：fixed = 不重抽，randomize = 每次重抽。
                # 运行时只处理这一个节点，不会去改你采样器的随机种。
                "生成后控制": (["fixed", "randomize", "increment", "decrement"],
                          {"default": "fixed",
                           "tooltip": "fixed=不重抽提示词（省钱）；randomize=每次重抽一次新提示词"}),
                "思考强度": (["关闭", "低", "中", "高"], {"default": "关闭"}),
                "你的意图": ("STRING", {"multiline": True, "default": ""}),
                "文件名标签": ("STRING", {"default": "prompt"}),
                # 「接口地址」和「API_KEY」的默认值取自 config.json —— 显示的就是实际生效的值。
                # （写死默认值会出现"节点上显示 deepseek、实际用的是火山方舟"这种误导，
                #   实测踩过：换了配置后这格还显示旧地址。）
                "接口地址": ("STRING", {"default": _current_base_url()
                                     or "https://api.deepseek.com/v1"}),
                # 注意：API_KEY 这一格**默认留空**，不要从配置自动填。
                # 原因：这一格的值会被存进工作流 json，发布/分享工作流时就把 key 带出去了。
                # 留空时运行时自动回落到 config.json 里的 key，所以不留空也能用。
                "API_KEY": ("STRING", {"default": "",
                    "tooltip": "留空 = 用 config.json 里的 key。这里填了会存进工作流文件，分享时注意。"}),
                # 文字/视觉模型：按「接口地址」对应的服务商给一份候选列表（下拉可选、也能手填）。
                # 选项在读取节点定义时按当前配置的接口地址算出来，所以换了接口、
                # 重开一次页面/重新加载工作流，候选就跟着换了。
                "文字模型": (_model_choices("text", prefix_provider=True),
                         {"default": _config_model("model")}),
                "视觉模型": (_model_choices("vision", prefix_provider=True),
                         {"default": _config_model("vision_model", "vision")}),
            },
            "optional": {
                "模型输入": ("MODEL", {"lazy": True}),
                # ── 参考图：用 ComfyUI 的「动态生长」输入 ──
                # 类型写成 COMFY_AUTOGROW_V3 并附 template，前端就会按需自己长出输入点：
                # 你接上第 1 个，它自动追加第 2 个，接满 max 就不再长。
                # 这跟 Image 2.1 的 Text Encode 节点是同一个机制（对照它的
                # /object_info 形状写出来的），不是自己编的协议。
                #
                # prefix 式：实际输入名是 参考图.image_1、参考图.image_2 …，
                # run() 里用 **kwargs 统一收，按序号排序（见 _collect_autogrow_images）。
                "参考图": ("COMFY_AUTOGROW_V3", {
                    "template": {
                        "input": {"required": {"image": ["IMAGE", {}]}},
                        # ⚠️ 必须给 names，别用 prefix 式。
                        # 前端解析序号时是「优先查 names 数组，没有才退回按名字尾部的数字」，
                        # 而它把存档里的槽位（参考图.image_1…）对回模板时用的是同一个函数。
                        # 只给 prefix 不给 names 时，前端解析不出序号 → 会自己造出
                        # image0、image2、image3… 一堆垃圾输入点（踩过）。
                        # 这里照官方 TextEncodeQwenImage21 的写法明确列出名字。
                        "names": [f"image_{i}" for i in range(1, 11)],
                        "min": 1,
                    },
                    "tooltip": "参考图，接一张长一个点，最多 10 张。"
                               "留空跳过；接了任意一张会自动切视觉模型。",
                }),
            },
            # prompt = 整张工作流图（含每个节点的 class_type 和 widgets_values），
            # 只读不执行 —— 靠它反查模型加载器，不用把模型读进显存
            # extra_pnginfo.workflow = 前端发来的整份画布，里面有每个节点
            # widgets_values_named.control_after_generate（就是你界面上那个「生成后控制」）
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
                "uniq_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("优化后提示词", "报告", "文件路径")
    FUNCTION = "run"
    CATEGORY = "提示词工具"
    OUTPUT_NODE = True

    def check_lazy_status(self, **kwargs):
        # 故意不认领「模型输入」→ 上游的 UNetLoader / 加载器不会被执行，模型不会被加载。
        # 目标模型是从连线关系和加载器的文件名读出来的，不需要真的把模型读进显存。
        return []

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """让 ComfyUI 每次都真正执行本节点。

        为什么不在"不重抽"时返回固定值让它跳过？试过了，不行：
        IS_CHANGED 被调用的时机很早，它拿到的 prompt 是空字典、extra_pnginfo 是 None
        （服务端 get_input_data(..., extra_data={})），所以**在那里读不到画布，
        也就读不到采样器的「生成后控制」**。更麻烦的是 ComfyUI 的缓存判定还会
        在本节点这一层之上短路，连 IS_CHANGED 都不再问，导致"改了采样器却不重抽"。

        所以这里放弃用 ComfyUI 的缓存来省事，改成永远让它执行；
        真正花不花 token 一律由下面的结果缓存决定 —— 这样每一条路径都确定、可控。
        本节点执行本身很便宜（只读连线、算哈希，实测 0.1 秒级）。
        """
        return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, 文字模型=None, 视觉模型=None, **kwargs):
        """让「文字模型 / 视觉模型」跳过 ComfyUI 的"值必须在候选列表里"校验。

        机制：execution.py 里那个判断是
            if x not in validate_function_inputs and not validate_has_kwargs:
                ...（枚举/上下限校验都在这里面）
        只要在这里把参数名声明出来，这两个输入就不再被枚举卡住 ——
        于是它们既是下拉（好选、不易填错），又能手填任意模型名（换服务商、上新模型都不怕）。
        """
        return True

    def run(self, 提示词="", 目标模型="自动判断", 优化模式="标准优化",
            生成后控制="fixed",
            思考强度="关闭", 你的意图="", 文件名标签="prompt",
            接口地址="", API_KEY="", 文字模型="", 视觉模型="",
            模型输入=None, 参考图=None,
            prompt=None, extra_pnginfo=None, unique_id=None, uniq_id=None,
            **kwargs):
        text = _extract_prompt_text(提示词).strip()
        if not text:
            msg = "提示词为空 —— 请在上面填，或把某个文本节点的输出接到「提示词」输入"
            print("[提示词优化] " + msg)
            return (msg, msg, "")

        cfg = load_config()
        # 节点上的设置优先于 config.json（方便临时换 key / 换模型）
        if str(接口地址).strip():
            cfg["base_url"] = str(接口地址).strip()
        if str(API_KEY).strip():
            cfg["api_key"] = str(API_KEY).strip()
        if str(文字模型).strip():
            cfg["model"] = _strip_provider_prefix(文字模型)
        if str(视觉模型).strip():
            cfg["vision_model"] = _strip_provider_prefix(视觉模型)

        # 模型名和接口地址明显错配时，当场说清楚 —— 别让它跑到服务商那边
        # 换来一句看不懂的报错（实测踩过：方舟地址 + deepseek-chat + deepseek 的 key）。
        _pairs = [("文字模型", cfg.get("model")), ("视觉模型", cfg.get("vision_model"))]
        for _label, _m in _pairs:
            _ok, _why = _model_matches_endpoint(_m, cfg.get("base_url"))
            if not _ok:
                _hint = _endpoint_models_hint(cfg.get("base_url"))
                _sug = ""
                if _hint:
                    _name, _t, _v = _hint
                    _good = _t if _label == "文字模型" else _v
                    _sug = (f"\n\n这个接口地址（{_name}）下应该用：\n  "
                            + "、".join(_good)
                            + f"\n把「{_label}」改成其中之一即可。")
                _msg = (f"配置对不上：{_why}。\n"
                        f"  接口地址 = {cfg.get('base_url')}\n"
                        f"  {_label} = {_strip_provider_prefix(_m)}"
                        + _sug
                        + "\n\n（换服务商要同时改：接口地址 + API_KEY + 模型名。")
                print("[提示词优化] " + _msg.replace("\n", " "))
                return (text, _msg, "")
        # 图片长边上限：从配置读，且做安全转换（前端可能传来非法值）
        try:
            max_side = int(cfg.get("max_image_side", 1024))
        except Exception:
            max_side = 1024
        max_side = max(256, min(4096, max_side))
        try:
            quality = int(cfg.get("image_quality", 85))
        except Exception:
            quality = 85

        # 「重新优化」= 无视缓存重调一次，改写强度仍按标准优化
        _force = (优化模式 == "重新优化")
        eff_mode = "标准优化" if _force else 优化模式
        _nodes = prompt if isinstance(prompt, dict) else {}

        # 要不要重抽提示词？
        #   自己节点上的「生成后控制」!= fixed  → 重抽
        #   或者采样器上那个「生成后控制」!= fixed（从界面运行时前端的画布里有这个值）→ 重抽
        _my_roll = str(生成后控制 or "fixed").strip() or "fixed"
        # 采样器的「生成后控制」在前端发来的 extra_pnginfo.workflow 里；
        # 没有就退回看执行图本身（有些调用方会把 workflow 直接塞在 prompt 里）
        _sampler_roll = _sampler_roll_state(extra_pnginfo) or _sampler_roll_state(_nodes)

        # 重抽规则：只在"模式发生变化"时重抽一次，而不是一直重抽。
        #   固定种 → 重跑：命中缓存，0 token
        #   把采样器（或本节点）切成 randomize 的那一次：重抽一份新提示词
        #   之后一直保持 randomize 重跑：仍然用同一份（不再花钱，直到你又切一次）
        _uid_key = str(unique_id if unique_id is not None else (uniq_id or "2"))
        _mode_now = (f"my={_my_roll}|sampler={_sampler_roll or '-'}",
                     (_my_roll != "fixed") or ((_sampler_roll or "") not in ("", "fixed")))
        _prev = _LAST_ROLL_MODE.get(_uid_key)
        _mode_changed = (_prev is not None and _prev[0] != _mode_now[0])
        _wild = _mode_now[1]
        _prev_wild = bool(_prev[1]) if _prev else False
        # 切进/切出"抽卡模式"的那一次都重抽：
        #   切进去 → 换一份新的；切出来 → 也换一份，免得又退回更早那份旧结果
        _reroll = _force or (_mode_changed and (_wild or _prev_wild))
        _LAST_ROLL_MODE[_uid_key] = _mode_now

        _roll_src = ("来自「优化模式＝重新优化」" if _force else
                     f"来自本节点的「生成后控制＝{_my_roll}」" if _my_roll != "fixed" else
                     f"来自你采样器的「生成后控制＝{_sampler_roll}」" if _sampler_roll not in (None, "", "fixed") else
                     "你刚把「生成后控制」切回 fixed，所以换回固定种该用的那一版" if _prev_wild else
                     "「生成后控制」发生了切换")
        # 重抽时 _ck 里会带一个时间戳（见下面），等于主动放弃缓存、真调一次 LLM

        # 选 skill：优先用连进来的模型加载器判定，其次节点上手填的名字，最后才看下拉
        _graph = prompt if prompt is not None else (模型输入 if isinstance(模型输入, dict) else None)
        _uid = unique_id if unique_id is not None else uniq_id
        model_key, model_src = resolve_model_skill(目标模型, _graph, _uid)

        if model_key is None:
            msg = ("没连模型，所以不自动判断（避免用错 skill）。任选一个：\n"
                   "  1. 最省事：把下面「目标模型」这个下拉从「自动判断」改成 "
                   "flux2-dev 或 qwen-image-2-1\n"
                   "  2. 或者把模型加载器的「模型」紫线接到本节点的「模型输入」上"
                   "（只读文件名，不会执行、不会加载模型）")
            print("[提示词优化] " + msg)
            return (text, msg, "")

        skill_name = SKILL_MAP.get(model_key)
        skill_text, err = load_skill(skill_name) if skill_name else (None, f"未知模型: {model_key}")
        if err:
            print("[提示词优化] " + err)
            return (text, err, "")

        # ---- 收集参考图（动态生长输入，按序号编号成 图1/图2/…）----
        # 注意：system prompt 必须等参考图收完再拼，因为"纯文生图一律英文"这条
        # 要按实际收到的图数量来判定（下面 build_system_prompt 的 n_images）。
        img_urls, img_notes, img_raws, slot_names = [], [], [], []
        _refs = _collect_autogrow_images(参考图, kwargs)
        for idx, im in enumerate(_refs, start=1):
            if im is None:
                continue
            urls, notes, raws = encode_images(im, max_side=max_side, quality=quality, max_count=1)
            if urls:
                img_urls.extend(urls)
                img_raws.extend(raws)
                slot_names.append(f"<image{idx}>")
                for n in notes:
                    img_notes.append(f"<image{idx}> → {n}")
            else:
                img_notes.append(f"<image{idx}> 编码失败: " + "; ".join(notes))

        user_parts = [f"【目标模型】{model_key}", f"【待优化提示词】\n{text}"]
        if 你的意图.strip():
            user_parts.append(f"【用户意图/额外要求】\n{你的意图.strip()}")

        if img_urls:
            pairs = "、".join(
                f"第 {i+1} 张图 = {nm}" for i, nm in enumerate(slot_names)
            )
            user_parts.append(
                f"【本次附带了 {len(img_urls)} 张参考图，按你收到的顺序编号】\n"
                f"{pairs}\n\n"
                "先仔细看这些图，把每张图当成**属性的来源**（人物/身份、环境/构图、材质、"
                "光源与色温各自来自哪张），而不是把某张图当成“要还原的那张”。"
                "**最终画面由用户的描述决定，参考图只供它负责的那部分**：\n"
                "  · 哪张图提供人物与身份（发型、脸部特征、身材、皮肤细节）\n"
                "  · 哪张图提供环境与构图（空间结构、陈设、物体位置）\n"
                "  · 哪张图提供光源方向、色温与色调\n"
                "  · 哪张图提供材质与表面行为\n"
                "  · 图里有文字就把文字写出来\n"
                "图是用来看清楚属性的，不是用来决定画面内容的；"
                "用户描述里没有的东西，不要自己加。\n"
                "写提示词时用 <image1>/<image2> 这样的标签指代对应的图（[dev] 用 image 1 这种写法）。"
            )

        user_prompt = "\n\n".join(user_parts)

        # 拼 system（规则本身在 build_system_prompt 里，方便在聊天室里离线评测同一份规则）
        system_prompt = build_system_prompt(model_key, skill_text, eff_mode, text,
                                            n_images=len(img_urls))

        # 仅预览模式：不调 LLM
        if 优化模式 == "仅输出将要发送的内容":
            ts = _ts()
            key = cfg.get("api_key") or ""
            key_note = ("（未填！调用会失败）" if not key
                        else f"{key[:6]}…{key[-4:]}（{len(key)} 位）")
            info = (
                f"=========== 本次将要发送给 LLM 的内容 ===========\n"
                f"接口地址 : {cfg.get('base_url')}\n"
                f"API_KEY  : {key_note}\n"
                f"目标模型 : {model_key}   ← {model_src}\n"
                f"使用模型 : {(cfg.get('vision_model') if img_urls else cfg.get('model'))}"
                + ("   ← 视觉模型（因为接了参考图）" if img_urls else "   ← 纯文字模型") + "\n"
                f"思考强度 : {思考强度}\n"
                f"参考图   : {len(img_urls)} 张"
                + (f"  ({', '.join(slot_names)})" if slot_names else "") + "\n"
                "\n"
                "=========== SYSTEM（整份 skill 会在这里）===========\n"
                f"{system_prompt}\n"
                "\n"
                "=========== USER（文字部分）===========\n"
                f"{user_prompt}\n"
                "\n"
                "=========== 附带的图像 ===========\n"
                + ("\n".join("  " + n for n in img_notes) if img_notes else "  （无）") + "\n"
            )
            p = _write(f"will-send-{slug(文件名标签)}-{ts}.txt", info)

            # 同时把图片本身导出来，方便肉眼核对发的是哪张
            exported = []
            for i, raw in enumerate(img_raws, start=1):
                try:
                    ip = os.path.join(_out_root(), f"will-send-{slug(文件名标签)}-{ts}-img{i}.jpg")
                    with open(ip, "wb") as fh:
                        fh.write(raw)
                    exported.append(os.path.basename(ip))
                except Exception as e:
                    print(f"[提示词优化] 导出预览图片 {i} 失败: {e}")

            print(f"[提示词优化] 仅预览，未调用 LLM → {p}")
            print(f"[提示词优化] 判定目标模型 = {model_key}（{model_src}）")
            return ("", f"未调用 LLM（不花钱）。\n\n"
                        f"目标模型: {model_key}   ← {model_src}\n"
                        f"接口: {cfg.get('base_url')}\n"
                        f"模型: {(cfg.get('vision_model') if img_urls else cfg.get('model'))}"
                        + ("  （视觉模式）" if img_urls else "") + "\n"
                        f"KEY : {key_note}\n"
                        f"思考: {思考强度}\n"
                        f"图片: {len(img_urls)} 张\n\n"
                        f"完整内容已写入:\n{p}\n"
                        + (f"\n图片已导出（可直接双击打开核对）:\n  " + "\n  ".join(exported)
                           if exported else ""),
                    p)

        # ---- 先查缓存：同一份输入第二次跑就直接给上次结果，一字不差、0 token ----
        # 要重抽时换一个 key（用时间戳），等于主动放弃缓存 → 真调一次 LLM。
        _ck = _cache_key(
            ["prompt", text, "model", model_key, "mode", eff_mode,
             "think", 思考强度, "intent", 你的意图.strip(),
             "vision", cfg.get("vision_model") if img_urls else cfg.get("model"),
             "reroll", _ts() if _reroll else "no"],
            img_raws, skill_text)
        cached = None if _reroll else _cache_read(_ck)

        if cached:
            # 老缓存里可能留着没软化过的否定句式，读出来也过一遍，保证一致
            _cached_result, _ = _soften_negation(cached["result"])
            # 旧缓存可能是"加压缩兜底之前"存的超长结果 —— 走缓存路径也要压一次，
            # 否则命中缓存就永远拿不到压缩后的版本。
            _cached_result, _cn = compress_if_too_long(
                _cached_result, cfg, build_system_prompt(model_key, skill_text, eff_mode, text,
                                                        n_images=len(img_urls)),
                word_budget(model_key, len(img_urls)), 思考强度)
            if _cn:
                print(f"[提示词优化] 缓存条目{_cn}")
                try:                       # 压缩后写回缓存，下次直接就拿到压缩版
                    _cache_write(_ck, _cached_result, {
                        "model_key": model_key, "mode": eff_mode, "thinking": 思考强度,
                        "skill": skill_name, "src": model_src, "images": len(img_urls)})
                except Exception:
                    pass
            _c_at = cached.get("created", "?")
            _c_pretty = (_c_at[:4] + "-" + _c_at[4:6] + "-" + _c_at[6:8] + " " +
                         _c_at[9:11] + ":" + _c_at[11:13]) if len(str(_c_at)) >= 13 else str(_c_at)
            cfile = _write(f"{slug(文件名标签)}-{_ts()}-用缓存.txt", _cached_result)
            print(f"[提示词优化] 命中缓存（未调 LLM，未花钱）原稿生成于 {_c_pretty} → {cfile}")
            creport = (f"目标模型: {model_key}   ← {model_src}\n"
                       f"未调用 LLM —— 命中缓存（0 token）\n"
                       f"原因: 「生成后控制」是 fixed，不重抽提示词\n"
                       f"这份结果生成于: {_c_pretty}\n"
                       f"缓存文件: {cached['_path']}\n"
                       f"参考图: {len(img_urls)} 张\n"
                       f"原文 {len(text)} 字 → 结果 {len(_cached_result)} 字\n"
                       f"文件: {cfile}\n\n"
                       f"想换一份新提示词？把本节点或采样器的「生成后控制」改成 randomize，\n"
                       f"或把「优化模式」选成「重新优化」。\n"
                       f"想清空全部缓存：删掉这个文件夹\n  {_cache_dir()}")
            return (_cached_result, creport, cfile)

        result, err = call_llm(system_prompt, user_prompt, cfg,
                               image_urls=img_urls or None, thinking=思考强度)
        if err:
            print(f"[提示词优化] 调用失败: {err}")
            return (text, f"调用失败: {err}\n\n配置文件: {CONFIG_PATH}", "")

        result = _strip_fence(result).strip()
        result, _negfix = _soften_negation(result)
        # 失控重复兜底：实测 qwen 那条"把排除写进负向通道"的指令会让模型
        # 罗列到退化循环（一路重复"…陈设元素与陈设元素与…"到 7000 词当量）
        result, _gwhy = _guard_runaway(result)
        if _gwhy:
            print(f"[提示词优化] {_gwhy}")
        # 篇幅兜底：提示词里写上限只能降低概率、压不住，超长就自动再压一次
        _budget = word_budget(model_key, len(img_urls))
        result, _cmpnote = compress_if_too_long(result, cfg, system_prompt,
                                                _budget, 思考强度)
        if _cmpnote:
            print(f"[提示词优化] {_cmpnote}")
        _cache_write(_ck, result, {"model_key": model_key, "mode": eff_mode,
                                   "thinking": 思考强度, "skill": skill_name,
                                   "src": model_src, "images": len(img_urls)})
        ts = _ts()
        p = _write(f"{slug(文件名标签)}-{ts}.txt", result)
        used_model = (cfg.get("vision_model") or cfg.get("model")) if img_urls else cfg.get("model")
        report = (f"目标模型: {model_key}   ← {model_src}\n"
                  f"模型: {used_model}"
                  + ("  （视觉模式）" if img_urls else "") + "\n"
                  f"思考强度: {思考强度}\n"
                  f"skill: {skill_name} ({len(skill_text)} 字符)\n"
                  f"参考图: {len(img_urls)} 张"
                  + (f"  ({', '.join(slot_names)})" if slot_names else "") + "\n"
                  + ("".join("  " + n + "\n" for n in img_notes) if img_notes else "")
                  + f"原文 {len(text)} 字 → 优化后 {len(result)} 字\n"
                  + (f"本次调了 LLM（重抽原因：{_roll_src}）\n" if _reroll else
                     "本次调了 LLM 并把结果存进缓存；\n"
                     "「生成后控制」是 fixed，下次同样输入直接复用，不再花钱\n")
                  + f"文件: {p}")
        print(f"[提示词优化] {model_key} / {eff_mode}"
              + (f" / 重抽({_roll_src})" if _reroll else " / 不重抽")
              + (f" / 视觉({len(img_urls)}图)" if img_urls else "") + f" → {p}")
        return (result, report, p)


def _strip_fence(s):
    """剥掉 markdown 外壳，只留提示词正文。

    为什么要做这么多层：qwen 那份规范的「默认模式」本身要求三段式输出
    （💡 优化解析 / #### 📋 提示词（可直接复制）用代码块包住正文 / 🎨 微调建议）。
    我们在 system 里已经声明"只输出正文"，但规范文件写在后面、说得更具体，
    模型偶尔会照着规范走。这里做程序化兜底，保证接到节点上的永远是纯正文。

    顺序很重要：先认三段式（能精确定位 📋 那一段），再退到"整段就是一个代码块"。
    """
    s = s.strip()
    if not s:
        return s
    # ① 三段式：取「📋 提示词」标题之后、第一个代码块里的内容
    m = re.search(r'📋[^\n]*\n.*?```[a-zA-Z]*[ \t]*\n(.*?)\n?[ \t]*```', s, flags=re.S)
    if m and m.group(1).strip():
        return m.group(1).strip()
    # ② 三段式但没用代码块：以「📋 提示词」那一行为界，后面就是正文；
    #    再切掉紧跟其后的 🎨 建议段。有 🎨 却没 📋 的不动（可能是正常正文）。
    if "📋" in s:
        body = s.split("📋", 1)[1]
        body = body.split("\n", 1)[1] if "\n" in body else ""
        for marker in ("🎨", "\n## ", "\n### "):
            if marker in body:
                body = body.split(marker, 1)[0]
        body = re.sub(r'^```[a-zA-Z]*[ \t]*\n?', '', body.strip())
        body = re.sub(r'\n?[ \t]*```\s*$', '', body).strip()
        if body:
            return body
    # ③ 剥掉开头/结尾的散装代码块围栏
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s


# ------------------------------------------------- 篇幅兜底压缩
# 为什么需要程序化兜底：prompt 里写"不许超过 N 词"只能降低概率，压不住。
# 实测同一份输入，写 250 词的输出会跑到 295 词。超长时自动再叫模型压缩一次，
# 并明确列出**不许删**的东西。

def _count_words(t):
    """英文按空格词数；中文按字符数折算（1 汉字 ≈ 1.6 词）"""
    words = len([w for w in t.split() if w.strip()])
    cjk = len(re.findall(r'[\u4e00-\u9fff]', t))
    return max(words, int(cjk / 1.6)), words, cjk


# ------------------------------------------------- 失控重复检测
# 实测踩过一次很严重的：qwen 那条"把排除集中写进负向通道"的指令，
# 让模型开始罗列排除项，然后陷入退化循环 ——
# 「…陈设元素与陈设元素与陈设元素与陈设…」一直重复到 7000 词当量。
# 生成类模型在"穷举同类项"这种任务上很容易这样。这里做兜底截断。

_REP_PATTERNS = [
    # 同一短语连续重复（间隔很短）
    re.compile(r'(.{6,40}?)\1{2,}'),
    # "X与X" / "X和X" 式堆叠
    re.compile(r'([\u4e00-\u9fff]{2,12})[与和](?:\1[与和]){2,}'),
    # 英文同词反复
    re.compile(r'(?i)\b(\w{3,})\b(?:\s+\1\b){3,}'),
]


def _find_runaway(text, max_passes=6):
    """找出失控重复的起点。返回 (截断位置, 说明) 或 (None, '')"""
    for _ in range(max_passes):
        best = None
        for rx in _REP_PATTERNS:
            for m in rx.finditer(text):
                # 从这次重复开始到结尾，检查是否还有足够多的重复
                n = len(m.group(0))
                if best is None or m.start() < best[0]:
                    best = (m.start(), n, m.group(0)[:60])
        if best is None:
            return None, ""
        start, n, sample = best
        # 只在"重复占比很高"时才认为是失控，避免误砍正常的轻微重复
        if len(text) - start < 200:
            return None, ""
        return start, f"疑似失控重复（“…{sample}…”），已从该处截断"


def _guard_runaway(text):
    """截断失控重复，并补回一个像样的收尾。返回 (文本, 说明)"""
    if not text:
        return text, ""
    pos, why = _find_runaway(text)
    if pos is None:
        return text, ""
    head = text[:pos].rstrip()
    # 截断处可能在半个句子里，收一下尾
    head = re.sub(r'[，、,;；:：\s]+$', '', head)
    if head and head[-1] not in '。.!！?？':
        head += '。'
    return head, why


def decide_output_lang(model_key, n_images, user_text):
    """本次成品正文该用哪种语言：返回 "zh" 或 "en"。

    ⚠️ 必须只有这一处判定，而且要在拼 system **之前**就定好、把结论直接写进 system。
    早先的踩坑：写成"看输入语言"的规则 → 英文输入 0/5 全串成中文，
    因为 system 本身是中文，模型跟随 system 的语言而不去查输入。

    规则只有两条：
      · qwen + 没有任何参考图（纯文生图）→ 强制英文
        （官方 PE-T2I 规格：The description is **always in English**,
         whatever language the request arrives in）
      · 其它一律跟随输入语言
        （flux2 规范里没有英文规定，还明说它 handles non-English prompts natively）
    """
    if not n_images and model_key == "qwen-image-2-1":
        return "en"
    return detect_lang(user_text)


def word_budget(model_key, n_images):
    """返回 (词数上限, 中文字数上限)，超出任一条就触发压缩兜底。

    数值来源（不是拍脑袋）：
      · flux2 规范自己写的区间：默认 30–80 词，长区间 80–300+（多主体/强指定时用）。
        上限取 250（单图/文生图）与 350（多参考图），落在官方长区间里、不越界。
      · qwen 官方 PE-T2I 规格本来就要 400–500 词，编辑任务官方没有篇幅规定，
        所以上限放宽到 800/900，只当"失控兜底"用，不参与日常风格控制。
      · 中文另给一条字符上限：中文字数上限 ≈ 词上限 × 1.2。
        （早先按 cjk/1.6 折算词当量、再跟词预算比，单位不统一，
          中文经常在没超官方的位置就被压一次。现在两条线各管各的语言。）
    """
    if model_key == "qwen-image-2-1":
        words = 900 if n_images >= 2 else 800
    else:                                     # flux2-dev
        words = 420 if n_images >= 2 else 300
    return words, int(words * 1.2)


def compress_if_too_long(result, cfg, system_prompt, budget, thinking="关闭"):
    """超长就压一次。返回 (文本, 说明)。压缩失败就原样返回，不阻塞。

    budget 是 (词数上限, 中文字数上限) 二元组 —— 中文看字数、英文看词数，
    避免拿折算出来的"词当量"去跟按词写的预算比（单位不统一，会误压中文）。
    """
    words_max, cjk_max = budget
    eff, words, cjk = _count_words(result)
    over_words = words > words_max
    over_cjk = cjk > cjk_max
    if not over_words and not over_cjk:
        return result, ""
    if over_cjk and not over_words:
        limit_txt = f"{cjk_max} 个汉字（现在 {cjk} 字，超了 {cjk - cjk_max} 字）"
    else:
        limit_txt = f"{words_max} 词（现在约 {words} 词，超了 {words - words_max} 词）"
    ask = (
        f"你上面那版提示词太长了：必须压到 **{limit_txt}以内**。"
        "这不是建议，是交付标准。\n"
        "先想清楚哪些是真正改变像素的内容，然后**大刀阔斧地砍**：\n"
        "  · 把「X occupying the upper portion」这类结构说明压成最少的词\n"
        "  · 同一件事只在一处说，删掉所有重复表述（例如已经写了“只保留图2的设定”，"
        "就不用再补“画面中不存在图1”之类）\n"
        "  · 删掉一切可以从画面推断的修饰词（rounded / simple / soft / smooth 这类）\n"
        "  · 删掉解释性的从句，只留名词短语和必要动词\n"
        "  · 句子能合并就合并，不要为了分段而分段\n"
        "**必须原样保留**：\n"
        "  · 所有十六进制色值，一个都不许删\n"
        "  · 光照描述（光源、方向、色温）\n"
        "  · carry-over 声明（“图N 的设定被整个替换/占据”那句）\n"
        "  · 用户点明的硬约束（保留什么、改变什么、哪张是画布）\n"
        f"写完自己数一遍；超了就继续删，直到 ≤ {limit_txt.split('（')[0]}再输出。\n"
        "只输出压缩后的提示词正文，不要解释，不要 markdown 代码块标记。"
    )
    out, err = call_llm(system_prompt, ask, cfg, thinking=thinking,
                        first_reply=result)
    if err or not out:
        return result, f"压缩失败({err})，保留原稿"
    out = _strip_fence(out)
    out, _ = _soften_negation(out)
    eff2, words2, cjk2 = _count_words(out)
    # 没压动（长度没降）就退回原稿
    if eff2 >= eff:
        return result, f"压缩无效({eff}→{eff2} 当量)，保留原稿"
    # 落进预算了就收货 —— 这才是压缩的目的。
    # ⚠️ 别在这里加"压缩比例不得低于 X%"的闸门：实测模型会一次砍得比要求更狠
    # （774 字 → 150 字，上限是 360 字，完全合格），按比例判会被误丢，
    # 结果超长稿原样返回、白花一次调用。判据只能是"是否真的落进预算"。
    if not (words2 > words_max or cjk2 > cjk_max):
        return out, f"已压缩 {eff}→{eff2} 词当量（中文 {cjk}→{cjk2} 字）"
    # 压了但还是超限：只在"还超很多"时才算失败，否则收货
    if (words2 <= words_max * 1.3) and (cjk2 <= cjk_max * 1.3):
        return out, f"已压缩 {eff}→{eff2} 词当量，仍略超（中文 {cjk2} 字）"
    # 只砍掉一小部分、离预算还很远 → 说明它没照做，退回原稿更安全
    return result, f"压缩不到位({eff}→{eff2} 当量，仍超限)，保留原稿"


# ------------------------------------------------- 否定句式兜底改写
# 为什么需要它：FLUX 没有负向通道，"no X" 既无效又会让模型去关注 X。
# 实测四轮下来，光靠指令治不住——模型总在风格/光照句尾追加 "no gradients /
# no shadows / 无投影"。
#
# 做法：**整句判定替换**。原因是一开始用"短语替换"，匹配到了却留下残渣
# （"no shading or gradient," 只剩 "or gradient,"；"无明暗过渡与投影" 只剩 "与投影"），
# 因为删掉短语后的连接词没法自动拼干净。这些否定残留几乎都出现在**独立的风格/光照句**里，
# 所以整句换成等价的正向句最干净。故意保守：只认表里列出的明确句式，不猜。
NEG_SENT_EXACT = [
    # ── 英文（按长短顺序，先匹配具体的）──
    (r'(?i)flat vector illustration style,?\s*(?:with\s+)?(?:uniform\s+|solid\s+|evenly filled\s+)?'
     r'colour\s+blocks?\s+and\s+no\s+shading\b',
     'Flat vector illustration style with uniform flat colour fills'),
    (r'(?i)with\s+(?:uniform\s+|solid\s+|evenly\s+filled\s+)?(?:flat\s+)?(?:colour|color)\s+'
     r'(?:blocks?|fills?|fields?)\s+and\s+no\s+shading\b',
     'with uniform flat colour fills'),
    (r'(?i)with\s+no\s+directional\s+(?:shadows?|shading|falloff|lighting)\b',
     'with even shadowless illumination'),
    (r'(?i)nothing\s+else\s+from\s+(<[^>]+>|image\s*\d+|图\s*\d+)\s+appears\b',
     r'the rest of \1 stays out of the frame'),
    (r'(?i)\bnothing\s+else\s+from\s+(<[^>]+>)\b', r'only that part of \1'),
    (r'(?i)with\s+no\s+directional\s+sources?\b', 'with even ambient light'),
    (r'(?i)no\s+directional\s+sources?\b', 'even ambient light'),
    (r'(?i)no\s+visible\s+light\s+sources?\b', 'even directionless ambient light'),
    (r'(?i)no\s+(?:visible\s+)?light\s+sources?\b', 'even directionless ambient light'),
    (r'(?i)no\s+directional\s+(?:shadows?|shading|falloff|lighting)\b', 'even shadowless illumination'),
    # 替换词要带上名词，否则会出现 "with shadowless on any surface" 这种读不通的句子。
    # 把后面的整个介词短语一起吃掉，整段换成通顺的正向说法。
    (r'(?i)(?:and\s+)?no\s+cast\s+shadows?\s+(?:on|across|over|anywhere\s+on)\s+'
     r'(?:any|every|all|the)?\s*\w+', 'even shadowless lighting throughout'),
    (r'(?i)(?:and\s+)?no\s+cast\s+shadows?\b', 'even shadowless lighting'),
    (r'(?i)(?:with\s+)?(?:and\s+)?no\s+shading\s+gradients?\b', 'uniform flat colour fills'),
    (r'(?i)(?:with\s+)?(?:and\s+)?no\s+shading\s+or\s+gradient\b', 'uniform flat colour fills'),
    (r'(?i)(?:with\s+)?(?:and\s+)?no\s+shading\b', 'uniform flat colour fills'),
    (r'(?i)(?:with\s+)?(?:and\s+)?no\s+gradients?\b', 'uniform flat colour fills'),
    # 替换词 "matte surfaces" 是名词短语，直接接在后面会读不通
    # （"with even shadowless lighting matte surfaces"）—— 所以让它自带前置逗号。
    # 后面的整个介词短语（on the glass / across every surface）要一起吃掉：
    # 只吃一个单词会留下悬空的 "glass"（"matte surfaces glass"）。
    (r'(?i)[\s,，、]*(?:and\s+)?(?:with\s+)?(?:and\s+)?no\s+specular\s+'
     r'(?:highlights?|reflections?)\b(?:\s+(?:on|across|over|anywhere\s+on)\b[^,.;，。；]*)?',
     ', matte surfaces'),
    (r'(?i)[\s,，、]*(?:and\s+)?(?:with\s+)?(?:and\s+)?no\s+highlights?\b'
     r'(?:\s+(?:on|across|over|anywhere\s+on)\b[^,.;，。；]*)?', ', matte surfaces'),
    (r'(?i)(?:with\s+)?(?:and\s+)?no\s+tonal\s+variation\b', 'flat even tone'),
    (r'(?i)(?:with\s+)?(?:and\s+)?no\s+texture\b', 'smooth surfaces'),
    (r'(?i)(?:with\s+)?(?:and\s+)?no\s+outlines?\b', 'clean hard edges'),
    # ── 中文 ──
    # 尾巴上的连接词很关键：中文习惯写「无明暗过渡与投影」，
    # 只吃掉第一个词会留下「与投影」这种残渣。
    # 注意中文这里用 [，,、]?\s*[与和及]? 而不是把「无」也塞进第三分支 ——
    # 「均匀平涂，无高光与阴影变化」这种带逗号的才不会漏。
    (r'无(?:方向性)?(?:明暗过渡|明暗渐变|渐变过渡|渐变|投影|阴影变化|阴影|高光|纹理颗粒|纹理|方向性阴影)'
     r'(?:\s*[，,、]?\s*[与和及]?\s*'
     r'(?:方向性)?(?:明暗过渡|明暗渐变|渐变过渡|渐变|投影|阴影变化|阴影|高光|纹理颗粒|纹理|方向性阴影))*',
     '均匀平涂、保持固有色'),
]
_NEG_SENT = [(re.compile(p), r) for p, r in NEG_SENT_EXACT]

# 句尾连接词/助词，替换后可能悬空 —— 顺手清掉
_DANGLE = re.compile(r'(?i)(?:\s*[,，、]\s*)?(?:\b(?:and|or|with)\b|与|和|及)\s*(?=[,，、.。;；]|$)')
_MULTI_SEP = re.compile(r'([,，、])\s*(?=[,，、])')
# 替换后常出现"同一句里同一段正向说法说了两遍"（原句本来就有 + 我替换进去的），清一次。
# 分隔段要排除 . ! ? ; ，否则会把两个独立句子里的重复误判成同句重复。
_DUP_TERMS = (r'uniform flat colou?r fills|matte surfaces|even shadowless illumination|'
              r'flat even tone|smooth surfaces|clean hard edges|均匀平涂、保持固有色|shadowless')
_DUP_FRAG = re.compile(r'(?i)\b(' + _DUP_TERMS + r')\b'
                       r'((?:[^.!?;。！？；]|\b(?:and|or|with)\b){0,40}?)\1')
# 同义反复："shadowless illumination" 后紧跟 "shadowless surfaces" —— 语义已覆盖，去掉后半。
# ⚠️ 必须把中间那个 _TERM 一起吃掉，否则会留下孤立的 "shadowless"
# （实测漏过一次：变成 "even shadowless illumination shadowless"，读不通）。
# 尾部的 (?:on|across|over)\s+(?:any|every|all)\s+\w+ 也要吃掉。
_SL_TERM = (r'(?:shadowless|cast\s+shadows?|shadows?|shading|lighting|light|highlights?|'
            r'tonal\s+variation|surfaces?)')
_SHADOWLESS_DUP = re.compile(r'(?i)(even\s+)?shadowless\s+(illumination|lighting|light)\b'
                             r'[\s,，、]*(?:and\s+)?'
                             r'(?:(?:on|across|over)\s+)?(?:any\s+|every\s+|all\s+)?'
                             r'(?:' + _SL_TERM + r'[\s,，、]*)+'
                             r'(?:(?:on|across|over)\s+(?:any|every|all)\s+\w+)?')
_BARE_SHADOWLESS = re.compile(r'(?i)\bshadowless\s+surfaces?\b')


def _soften_negation(text):
    """把风格/光照类否定句换成正向等价说法。返回 (新文本, 改了几处)"""
    n = 0
    for rx, rep in _NEG_SENT:
        text, k = rx.subn(rep, text)
        n += k

    if n:
        # ── 只在真的替换过时才做的收尾（这些是替换的副作用）──
        text = _DANGLE.sub('', text)
        text = _MULTI_SEP.sub(r'\1', text)
        # 替换掉短语后可能留下孤零零的冒号（"uniform flat colour fills: hair #F5E14B"）。
        # 按语言选标点：英文用 ", "，中文用 "，" —— 别把中文逗号塞进英文正文。
        def _fix_colon(m):
            before = text[max(0, m.start() - 12):m.start()]
            cjk = len(re.findall(r'[\u4e00-\u9fff]', before)) > 0
            return '，' if cjk else ', '
        text = re.sub(r'\s*[:：]\s*(?=[A-Za-z\u4e00-\u9fff])', _fix_colon, text)
        # 替换后常并列出现两个近义说法（"uniform flat colour blocks uniform flat colour fills"）
        for _ in range(3):
            new = _DUP_FRAG.sub(r'\1\2', text, count=1)
            if new == text:
                break
            text = new
        text = re.sub(r'(?i)\buniform\s+flat\s+colou?r\s+(?:blocks?|fills?|fields?)\b[\s,，、]*'
                      r'(?=(?:uniform\s+)?flat\s+colou?r\s+(?:blocks?|fills?|fields?)\b)', '', text)
        text = re.sub(r'(?i)\bflat\s+colou?r\s+(?:blocks?|fills?)\b[\s,，、]*'
                      r'(?=(?:uniform\s+)?flat\s+colou?r\s+(?:blocks?|fills?|fields?)\b)', '', text)

    # ── 无条件清理：这些是通用卫生问题，不是替换的副作用 ──
    # 同一句里并列出现多个"…color fills"说法（uniform solid color fills … uniform flat
    # colour fills）—— 近义重复，句内只留第一个。
    # 用"相邻去重"不行：两个片段中间往往还夹着别的词（", simple rounded shapes, "），
    # 所以改成按句拆分、句内只保留第一处。
    _COL = r'colo(?:u)?r'
    _FILLS_ALL = re.compile(r'(?i)(?:uniform|solid|flat|even|smooth)\s+(?:flat\s+|solid\s+|even\s+)?'
                            + _COL + r'\s+fills?')
    _BLOCKS_ALL = re.compile(r'(?i)(?:uniform|solid|flat)\s+(?:flat\s+)?' + _COL
                             + r'\s+(?:blocks?|fields?)')
    # 中文没有 \b 词边界，单独一条（否则「均匀平涂、保持固有色」重复时认不出来）
    _CJK_FLAT_ALL = re.compile(r'均匀平涂(?:、保持固有色)?')
    # 同一句里「保持固有色」的各种措辞重复（"色彩保持固有色" / "保持固有色" / "保持自身固有色"）
    # ——实测漏过一次："平涂色块上均匀平涂、保持固有色，色彩保持固有色。"
    _CJK_COLOR_KEEP = re.compile(r'(?:色彩|颜色|色泽)?保持(?:自身)?固有色')

    def _keep_first(rx, s, prefer=None):
        """句内重复只留一处；留**信息最全**的那一处（prefer 指定优先形式），
        其余删除。比"盲删后面的"安全：不会把"保持固有色"这种附加信息丢掉。"""
        parts = re.split(r'((?<=[.!?;。！？；])\s*)', s)
        out = []
        for seg in parts:
            if not seg or not seg.strip() or re.fullmatch(r'\s+', seg):
                out.append(seg)
                continue
            hits = list(rx.finditer(seg))
            if len(hits) <= 1:
                out.append(seg)
                continue
            keep = hits[0]
            if prefer:
                for h in hits:
                    if h.group(0) == prefer:
                        keep = h
                        break
            res, last = [], 0
            for m in hits:
                if m is keep:
                    continue
                start = m.start()
                while start > last and seg[start - 1] in ' \t,，、':
                    start -= 1
                if start <= last:
                    continue
                res.append(seg[last:start])
                last = m.end()
            res.append(seg[last:])
            out.append(''.join(res))
        return ''.join(out)

    # 英文留 "uniform flat colour fills"（信息最全）；中文留「均匀平涂、保持固有色」。
    text = _keep_first(_FILLS_ALL, text, prefer='uniform flat colour fills')
    text = _keep_first(_BLOCKS_ALL, text)
    text = _keep_first(_CJK_FLAT_ALL, text, prefer='均匀平涂、保持固有色')
    # 「保持固有色」的各种措辞在句内只说一次
    text = _keep_first(_CJK_COLOR_KEEP, text, prefer='保持固有色')
    # 上面删掉重复后可能留下"均匀平涂、保持固有色"→"均匀平涂、"，收一下尾
    text = re.sub(r'均匀平涂[、，,]\s*(?=[，,。；;]|$)', '均匀平涂', text)
    # 同前缀重复："flat 2D cartoon illustration … flat cartoon illustration"
    text = re.sub(r'(?i)\bflat\s+2D\s+cartoon\s+illustration\b[\s,，、]*'
                  r'(?=flat\s+cartoon\s+illustration\b)', '', text)

    # ⚠️ 下面这两条重复清理必须放在 if not n 之前 ——
    # 重复可能是**模型自己写出来的**（不经过我的替换），只看 n 会整个跳过。
    # 踩过的坑：放在 if not n 之后，于是 "even shadowless illumination shadowless surfaces"
    # 这种没有否定词、但有重复的输入完全没被处理。
    text = _SHADOWLESS_DUP.sub(r'\1shadowless illumination throughout', text)
    hits = list(re.finditer(r'(?i)\bshadowless\s+surfaces?\b', text))
    if len(hits) > 1:
        keep, last, out = 0, 0, []
        for m in hits:
            keep += 1
            if keep == 1:
                continue
            out.append(text[last:m.start()])
            last = m.end()
        out.append(text[last:])
        text = ''.join(out)

    # 改写残留的孤立助词「的」：
    #   「云朵为的白色」→「云朵为白色」
    #   「以的色块呈现」→「以色块呈现」
    # 为什么会发生：模型把「以平涂的色块呈现」这类结构压缩时把中心名词删了，
    # 只剩下介词 + 的。这里把「介词/系词 + 的 + 名词/色值」里的的去掉。
    # ⚠️ 必须在 if not n 之前 —— 这个瑕疵跟"有没有否定词"无关（踩过三次）。
    # 注意排除「成为的一部分 / 作为的一部分」这类正常说法 —— 那里的「为」是动词的一部分，
    # 不是孤立的介词（实测误伤过一次）。
    # 字符类要含动词（实测漏过「各表面呈的色块」），但**不要放「和」** ——
    # 「柔和的过渡」里「和的」会被误当成孤立助词（也误伤过一次）。
    # ⚠️ 还要排除「原有的 / 固有的 / 现有的 / 所有的」这类**正常所有格** ——
    # 那里的「有的」是「原有 + 的」，不是「有 + 的」。实测踩过：
    # 「图1原有的场景不进入结果」被删成「图1原有场景…」。
    _ORPHAN = (r'(?<!成)(?<!作)(?<!原)(?<!固)(?<!现)(?<!所)'
               r'(?:为|以|用|是|与|及|由|被|把|将|在|从|对|'
               r'呈|现|显|有|带|配|覆盖|包含|提供)的')
    text = re.sub(_ORPHAN + r'(?=[\u4e00-\u9fff#])', lambda m: m.group(0)[:-1], text)
    # 更窄的一条：只认「呈/现/显 + 的 + 短名词 + 断句」。
    # ⚠️ 不要把「为/以/用/由/被/把/将」也放进这一条 —— 那样会把
    # 「呈现出柔和的过渡」误判成「出的 + 柔和」删掉（实测误伤过）。
    text = re.sub(r'((?:呈|现|显))的([\u4e00-\u9fff]{1,6})(?=[，、。；,;.]|$)', r'\1\2', text)

    # 删掉「某张图没被用上」这类多余声明：
    #   「图3与图4不参与本次合成。」「图3 的内容不会出现在结果里。」
    #   "image 3 contributes nothing to the result" / "image 3 is not used here"
    # 为什么必须程序化删：gate 里已经明说「没用的图干脆一个字都不提」，
    # 但模型仍然爱补这么一句（实测输出过「图3与图4不参与本次合成」）。
    # 它有两个害处：① 是多余的（没人问）② 带否定词（"不参与"），
    # 而 FLUX.2 没有负向通道，写出来只会让模型去注意那张图。
    # ⚠️ 必须在 if not n 之前 —— 这句本身就可能带「不」，但也可能是
    # 「contributes nothing」这种没有中文否定的写法，只跟着 n 走会漏。
    _UNUSED_IMG = re.compile(
        r'(?:^|(?<=[。．.！!？?；;]))\s*'
        r'(?:'
        # 中文：图号后面**紧接着**就是否定 —— 这才是"这张图没用到"。
        # ⚠️ 不能在「图N」和否定词之间允许任意文字：那样会把合法的
        #    carry-over 声明（「图1原有的场景不进入结果」）也误删 —— 实测踩过。
        #    合法声明的结构是「图N + 的/原有… + 名词 + 不进入」，中间有名词；
        #    多余声明的结构是「图N(、图M…) + 不参与/未使用」，图号后直接跟否定。
        r'图\s*\d+\s*(?:(?:、|,|，|与|和|及)\s*图\s*\d+\s*)*'
        r'(?:均|都|也)?\s*'
        r'(?:不参与|不参加|未参与|未使用|没用上)'
        r'|image\s*\d+\s*(?:(?:and|,|、)\s*image\s*\d+\s*)*'
        r'(?:contributes?\s+nothing|is\s+not\s+used|are\s+not\s+used)'
        r')'
        r'[^。．.！!？?；;\n]{0,40}[。．.！!？?；;]?',
        re.I)
    text = _UNUSED_IMG.sub('', text)
    # 删句后可能留下多余空格或空标点，收一下尾
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'([。．！!？?；;])[，,、；;]', r'\1', text)
    text = re.sub(r'^[，,、；;\s]+', '', text)

    # ⚠️ 下面这些收尾只在**真的替换过**时才做。
    # 踩过的坑：无条件跑过一次，结果对没有任何否定的输入也做字符串拼接/删除，
    # 把句间空格吃掉了（"Sky: #6B8FF5. Hill:" → "#6B8FF5.Hill:"）。
    if not n:
        return text, n

    # 同一句里已经说过 shadowless，后面再冒出来的 "shadowless surfaces" 是赘述
    text = re.sub(r'(?i)\beven\s+even\b', 'even', text)
    text = re.sub(r'(?i)\bunifo\w+\s+unifo\w+\b', 'uniform', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'([，、])\1+', r'\1', text)
    text = re.sub(r'。{2,}', '。', text)
    text = re.sub(r'(?<![.!?。！？])\s+([,;])', r'\1', text)   # 只动逗号/分号，不动句号
    text = re.sub(r'(?i),\s*and\s+(?=[,.;])', '', text)
    text = re.sub(r'([，、])\s*([。.])', r'\2', text)
    text = re.sub(r',\s*,+', ',', text)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r'([,，、])\s*([,，、.。])', r'\2', text)
    # 拼接残留的"标点前空格"（"smooth surfaces ."）—— 只可能在拼接后出现，清掉是安全的。
    # 方向是"标点**前**的空格"，不是"标点后的空格"，别搞反。
    text = re.sub(r'[ \t]+([,.;，。；!?！？])', r'\1', text)
    # 替换后可能留下句首逗号（"…unshaded areas, colors… , matte surfaces"）或连续标点
    text = re.sub(r'([.。!！?？;；])\s*[,，、]\s*', r'\1 ', text)
    text = re.sub(r'(?m)^[\s,，、]+', '', text)
    # ⚠️ 只压空格和制表符，**不要碰换行** —— 用 [\s]{2,} 会把段落结构一起压掉
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip(), n


# ------------------------------------------------- system prompt（规则本体）
# 抽成模块级函数，好处：聊天室里离线评测调用的就是插件运行时用的同一份规则，
# 不会出现"测的是旧副本"。

MODE_HINTS = {
    "标准优化": "按 skill 规则全面优化，保持原意与关键约束。",
    "保守改写": "只修正明确违反 skill 规则的地方，其余尽量保留原文措辞。",
    "大幅重构": "可以重写结构与表达方式，但必须保留全部意图与硬约束。",
}

# 模型相关的硬要求，放在整份规范之前。
# 为什么提到 system 层：规范文档 30+ KB，关键约束埋在中间会被"注意力稀释"——
# 实测过一次：新加规则生效了，但原有的 hex 绑定被整个丢掉（8 个 -> 0 个）。
# 短、显眼、排前面的硬清单比长清单更压得住。
GATES = {
    "flux2-dev": (
        "【硬要求 · 逐条满足 · 输出前自查】\n"
        "0. 下面那份规范是「规则」，不是「素材」。它的标题、清单、术语、"
        "“no X / 不许 X”这类表述都不许抄进成品的提示词里——"
        "成品里出现它们就是错的，即使规范文档里到处都写着。\n"
        "0b.【语言】**用户用什么语言写的，成品就用什么语言**，不要在语言之间跳。"
        "中文指令写中文提示词，英文指令写英文提示词。（本次该用哪种语言，"
        "上面「语言」那一条已由系统判定好，直接照它执行，不要自行改判。）\n"
        "1. 色值绑定：必须给出 3~7 个关键色并各自绑定到具体物体。"
        "用户给了十六进制色值就逐个沿用，给了几个留几个，一个都不许丢；"
        "用户没给就自己定一组协调的关键色——有参考图就从画面显而易见的颜色来，"
        "没有参考图（纯文生图）就按描述自己定。"
        "任何情况下都不要省略色值这一节。\n"
        "2. 全文不许出现排除式否定，包括风格和光照的描述。"
        "用户说的任何「删除/移除/不要/没有/抛弃」都要改写成“那个位置实际是什么”的正向描述。"
        "（编辑时的「其余保持不变」属于保留声明，不算否定，照原样保留。）"
        "把风格和光照写成“是什么样子”：颜色平涂就写“均匀平涂的色块”，"
        "光均匀就写“均匀无方向的环境光”——凡是写成“no gradients / no shadows / "
        "no highlights / nothing else appears / 无明暗过渡 / 无投影”这类“不是什么”的句子，"
        "一律改写掉，一句都不留。\n"
        "3. 光照必须写全：光源、方向、色温、在表面上的表现。用户没提光照就按其场景推断后写明。"
        "但“表面上的表现”必须跟风格一致：成品若是扁平/矢量/无光影/图解风格，"
        "就写“均匀平涂的色块、色彩保持固有色”，不要硬编“清晰的投影”“柔和的光影过渡”。\n"
        "4.【多参考图 · 最容易踩的坑】给每个元素指定**唯一的来源**，用正向声明写；"
        "参考图是整帧，不写清来源就等于默许它的背景/道具/构图一起跟过来。\n"
        "   ※ 为什么必须写：你描述的 X 如果在某张参考图里也存在，就有两条路径同时指向同一个画面元素——"
        "文字一条、那张图的 latent 一条；没有一句话告诉它这个 X 该照描述画、而不是从图 N 拿，"
        "它就取图 N 的。本模型没有负向通道，所以**“说明成品里没有什么”的从句接不住内容**。\n"
        "   ※ 两种否定要分开（实测结论，别一刀切）：\n"
        "     · **命令式删除 = 有效，保留**：「抛弃图1 的场景」「删除图2 的人物」——"
        "它有动词、有对象，模型当成一次编辑操作去执行。这是训练里见过的编辑指令形态。\n"
        "     · **状态从句 = 无效，删掉**：「图1 原有的场景不进入结果」"
        "“the setting of image 1 does not carry over”“nothing else from image N appears”——"
        "它先把那个内容表征出来，又没有动作可执行，内容留下来了。一句都不要写。\n"
        "   ※ 命令与归权**并用**（缺一个都会漏）：命令式删除实测稳定，但如果被抛弃的内容"
        "和你在别处描述的内容重合，那张参考图的 latent 会把它拉回来。所以后面必须跟一句归权声明："
        "「抛弃图1 的场景；画面环境只由图2 提供。」"
        "英文：「Discard the scene of image 1; the frame's environment is image 2's own.」\n"
        "   ※ 归权声明模板（元素名和图片号换成实际的，一个元素一句）："
        "「人物与身份 = 图1；环境、材质、构图 = 图2；光源方向与色温 = 图2。」"
        "英文：「Person and identity: image 1. Environment, materials and composition: image 2. "
        "Key direction and colour temperature: image 2.」（某张图只提供风格/材质，就写它只提供那一样。）\n"
        "   ※ 先写**目标画面**，再写“来自图N”。语序=优先级：开头就出现的参考图会被当成“要还原的那张”。\n"
        "     弱：“Take the person from image 1 and put them in image 2.”\n"
        "     强：“An empty interior of image 2's kind. The person of image 1 stands in it at full scale.”\n"
        "   ※ **命令式删除的对象必须笼统**：「抛弃图1 的场景」可以（“场景”是笼统词），"
        "「抛弃图1 的天空、云朵和草地」不行——你经常把源图内容记错（那三样其实在图2 里），"
        "于是命令指向了错的图，真正的内容反而没人管。\n"
        "   ※ **绝不写 extract / cut out / re-composite / inpaint**（除非任务真是抠图）："
        "这些是操作词，它会按贴图逻辑还你一张拼接图（边缘硬、光不一致、风格还是源图的）。"
        "直接描述成品帧：人已经站在那个空间里、已经受好光、尺度已经对上。\n"
        "   ※ **某张图完全没用上时，那张图干脆一个字都不提**——"
        "不要写“图3 不参与”“图3 的内容不会出现在结果里”“image 3 contributes nothing”这类句子："
        "既是否定从句，又是在描述你其实没看过的源图。\n"
        "5. 这条只管“物体和地点”：不要编造用户没给的具体主体、物种、地点或场景物件；"
        "指代保持通用（“图1 里的人物”“图2 里的场景”），让描述能被反复套用。"
        "颜色、材质、光照这类能从画面直接推断的属性，照第 1、3 条正常写。\n"
        "6. 篇幅硬上限。写完数一遍词数，超了就删到线内（删的优先顺序：重复的近义短语、"
        "可推断的修饰词、次要细节；不许删色值、光照、carry-over 声明和硬约束）：\n"
        "   · 单图编辑 / 纯文生图：≤ 300 词（中文按 ≤ 360 字）\n"
        "   · 多参考图编辑：≤ 420 词（中文按 ≤ 500 字）\n"
        "   典型场景的默认区间是 30–80 词；长区间（80–300+）是给"
        "「多主体 + 多参考图 + 色板」用的，不要因为能写就写长。\n"
        "   中文不必因为「字多」而删——按汉字个数算上限（360 / 500 字），"
        "上限定得比英文区间宽，就是为了不逼你为了凑词数硬压中文。\n"
        "7. 不许引用不存在的图。提示里说带了几张参考图，就只有那几张："
        "只有 1 张时不许出现 <image2>、<image3>；"
        "用户提到的图号超出实际张数时，按“用户想要的效果”正常写，"
        "但不要凭空描述那张不存在的图里有什么。\n"
        "8.【风格不许留空】纯文字描述内容、不写镜头与风格时，模型会自己填一套审美先验"
        "（常见是影棚/时尚大片感），这是“指定的背景被它自己的风格压掉”的首要原因。"
        "必须写明**媒介 + 镜头焦段与光圈 + 画幅 + 色调倾向**；"
        "能用参考图锁就用参考图锁（“色调与光线的唯一来源 = 图N”）——"
        "图像锚比文字描述更能压住它自己的审美。\n"
        "9.【命令式删除可以留，状态从句必须删】“抛弃/删除/移除 + 笼统对象”是有效编辑指令，"
        "照原样保留；但绝不要写成描述成品状态的从句（“某张图的内容不进入结果”）。"
        "写完通读一遍：凡是**在描述成品里没有什么**的句子，全部删掉或改成正向归权。\n"
    ),
    "qwen-image-2-1": (
        "【硬要求 · 逐条满足 · 输出前自查】\n"
        "0. 下面那份规范是「规则」，不是「素材」，不许把它的话抄进成品。\n"
        "0b.【语言】见最开头的语言总则。此处只补充一点：规范里有两套语言规则，"
        "按任务选，别选错——带参考图的编辑/合成按用户输入语言，纯文生图才一律英文。\n"
        "1. 描述正文保持正向观察式，正文里不散落否定。"
        "**不要输出 Negative prompt 段、也不要罗列“要避免的东西”** —— "
        "本 skill 的输出负载里根本没有负向字段，而且罗列近似短语会让模型退化成"
        "不断重复（“与元素与元素与元素”）直到输出不可用。"
        "用户说的「删除/移除/不要」直接在描述里落成正向观察即可"
        "（“不要现代元素”→“period-accurate furniture”）。\n"
        "2. 多参考图：说明每张图的角色（谁是画布、谁提供素材）。"
        "**并且必须写出素材图原来那个位置由什么占据——用正向填空写，不要只写否定式。**"
        "实测“说出那个位置是什么”比“说出什么不进来”更难漏；只写后者时会漏掉。\n"
        "   照这个模板写（把图号换成实际的，命令 + 归权两句并用）：\n"
        "   「抛弃图1 的场景；以图2 为画布，画面环境只由图2 提供，"
        "人物直接落在图2 的地面上。」\n"
        "   英文对应：「Discard the scene of image 1; the frame is image 2's environment, "
        "and the person of image 1 stands on image 2's ground.」\n"
        "   ❗ **不要再追加“图1 原有的场景不进入结果”这类收尾**：那是状态从句，"
        "既是否定式，又把源图内容重说了一遍，起反作用。\n"
        "   命令式删除实测稳定（基本每次都能把图1 的场景去掉），但它**不完全管用**——"
        "被抛弃的内容若与你在别处描述的重合，图1 的 latent 会把它拉回来，"
        "所以后面那句归权声明必须有。\n"
        "   不写就等于默许素材图的背景、道具、构图一起跟过来。\n"
        "3. 光照必须在描述里有明确交代：光源、方向、质感、以及留下的明暗。不许留空。\n"
        "3b.【篇幅 · 只对纯文生图】没有任何参考图时，按官方 PE-T2I 规格写"
        "**约 20 句、400–500 词的完整描述**：开头一句约 20 词点明媒介/风格/主体/背景，"
        "中间用 8–14 个方位短语走完画面（到四角、四边、中心），光照单独一句，"
        "结尾一句总览构图与色调。短简报不等于短描述——简报越短，越是你自己把画面补全。"
        "（带参考图的编辑任务不适用这条；**编辑任务官方没有篇幅上限，别自己往下压**"
        "——该写清的材质、光照、carry-over 都写全，宁可长一点。）\n"
        "4. 用户用 <image1>/<image2> 之类的标签指代图片时，照原样保留，不要改成自然语言。\n"
        "5. 不要编造用户没给的具体主体、物种、地点或场景物件；"
        "颜色、材质、光照这类能从画面推断的属性正常写。\n"
    ),
}


# 语言总则：**结论式**，不是选择式。
# 踩过的坑（重要）：
#   一开始写成"中文输入→中文 / 英文输入→英文"的决策树，放在 system 最前面，
#   结果英文输入 0/5 全串成中文 —— 因为 system 提示本身是中文，
#   模型倾向跟随 system 的语言，而不是去查输入语言。
#   改成在 Python 侧先判断输入语言，把结论硬写进 system，才是可靠的。
LANG_PRIMER = (
    "【语言 · 已由系统判定，直接执行】\n"
    "  本次成品正文必须用 **{lang}** 写。这是硬性要求，不要自行改判。\n"
    "  {reason}\n"
)


def detect_lang(text):
    """判断用户指令的语言：中文字多于拉丁字母 → zh，否则 en。

    ⚠️ 必须先剥离非语言内容再数：`<image1>` 这类标签、hex 色值、以及
    规范术语（hex/color/prompt/image）都是拉丁字母，会把中文指令误判成英文。
    实测踩过：`用 <image1> 做画布` 被判成 en。
    """
    if not text:
        return "zh"
    s = re.sub(r'<image\s*\d+>|image\s*\d+', ' ', str(text), flags=re.I)
    s = re.sub(r'#[0-9A-Fa-f]{3,8}', ' ', s)
    s = re.sub(r'(?i)\b(hex|color|colour|prompt|image|skill|aspect|ratio)\b', ' ', s)
    cjk = len(re.findall(r'[\u4e00-\u9fff]', s))
    latin = len(re.findall(r'[A-Za-z]', s))
    return "zh" if cjk >= latin else "en"


def build_system_prompt(model_key, skill_text, mode="标准优化", user_text="", n_images=0):
    """插件运行时和离线评测共用同一份 system prompt

    n_images：本次附带的参考图数量。为 0 = 纯文生图。
    ⚠️ 纯文生图必须由这里直接把语言定成英文，不能只写一条"纯文生图除外"的例外——
    那样 system 里同时出现「必须用中文」和「纯文生图一律英文」两句，
    中文输入时模型跟第一句走，实测 Q3 连续两次输出中文（472/503 个中文字）。
    """
    gates = GATES.get(model_key, "")
    mode_hint = MODE_HINTS.get(mode, "")
    _lang = decide_output_lang(model_key, n_images, user_text)
    if _lang == "en":
        lang = "英文"
        if not n_images and model_key == "qwen-image-2-1":
            reason = ("本次没有任何参考图，属于纯文生图；本模型规范的总则要求这类提示词"
                      "一律用英文，与用户指令本身用什么语言无关。")
            if detect_lang(user_text) == "zh":
                reason += "（用户写的是中文，但成品正文仍要英文。）"
        elif n_images:
            reason = "本次带了参考图，属于编辑/合成任务，语言跟随用户指令的语言。"
        else:
            reason = "本次属于纯文生图；本模型规范未要求必须用英文，语言跟随用户指令的语言。"
    else:
        lang = "中文"
        if n_images:
            reason = ("本次带了参考图，属于编辑/合成任务，语言跟随用户指令的语言。"
                      "规范里 \"always in English\" 只管纯文生图，不要拿它改这里的语言。")
        else:
            reason = "本次属于纯文生图；本模型规范未要求必须用英文，语言跟随用户指令的语言。"
    return (
        "你是一个提示词工程助手。下面是一份必须严格遵守的专业规范（skill），"
        "请完全按照它来改写用户给出的提示词。\n"
        "只输出改写后的提示词正文，不要任何解释、前言、markdown 代码块标记。\n"
        "⚠️ 规范里如果有「输出格式」章节（比如要求分 💡 解析 / 📋 提示词 / 🎨 建议 三段，"
        "或要求用 JSON 字段包起来），那一节**不适用于本次**：这里要的是能直接接进"
        "采样器的纯文本，所以只给正文，别加标题、别加代码块、别加前后说明。\n"
        + LANG_PRIMER.format(lang=lang, reason=reason)
        + (gates + "\n" if gates else "")
        + f"本次改写强度：{mode_hint}\n"
        f"\n================ 规范开始 ================\n{skill_text}\n================ 规范结束 ================"
    )


# ------------------------------------------------- 从连线识别"目标模型"

# 加载器 → 哪个槽位放着模型文件名（新老节点都覆盖）
LOADER_SLOTS = {
    "UNETLoader": [0],
    "CheckpointLoaderSimple": [0],
    "CheckpointLoader": [0],
    "unet_loader": [0],
    "DiffusionLoader": [0],
    "DiffusionModelLoader": [0],
    "LoraLoader": [0],
    "LoraLoaderModelOnly": [0],
    "CheckpointLoaderNF4": [0],
    "ImageOnlyCheckpointLoader": [0],
}

# 加载器 → 装模型文件名的输入名（执行时的图里没有 widgets_values，只有 inputs）
LOADER_INPUT_NAMES = {
    "UNETLoader": ("unet_name",),
    "unet_loader": ("unet_name",),
    "CheckpointLoaderSimple": ("ckpt_name",),
    "CheckpointLoader": ("ckpt_name",),
    "CheckpointLoaderNF4": ("ckpt_name",),
    "ImageOnlyCheckpointLoader": ("ckpt_name",),
    "DiffusionLoader": ("model_name", "diffusion_model", "unet_name"),
    "DiffusionModelLoader": ("model_name", "diffusion_model", "unet_name"),
    "LoraLoader": ("lora_name",),
    "LoraLoaderModelOnly": ("lora_name",),
}


def _loader_file(node, ct):
    """从加载器节点里取出模型文件名。
    执行时的图（API 提交）文件名在 inputs 里；前端保存的图在 widgets_values 里。"""
    if not isinstance(node, dict):
        return ""
    inp = node.get("inputs")
    if isinstance(inp, dict):
        for nm in LOADER_INPUT_NAMES.get(ct, ()):
            v = inp.get(nm)
            if isinstance(v, str) and v.strip():
                return v.strip()
    vals = node.get("widgets_values")
    if isinstance(vals, list):
        for i in LOADER_SLOTS.get(ct, [0]):
            if i < len(vals) and _wstr(vals[i]):
                return _wstr(vals[i])
    return ""


# 按顺序匹配：命中即判定（qwen 排在前面，避免 "qwen…flux" 之类被误判）
MODEL_HINTS = [
    ("qwen", "qwen-image-2-1"),
    ("千问", "qwen-image-2-1"),
    ("通义", "qwen-image-2-1"),
    ("wan2", "qwen-image-2-1"),
    ("flux", "flux2-dev"),
    ("黑森林", "flux2-dev"),
]


def _wstr(v):
    """widgets_values 里的值 → 干净字符串（兼容 None / NaN / 数字）"""
    if v is None:
        return ""
    if isinstance(v, float) and v != v:   # NaN
        return ""
    if isinstance(v, bool):
        return ""
    return str(v).strip()


def _is_link(v, nodes):
    """判断是不是一条真实连线：[来源节点id, 输出槽]。widget 值(如 [文件名, "default"])不算。"""
    if not (isinstance(v, list) and len(v) == 2):
        return False
    if isinstance(v[1], (dict, list)):
        return False
    return str(v[0]) in nodes


def _node_model_inputs(node, nodes):
    """列出这个节点里所有 MODEL 类型的输入（用来判断它是加载器还是中间节点）"""
    if not isinstance(node, dict):
        return []
    raw = node.get("inputs")
    out = []
    if isinstance(raw, dict):
        for k, v in raw.items():
            if _is_link(v, nodes):
                out.append(k)
    elif isinstance(raw, list):
        for d in raw:
            if isinstance(d, dict) and d.get("type") == "MODEL":
                out.append(d.get("name"))
    return out


def _start_node(prompt, uid):
    """本节点的「模型输入」插槽，连的是哪个节点"""
    nodes = prompt.get("prompt") or prompt
    if not isinstance(nodes, dict):
        return None
    return _node_input_link(prompt, nodes.get(str(uid)), "模型输入")


def _node_input_link(prompt, node, key):
    """取某个节点某个输入口连的上游 → 返回 "节点号" 或 None。
    执行图里 inputs 是 dict（值就是 [来源节点, 槽]）；
    前端保存的画布里 inputs 是 list（要对 links 表查一次）。"""
    if not isinstance(node, dict):
        return None
    ins = node.get("inputs")
    if isinstance(ins, dict):
        v = ins.get(key)
        if isinstance(v, list) and len(v) == 2 and not isinstance(v[1], (dict, list)):
            return str(v[0])
        return None
    if isinstance(ins, list):
        link_id = None
        for d in ins:
            if isinstance(d, dict) and d.get("name") == key:
                link_id = d.get("link")
                break
        if link_id is None:
            return None
        for L in (prompt.get("links") or []):
            if isinstance(L, list) and len(L) >= 5 and L[0] == link_id:
                return str(L[1])
    return None


def _climb_to_loader(prompt, uid, depth=0, seen=None):
    """从 uid 这个节点一路往上游爬，直到找到真正的加载器。
    规则：自己有 MODEL 输入的节点 = 中间节点（LoRA 之类），继续往上游；
          没有 MODEL 输入的节点 = 加载器，就是终点。"""
    if seen is None:
        seen = set()
    if depth > 12 or not isinstance(prompt, dict):
        return None
    nodes = prompt.get("prompt") or prompt
    if not isinstance(nodes, dict) or str(uid) in seen:
        return None
    seen.add(str(uid))
    me = nodes.get(str(uid))
    if not isinstance(me, dict):
        return None

    # 先尝试往上游：只要它自己吃着 MODEL，就一定不是加载器
    for key in _node_model_inputs(me, nodes):
        up_uid = _node_input_link(prompt, me, key)
        if up_uid is not None:
            got = _climb_to_loader(prompt, up_uid, depth + 1, seen)
            if got:
                return got

    ct = me.get("class_type") or me.get("type") or ""
    if ct in LOADER_SLOTS and not _node_model_inputs(me, nodes):
        return (ct, str(uid))
    return None


def _walk_loader_chain(prompt, uid):
    """本节点的「模型输入」→ 找到真正的模型加载器"""
    if not isinstance(prompt, dict):
        return None
    start = _start_node(prompt, uid)
    if start is None:
        return None
    return _climb_to_loader(prompt, start)


def _self_uid(prompt, uid=None):
    """确定"我自己"在图里的节点号。拿不到 uid 就靠特征反查：
    唯一一个带「模型输入」这个口的节点就是自己（比看 properties 可靠 —— 
    CR Text 之类节点的 properties 里也会写 PromptOptimizerSkill）。"""
    if uid is not None and str(uid) not in ("", "None"):
        return str(uid)
    nodes = prompt.get("prompt") if isinstance(prompt, dict) else None
    if not isinstance(nodes, dict):
        nodes = prompt if isinstance(prompt, dict) else {}

    def _has_slot(v):
        if not isinstance(v, dict):
            return False
        ins = v.get("inputs")
        if isinstance(ins, dict):
            return "模型输入" in ins
        if isinstance(ins, list):
            return any(isinstance(d, dict) and d.get("name") == "模型输入" for d in ins)
        return False

    slot_hits = [k for k, v in nodes.items() if _has_slot(v)]
    if len(slot_hits) == 1:
        return str(slot_hits[0])

    hits = [k for k, v in nodes.items()
            if isinstance(v, dict)
            and (v.get("class_type") or v.get("type")) == "PromptOptimizerSkill"]
    if len(hits) == 1:
        return str(hits[0])
    return "2"


SAMPLER_TYPES = ("KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced",
                 "KSamplerSelect", "RandomNoise", "SamplerCustomAdvancedT8")

# 记录每个节点上次见到的「生成后控制」组合，用来判断"是不是刚切过去的"。
# 例：fixed → randomize 的那一次算重抽；之后一直是 randomize 就不再重复花钱。
_LAST_ROLL_MODE = {}


def _collect_autogrow_images(group, extra):
    """把「动态生长」输入里的参考图按序号收成一个列表。

    前端把生长出来的点发过来时，键名形如 `参考图.image_1`（V1 节点不会嵌成字典），
    所以这里用 **kwargs 全收，认这几种形态：
      · 平键：参考图.image_1                  （当前实测形态）
      · dict：{ "image_1": 张量, ... }        （未来若改成嵌套就兜住）
      · 组本身给了单个张量                    （只有一张时）
      · 旧版固定槽 参考图1..参考图N           （老画布迁移期，见下）
      · 旧定义残留下来的 image0                （同上）

    为什么要认旧的：canvas 上换节点定义时，旧连线有可能以旧名字发过来。
    不认的话那些图会被静默丢掉 —— 提示词照写，但不带那张图的信息，很难发现。
    """
    items = []
    if group is not None:
        if isinstance(group, dict):
            items = list(group.items())
        else:
            items = [("image_1", group)]     # 组里直接给了单张
    for k, v in (extra or {}).items():
        if v is None:
            continue
        s = str(k)
        if s == "参考图" or s.startswith("参考图."):
            items.append((s.split(".")[-1], v))
        elif s.startswith("参考图") and s[3:].isdigit():
            items.append((f"image_{s[3:]}", v))          # 旧固定槽 参考图N
        elif s == "image0":
            items.append(("image_1", v))                 # 旧定义残留

    def order(kv):
        tail = kv[0].rsplit("_", 1)[-1]
        return (0, int(tail)) if tail.isdigit() else (1, 0)

    return [v for _, v in sorted(items, key=order) if v is not None]


def _sampler_roll_state(prompt):
    """从整份画布里读出采样器的「生成后控制」设成了什么。

    从 ComfyUI 界面点运行时，前端会把整份画布放在 extra_pnginfo.workflow 里发过来
    （见前端 bundle：extra_data.extra_pnginfo.workflow），里面每个节点的
    widgets_values_named.control_after_generate 就是你截图里那个下拉的值。
    从 API 直接提交时没有这个字段 → 返回 None（那就只认我自己节点上的下拉）。
    """
    if not isinstance(prompt, dict):
        return None
    wf = prompt.get("workflow")
    if not isinstance(wf, dict):
        return None
    nodes = wf.get("nodes")
    if not isinstance(nodes, list):
        return None
    found = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        if (n.get("type") or "") not in SAMPLER_TYPES:
            continue
        if n.get("mode") == 4:          # 禁用的采样器不算
            continue
        v = (n.get("widgets_values_named") or {}).get("control_after_generate")
        if isinstance(v, str) and v.strip():
            found.append(v.strip())
    return found[0] if found else None


def resolve_model_skill(目标模型, prompt, uid=None):
    """返回 (model_key, 来源说明)。只有两条来源，顺序固定：

      1. 连进来的模型加载器（读文件名，只读不执行）
      2. 「目标模型」下拉（手动固定）

    没连线又没选下拉 → 返回 (None, None)，由调用方提示用户，**不猜**。
    （以前这里会扫整张画布找加载器，但双模型工作流里 flux 和 qwen 的加载器
      同时存在，扫到的那个不一定是这次要用的，会静默用错 skill。
      另外曾经还有「模型名称」手填这条中间来源，随手填个字就能盖掉连线的判定，
      属于最难查的一类错 —— 已整个去掉。）
    """
    if not isinstance(prompt, dict):
        prompt = {}
    uid = _self_uid(prompt, uid)

    found = _walk_loader_chain(prompt, uid)
    if found:
        ct, nid = found
        node = (prompt.get("prompt") or prompt).get(nid) or {}
        name = _loader_file(node, ct)
        off = "（注意：这个加载器在图里是「已禁用/静音」状态，可能不是这次真正在用的那个）" \
              if node.get("mode") == 4 else ""
        if name:
            low = name.lower()
            for k, v in MODEL_HINTS:
                if k in low:
                    return v, f'来自连进来的模型加载器 #{nid} {ct}：「{name}」{off}'
            return "flux2-dev", (f'来自连进来的加载器 #{nid} {ct}：「{name}」——'
                                 f"名字里认不出是哪家，先按 flux2-dev 处理{off}")
        return "flux2-dev", f"已连到加载器 #{nid} {ct}，但它里面还没选模型文件{off}"

    if 目标模型 != "自动判断":
        return 目标模型, "来自「目标模型」下拉（手动固定）"

    return None, None


# ------------------------------------------------------------------ 节点 2

FILLER = ["photorealistic", "hyperrealistic", "masterpiece", "8k", "4k",
          "octane render", "award-winning", "highly detailed", "超写实", "高清",
          "大师作品", "极致细节", "超高清"]
NEG_WORDS = ["no ", "not ", "without ", "never ", "none ", "don't ", "不要", "没有",
             "不出现", "移除", "删除", "抛弃"]
WARN = "[!]"      # 不用 emoji —— Windows 控制台 GBK 编码会崩


class PromptSkillCheck:
    """按 skill 规则本地自检（不调 LLM，零成本）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "提示词": ("STRING", {"multiline": True, "default": ""}),
                "目标模型": (["flux2-dev", "qwen-image-2-1"], {"default": "flux2-dev"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("检查报告", "文件路径")
    FUNCTION = "run"
    CATEGORY = "提示词工具"
    OUTPUT_NODE = True

    def run(self, 提示词="", 目标模型="flux2-dev"):
        text = _extract_prompt_text(提示词).strip()
        issues = []

        if not text:
            report = "提示词为空"
            print("[提示词自检] " + report)
            return (report, "")

        # 词数
        en = len(re.findall(r"[A-Za-z']+", text))
        cn = len(re.findall(r"[\u4e00-\u9fff]", text))
        words = en + cn
        if 目标模型 == "flux2-dev":
            if words < 30:
                issues.append(f"偏短（约 {words} 词）—— [dev] 不会自动扩展，考虑加内容或开启提示词上采样")
            elif words > 300:
                issues.append(f"偏长（约 {words} 词）—— 官方长档上限 300，注意是否塞了废话")
            else:
                issues.append(f"长度合适（约 {words} 词，落在 30–300 档）")

        # 填充词
        low = text.lower()
        hit = [w for w in FILLER if w in low]
        if hit:
            issues.append(f"⚠ 含填充词 {hit} —— 官方点名这些会把画面拉向塑料 CGI，建议删掉换成相机/镜头描述")

        # 排除式否定
        # ⚠️ 命令式删除（抛弃/删除/移除 + 对象）不在此列 —— 实测它能被当作编辑
        #    操作执行，是有效写法。真正接不住的是描述成品状态的从句
        #    （"…不进入结果" / "does not carry over"），那种由下面的状态从句检查报。
        _low = re.sub(r'(?:抛弃|删除|移除)(?=[\s\u4e00-\u9fff])', '', low)
        _low = re.sub(r'\b(?:remove|discard)\b', '', _low)
        neg_hit = [w for w in NEG_WORDS if w in _low]
        if neg_hit:
            issues.append(f"⚠ 含排除式否定 {neg_hit} —— [dev] 对否定处理差，"
                          f"建议改成正向描述（保留条款「保持X不变」是合法的，排除式「不要X」才是有问题的）")

        # 光照
        light_kw = ["light", "sun", "lighting", "shadow", "kelvin", "k ", "光", "光源", "照明", "阴影", "阳光"]
        if not any(k in low for k in light_kw):
            issues.append("⚠ 完全没提光照 —— 官方说光照对出图质量影响最大，建议补光源/方向/色温/作用")

        # 色值绑定
        hexes = re.findall(r"#[0-9a-fA-F]{6}", text)
        if hexes:
            issues.append(f"含 {len(hexes)} 个十六进制色值 {hexes[:6]}"
                          f" —— 确认每个都绑定了具体物体（官方实测色相会偏 5–10°，别指望精准）")


        # ── 参考图结构规则（与 skill 的 Exclusive sourcing 一节同源）──
        # 只在"有参考图"时才检查：纯文生图没有归权问题。
        _has_ref = bool(re.search(r'<image\d+>|\bimage\s*\d+\b|图\s*\d+', text))

        # (a) 状态从句：描述成品里"没有什么"——无负向通道，接不住内容，只会重提一遍
        _STATE_CLAUSE = re.compile(
            r'(?:不进入|不参与|不会出现|不带(?:过来)?|不跟(?:过来)?|不会带|不做保留)'
            r'|\b(?:does\s+not\s+carry\s+over|is\s+absent|are\s+absent|'
            r'contributes\s+nothing|nothing\s+else\s+from\s+image)'
            r'|\b(?:is|are)\s+not\s+(?:present|included|used|carried)',
            re.I)
        _hits = _STATE_CLAUSE.findall(text)
        if _hits:
            _sample = re.search(_STATE_CLAUSE, text)
            _ctx = text[max(0, _sample.start() - 18):_sample.end() + 18].replace("\n", " ")
            issues.append(
                f"状态从句 {len(_hits)} 处（如「…{_ctx.strip()}…」）——"
                "FLUX.2 没有负向通道，这类句子只是把源图内容又说了一遍；"
                "删掉它，或改成正向归权声明「画面环境只由图N 提供」")

        # (b) 操作动词：会被当成抠图/粘贴指令，返回拼贴感（边缘硬、光不一致、风格随源图）
        _OP_VERB = re.compile(
            r'提取|抠出|抠图|拼合|重新拼合|粘贴|抠取'
            r'|\b(?:extract|cut\s*out|re-?composite|re-?compose|inpaint|paste)\b', re.I)
        _hits = _OP_VERB.findall(text)
        if _hits:
            issues.append(
                f"操作动词 {len(_hits)} 处（{'、'.join(sorted(set(h if isinstance(h, str) else h[0] for h in _hits))) }）——"
                "这类词让模型按抠图粘贴执行，而不是描述成品帧；"
                "改成直接描述结果：人物已经站在那个空间里、已经受好光、尺度已经对上")

        # (c) 有多张图却没有归权声明
        _imgs = set(re.findall(r'<image(\d+)>|\bimage\s*(\d+)\b|图\s*(\d+)', text))
        _imgs = {next(x for x in m if x) for m in _imgs}
        _OWNER = re.compile(
            r'(?:只来自|只由|唯一来源|仅来自|只提供)\s*图\s*\d+'
            r'|图\s*\d+\s*只(?:提供|负责)'
            r'|\bonly\s+from\s+image\s*\d+|\bimage\s*\d+\s*(?:alone|only)\b'
            r'|\b(?:from|of)\s+image\s*\d+\s*$',
            re.I | re.M)
        # 也接受"元素：图N"这种归权表写法
        _OWNER_TABLE = re.compile(r'[：:]\s*(?:图|<image)\s*\d+|\bimage\s*\d+\b\s*[.。]', re.I)
        if len(_imgs) >= 2 and not (_OWNER.search(text) or _OWNER_TABLE.search(text)):
            issues.append(
                f"提到 {len(_imgs)} 张参考图，但没有任何一句把元素归给唯一的来源——"
                "内容与参考图重合时会被直接拿走。补一句归权声明："
                "「人物与身份 = 图1；环境、材质、构图 = 图2；光源与色温 = 图2」"
                "（英文：Person and identity: image 1. Environment: image 2.）")

        # (d) 语序：参考图出现在最前面的句子里，会被当成"要还原的那张"
        # 只在 >=2 张图时才判断：单图编辑本来就该先说"改图1 的什么"
        _first = re.split(r'[。．.!?！？\n]', text.strip())
        _first = next((s for s in _first if s.strip()), "")
        if len(_imgs) >= 2 and \
           re.search(r'<image\d+>|\bimage\s*\d+\b|图\s*\d+', _first) and \
           not re.search(r'\b(?:discard|remove)\b|抛弃|删除|移除', _first, re.I):
            issues.append(
                "第一句就出现了参考图编号——开头提到的图会被当成"
                "「要还原的那张」，抢走画布。先写目标画面，再写「来自图N」")

        # (e) 多图但缺归权声明（与上面 (c) 同源，但只认"明确的归属句"，更严）
        if len(_imgs) >= 2:
            _OWN_STRICT = re.compile(
                r'(?:只来自|只由|唯一来源|仅来自|只提供|唯一提供)\s*图\s*\d+'
                r'|图\s*\d+\s*只(?:提供|负责)'
                r'|\bonly\s+(?:from|provided by)\s+image\s*\d+'
                r'|\bimage\s*\d+\s+(?:alone|only)\b'
                r'|[：:]\s*(?:图|<image)\s*\d+',
                re.I)
            if not _OWN_STRICT.search(text):
                issues.append(
                    f"提到 {len(_imgs)} 张参考图，却没有一句明确的归属句——"
                    "内容与参考图重合时会被直接拿走。补一句："
                    "「画面的环境只由图2 提供」，或写成归权表"
                    "「人物与身份 = 图1；环境与材质 = 图2」")

        # (f) 画幅 / 镜头 / 光圈缺失：留空 = 模型用自己的审美先验填，指定的背景会被压掉
        _CAM = re.compile(
            r'\b\d{2,4}\s*mm\b|\bf/\s*\d(?:\.\d)?\b|\b(?:full[- ]frame|medium format|'
            r'35mm|50mm|85mm|telephoto|wide[- ]angle)\b'
            r'|\d{3,4}\s*[x×]\s*\d{3,4}|全画幅|中画幅|焦段|镜头|光圈|画幅|广角|长焦',
            re.I)
        if not _CAM.search(text):
            issues.append(
                "没有交代画幅 / 镜头 / 光圈——风格槽留空时，模型会用自己的审美先验把它填满"
                "（常见是影棚或时尚大片感），这是「指定的背景被它自己的风格压掉」的首要原因。"
                "补一句：Full-frame, 35mm, f/2.8, eye-level, fine film grain")

        # 参考图编号
        if re.search(r"<image\d+>", text) and 目标模型 == "flux2-dev":
            issues.append("用了 <imageN> 标签 —— 这是 Qwen 的写法；[dev] 用 image 1 / image 2（英文+数字）")

        # 精修/编辑段的重描述检测
        # 机制：图生图那段不是加滤镜，而是「看着你的图 + 读你的提示词重新生成」。
        # 所以描述 = 要求重画。把「保持不变」的东西又细描述一遍，等于给模型更多
        # 机会重新解释它们 —— 上游那层真实感（毛孔、雀斑、细微不平整）就是这样被
        # 磨平成塑料感的。判断依据不看总长度，只看「保持不变」之后跟了多少细节。
        KEEP_MARK = re.compile(
            r"(保持|维持|保留|沿用|不变|原样|unchanged|keep|preserve|retain|remain|same)", re.I)
        DETAIL_WORD = re.compile(
            r"(皮肤|毛孔|雀斑|质感|材质|纹理|褶皱|光泽|反光|氛围|色调|虚化|景深|构图|"
            r"表情|眼神|发型|妆容|细节|光斑|阴影过渡|"
            r"skin|pore|freckle|texture|material|fold|gloss|reflection|atmosphere|"
            r"tone|palette|bokeh|composition|expression|gaze|hairstyle|detail)", re.I)
        segs = re.split(r"(?<=[。．.!！?？；;\n])", text)
        bad_seg = None
        for s in segs:
            if not KEEP_MARK.search(s):
                continue
            n = len(DETAIL_WORD.findall(s))
            if n >= 3:
                bad_seg = (s.strip()[:46], n)
                break
        if bad_seg:
            issues.append(
                f"⚠ 精修段重描述（「保持不变」那句里跟了 {bad_seg[1]} 个细节词："
                f"“{bad_seg[0]}…”）—— 图生图/编辑那段是**重新生成**，不是刷滤镜："
                f"描述什么就等于要求重画什么，上游的真实感会被磨平。"
                f"改成「保持 A/B/C 不变，只改 D」，或直接用遮罩圈出要改的区域")

        report = f"【{目标模型} 合规自检】\n" + "\n".join("  · " + i for i in issues)
        p = _write(f"check-{slug(目标模型)}-{_ts()}.txt", report + "\n\n--- 原文 ---\n" + text)
        print(f"[提示词自检] {len(issues)} 条 → {p}")
        return (report, p)


# ------------------------------------------------------------------ 注册

NODE_CLASS_MAPPINGS = {
    "PromptOptimizerSkill": PromptOptimizerSkill,
    "PromptSkillCheck": PromptSkillCheck,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptOptimizerSkill": "提示词优化 (带 skill 调 LLM)",
    "PromptSkillCheck": "提示词合规自检 (本地)",
}

# 网页端扩展目录：里面的 js 会被前端自动加载。
# 这里放的是「让画布支持把 .json 工作流文件直接拖进来」的补丁 ——
# 官方前端 1.52.x 的拖入只处理图片（isImageFile 只认 image/*），
# 拖 json 毫无反应，所以用扩展补上。

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
