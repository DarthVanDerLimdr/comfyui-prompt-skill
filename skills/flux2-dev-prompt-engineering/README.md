# flux2-dev-prompt-engineering

Prompt-engineering spec for **FLUX.2 [dev]** (Black Forest Labs' 32B open-weight model,
local ComfyUI / self-hosted). Used by the **提示词优化 (带 skill 调 LLM)** node in the
parent repository.

## Install

The node looks for a folder named **exactly** `flux2-dev-prompt-engineering` in one of:

1. `ComfyUI/output/workflow_llm/skills/`
2. `~/.agents/skills/`  ← usual place
3. `~/Documents/agent-skills/`
4. `~/.claude/skills/`
5. `~/.dsh/skills/`

So: copy this folder into `~/.agents/skills/` — the file `SKILL.md` must sit directly
inside it. No restart needed; the spec is re-read on every run.

```bash
mkdir -p ~/.agents/skills
cp -r skills/flux2-dev-prompt-engineering ~/.agents/skills/
```

Windows (PowerShell):

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.agents\skills" | Out-Null
Copy-Item -Recurse "skills\flux2-dev-prompt-engineering" "$env:USERPROFILE\.agents\skills\"
```

## Using it without ComfyUI

It is plain Markdown — paste it into any chat assistant as context, or read it as a
reference for writing `[dev]` prompts by hand.

## Provenance

Written for FLUX.2 [dev] by Liang with an AI assistant, then edited and fact-checked
against official sources. A few sections are adapted from Marco Leder's MIT-licensed
`flux2-prompt-engineering` — see the *Provenance & Attribution* section at the end of
`SKILL.md` for the full notice.

MIT License.
