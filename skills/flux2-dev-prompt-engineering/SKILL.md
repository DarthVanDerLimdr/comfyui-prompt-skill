---
name: flux2-dev-prompt-engineering
description: Use when writing, debugging, or reformulating prompts for FLUX.2 [dev] (Black Forest Labs 32B open-weight model, black-forest-labs/FLUX.2-dev, local ComfyUI or self-hosted inference) for text-to-image, single-reference editing, or multi-reference editing. Also applies to prompt-upsampling decisions, hex color control, in-image text rendering, and multi-reference role assignment. Symptoms include ignored later instructions, short prompts underperforming, hex colors drifting, attribute drift across iterations, unwanted pedestals or platforms, or a full re-render when only a style change was asked for. Adapted for FLUX.2 [dev] by Liang; a few sections adapted from Marco Leder's MIT-licensed flux2 skill (see Provenance & Attribution).
---

# FLUX.2 [dev] Prompt Engineering

## Scope

This skill targets **FLUX.2 [dev]** specifically — the 32B open-weight model. Its prompting behaviour differs from the hosted `[pro]` / `[max]` / `[flex]` variants, and from `[klein]`. Where a rule is [dev]-only, it is marked.

**Do NOT apply these rules to:** Stable Diffusion (negative prompts work, keyword lists work), Midjourney (`--ar`, `--style`, weighting syntax), DALL-E / GPT-Image (verbose narrative preferred), Ideogram, Imagen, Qwen-Image, Seedream. The no-negatives rule and the hex-binding rule are FLUX.2-specific.

## Three Non-Negotiable Gates — apply to EVERY output, unprompted

These are default behaviour, not optional extras the user has to request. Run them against
your draft **before** you output anything. If the draft violates one, fix it first — never
ship it and never wait to be told.

1. **Zero exclusion-style negation — this one is FLUX-specific, not a general rule.** FLUX.2
   has no negative-prompt channel at all, so an exclusion is not merely weak, it is
   unimplementable; it must become a positive description of what occupies that space.
   Any `no / not / without / never / none / 不要 / 没有 / 移除 / 删除 / 抛弃` used to *exclude*
   something gets converted. Procedure and lookup table: **Negation Is Unsupported** below.
   Preservation clauses in edits ("keep the rest unchanged") are the one legitimate
   exception — they are not exclusions. A second one: an **imperative deletion with a generic
   object** (`discard the scene of image 1`, `remove the person in image 2`) is an edit
   operation and works — it is the state-describing clause (`image 1's setting does not carry
   over`) that fails. See *Exclusive sourcing* under Multi-reference.

   Do **not** carry this rule to models that do support negatives. Qwen-Image, for example,
   does support negative prompts, so there an exclusion is implementable and belongs in that
   channel instead of being forced into positive prose. Model capability decides this; the
   output style does not.

2. **Lighting is never left implicit.** Every prompt names a physical source, its direction,
   its temperature, and what it does to the surfaces it hits. If the input says nothing about
   light, **infer the lighting the described scene implies and state it** — do not omit it,
   and never emit a bare intensity word like "soft" without a named source.
   See **Lighting: Highest Return on Effort**.

3. **Ship a reusable template, not a one-off.** Write output so it can be reused across
   future rounds without knowing the specific subject or setting in advance. Never invent
   concrete subjects, species, locations, or scene objects the user did not supply — describe
   the *structure* (roles, relations, attribute slots) and let the user's own material fill
   it. Keep references generic: "the character in image 1", "the environment in image 2".

   **This gate is about objects and places only — it does not license dropping colour,
   material, or lighting.** Those are properties you are *expected* to read off the reference
   or infer from the scene, and they must still be stated concretely: a palette bound to
   specific objects, the materials of the surfaces involved, the full lighting description.
   "Do not invent a species" and "do not invent a hex" are opposite instructions — abstracting
   away the colours is a violation of the colour rule, not a display of restraint.

All three are enforced regardless of length or mode, and they compose — a multi-reference
edit still needs all three.

## The Three Facts That Explain Everything

Every rule below follows from these. If you remember nothing else, remember these.

1. **The text encoder is a 24B Mistral VLM, not T5+CLIP.** It reads sentences, clauses, and conditional constraints. It reads text overlaid on input images. It handles non-English prompts natively. Prompt like you are briefing a photographer, not tagging a database. Keyword soup wastes its capability.

2. **[dev] is guidance-distilled but NOT step-distilled.** Recommended 50 steps at guidance 4.0; 28 steps is the stated quality/speed tradeoff. Carrying over FLUX.1 schnell-era habits (4 steps) produces broken output — a production incident traced garbled in-image text and hallucinated real names to exactly this. Step-distillation LoRAs (e.g. a "Turbo" LoRA at 8 steps) are the legitimate way to go fast.

3. **The VAE is retrained.** FLUX.1 VAEs and FLUX.1 LoRAs are NOT compatible. "Trigger word produces a texture" folklore from FLUX.1 does not transfer.

## The Single Most Important Operational Fact: [dev] Does Not Auto-Upsample

**This is the biggest difference from the hosted variants and the most common cause of disappointing output.**

| Variant | Short prompt behaviour |
|---|---|
| `[pro]` / `[max]` / `[flex]` | **Automatically enhanced** — the service expands short prompts |
| `[klein]` | What you write is what you get |
| **`[dev]`** | **No auto-enhancement, but benefits significantly from explicit prompt upsampling** |

On the official inference repo:

```bash
python scripts/cli.py --upsample_prompt_mode=openrouter   # recommended, stronger expansion
python scripts/cli.py --upsample_prompt_mode=local        # local Mistral-Small-3.2-24B
```

Hosted endpoints of [dev] expose this as a boolean (e.g. `enable_prompt_expansion`, default `false`).

**Rule: if a short prompt underperforms on [dev], suspect the missing upsample step before you suspect the model.**

Upsampling pays off most for: reasoning-heavy prompts, generating creative in-image text, interpreting annotations/arrows in a reference image, and code or math visualisations. It is **not** worth it for simple direct prompts ("a red car").

## Length: Capacity vs. Advice

[dev] accepts up to **32K tokens**. That is capacity, not a target. Official bands:

| Band | Words | Use for |
|---|---|---|
| Short | 10–30 | Quick concepts, iteration, style exploration |
| **Medium** | **30–80** | **Most scenes — the working default** |
| Long | 80–300+ | Complex multi-subject scenes, strongly directed output |

Official warnings, verbatim: **"Start short. Add only what changes the image."** and **"Specific detail helps. Filler hurts."**

**What actually matters is density, not length.** Real-world [dev] prompts from official showcases run 18–30 words and still work — because the words are technical (`Dutch angle`, `desaturated dystopian color palette`, `dramatic harsh sunlight creating deep shadows`) rather than narrative. Narrative padding is the only thing safe to delete.

⚠️ One dissenting claim, treat as unverified: a Chinese community article states prompts over ~50 words cause attention scatter and recommends staying under 30. This **conflicts** with the official 80–300+ band. Unresolved. If you observe degradation, test it rather than assuming either side.

## Structure

Official template — a build aid, **not a mandatory checklist**:

```
[SUBJECT], [LOCATION],
[STYLE], [CAMERA SETTINGS], [LIGHTING], [COLORS], [EFFECT],
[ADDITIONAL ELEMENTS]
```

Slot meanings: **Image type** (portrait / landscape / macro / bird's-eye / abstract) · **Subject** · **Location** · **Style** · **Camera settings** · **Lighting** · **Colors** · **Effect** · **Additional elements**.

**Word order carries priority.** The official example pair is the clearest demonstration:

- Drifts to a wide shot: `Person standing inside a forest fire, strong determined attitude, close-up shot, realistic`
- Stays controlled: `Person with a strong determined expression, forest fire in the background, close-up shot, realistic`

Subject and expression first; environmental detail after. Use this deliberately when framing keeps drifting.

## Lighting: Highest Return on Effort

Official position: lighting has the **single greatest impact** on output quality, and "good lighting" is a non-instruction. Cover five attributes:

| Attribute | Write |
|---|---|
| **Source** | Where the light comes from and what it is: `large softbox key light at 45 degrees`, `golden hour sun from behind-right`, `flickering neon tube`, `warm desk lamp` |
| **Quality** | Soft vs hard, determined by source size: `soft diffused, broad even falloff` / `harsh direct, sharp-edged` |
| **Direction** | `frontal` flattens · `45° key` models form · `90° side` sculpts texture · `backlit` separates · `overhead` dramatises · `from below` is unnatural (fire, screens) |
| **Temperature** | 2000K candle · 2700K tungsten · 3200K warm standard · 4300K mixed · **5600K daylight** · 7500K overcast/blue hour · 10000K+ moonlight. **No more than 2 temperatures per frame**, each with a visible source |
| **Interaction** | What the light does to surfaces: `rim light tracing the hair`, `specular pin-points on wet asphalt`, `bounce from sand lifting the shadow side of the face`, `visible light rays through haze` |

Two further useful heuristics: **key:fill ratio** (`2:1` soft/commercial · `4:1` classic portrait · `8:1+` noir) and the **daylight rule** — sun warm, sky fill cool, so `warm sunlight, cool blue-tinted shadows`. A shadow is not black; it is the colour of the ambient light, so name its tint and name its source.

**"Soft" is a red flag.** It usually conceals a missing physical source. Replace it with the object casting the light and its angle.

**When the input says nothing about light — do not skip it, infer it.** This is the default
path, not an edge case, and it is the most common reason a rewrite underperforms: the user
described subject and scene but never mentioned light, so the draft silently omits the
highest-leverage attribute. Instead:

1. Read what the scene *implies* — an interior with a window means daylight through that
   window; an overcast street means flat 7500K ambient; a candlelit room means 2000K with a
   visible falloff; a studio product shot means a large source at 45°.
2. State the inferred lighting explicitly across all five attributes, chosen to suit the
   scene rather than pasted from a default.
3. If the scene truly implies nothing usable, fall back to neutral daylight: a single large
   source at ~45° from the upper left, 5600K, soft-edged falloff, with shadows tinted by
   ambient rather than black — and say so positively, never as an absence.

Never resolve missing lighting by deleting the requirement, and never leave a note like
"lighting unspecified" in the output.

## Colour: Hex Works, Precision Does Not

Use the keyword `color` or `hex` and **bind every code to a specific object**. Official warning: *vague references like "use #FF0000 somewhere" may produce inconsistent results.*

```
Object + colon:   Apple: #0047AB
Inline:           walls in hex #C4725A / in color #0047AB
Multiple:         #00FF2F #0D00FF #FF0000
Linear gradient:  the vase is a gradient, starting with color #02eb3c and finishing with color #edfa3c
Radial gradient:  radial gradient from rich purple (#6A0DAD) at the center fading to warm gold (#FFD700) at the edges
Multi-stop:       use upper / middle / lowest + "gradually transitioning through"
```

**Measured accuracy — do not promise precision.** Pixel analysis of the official hex demonstration (`Change the colours of the paint on the wall to #F0FF41 and #172DD7`):

| Target | Target HSV | Result | Result HSV | Deviation |
|---|---|---|---|---|
| `#F0FF41` | 64.7°, 74.5%, 100% | `#D2C85D` | 54.9°, 55.7%, 82.4% | **hue −9.8°, sat −18.8** |
| `#172DD7` | 233.1°, 89.3%, 84.3% | `#1F3BA8` | 227.7°, 81.5%, 65.9% | **hue −5.4°, sat −7.8** |

**Pattern: the more unusual the hue, the larger the drift.** A standard deep blue holds; a rare chartreuse gets pulled toward a commoner gold. Lighting and material rendering modulate the result further.

**Rule: hex locks the hue direction and the palette relationship. It does not reproduce brand colour.** For brand-accurate work, colour-correct in post.

**Palette hierarchy (field practice, not official):** assign roughly 60% dominant / 30% secondary / 10% accent, keep the palette to 3–7 distinct colours, never leave the accent slot empty, and allow exactly one most-saturated area per frame — put the accent there. Background elements more saturated than the subject steal attention.

## Text in Images

Three official steps:

1. **Wrap exact text in quotation marks** — `"COFFEE SHOP"`, `"Est. 1952"`. Quotes separate "this is the content to render" from scene description.
2. **State position** — `The text 'OPEN' appears in red neon letters above the door`
3. **Name the font character** — `elegant serif typography`, `bold industrial sans-serif lettering`

Additional working rules: **front-load the text description**; **keep strings short**; hex works for brand colours (`The logo text 'ACME' in color #FF5733`); font semantics — serif = traditional, sans-serif = modern, script = elegant, display = bold.

**Verified capability: long text works.** A validated official example renders a ~25-word sentence with a comma, two exclamation marks, an apostrophe, and a trailing hand-drawn smiley — correct word for word, zero errors, natural 5-line wrap. Do not artificially shorten a sentence that needs to be long; do keep it legible.

For CJK text specifically: **evidence says English outperforms Chinese** in FLUX.2 text rendering (academic OCR benchmark + independent Chinese-language hands-on reports: "English works well, Chinese works but is a bit dumb"). **Route Chinese in-image typography to Qwen-Image or a domestic model instead.** Use [dev] for photorealism, not for Chinese lettering.

## Negation Is Unsupported — Rewrite It as Positive Content

FLUX.2 does not support negative prompts, and the mechanism is why: the model handles negation poorly, so `a person without glasses` makes it attend to "glasses" and render them. Official replacement procedure:

1. **Identify** the unwanted element: `no crowds`
2. **Ask** what would be visible in that space instead
3. **Describe that positive content**: `peaceful solitude`, `empty pathways`

| Instead of | Write |
|---|---|
| no people | empty, deserted, solitary |
| no colours | monochrome, black and white, grayscale |
| no text | clean surfaces, unmarked, blank |
| no modern elements | traditional, historical, period-accurate |
| not dark | brightly lit, sun-drenched |
| not sad | joyful, content |
| not many | few, single, minimal |

If a positive rewrite still fails: add specificity, **front-load the positive description** (order signals priority), add detail, or ground it in environmental context.

**One important distinction:** "keep X unchanged" as a preservation clause in an edit prompt is fine and official (`while keeping the rest of the image unchanged`). What fails is "do not include X" as an exclusion. Preserve-by-negation works; exclude-by-negation does not.

## Editing

**Five rules that hold for every edit, single or multi reference:**

1. **One instruction per pass.** Bundling several changes into one prompt causes *selective*
   execution — the model applies some and silently drops others, and which ones is not
   predictable. Chain separate passes instead.
2. **Do NOT enable prompt upsampling for edits.** Upsampling is for generating a scene from a
   short brief. On an edit it rewrites your carefully stated preservation clauses into new
   descriptive content, which is exactly the thing you were trying not to change.
3. **Do not overuse the word "same".** Heavy repetition of "same" anchors the model to exact
   reproduction of the input and *suppresses the change you asked for*. State what changes,
   then state what is preserved once.
4. **Describe the target state, not the transformation.** Declarative description of the
   finished frame outperforms metalanguage; the pose example under *Single reference* below is
   a verified case of this.
5. **Preservation clause and carry-over clause are different things** (see *Multi-reference*) —
   an edit usually needs both, and neither substitutes for the other.

### Single reference

State what changes, and state what stays as the finished frame's own content. Official anti-patterns, all ineffective: `Make it better`, `Improve the lighting`, `Make it more professional`, `Fix the image`.

Official patterns:

```
Remove all of the sprinkles while keeping the rest of the image unchanged
Replace the cherries in the right-most jar with multi-colored sprinkles. Change nothing else
Change the cow's white fur to the color #8bc4bb and its black spots to #de4528
Change the color of the woman's lace wedding dress to sky blue (light blue, #87CEEB), while keeping all
lace embroidery details white and fully visible. Preserve the original fabric texture, transparency,
patterns, highlights, and natural folds.
```

Short edits work too: `Change it to Night`, `Change this to Winter`, `The butterfly is now made of shiny silver`.

**Pose and expression: describe the target state, not the transformation.** A verified official example drove a complete change — smile, eye contact, head tilt, raised peace sign, hip angle — using only declarative sentences, with **no** "change the pose to..." metalanguage. It also carried the expression and gaze along with the pose. Verified result: facial identity preserved, no degradation.

### Multi-reference

[dev] accepts up to **10 reference images**. ⚠️ NVIDIA's launch post says six; BFL's own documentation says 10 — **trust BFL**, but note the practical ceiling is bounded by VRAM, not by the documented number, since each reference is encoded as its own conditioning latent.

**Always address references by number** and assign each one a role:

```
A photograph of the woman in image 2 sitting on the swing in image 1 and the cat from image 3
sitting on her lap, all in the style of image 4
```

```
Create a house for the chickens from image 1 using materials from images 2, 3, 4, and 5. Use the wood
from image 5 for the base, the materials from images 2 and 4 for the walls and floor, and the material
from image 3 for a small pillow nest.
```

For composites, state per element: which image it comes from, where it goes, and **match scale, lighting, and perspective** explicitly.

**Carry-over is a separate decision — state it every time.** «Take X from image N» does **not**
mean image N's surroundings stay behind. A reference image is a whole frame, and a model told
to use "the character from image 1" will happily drag image 1's background, props, and framing
along with it. Keeping image 2's environment and *removing* image 1's environment are two
different instructions, and the second one is the one that gets forgotten.

So for every "take X from image N", also say explicitly what occupies the space X left behind
in that reference:

```
The frame is the environment of image 2 — its architecture, materials and composition are
image 2's own — and the character of image 1 stands in it at full scale, lit by image 2's
ambient light.
```

```
The vase of image 1 sits on the table of image 2; the table, its room and its furniture are
image 2's own surfaces and everything in the frame belongs to image 2's setting.
```

### Exclusive sourcing: name which reference is the source, never which one is not

This is the failure everyone hits, and the fix is a wording change, not a stronger wording.

Shared content with a reference image gets **doubled**: the text says "X", the latent of image
N also says "X", and no sentence tells the model that this X is built from the description
rather than lifted from image N. FLUX.2 resolves it by taking image N's X. The parts of your
description that overlap a reference are exactly the parts it hands over to that reference.

**Two kinds of negation, and they behave differently.** This is a load-bearing distinction,
because the obvious rule ("no negations at all") is wrong and costs you a working tool.

| | Example | What the model does | Verdict |
|---|---|---|---|
| **Imperative deletion** — a verb with an object | `discard the scene of image 1`, `remove the person in image 2`, `抛弃图1的场景`, `删除图2的人物` | Treats it as an edit operation and **performs it**. This is the shape of the edit instructions the model was trained on | **Keep it.** Field-tested as reliable for dropping the source scene |
| **State-describing clause** — a description of the output | `the setting of image 1 does not carry over`, `image 1's background is absent`, `nothing else from image 3 appears` | Builds a representation of the named content, then has nothing to act on. The content stays and gets a second, image-side path — the reference latent | **Delete it.** It does no work and re-names the content |

So the rule is not "no negations" — it is **no negations that describe the output**. An
imperative that names an object and a deletion is doing work; a clause that describes what the
finished frame lacks is not.

| Do not write (state clause) | Write instead |
|---|---|
| "the setting of image 1 does not carry over" | "the frame's setting is image 2's own" |
| "image 1's background is absent" | "environment: image 2" |
| "nothing else from image 3 appears" | "texture: image 3" |
| "only image 2's setting is present" | "the frame is image 2's setting" |

**Use the imperative *and* the ownership sentence.** They cover different failure paths and the
imperative alone is the one that leaks: it reliably removes the source scene, but if the
content you dropped also matches something your text describes, the reference latent pulls it
back. So pair them — command first, owner second:

```
Discard the scene of image 1; the frame's environment is image 2's own.
The person of image 1 stands in it at full scale, lit by image 2's ambient light.
```

That is the whole fix for this failure mode: the command does the work, and the ownership
sentence closes the way back.

**The pattern: `[element] is the one in image N`.** One positive sentence per element, and the
element is thereby given a single owner. If an element is not listed, it is not claimed —
and because silence is read as permission, everything the reference contains that you do not
assign elsewhere becomes free for the model to reuse. A workable prompt therefore reads as a
short authority table, not a list of prohibitions:

```
Person and facial identity: image 1.            Environment, materials, composition: image 2.
Key direction and colour temperature: image 2.  Figure and pose: image 1.
```

That table is the whole trick. It costs about twenty words and replaces every suppression
clause that was pulling the model the other way.

**Write the target frame first, and put every "from image N" after it.** Word order signals
priority, and a reference image mentioned in the opening sentence is read as the thing being
reproduced. Establish the result, then slot the sources into it:

- Weak: `Take the person from image 1 and put them in image 2.`
- Strong: `An empty interior of image 2's kind. The person of image 1 stands in it at full scale.`

**Never write `extract` / `cut out` / `re-composite` / `inpaint`** unless the task really is a
cut-out. FLUX.2 reads those as operation words and returns a paste-up: cut edges, mismatched
light, and the source image's own rendering style. Describe the finished frame instead — a
person standing in a space, already lit and already at the right scale.

**Never name the source image's contents — including inside an imperative.** `抛弃图1的场景`
is safe because "场景" is generic; `抛弃图1的天空、云朵和草地` is not, because you will
assign an element to the wrong image — models routinely write "the sky, clouds and hill of
image 1 do not appear" when those elements are actually in image 2. The command now points at
the wrong picture, and the real content goes ungoverned. Keep the object of a deletion
**generic** (`the scene of image 1`, `the background of image 1`), or skip the imperative and
just let the ownership sentence place everything.

Do not confuse this with a preservation clause. "The rest of image 2 is unchanged" preserves;
"image 1's setting does not carry over" is a state clause. A multi-reference composite usually
needs both a preservation clause and an ownership table, and they are not interchangeable.

### Attempting geometric change — do not

**"Make the cards 30% larger", "move the subject 10% left", "crop tighter" have no meaning in
diffusion space.** The model reasons over semantics, not pixels; a percentage change to a
layout is not something it can measure or honour. Asking for one produces a *different image*
that happens to resemble the request, not the same image shifted.

The reliable route is on the pixel side, before or after generation — crop inward and upscale:

```python
from PIL import Image

img = Image.open("input.png")
w, h = img.size
pct = 15                      # 10 = subtle · 15 = moderate · 20 = aggressive
cx, cy = int(w * pct / 100), int(h * pct / 100)
cropped = img.crop((cx, cy, w - cx, h - cy))          # tighter framing
result = cropped.resize((2048, 1324), Image.LANCZOS)   # then resample up
result.save("output.png")
```

So when a user asks to "make X bigger", split the request: if they want **more of the frame
devoted to X**, crop and upscale. If they want **X re-rendered at a larger apparent size in a
new composition**, that is a compositional instruction and belongs in the prompt as a
description of the target frame.

### Engineering note

In chained edit workflows, quality degrades after a few frames if the sampler's target latent size and the reference latent size differ. **Decide the frame size once and hold it constant for the whole chain.**

## JSON Structured Prompting

Officially supported. Base schema:

```json
{
  "scene": "",
  "subjects": [ { "description": "", "position": "", "action": "" } ],
  "style": "",
  "color_palette": ["#hex1", "#hex2"],
  "lighting": "",
  "mood": "",
  "background": "",
  "composition": "",
  "camera": { "angle": "", "lens": "", "depth_of_field": "" }
}
```

Official position, verbatim: the model **"understands both formats equally well."** JSON can be passed directly or flattened to prose.

**Be honest with the user about what JSON buys.** Official documentation never claims JSON is more accurate. Its value is:
- consistent structure across a production pipeline
- programmatic/automated generation
- complex multi-subject scenes with explicit relationships
- iterating one element at a time while holding everything else fixed

Natural language is better for: fast exploration, single-subject scenes, and creative flexibility.

**An open question worth flagging to the user:** a public prompt-adherence benchmark measured FLUX.2 at 5/15 against proprietary models — but it **tested prose prompts only**, and the author himself raised the untested possibility that JSON could improve accuracy. Nobody has resolved this. If a user is investing in a JSON pipeline, that is the highest-value experiment available to them.

⚠️ One implementation caveat: some API layers (notably Replicate) double-escape embedded JSON, corrupting it. If JSON arrives mangled, flatten to prose.

## Parameters and Hardware

| Source | Steps | Guidance |
|---|---|---|
| BFL official recommendation | 50 | 4.0 |
| BFL stated tradeoff | 28 | 4.0 |
| Official ComfyUI template default | 20 | 4 |
| Official ComfyUI Turbo LoRA mode | 8 | 4 |
| A hosted endpoint's default | 28 | 2.5 |

**Hardware reality — state the right number.** The binding constraint is often **system RAM, not VRAM**: quantising requires holding the transformer *and* the 24B text encoder in memory simultaneously, roughly **90GB of RAM**. A team that tried 2×RTX 4070 Ti SUPER, then two RTX 5090s, and then an A100, reported the 24B model "would not load a single layer even at the smallest resolution with everything offloaded to CPU."

Reference figures: NVIDIA documents **90GB VRAM** for a full load, **64GB** in lowVRAM mode, and **~40% reduction** with FP8. A single 96GB card still hit CUDA OOM on the BF16 text encoder — because the text encoder is a full LLM in its own right. Weight streaming from system RAM (the ComfyUI RAM-offload path NVIDIA helped optimise) is what makes consumer cards viable, at a speed cost.

**Do not conflate these two claims**, which are constantly mixed up online:
- "A 24GB card can run a 4-bit quant" → single-image inference, heavy offload, slow
- "Two 5090s failed" → LoRA **training**, which needs both models resident

Tokens-per-second and trainability are entirely different questions.

### Hosted variants — pick the right one

Most of this skill is about local `[dev]`, but the same prompting rules transfer to the hosted
endpoints. Which variant you are calling changes one behaviour that matters: **whether the
service expands your short prompt for you** (see *[dev] Does Not Auto-Upsample* above).

| Variant | Best for | Reference images | Auto-expands short prompts? |
|---|---|---|---|
| `[max]` | Highest fidelity, complex edits, character consistency | up to 8 | yes |
| `[pro]` | Production at scale, quality/cost balance | up to 8 | yes |
| `[flex]` | Typography and fine-grained control (steps, guidance) | up to 10 | yes |
| `[klein]` | Sub-second generation, real-time apps | up to 5 | no |
| `[dev]` | Local / self-hosted, open weights | up to 10 (VRAM-bound) | **no** — upsample explicitly |

Practical consequences:

- On `[pro]` / `[max]` / `[flex]`, a terse brief is acceptable because the service rewrites it.
  On `[dev]` and `[klein]`, what you write is what you get — write the full description.
- Reference counts are **documentation, not capability**. The real ceiling is conditioning
  latents vs available VRAM.

### Hosted API parameters worth knowing

```json
{
  "prompt": "...",
  "width": 2048, "height": 1324,
  "output_format": "png", "output_quality": 80,
  "safety_tolerance": 2,
  "prompt_upsampling": true
}
```

- **`input_images`** — omit the field entirely for text-to-image. Do **not** send an empty
  array; several wrappers reject it.
- **`safety_tolerance`** — 0–6, default 2. It filters output, it does not affect layout or
  quality. Raising it is not a quality knob.
- **`prompt_upsampling`** — a boolean on hosted `[dev]` endpoints (sometimes named
  `enable_prompt_expansion`), default **`false`**. Turn it on for short briefs, off for edits.
- **⚠️ Replicate-specific trap: do not send the JSON prompt *as* JSON inside the payload.**
  Inner JSON gets double-escaped and arrives corrupted. Flatten structured prompts to natural
  language prose before sending. (The JSON schema documented below is for the official
  inference stack and BFL's own API, where this does not apply.)

## Failure Modes to Pre-empt

These are field- and image-verified. Warn the user before they hit them.

| Failure | What happens | Prevention |
|---|---|---|
| **Hands and anatomy** | Multiple independent users report anatomy **worse than FLUX.1 [dev]**: wrong finger counts, distorted limbs. One user noted the official showcase images "very carefully avoid hands" — and a check of four official showcase images confirmed **4 of 4 avoid or obscure hands**, in prompts that never asked to exclude them. | Do not promise reliable hands. Budget extra generations for hand-forward compositions. Prefer framing that does not put hands in the foreground. |
| **Unrequested pedestals / platforms** | In the official 7-step iteration demo, a pedestal appears under the subject from step 5 and escalates (cobblestone → cylindrical plinth → metal base), **never once mentioned in any prompt**. It coincided with adding a dog. The composition shifted from environmental portrait to displayed figurine. | After adding a prop or animal, re-inspect the whole frame. Explicitly describe the ground plane and footing. |
| **Attribute drift across iterations** | In that same demo the coat shortened from ankle-length to mid-thigh, and in one step shifted from `charcoal` to visibly blue — neither requested. | **Re-state must-keep attributes in every iteration.** Do not declare them once at the start and assume they persist. |
| **Style words re-render everything** | Adding `watercolor illustration style` produced a genuine full medium transfer (paper texture, wet-in-wet blooms, gouache-like hard edges) — but also silently changed the subject's pose, hair, and expression, and introduced equal-spaced repetition artefacts in a row of cars. | Treat a style addition as a fresh generation, not a filter. To preserve composition, lock the frame first then restyle, and re-verify every detail. |
| **Filler words** | `photorealistic, hyperrealistic, 8k, Octane Render, masterpiece` pull output toward plastic CGI. | Delete them. Describe camera, lens, lighting, and imperfections instead. |
| **Keyword-soup prompting** | Migrating SD-era habits (tag stacks, weights, negative prompts) underperforms. | Write structured prose. |

## Domain Tactics (Field-Verified, Product/Flatlay Work)

Useful in product, flatlay, and still-life work. **Domain tactics — not universal laws.**

- **Named positions per item.** Abstract layout words ("scattered", "casual arrangement") let the model default to a grid. Give every item a position: upper left, far lower right corner, center-right.
- **Per-item angles, alternating.** Replace an angle *range* with specific degrees per item, and make some explicitly straight.
- **Exact counts, not ranges.** `six items`, never `5–6 items`. Front-load the count — it anchors composition.
- **Material terminology.** "Concrete" yields heavily pitted surfaces; use the real interior finish term (`micro-cement`) when a smooth modern surface is wanted.
- **Camera model + lens + aperture.** `Hasselblad X2D 80mm f/5.6` outperforms "medium format camera" or "photorealistic". Focal length controls apparent crop: 14–24mm wide/dramatic · 35–50mm natural · 70–85mm portrait/product · 100mm+ tight telephoto.
- **Geometric resizing does not work in editing.** "Make cards 30% larger" is meaningless in latent space. Crop inward 10–20% and upscale (LANCZOS) instead.

## Reformulation Workflow

Given a verbose, unstructured, or underperforming prompt:

1. **Decide the target band.** 30–80 words for most scenes; only go long for genuinely complex multi-subject work.
2. **Delete filler.** Remove preamble, aesthetic self-description, hedges, redundant modifiers, and generic quality words. Keep only words that change pixels.
3. **Convert every negation.** Find `no`, `not`, `without`, `never`, `none` used as exclusions and rewrite as positive description. **Except** preservation clauses in edits ("keep the rest unchanged"), which are legitimate.
4. **Reorder to priority.** Subject and expression first, environment after, technical direction last.
5. **Fix the lighting.** Name a physical source, its direction, its temperature, and what it does to surfaces. Delete "soft" unless a source is named.
6. **Bind colours.** Each hex gets an object. Reduce to 3–7 distinct colours with a 60/30/10 hierarchy.
7. **Quote any in-image text**, state placement, font character, and size.
8. **If it is an edit:** state the change, then explicitly enumerate what must be preserved. One instruction set per pass. Do not enable upsampling on edits.
9. **If it is multi-reference:** number every reference and assign each a role; state scale/lighting/perspective matching for composites.
10. **Decide on upsampling.** [dev] does not do it for you. Enable for reasoning-heavy or short T2I prompts; skip for simple prompts and for edits.
11. **Re-read for attribute risk.** Anything that must survive iteration should be stated explicitly and repeated next round.
12. **Walk the Pre-Flight Checklist below against the finished draft, line by line, and fix
    whatever fails.** This step is mandatory and comes before outputting — not after, and not
    only when the user asks for a thorough pass. Steps 3 and 5 in particular are the two that
    get silently skipped, which is exactly why the checklist exists as a separate gate.

## Pre-Flight Checklist

**This is a gate, not a reminder.** After drafting (and before outputting), walk every line
below against the draft text. For each unchecked line, fix the draft and re-walk it. Do not
output a draft you have not walked. If a line genuinely does not apply (e.g. no hex colours
are in play), that is fine — but say so to yourself explicitly rather than skipping silently.

- [ ] Word band chosen deliberately (30–80 default; longer only with justification)
- [ ] No filler: no `photorealistic / 8k / masterpiece / hyperrealistic`
- [ ] **Zero exclusion-style negation** — search the draft for `no / not / without / never /
      none / 不要 / 没有 / 移除 / 删除 / 抛弃`. Every hit is either converted to a positive
      description, or is a preservation clause in an edit (the only allowed exception)
- [ ] **Lighting present even if the input never mentioned it** — source + direction +
      temperature + surface interaction, inferred from the scene when necessary
- [ ] Subject and expression precede environmental detail
- [ ] No bare "soft" lighting without a named physical source
- [ ] Every hex bound to a specific object; palette within 3–7 colours
- [ ] In-image text in quotation marks, with placement and font character
- [ ] Camera specified as body + lens + aperture when photorealism matters
- [ ] Exact counts, not ranges
- [ ] Editing: change stated, preservation explicitly enumerated, one pass only
- [ ] Multi-reference: every reference numbered and given a role
- [ ] **Every reference given an exclusive source, positively** — for every "take X from
      image N", the prompt also says which reference owns the rest ("the frame is image 2's
      setting"). Never left silent; silence reads as permission
- [ ] **Zero state-describing negations** — scan for "image N does not / is absent / nothing
      else from image N" and delete it. An imperative deletion (`discard the scene of image 1`)
      is allowed and works; a clause describing what the output lacks does not. If a state
      clause seems to be doing real work, its content was never assigned an owner: go back and
      assign one instead of re-strengthening the clause
- [ ] **No operation verbs** — no `extract` / `cut out` / `re-composite` / `inpaint`; the
      finished frame is described, not the procedure
- [ ] **Style, lens and frame declared** — an empty style slot is filled by the model's own
      photographic prior, which is the usual reason a specified environment does not survive
- [ ] **Reusable as a template** — no invented subjects/species/locations/scene objects;
      references stay generic ("the character in image 1")
- [ ] Upsampling decision made explicitly (it is NOT automatic on [dev])
- [ ] Steps/guidance sane: 28–50 steps at guidance ~4.0 (or a Turbo LoRA at 8)
- [ ] Hand-forward compositions flagged as high-variance
- [ ] Must-keep attributes noted for repetition in the next iteration

## Worked Example

**Before** (~150 words, SD-era habits, multiple violations):

> A top-down flatlay photograph of a designer's moodboard on a dark charcoal concrete wall with fine porous texture. Warm directional sunlight enters from upper left, filtered through horizontal window blinds, casting soft diagonal bands across the scene. A monstera deliciosa leaf casts its organic shadow in the upper-left corner. Scattered across the wall are 5–6 matte paper elements, each rotated at subtle casual angles (2–8°): several blank Polaroid frames in varying sizes with classic thick white borders. The photo areas contain no image, just soft gray gradients. Two Pantone swatch cards, one anthracite/dark olive, one with two blocks of dark charcoal and cool silver-gray. One cream-colored business card reading "MARCO LEDER / CREATIVE DIRECTOR / STUDIO" in minimal sans-serif. Color palette is muted warm neutrals. No bright or saturated colors. Shot with a medium format camera. Photorealistic.

**Violations:** ~150 words of mostly narrative · two exclusion negations · range count (`5–6`) · range angle (`2–8°`) · abstract layout (`scattered`) · no hex binding · abstract camera · "soft" with no named source · "concrete" for a smooth surface · generic quality word.

**After** (~95 words, [dev]-conformant):

> Top-down flat lay, six items on smooth matte dark gray micro-cement surface hex #363636. Upper left: tall Pantone card, white stock, solid #B8B8B8 top block and anthracite #6B5F58 bottom block, washi tape, straight. Upper right: large straight polaroid, white border, solid #A8B0B8 interior. Center: medium polaroid tilted 8 degrees, solid #CCCCCC interior. Lower left: Pantone card, charcoal #3A3A3A and steel gray #8A9199 blocks, text 'PANTONE ANTHRACITE' in small sans-serif, tilted 12 degrees. Lower right: large straight polaroid, solid #B0B0B0 interior. Bottom center: off-white #E8E4DF business card, text 'MARCO LEDER' and 'CREATIVE DIRECTOR' in small sans-serif, tilted 5 degrees. Monstera leaf shadow diagonally across upper left, venetian blind stripes at 30 degrees across surface, warm afternoon light from upper left 3200K, warm shadows. Hasselblad X2D 80mm f/5.6, dark minimalist editorial. 2048x1324.

**What changed:** narrative padding removed · negations converted to positive description (`no image` → `solid #B8B8B8 interior`) · range count → exact count, front-loaded · angle range → per-item degrees, alternating with straight · every colour bound to an object as hex · physical shadow source named **with angle** · light given temperature and a warm-shadow statement · camera made specific · surface term corrected · word count brought into band.

**Then:** enable upsampling for this prompt (`--upsample_prompt_mode=openrouter`), because it is a dense short prompt for text-to-image on [dev]. Upsampling supplies material and atmospheric richness the compressed structure omits.

## Common Mistakes

| Symptom | Cause | Fix |
|---|---|---|
| Short prompt looks weak on [dev] | Upsampling not enabled — [dev] never auto-enhances | Enable upsampling, or lengthen the prompt yourself |
| Later instructions silently ignored | Prompt padded with narrative; priority buried | Cut filler, move the important content earlier |
| Grid or catalogue layout | Abstract layout words | Named position per item |
| Flat, lifeless lighting | "soft light" / "warm sunlight" with no source | Name the physical source, direction, temperature, and surface interaction |
| Colours drift from brand spec | Hex is a direction, not a measurement (measured −5° to −10° hue) | Accept drift or colour-correct in post |
| Garbled in-image text, real names appearing | Steps too low (schnell-era value) or raw prose dumped into the prompt | 28–50 steps; write directed instructions |
| Unwanted platform under the subject | Model hallucinated it after new elements were added | Describe the ground plane; re-inspect after each addition |
| Previously correct attribute changed | Attribute drift across iterations | Re-state must-keep attributes every round |
| Whole image changed when only style was requested | Style words trigger re-rendering | Lock composition first, or accept and re-verify |
| Broken hands | Known [dev] weakness | Reframe, or budget extra generations |
| Chinese lettering unreliable | English outperforms Chinese in text rendering | Route Chinese typography to Qwen-Image |
| JSON arriving corrupted | API layer double-escaping | Flatten to prose |
| Chained edits fading | Reference and sampler latent sizes diverged | Fix frame size once, hold it |

## Do Not Overclaim

**Everything in this section is meta-information about the model — it is never prompt text.**
Nothing here may be copied, paraphrased, or echoed into the output prompt. It exists only to
calibrate how you *talk about* [dev] when advising a user.

When advising a user, be precise about what is established versus claimed:

- **Established:** the model has no negative-prompt channel (so exclusions must be rewritten
  as positive content) · hex values bind to objects · quoted strings render as text · up to 10
  references, addressed by number · [dev] does not auto-upsample · 28–50 steps at guidance ~4 ·
  editing preserves unrequested content well · long in-image text renders accurately ·
  description-only pose/expression control works · identity persists across scene changes in
  text-only generation
- **Measured with caveats:** hex accuracy drifts by hue rarity · hands are a weak point
- **Field practice, not official:** 60/30/10 palette hierarchy · key:fill ratios · Kelvin values · SQTSI decomposition · "micro-cement" terminology · per-item angle alternation
- **Unresolved:** whether JSON beats prose for accuracy · whether long prompts genuinely degrade attention on [dev] · the true stable multi-reference count under limited VRAM

---

## Provenance & Attribution

This skill was written for FLUX.2 **[dev]** by **Liang** (DarthVanDerLimdr), working
with an AI assistant, and then edited and fact-checked against official sources. It is
distributed as part of the `comfyui-prompt-skill` project.

**A few sections are adapted from an MIT-licensed work** — the *Hosted variants* table,
the hosted API parameters and the Replicate JSON double-escape warning, the crop-and-upscale
(LANCZOS) note under *Attempting geometric change*, and the `safety_tolerance` /
`prompt_upsampling` descriptions. Those derive from:

> **flux2-prompt-engineering** by Marco Leder — https://github.com/marcoleder/claude-plugins
>
> MIT License — Copyright (c) 2026 Marco Leder
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this
> software and associated documentation files (the "Software"), to deal in the Software
> without restriction, including without limitation the rights to use, copy, modify,
> merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject to the following
> conditions: The above copyright notice and this permission notice shall be included in
> all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
> INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
> PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
> FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
> OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

**Rule-level facts** (band widths, Kelvin values, step/guidance pairs, hardware figures,
in-image text behaviour) are cross-checked against Black Forest Labs' official
documentation and repos. Where a claim is unverified or contested it is marked as such in
the text — please do not cite unmarked passages as official.

Everything else is original to this project, released under the MIT License
(see `LICENSE` in the repository root).
