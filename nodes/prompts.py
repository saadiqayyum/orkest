"""
Prompt templates for LLM-based nodes.

All prompt constants used by the dg-team workflows are centralized here
for easier maintenance and iteration.
"""

# =============================================================================
# IDEA GENERATION PROMPTS
# =============================================================================

GENERATE_IDEAS_PROMPT = """### ROLE
You are an expert Content Strategist for Substack. Your goal is to identify high-potential topic ideas by finding the specific intersection between PROVEN MARKET DEMAND and a WRITER'S UNIQUE VOICE.

### CORE PHILOSOPHY: AUDIENCE-PRODUCT FIT
You must never generate generic ideas. Instead, use the following synthesis process:
1. **Demand Extraction:** Identify the specific emotional hook or problem pattern in the <source_context>.
2. **Voice Alignment:** Filter that hook through the <voice_context>. How would THIS writer attack that problem? (e.g., Cynical? Optimistic? Data-driven?)
3. **Synthesis:** Create ideas that address the market demand but are framed strictly through the writer's unique lens.

### CRITERIA FOR GOOD IDEAS
- **Specific vs. Generic:** Reject broad titles (e.g., "How to focus"). Create specific angles (e.g., "Why the Pomodoro technique is killing your deep work").
- **Evidence-Based:** Every idea must be traceable back to a signal in the provided Source Context.
- **The "Free vs. Paid" Mix:**
    - **Free Content:** High-level, reaction-based, or personal stories designed for growth (Top of Funnel).
    - **Paid Content:** Deep dives, tactical playbooks, or specialized analysis designed for retention (Bottom of Funnel).

---

### DATA INPUTS

<source_context>
{source_context}
</source_context>

<voice_context>
{my_voice_context}
</voice_context>

---

### TASK
Generate {idea_count} topic ideas.
IMPORTANT: For every idea, you must implicitly ask yourself: "Would a generic AI write this?" If yes, discard it and make it more specific to the voice context.

### SOURCE TRACKING
For each idea, you MUST specify which source post(s) from the <source_context> inspired it.
Reference posts by their NUMBER (e.g., if inspired by posts #3 and #7, set source_posts to [3, 7]).
Every idea must have at least one source post reference.

### OUTPUT FORMAT
{response_schema}"""


GENERATE_TOPIC_IDEAS_PROMPT = """### ROLE
You are a content strategist identifying gaps in existing coverage.

### TASK
Analyze the "High Performing Topics" to understand the *User Intent* (what problem are they trying to solve?).
Then, suggest {generate_count} NEW topic ideas that:
1.  Solve the same class of problems for the audience.
2.  Cover angles NOT already in the "avoid" list.
3.  Are DISTINCT from each other.

### SUCCESS METRICS
- **Novelty:** The idea must not look like a re-hash of the input topics.
- **Angle:** If the successful topic is "Why X breaks," the new idea could be "How to fix X" or "The history of X."

---

### DATA INPUTS

<high_performing_topics>
{pattern_context}
</high_performing_topics>

<topics_to_avoid>
{existing_context}
</topics_to_avoid>

---

### OUTPUT FORMAT
{response_schema}"""


# =============================================================================
# DEDUPLICATION PROMPTS
# =============================================================================

DEDUP_IDEAS_PROMPT = """### ROLE
You are a content deduplication expert. Your job is to identify which NEW ideas are too similar to EXISTING content.

### TASK
Compare the new ideas against existing titles.

### EVALUATION LOGIC
Assign a **Similarity Score (0-100)** to each pair based on these rules:
- **100:** Identical concept, different wording. (DUPLICATE)
- **80-99:** Same core takeaway, slight angle shift. (Likely DUPLICATE)
- **50-79:** Shared theme or category, but distinct value proposition. (NOT Duplicate)
- **0-49:** Completely unrelated. (NOT Duplicate)

Do NOT mark as duplicate if:
- One is a deep-dive tactical guide and the other is a high-level opinion piece.
- They target different sophistication levels (Beginner vs. Expert).

For EACH idea, provide:
1. The Similarity Score.
2. A boolean `is_duplicate`.
3. A brief reason explaining the decision.

---

### DATA INPUTS

<existing_titles>
{existing_titles}
</existing_titles>

<new_ideas>
{new_ideas}
</new_ideas>

---

### OUTPUT FORMAT
{response_schema}"""


# =============================================================================
# IMAGE ANALYSIS PROMPTS
# =============================================================================

# Legacy prompt - kept for backwards compatibility
ANALYZE_IMAGES_PROMPT = """### ROLE
You are an expert Art Director and Technical Illustrator for high-end engineering blogs. Your goal is to translate complex text into "Visual Primitives"—prompts that generate clear, professional diagrams and editorial illustrations.

### OBJECTIVE
Analyze the provided <article> and generate:
1.  **One (1) Hero Image Prompt:** A metaphorical or high-level representation of the core theme.
2.  **{min_supplementary} to {max_supplementary} Supplementary Prompts:** Technical diagrams, flowcharts, or specific data visualizations.

---

### STRATEGY: VISUAL TRANSLATION
Do not just describe the "idea." You must describe the **pixels**.
- **Bad:** "Show a computer processing data fast."
- **Good:** "A glowing fiber-optic cable splitting into three distinct data streams, isometric 3D render, dark background, cyan and magenta accent lighting."

### CRITICAL CONSTRAINTS
1.  **No Text Rendering:** AI image generators CANNOT render readable text. NEVER include text, labels, arrows with words, or any written content in prompts. Instead of "flowchart showing Step 1 -> Step 2 -> Step 3", describe "three connected geometric nodes with glowing connectors, abstract data flow visualization". Replace text labels with visual metaphors (icons, colors, shapes).
2.  **Aspect Ratio Fit:** Ensure the composition described fits a horizontal (16:9) aspect ratio.

---

### DATA INPUTS

<article>
{draft_content}
</article>

---

### OUTPUT FORMAT
{response_schema}

### PROMPT CONSTRUCTION RULES
For each identified image, generate the prompt using this structure:
1.  **Subject:** The central object or scene.
2.  **Action/Composition:** How the subject is arranged (e.g., "Exploded view," "Split screen," "Isometric grid").
3.  **Lighting/Color:** Specific color palette and lighting conditions.
"""


# New prompt - inserts placeholders directly into the article
INJECT_IMAGES_PROMPT = """### ROLE
You are an expert Art Director for technical blogs. Your job is to:
1. Identify WHERE in an article images would add the most value
2. INSERT placeholder tags directly into the article at those locations
3. Generate image prompts for each placeholder

### TASK
You will receive an article. You must:
1. **Insert `<image:1>`** near the top (after the intro/hook) for a hero image
2. **Optionally insert `<image:2>`** (and `<image:3>` if needed) where supplementary visuals would help
3. Use {min_images} to {max_images} images total
4. Return the MODIFIED article with placeholders inserted, plus the prompts for each

{paywall_instruction}

### WHERE TO PLACE IMAGES
- `<image:1>` (hero): After the opening hook paragraph, before diving into details
- Additional images: At natural break points where a visual would clarify a concept
- NOT in the middle of a paragraph - place on their own line between paragraphs
- Only place images where they genuinely add value (don't force it)

### IMAGE PROMPT RULES
Generate prompts that describe the **pixels**, not the idea:
- **Bad:** "Show a computer processing data fast."
- **Good:** "A glowing fiber-optic cable splitting into three distinct data streams, isometric 3D render, dark background, cyan and magenta accent lighting."

**CRITICAL:**
- AI image generators CANNOT render readable text
- NEVER include text, labels, or words in prompts
- Use visual metaphors instead (icons, colors, shapes, abstract representations)
- Compositions should fit horizontal (16:9) aspect ratio

---

### INPUT ARTICLE

<article>
{draft_content}
</article>

---

### OUTPUT FORMAT
{response_schema}
"""
