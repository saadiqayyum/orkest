"""
Utility functions for DG-Team workflows.

These are small helper functions that don't fit in other modules.
"""
from typing import Optional
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger()


# =============================================================================
# INPUT/OUTPUT SCHEMAS
# =============================================================================

class CalculateWordCountInput(BaseModel):
    """Input for calculate_target_word_count function."""
    length_mode: str = Field(
        default="standard",
        description="Length mode: brief, standard, long, or custom"
    )
    custom_word_count: Optional[int] = Field(
        default=None,
        description="Custom word count (only used when length_mode is 'custom')"
    )
    source_content: Optional[str] = Field(
        default=None,
        description="Source content to base word count on (for source-derived articles)"
    )


class CalculateWordCountOutput(BaseModel):
    """Output from calculate_target_word_count function."""
    status: str = "success"
    target_word_count: int
    length_mode: str
    source_word_count: Optional[int] = None
    calculation_method: str  # "custom", "source_relative", "preset"


# =============================================================================
# LENGTH PRESETS
# =============================================================================

# Absolute presets (for original ideas with no source)
ABSOLUTE_PRESETS = {
    "brief": 700,
    "standard": 1250,
    "long": 2500,
}

# Relative multipliers (for source-derived articles)
RELATIVE_MULTIPLIERS = {
    "brief": 0.7,      # ~70% of source length
    "standard": 1.0,   # Match source length
    "long": 1.3,       # ~130% of source length
}


# =============================================================================
# FUNCTIONS
# =============================================================================

async def calculate_target_word_count(
    ctx,
    params: CalculateWordCountInput
) -> CalculateWordCountOutput:
    """
    Calculate target word count based on length mode and optional source content.

    For source-derived articles:
      - brief: ~70% of source length
      - standard: match source length
      - long: ~130% of source length

    For original ideas (no source):
      - brief: ~700 words
      - standard: ~1250 words
      - long: ~2500 words

    Custom mode always uses the specified word count.
    """
    length_mode = params.length_mode or "standard"
    custom_word_count = params.custom_word_count
    source_content = params.source_content

    ctx.report_input({
        "length_mode": length_mode,
        "custom_word_count": custom_word_count,
        "has_source": bool(source_content),
    })

    # Custom mode - use specified count
    if length_mode == "custom" and custom_word_count:
        logger.info(
            "word_count_calculated",
            method="custom",
            target_word_count=custom_word_count,
        )
        ctx.report_output({
            "target_word_count": custom_word_count,
            "method": "custom",
        })
        return CalculateWordCountOutput(
            status="success",
            target_word_count=custom_word_count,
            length_mode=length_mode,
            calculation_method="custom",
        )

    # Source-derived - calculate relative to source length
    if source_content:
        source_word_count = len(source_content.split())
        multiplier = RELATIVE_MULTIPLIERS.get(length_mode, 1.0)
        target_word_count = int(source_word_count * multiplier)

        # Apply reasonable bounds
        target_word_count = max(300, min(5000, target_word_count))

        logger.info(
            "word_count_calculated",
            method="source_relative",
            source_word_count=source_word_count,
            multiplier=multiplier,
            target_word_count=target_word_count,
        )
        ctx.report_output({
            "target_word_count": target_word_count,
            "source_word_count": source_word_count,
            "multiplier": multiplier,
            "method": "source_relative",
        })
        return CalculateWordCountOutput(
            status="success",
            target_word_count=target_word_count,
            length_mode=length_mode,
            source_word_count=source_word_count,
            calculation_method="source_relative",
        )

    # Original idea - use absolute preset
    target_word_count = ABSOLUTE_PRESETS.get(length_mode, 1250)

    logger.info(
        "word_count_calculated",
        method="preset",
        target_word_count=target_word_count,
    )
    ctx.report_output({
        "target_word_count": target_word_count,
        "method": "preset",
    })
    return CalculateWordCountOutput(
        status="success",
        target_word_count=target_word_count,
        length_mode=length_mode,
        calculation_method="preset",
    )
