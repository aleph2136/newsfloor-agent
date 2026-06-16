"""
config.py
 
Central configuration loaded from environment variables at Lambda cold start.
All table names, email addresses, and tuning parameters live here.
 
Local development: create a .env file in the project root (never commit it).
Lambda: set these as environment variables in the function configuration.
 
Usage:
    from config import settings
    table = dynamodb.Table(settings.dynamodb_runs_table)
"""
 
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
 
 
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
 
    # -------------------------------------------------------------------------
    # AWS region
    # -------------------------------------------------------------------------
    aws_region: str = Field(default="us-east-1")
 
    # -------------------------------------------------------------------------
    # DynamoDB table names
    # Defaults match what we'll create in the infrastructure phase.
    # -------------------------------------------------------------------------
    dynamodb_runs_table:    str = Field(default="digest-run-records")
    dynamodb_weekly_table:  str = Field(default="digest-weekly-synthesis")
    dynamodb_trends_table:  str = Field(default="digest-trends")
    dynamodb_sources_table: str = Field(default="digest-sources")
 
    # -------------------------------------------------------------------------
    # Gmail SMTP config
    # Use a Gmail App Password — generate one at:
    # https://myaccount.google.com/apppasswords
    # -------------------------------------------------------------------------
    smtp_sender_email:    str = Field(default="you@gmail.com")
    smtp_recipient_email: str = Field(default="you@gmail.com")
    smtp_app_token:       str = Field(default="")
 
    # -------------------------------------------------------------------------
    # Model config — Bedrock (default) + Gemini fallback for synthesis
    #
    # - topic:    Llama 4 Maverick — topic selection requires reasoning over a rotating recency
    #             window and structured JSON output; Maverick handles this reliably at lower cost
    #             than Sonnet and with more reasoning depth than Haiku
    # - fetch:    Amazon Nova 2 Lite — article enrichment is high-volume, low-complexity extraction;
    #             cheapest Bedrock-native model, low latency per article
    # - scoring:  Llama 4 Maverick — relevance classification requires nuanced judgment against the
    #             chosen topic; Maverick's structured output and reasoning justify the step up from Haiku
    # - synthesis writer: Claude Sonnet 4.6 (primary) / Gemini 3.5 Flash (near-zero cost fallback)
    #             — the digest prose is the user-visible output; Sonnet 4.6 is the best quality/cost
    #             balance for long-form writing; Gemini Flash is available when cost is the constraint
    # - synthesis support agents: Llama 4 Maverick — contextualizer + signal extractor are structured
    #             extraction tasks feeding the writer; Maverick handles multi-turn JSON reliably at scale
    # - trend signal clustering: Llama 4 Scout — batch pattern recognition over extended signal windows;
    #             Scout is optimized for throughput and structured JSON on large contexts
    # - trend weekly synthesis: Llama 4 Maverick — reasoning over accumulated trend data to produce a
    #             weekly narrative; Maverick provides the depth this task needs (Haiku was under-resourced)
    # - supervisors: Nova Pro (both input + output) — fastest capable Bedrock-native model for gate/
    #             validation tasks; cross-family diversity is achieved via Llama 4 / Anthropic / Nova
    #             mix elsewhere in the pipeline rather than splitting supervisor models
    # -------------------------------------------------------------------------
    gemini_model_synthesis:   str = Field(default="gemini/gemini-3.5-flash", description="Gemini model for the digest writer agent. Near-zero cost alternative to Bedrock Sonnet.")
    bedrock_model_synthesis:  str = Field(default="bedrock/us.anthropic.claude-sonnet-4-6")
    bedrock_model_synthesis_support: str = Field(default="bedrock/us.meta.llama4-maverick-17b-instruct-v1:0")
    bedrock_model_topic:   str = Field(default="bedrock/us.meta.llama4-maverick-17b-instruct-v1:0")
    bedrock_model_fetch:   str = Field(default="bedrock/us.amazon.nova-2-lite-v1:0")
    bedrock_model_scoring: str = Field(default="bedrock/us.meta.llama4-maverick-17b-instruct-v1:0")
    bedrock_model_trend:        str = Field(default="bedrock/us.meta.llama4-scout-17b-instruct-v1:0")
    bedrock_model_trend_weekly: str = Field(default="bedrock/us.meta.llama4-maverick-17b-instruct-v1:0")
    bedrock_model_input_supervisor:  str = Field(default="bedrock/us.amazon.nova-pro-v1:0")
    bedrock_model_output_supervisor: str = Field(default="bedrock/us.amazon.nova-pro-v1:0")
 
    # -------------------------------------------------------------------------
    # Pipeline tuning parameters
    # Changing these here affects all nodes — no hunting through agent code.
    # -------------------------------------------------------------------------
 
    # Fetch node
    fetch_min_articles:  int   = Field(default=3)
    fetch_max_articles:  int   = Field(default=10)
 
    # Scoring node
    score_threshold:            float = Field(default=0.5)   # minimum combined score to pass
    scoring_relevance_weight:   float = Field(default=0.55)  # weight for content relevance in combined score
    scoring_reputation_weight:  float = Field(default=0.25)  # weight for source reputation in combined score
    scoring_recency_weight:     float = Field(default=0.20)  # weight for article age in combined score
    scoring_default_reputation: float = Field(default=0.5)   # fallback reputation for unknown sources
 
    # Orchestrator gate
    max_retries_per_node: int  = Field(default=2)     # failures before degraded mode
 
    # Trend node
    #
    # Decay is calendar-time-based — elapsed days since the trend was last
    # touched (boosted or decayed) × trend_decay_rate_per_day — not a flat
    # amount per run. Topic rotation (topic_recency_window, default 30 days)
    # guarantees long gaps between chances to reinforce a given trend, so
    # decay has to be keyed to real elapsed time or it outpaces the rotation
    # window and every trend gets archived before its topic is even eligible
    # again. See strength.py for the calculation.
    #
    # 0.01/day means a never-reinforced trend takes ~90 days to fully decay
    # to trend_archive_threshold, and a STRONG trend (0.65) survives one full
    # topic_recency_window (30 days) idle and still stays above
    # trend_active_min_strength below — i.e. rotation alone won't drop a
    # trend out of "active" before it gets a chance to come back around.
    trend_decay_rate_per_day: float = Field(default=0.01)  # strength lost per elapsed day without reinforcement
    trend_boost_rate:        float = Field(default=0.25)   # strength gained when reinforced (instantaneous, not time-scaled)
    trend_archive_threshold: float = Field(default=0.1)    # archived below this strength
    trend_active_min_strength: float = Field(default=0.3)  # floor for a trend to count as "active" in get_active_trends — calibrated against trend_decay_rate_per_day above
 
    # Source reputation
    reputation_recency_weight: float = Field(default=0.2)  # how much a new article shifts domain score

    # Topic rotation — single source of truth for two things that must move
    # together:
    #   1. load_context.run() — days of RunRecord history fetched to build
    #      recent_topics (the window a topic must clear before it's eligible
    #      again).
    #   2. trend/db_writer.py write_run_record — the RunRecord's DynamoDB TTL.
    # The TTL must be >= the lookback window, or DynamoDB deletes records
    # before load_context's query window has had a chance to read them,
    # silently shrinking the effective lookback. Tying both to this one
    # setting makes that impossible instead of relying on comments.
    topic_recency_window: int = Field(default=30)     # days

    # -------------------------------------------------------------------------
    # Personal site publishing (S3 + CloudFront)
    # Set these to enable the publish node. If personal_site_bucket is empty
    # the publish node skips gracefully — useful while the site is not yet live.
    # -------------------------------------------------------------------------
    personal_site_bucket:      str = Field(default="")   # e.g. "mysite.com"
    personal_site_cf_dist_id:  str = Field(default="")   # CloudFront distribution ID
    personal_site_domain:      str = Field(default="")   # e.g. "mysite.com"
    personal_site_author_name: str = Field(default="")   # e.g. "Tester O'Testigan"


# Single shared instance — import this everywhere
settings = Settings()
 