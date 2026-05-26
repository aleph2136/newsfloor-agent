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
    # Bedrock model config
    # - topic:    Amazon Nova Micro (lightweight structured selection task)
    # - fetch:    Amazon Nova Lite v2 (article enrichment scraping)
    # - scoring:  Llama 3.3 70B (relevance analysis)
    # - synthesis writer: Sonnet (highest quality for final output)
    # - synthesis support agents: Haiku (contextualizer + signal extractor)
    # - trend tracker: Llama 4 Scout (signal clustering + weekly synthesis)
    # - supervisors: Nova Pro (input) and Llama 3.3 70B (output) to avoid Claude monoculture
    # -------------------------------------------------------------------------
    bedrock_model_haiku:   str = Field(default="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0")
    bedrock_model_sonnet:  str = Field(default="bedrock/us.anthropic.claude-sonnet-4-6")
    bedrock_model_topic:   str = Field(default="bedrock/us.amazon.nova-micro-v1:0")
    bedrock_model_fetch:   str = Field(default="bedrock/us.amazon.nova-2-lite-v1:0")
    bedrock_model_scoring: str = Field(default="bedrock/us.meta.llama3-3-70b-instruct-v1:0")
    bedrock_model_trend:   str = Field(default="bedrock/us.meta.llama4-scout-17b-instruct-v1:0")
    bedrock_model_input_supervisor:  str = Field(default="bedrock/us.amazon.nova-pro-v1:0")
    bedrock_model_output_supervisor: str = Field(default="bedrock/us.meta.llama3-3-70b-instruct-v1:0")
 
    # -------------------------------------------------------------------------
    # Pipeline tuning parameters
    # Changing these here affects all nodes — no hunting through agent code.
    # -------------------------------------------------------------------------
 
    # Fetch node
    fetch_min_articles:  int   = Field(default=3)
    fetch_max_articles:  int   = Field(default=10)
 
    # Scoring node
    score_threshold:            float = Field(default=0.5)   # minimum combined score to pass
    scoring_relevance_weight:   float = Field(default=0.65)  # weight for content relevance in combined score
    scoring_reputation_weight:  float = Field(default=0.35)  # weight for source reputation in combined score
    scoring_default_reputation: float = Field(default=0.5)   # fallback reputation for unknown sources
 
    # Orchestrator gate
    max_retries_per_node: int  = Field(default=2)     # failures before degraded mode
 
    # Trend node
    trend_decay_rate:    float = Field(default=0.15)  # strength lost per run without reinforcement
    trend_boost_rate:    float = Field(default=0.25)  # strength gained when reinforced
    trend_archive_threshold: float = Field(default=0.1)  # archived below this strength
 
    # Source reputation
    reputation_recency_weight: float = Field(default=0.2)  # how much a new article shifts domain score
 
    # Topic rotation — how many recent topics to exclude from selection
    topic_recency_window: int = Field(default=14)     # days
 
 
# Single shared instance — import this everywhere
settings = Settings()
 