# AI Agent Morning Digest

A self-running daily briefing on AI agentic engineering, delivered to your inbox every morning. You configure the topics, sources, and recipient profile once — then it runs itself on a schedule, pulling fresh articles, scoring them, writing a focused digest, and sending it via email.

Built on AWS Lambda, LangGraph, and CrewAI. No ongoing maintenance required once deployed.

---

## What It Does

Every morning, the pipeline wakes up and runs ten steps:

1. **Loads context** from its own history — what topics have been covered recently, which sources have been reliable, what trends are emerging
2. **Selects a topic** from a rotating list, using an LLM to pick the most timely choice and sharpen it into a specific focus angle
3. **Fetches articles** from 15 curated RSS feeds relevant to AI engineering
4. **Scores each article** for relevance to today's topic and the credibility of its source
5. **Reviews the quality** of what it found — if the results aren't good enough, it loops back and tries again (up to twice)
6. **Writes the digest** — a focused HTML email that summarizes the best articles and highlights what's meaningful about them
7. **Reviews the digest** — if the writing doesn't meet its own standard, it rewrites (up to twice)
8. **Sends the email** via AWS SES
9. **Updates its memory** — trend strength scores, source reputation, and a weekly synthesis for long-term context

The whole thing runs in under 15 minutes on a cold Lambda start. You can also trigger it manually with a single CLI command.

---

## How the Pipeline Works

Understanding the flow makes it much easier to configure and debug. Here's the shape of the pipeline:

```
START
  └─► load_context
        └─► topic ─► fetch ─► scoring
                                  └─► input_supervisor
                                           ├─ rework ─► topic (loops back, up to 2x)
                                           └─ proceed ─► synthesis
                                                            └─► output_supervisor
                                                                     ├─ rework ─► synthesis (loops back, up to 2x)
                                                                     └─► proceed ─► delivery
                                                                                        └─► trend
                                                                                              └─► END
```

### The Two Quality Gates

The pipeline has two supervisor nodes that act as quality checkpoints:

**Input Supervisor** sits between scoring and synthesis. It evaluates whether the fetched and scored articles are good enough to write a digest from. If the article quality is too low, it sends the pipeline back to re-select a topic and re-fetch. It uses Amazon Nova Pro (deliberately not Claude — see below).

**Output Supervisor** sits between synthesis and delivery. It reads the draft digest and decides whether it meets the standard. If not, it routes back to synthesis for a rewrite. It uses Meta Llama 3.3 70B.

Both supervisors cap at two retries before forcing a proceed in degraded mode — the digest gets sent even if it's not perfect, rather than failing silently.

### Why Multiple LLM Models?

Each step in the pipeline uses the model that makes the most sense for that task:

| Node | Model | Reason |
|------|-------|--------|
| Topic, Fetch, Scoring | Claude Haiku 4.5 | Fast and cheap — these are structured tasks |
| Synthesis | Claude Sonnet 4.6 | The actual writing deserves higher quality |
| Input Supervisor | Amazon Nova Pro | Deliberate diversity — avoids Claude reviewing Claude |
| Output Supervisor | Meta Llama 3.3 70B | Same reason — independent second opinion |

Using multiple model families at the quality gates provides genuine independence. A supervisor built on the same model that wrote the content isn't a real check.

### What Gets Remembered Between Runs

The pipeline keeps a running memory in DynamoDB:

- **Trends** — topic areas that keep appearing get stronger scores over time; those that fade get archived
- **Source reputation** — each domain builds a reputation score based on the relevance of its articles
- **Recent topics** — the topic selector avoids repeating the last 14 days of coverage
- **Run history** — signals from recent runs feed into synthesis context so the digest can reference what's changed

---

## Project Structure

```
agentic_learning/
├── config.py                         # All operational settings (env var-backed)
├── deploy.ps1                        # Deployment script
├── pyproject.toml                    # Python dependencies
├── cft/
│   └── stack.yaml                    # CloudFormation — all AWS infrastructure
└── newsfloor/
    ├── handler.py                    # Lambda entry point
    ├── config_loader.py              # Loads JSON config files (cached per cold start)
    ├── config_data/                  # Edit these to customize without touching code
    │   ├── topics.json               # Topic rotation list
    │   ├── sources.json              # RSS/Atom feed URLs
    │   └── profile.json             # Recipient profile and focus areas
    ├── contracts/
    │   ├── primitives.py             # Shared data types (ArticleRaw, ArticleScored, etc.)
    │   ├── nodes.py                  # Input/output contracts for every node
    │   └── state.py                  # The graph state shape (what flows between nodes)
    ├── data/
    │   ├── db.py                     # DynamoDB read/write operations
    │   └── load_context.py           # Assembles historical context from DynamoDB
    └── graph/
        ├── graph.py                  # LangGraph wiring — nodes and edges only
        ├── nodes.py                  # Node functions (thin wrappers, call node_definitions/)
        ├── state.py                  # DigestGraphState TypedDict
        └── node_definitions/         # Where the actual work happens
            ├── topic.py              # Two-agent topic selection crew
            ├── fetch.py              # RSS fetching + thin article enrichment
            ├── scoring.py            # Relevance scoring + reputation weighting
            ├── input_supervisor.py   # Pre-synthesis quality gate
            ├── synthesis.py          # Three-agent digest writing crew
            ├── output_supervisor.py  # Pre-delivery quality gate
            ├── delivery.py           # SES email send
            └── trend/                # Trend tracking, strength decay, weekly synthesis
tests/
├── conftest.py                       # sys.path setup for all tests
├── unit/                             # Tier 1 — pure deterministic logic (67 tests)
│   ├── test_state.py
│   ├── test_strength.py
│   └── test_routing.py
├── integration/                      # Tier 2 — full graph with mocked nodes (4 tests)
│   └── test_graph_flow.py
├── tier3/                            # Tier 3 — schema contracts and scoring math (~90 tests)
│   ├── test_scoring_pipeline.py
│   └── test_schema_contracts.py
└── tier4/                            # Tier 4 — LLM-as-judge quality tests
    ├── conftest.py                   # Judge helper, LLM availability check, shared fixtures
    ├── test_topic_quality.py
    ├── test_synthesis_quality.py
    └── test_scoring_quality.py
```

---

## Configuration

Configuration lives in three places, each with a different scope:

### 1. Content Configuration (edit to customize)

These JSON files in `newsfloor/config_data/` control what the digest covers and who it's written for. Edit them directly — no code changes needed.

**`config_data/topics.json`** — the topic pool the pipeline rotates through. The strategist agent picks from this list each run, avoiding recent repeats.

```json
[
  "multi-agent orchestration patterns",
  "agent observability and tracing",
  "human-in-the-loop agent design",
  ...
]
```

To add a topic, append a string. To remove one, delete it. The list ships with 20 topics focused on AI agentic engineering — replace all of them if you want a digest on a completely different subject.

**`config_data/sources.json`** — the RSS/Atom feed URLs the fetcher harvests from. The pipeline ships with 15 curated feeds covering AI engineering, model providers, and practitioner blogs.

```json
[
  "https://simonwillison.net/atom/everything/",
  "https://blog.langchain.dev/rss/",
  ...
]
```

Add any RSS or Atom feed URL. Remove feeds that aren't relevant to your domain. Standard feed format is all that's required.

**`config_data/profile.json`** — the recipient profile that personalizes the synthesis. The writing agent uses this to calibrate depth, framing, and emphasis.

```json
{
  "name": "Sam",
  "focus_areas": [
    "AI agentic architecture",
    "agent observability and governance",
    ...
  ],
  "background_summary": "...",
  "experience_level": "senior engineer specializing in AI agentic architecture..."
}
```

Change `name`, `focus_areas`, `background_summary`, and `experience_level` to match the actual recipient. The synthesis node uses these fields directly in its prompts.

---

### 2. Operational Settings (`.env` for local dev, Lambda env vars for production)

`config.py` at the project root defines all tunable parameters. Values are read from environment variables with sensible defaults built in.

For **local development**, create a `.env` file in the project root:

```bash
# .env — never commit this file

# AWS
AWS_REGION=us-east-1

# Email
SES_SENDER_EMAIL=you@yourdomain.com
SES_RECIPIENT_EMAIL=you@yourdomain.com

# DynamoDB tables (must match stack.yaml)
DYNAMODB_RUNS_TABLE=digest-run-records-prod
DYNAMODB_WEEKLY_TABLE=digest-weekly-synthesis-prod
DYNAMODB_TRENDS_TABLE=digest-trends-prod
DYNAMODB_SOURCES_TABLE=digest-sources-prod

# Bedrock models — change these if you want to swap providers
BEDROCK_MODEL_HAIKU=bedrock/anthropic.claude-haiku-4-5-20251001-v1:0
BEDROCK_MODEL_SONNET=bedrock/anthropic.claude-sonnet-4-6
BEDROCK_MODEL_INPUT_SUPERVISOR=bedrock/us.amazon.nova-pro-v1:0
BEDROCK_MODEL_OUTPUT_SUPERVISOR=bedrock/us.meta.llama3-3-70b-instruct-v1:0

# Pipeline tuning (optional — these are the defaults)
SCORE_THRESHOLD=0.5
SCORING_RELEVANCE_WEIGHT=0.65
SCORING_REPUTATION_WEIGHT=0.35
TREND_DECAY_RATE=0.15
TREND_BOOST_RATE=0.25
MAX_RETRIES_PER_NODE=2
```

For **production**, the CloudFormation stack injects SES addresses as Lambda environment variables. All other values use their defaults unless you override them via the Lambda console or stack parameters.

---

### 3. Deployment Settings (environment variables for `deploy.ps1`)

Set these before running the deployment script. They control where the code is uploaded and what email addresses to use.

```powershell
# Required
$env:NEWSFLOOR_DEPLOYMENT_BUCKET = "newsroom-agent-deploy-123456789012"  # globally unique S3 bucket name
$env:NEWSFLOOR_SENDER_EMAIL      = "digest@yourdomain.com"
$env:NEWSFLOOR_RECIPIENT_EMAIL   = "you@yourdomain.com"

# Optional (these are the defaults)
$env:NEWSFLOOR_AWS_REGION        = "us-east-1"
$env:NEWSFLOOR_STACK_NAME        = "newsroom-agent"
$env:NEWSFLOOR_ENVIRONMENT       = "prod"
$env:NEWSFLOOR_SCHEDULE          = "cron(0 12 * * ? *)"   # 7am Eastern (UTC noon)
```

---

## Prerequisites

Before deploying or running locally, you need:

1. **Python 3.12** — the Lambda runtime and local dev environment
2. **uv** — the package manager used throughout this project
   ```powershell
   pip install uv
   ```
3. **AWS CLI** — configured with credentials that have permission to create Lambda, DynamoDB, SES, IAM, S3, and CloudFormation resources
   ```powershell
   aws configure
   ```
4. **Bedrock model access** — in the AWS console, navigate to Bedrock > Model access and enable:
   - Anthropic Claude Haiku 4.5
   - Anthropic Claude Sonnet 4.6
   - Amazon Nova Pro
   - Meta Llama 3.3 70B Instruct

   Model access is per-region. Enable it in the same region you're deploying to.

---

## Local Development

**Install dependencies:**
```powershell
uv sync
```

**Run the test suite:**
```powershell
uv run pytest -q
```

You should see 170 tests pass. These cover pure logic and schema contracts only — no LLM calls, no AWS. See the [Testing](#testing) section for when to run the Tier 4 LLM-as-judge tests.

**Verify JSON config files load correctly:**
```powershell
uv run python -c "from newsfloor.config_loader import load_topics, load_sources, load_profile; print(len(load_topics()), 'topics,', len(load_sources()), 'sources,', load_profile().name)"
```

---

## Testing

The test suite has four tiers. **Tiers 1–3 run on every change. Tier 4 is opt-in — run it when you make material changes to LLM-adjacent code.**

```
tests/
├── unit/          Tier 1 — pure functions, no I/O
├── integration/   Tier 2 — full graph with mocked nodes
├── tier3/         Tier 3 — scoring math, Pydantic contracts, edge cases
└── tier4/         Tier 4 — live LLM calls evaluated by a judge model
```

The default `uv run pytest` command covers Tiers 1–3 (configured via `testpaths` in `pyproject.toml`). Tier 4 requires explicit invocation.

---

### Tier 1 — Unit Tests (67 tests)

Pure deterministic logic. No LLM calls, no AWS, no I/O.

| File | What it covers | Count |
|------|----------------|-------|
| `tests/unit/test_state.py` | State merging, `rework_counts` reducer | 22 |
| `tests/unit/test_strength.py` | Trend strength boost/decay/archive math | 31 |
| `tests/unit/test_routing.py` | Supervisor routing decisions | 14 |

```powershell
uv run pytest tests/unit/ -q
```

**Run on:** every commit.

---

### Tier 2 — Integration Tests (4 tests)

Builds the compiled LangGraph and runs it end-to-end with all node functions mocked. Verifies the graph topology, supervisor rework loops, and retry cap enforcement.

```powershell
uv run pytest tests/integration/ -q
```

**Run on:** every commit. Together with Tier 1, this is what `uv run pytest -q` covers.

---

### Tier 3 — Schema and Assertion Tests (~90 tests)

Validates the deterministic scoring math, Pydantic contract enforcement, and JSON parsing/fallback behavior. No LLM calls — all `_score_relevance` invocations are patched out.

| File | What it covers |
|------|----------------|
| `tests/tier3/test_scoring_pipeline.py` | `_combine_scores` arithmetic, `_parse_relevance_output` parsing, `_apply_retry_adjustments` threshold logic, `run()` integration |
| `tests/tier3/test_schema_contracts.py` | Required fields, float bounds (`ge=0.0, le=1.0`), alias handling, default values, count consistency |

Included in the default `uv run pytest` run — no separate invocation needed.

---

### Tier 4 — LLM-as-Judge Tests

Runs the actual CrewAI crews against live Bedrock models and evaluates output quality using Claude Haiku as a judge. These catch **prompt regressions** — changes that pass all structural checks but produce noticeably worse output.

| File | Node tested | What the judge evaluates |
|------|-------------|--------------------------|
| `tests/tier4/test_topic_quality.py` | `topic` | Topic from rotation, not recently repeated, focus angle is specific, rationale references context |
| `tests/tier4/test_synthesis_quality.py` | `synthesis` | HTML structure, topic addressed, technical depth for Sam, no phantom URLs, signals are specific |
| `tests/tier4/test_scoring_quality.py` | `scoring` | Relevant articles score higher than irrelevant, score gap > 0.3, rationales don't contradict scores |

**Prerequisites:** AWS credentials with Bedrock access enabled for Claude Haiku 4.5. Tests auto-skip (via `pytest.mark.skipif`) when credentials are absent — they will not fail, they simply will not run.

```powershell
uv run pytest tests/tier4/ -q
```

Each test file caches its node output at `scope="module"`, so each crew runs once per session regardless of how many assertions check it.

**Run when you change:**
- Any prompt string in `node_definitions/topic.py`, `synthesis.py`, or `scoring.py`
- Scoring weights (`RELEVANCE_WEIGHT`, `REPUTATION_WEIGHT`) or the default threshold
- The Bedrock model ID for any tested node (`BEDROCK_MODEL_HAIKU`, `BEDROCK_MODEL_SONNET`)
- `config_data/profile.json` — the engineer profile is used directly in synthesis criteria
- `config_data/topics.json` — the rotation list is checked deterministically by the topic test

**Do not run on every commit.** Each full Tier 4 session makes approximately 15–20 Bedrock inference calls. Run it before merging a branch that touches prompt text, scoring logic, or model configuration.

To iterate on a single node without running all three files:
```powershell
uv run pytest tests/tier4/test_synthesis_quality.py -v
```

To run only the deterministic checks within Tier 4 (no judge calls):
```powershell
uv run pytest tests/tier4/ -v -k "not quality"
```

---

## Deployment

### First-Time Setup

**Step 1 — Set your deployment environment variables:**
```powershell
$env:NEWSFLOOR_DEPLOYMENT_BUCKET = "your-unique-bucket-name-here"
$env:NEWSFLOOR_SENDER_EMAIL      = "digest@yourdomain.com"
$env:NEWSFLOOR_RECIPIENT_EMAIL   = "you@yourdomain.com"
```

**Step 2 — Run the first-deploy script:**
```powershell
.\deploy.ps1 -FirstRun
```

This will:
1. Create the S3 deployment bucket
2. Send SES verification emails to both addresses
3. Pause and ask you to confirm you've clicked both verification links
4. Build the deployment package (pip-compatible Linux wheels via `uv`)
5. Upload the package to S3
6. Deploy the CloudFormation stack (Lambda, DynamoDB tables, EventBridge schedule, IAM role)

> **SES sandbox note:** New AWS accounts are in SES sandbox mode, which means you can only send to verified addresses. The first-run script verifies both the sender and recipient. If you want to send to unverified recipients, you'll need to request SES production access from the AWS console.

---

### Subsequent Deploys

After code or config changes:
```powershell
.\deploy.ps1
```

To update infrastructure only (no repackage):
```powershell
.\deploy.ps1 -InfraOnly
```

---

### Manual Trigger

To run the digest immediately without waiting for the schedule:
```powershell
aws lambda invoke --function-name digest-agent-prod --region us-east-1 response.json
cat response.json
```

To watch the logs in real time:
```powershell
aws logs tail /aws/lambda/digest-agent-prod --follow --region us-east-1
```

---

## AWS Infrastructure

All infrastructure is defined in `cft/stack.yaml` and managed by CloudFormation. No resources need to be created manually after the first-run S3 bucket.

| Resource | Purpose |
|----------|---------|
| **Lambda function** | Runs the pipeline; 15-minute timeout, 1024 MB memory |
| **EventBridge rule** | Triggers Lambda on schedule (7am Eastern by default) |
| **digest-run-records** | One record per daily run; 30-day TTL |
| **digest-weekly-synthesis** | Weekly distilled signals; 90-day TTL |
| **digest-trends** | Active trend records with strength scores |
| **digest-sources** | Domain-level reputation scores |
| **IAM role** | Least-privilege: only the specific DynamoDB, SES, and Bedrock actions actually used |

The DynamoDB tables use on-demand billing — you pay per read/write, not per hour. At one run per day, the cost is negligible.

---

## Scoring Explained

Each article gets a **combined score** from two components:

```
combined = (relevance × 0.65) + (reputation × 0.35)
```

- **Relevance** (0.0–1.0) — an LLM reads the article title and summary and scores it against today's topic and focus angle
- **Reputation** (0.0–1.0) — the domain's historical score from the sources table; unknown sources start at 0.5

The default passing threshold is **0.50**. An article from an unknown source that's highly relevant (0.9) scores `(0.9 × 0.65) + (0.5 × 0.35) = 0.76` — a clear pass. A low-relevance article from a trusted source (0.3 relevance, 0.9 reputation) scores `0.51` — a marginal pass that the input supervisor may flag.

The reputation score is adjusted based on the relevance score for the day.  High relevance scores increase the reputation, low relevance scores decrease the reuptation.

These weights are tunable via environment variables: `SCORING_RELEVANCE_WEIGHT`, `SCORING_REPUTATION_WEIGHT`, `SCORE_THRESHOLD`.

---

## Trend Tracking

The pipeline tracks emerging topics across runs using a decay/boost model:

- **Boost** (+0.25 by default): when an article reinforces an existing trend
- **Decay** (-0.15 by default): applied to every trend on every run, whether reinforced or not
- **Archive threshold** (0.10): trends below this are moved to archived status

This means a trend that stops appearing in articles fades naturally over 4–6 runs. A trend that appears repeatedly in quality articles grows toward a `dominant` rating.

Trend strength bands: `emerging` (0.1–0.39) → `growing` (0.4–0.64) → `strong` (0.65–0.84) → `dominant` (0.85–1.0)

---

## Troubleshooting

**"Missing required environment variables" error on deploy:**
Set `NEWSFLOOR_DEPLOYMENT_BUCKET`, `NEWSFLOOR_SENDER_EMAIL`, and `NEWSFLOOR_RECIPIENT_EMAIL` before running `deploy.ps1`.

**Lambda times out:**
The function has a 15-minute timeout. If it's consistently running close to the limit, check CloudWatch logs for which node is slow. The synthesis and supervisor nodes are the most LLM-intensive. You can also reduce `FETCH_MAX_ARTICLES` (default: 10) to process fewer articles per run.

**No email received:**
1. Check CloudWatch logs for the Lambda run: `aws logs tail /aws/lambda/digest-agent-prod --follow`
2. Verify both email addresses are confirmed in SES: AWS console → SES → Verified identities
3. If your account is in SES sandbox, only verified addresses can receive mail

**LLM errors in logs:**
Bedrock model access must be enabled per-region before first use. Go to AWS console → Bedrock → Model access and enable all four models used by the pipeline.

**Tests fail after modifying config_data JSON files:**
Run the manual verification command to catch syntax errors before running the test suite:
```powershell
uv run python -c "from newsfloor.config_loader import load_topics, load_sources, load_profile; print('OK')"
```
