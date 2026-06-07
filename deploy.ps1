
# =============================================================================
# deploy.ps1
#
# Full deployment script for the AI Agent Morning Digest.
# Run from the project root directory.
#
# Prerequisites:
#   - Docker Desktop running (for container image build)
#   - AWS CLI configured with credentials
#   - uv installed (for generating requirements.txt from pyproject.toml)
#
# First-time setup:
#   1. Set the required environment variables listed in the CONFIG section below
#   2. Run: .\deploy.ps1 -FirstRun
#      This creates the ECR repository and deploys the stack.
#
# Subsequent deploys (code changes only):
#   .\deploy.ps1
#
# Deploy infrastructure changes only (no image rebuild):
#   .\deploy.ps1 -InfraOnly
# =============================================================================

param(
    [switch]$FirstRun,    # creates ECR repository on first deploy
    [switch]$InfraOnly,   # skips image build/push, just updates the CFT
    [switch]$SkipTests    # skips the pre-deploy test suite (not recommended)
)

# =============================================================================
# CONFIG — values are loaded from .env in the project root.
# Shell environment variables override .env when both are present.
# =============================================================================
# Required (.env keys):
#   NEWSFLOOR_SENDER_EMAIL             Gmail address used to send the digest
#   NEWSFLOOR_RECIPIENT_EMAIL          Address the digest is delivered to
#   NEWSFLOOR_SMTP_PASSWORD            Gmail App Password (not your account password)
#                                      Generate at: https://myaccount.google.com/apppasswords
#   NEWSFLOOR_PERSONAL_SITE_BUCKET     S3 bucket name for the personal site
#   NEWSFLOOR_PERSONAL_SITE_CF_DIST_ID CloudFront distribution ID
#   NEWSFLOOR_PERSONAL_SITE_DOMAIN     Custom domain, e.g. "my-domain.com"
#   NEWSFLOOR_PERSONAL_SITE_AUTHOR_NAME  Display name rendered in article pages
#
# Optional (.env keys with defaults shown):
#   GEMINI_API_KEY                     Gemini API key for AI content generation (if using Gemini instead of Bedrock)
#   NEWSFLOOR_AWS_REGION               default: "us-east-1"
#   NEWSFLOOR_STACK_NAME               default: "newsroom-agent"
#   NEWSFLOOR_ENVIRONMENT              default: "prod"
#   NEWSFLOOR_SCHEDULE                 default: "cron(0 12 * * ? *)"  (7am Eastern)
# =============================================================================

# ---------------------------------------------------------------------------
# .env loader — reads KEY=VALUE pairs, strips quotes, ignores comments/blanks.
# Shell env always wins over .env so CI/CD can override without touching the file.
# ---------------------------------------------------------------------------
function Read-DotEnv {
    param([string]$Path)
    $map = @{}
    if (-not (Test-Path $Path)) { return $map }
    Get-Content $Path | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq '' -or $line.StartsWith('#')) { return }
        $idx = $line.IndexOf('=')
        if ($idx -lt 1) { return }
        $key   = $line.Substring(0, $idx).Trim()
        $value = $line.Substring($idx + 1).Trim()
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        $map[$key] = $value
    }
    return $map
}

$dotenv = Read-DotEnv (Join-Path $PSScriptRoot ".env")

function Get-EnvVal {
    param([string]$Name, [string]$Default = "")
    $fromShell = [System.Environment]::GetEnvironmentVariable($Name)
    if ($fromShell) { return $fromShell }
    if ($dotenv.ContainsKey($Name)) { return $dotenv[$Name] }
    return $Default
}

$STACK_NAME             = Get-EnvVal "NEWSFLOOR_STACK_NAME"    "newsroom-agent"
$ENVIRONMENT            = Get-EnvVal "NEWSFLOOR_ENVIRONMENT"   "prod"
$AWS_REGION             = Get-EnvVal "NEWSFLOOR_AWS_REGION"    "us-east-1"
$SENDER_EMAIL           = Get-EnvVal "NEWSFLOOR_SENDER_EMAIL"
$RECIPIENT_EMAIL        = Get-EnvVal "NEWSFLOOR_RECIPIENT_EMAIL"
$SMTP_PASSWORD          = Get-EnvVal "NEWSFLOOR_SMTP_PASSWORD"
$SCHEDULE               = Get-EnvVal "NEWSFLOOR_SCHEDULE"      "cron(0 12 * * ? *)"
$PERSONAL_SITE_BUCKET   = Get-EnvVal "NEWSFLOOR_PERSONAL_SITE_BUCKET"
$PERSONAL_SITE_CF_DIST  = Get-EnvVal "NEWSFLOOR_PERSONAL_SITE_CF_DIST_ID"
$PERSONAL_SITE_DOMAIN   = Get-EnvVal "NEWSFLOOR_PERSONAL_SITE_DOMAIN"
$PERSONAL_SITE_AUTHOR   = Get-EnvVal "NEWSFLOOR_PERSONAL_SITE_AUTHOR_NAME"
$GEMINI_API_KEY         = Get-EnvVal "GEMINI_API_KEY"


# ECR repository name is derived from the environment — no env var needed
$ECR_REPO_NAME = "digest-agent-$ENVIRONMENT"

# Validate required variables
$missing = @()
if (-not $SENDER_EMAIL)           { $missing += "NEWSFLOOR_SENDER_EMAIL" }
if (-not $RECIPIENT_EMAIL)        { $missing += "NEWSFLOOR_RECIPIENT_EMAIL" }
if (-not $SMTP_PASSWORD)          { $missing += "NEWSFLOOR_SMTP_PASSWORD" }
if (-not $PERSONAL_SITE_BUCKET)   { $missing += "NEWSFLOOR_PERSONAL_SITE_BUCKET" }
if (-not $PERSONAL_SITE_CF_DIST)  { $missing += "NEWSFLOOR_PERSONAL_SITE_CF_DIST_ID" }
if (-not $PERSONAL_SITE_DOMAIN)   { $missing += "NEWSFLOOR_PERSONAL_SITE_DOMAIN" }
if (-not $PERSONAL_SITE_AUTHOR)   { $missing += "NEWSFLOOR_PERSONAL_SITE_AUTHOR_NAME" }
if (-not $GEMINI_API_KEY)         { $missing += "GEMINI_API_KEY" }

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "ERROR: The following values are missing from .env (or shell environment):" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    Write-Host ""
    Write-Host "Add them to .env in the project root. Example:" -ForegroundColor Yellow
    Write-Host '  NEWSFLOOR_SENDER_EMAIL             = "you@gmail.com"'        -ForegroundColor Yellow
    Write-Host '  NEWSFLOOR_RECIPIENT_EMAIL          = "you@gmail.com"'        -ForegroundColor Yellow
    Write-Host '  NEWSFLOOR_SMTP_PASSWORD            = "xxxx xxxx xxxx xxxx"'  -ForegroundColor Yellow
    Write-Host '  NEWSFLOOR_PERSONAL_SITE_BUCKET     = "my-site.com"'          -ForegroundColor Yellow
    Write-Host '  NEWSFLOOR_PERSONAL_SITE_CF_DIST_ID = "ABCDEF123456"'         -ForegroundColor Yellow
    Write-Host '  NEWSFLOOR_PERSONAL_SITE_DOMAIN     = "my-site.com"'          -ForegroundColor Yellow
    Write-Host '  NEWSFLOOR_PERSONAL_SITE_AUTHOR_NAME= "Your Name"'            -ForegroundColor Yellow
    Write-Host '  GEMINI_API_KEY                     = "AIza..."'              -ForegroundColor Yellow
    Write-Host ""
    Write-Host "NEWSFLOOR_SMTP_PASSWORD must be a Gmail App Password, not your account password." -ForegroundColor Yellow
    Write-Host "Generate one at: https://myaccount.google.com/apppasswords" -ForegroundColor Yellow
    Write-Host "GEMINI_API_KEY can be generated at: https://aistudio.google.com/apikey" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

$ErrorActionPreference = "Stop"
$PROJECT_ROOT = $PSScriptRoot


# -----------------------------------------------------------------------------
# STEP 0 — Run the test suite
# Skipped for -InfraOnly (no code changes) and -SkipTests (explicit opt-out).
# Tiers 1–3 only — Tier 4 requires live Bedrock credentials and is opt-in.
# -----------------------------------------------------------------------------
if (-not $InfraOnly -and -not $SkipTests) {
    Write-Host "[0/4] Running tests (unit, integration, schema_and_assertion)..." -ForegroundColor Yellow

    uv run pytest tests/unit/ tests/integration/ tests/schema_and_assertion/ -q

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Tests failed. Fix failing tests before deploying." -ForegroundColor Red
        Write-Host "To skip tests (not recommended): .\deploy.ps1 -SkipTests" -ForegroundColor Yellow
        Write-Host ""
        exit 1
    }
    Write-Host "    Tests passed." -ForegroundColor Green
    Write-Host ""
}


# Resolve AWS account ID and ECR URI — needed for the image build and CFT parameter
$ACCOUNT_ID = (aws sts get-caller-identity --query Account --output text)
if ($LASTEXITCODE -ne 0) {
    Write-Host "Could not resolve AWS account ID. Are your AWS credentials configured?" -ForegroundColor Red
    exit 1
}
$ECR_URI = "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"

if ($InfraOnly) {
    # Reuse the image already deployed — don't generate a new tag that doesn't exist in ECR.
    $IMAGE_URI = (aws lambda get-function `
        --function-name "digest-agent-$ENVIRONMENT" `
        --region $AWS_REGION `
        --query "Code.ImageUri" `
        --output text)
    if ($LASTEXITCODE -ne 0 -or -not $IMAGE_URI) {
        Write-Host "Could not retrieve the current Lambda image URI. Run without -InfraOnly first to push an image." -ForegroundColor Red
        exit 1
    }
} else {
    # Use the git short SHA as the image tag so each push has a unique URI.
    # CloudFormation only updates the Lambda function when ImageUri changes — using
    # "latest" every time means CFT sees no change and skips the function update.
    $IMAGE_TAG = (git rev-parse --short HEAD 2>$null)
    if (-not $IMAGE_TAG) { $IMAGE_TAG = (Get-Date -Format 'yyyyMMddHHmmss') }
    $IMAGE_URI = "${ECR_URI}:${IMAGE_TAG}"
}

Write-Host ""
Write-Host "=== Digest Agent Deploy ===" -ForegroundColor Cyan
Write-Host "Stack:       $STACK_NAME-$ENVIRONMENT"
Write-Host "Region:      $AWS_REGION"
Write-Host "ECR image:   $IMAGE_URI"
Write-Host ""


# -----------------------------------------------------------------------------
# STEP 1 — First-run setup (ECR repository creation)
# -----------------------------------------------------------------------------
if ($FirstRun) {
    Write-Host "[1/4] Creating ECR repository..." -ForegroundColor Yellow

    aws ecr create-repository `
        --repository-name $ECR_REPO_NAME `
        --region $AWS_REGION `
        --image-scanning-configuration scanOnPush=true

    if ($LASTEXITCODE -ne 0) {
        Write-Host "    ECR repository may already exist. Continuing." -ForegroundColor Yellow
    } else {
        Write-Host "    ECR repository created." -ForegroundColor Green
    }
}


# -----------------------------------------------------------------------------
# STEP 2 — Build and push the container image
# -----------------------------------------------------------------------------
if (-not $InfraOnly) {
    Write-Host "[2/4] Building and pushing container image..." -ForegroundColor Yellow

    $REQS_PATH = Join-Path $PROJECT_ROOT "requirements.txt"

    # Generate requirements.txt from pyproject.toml.
    # The Dockerfile COPYs this and installs via pip inside the container.
    # --no-hashes: pip-compatible format  --no-dev: exclude test/lint tools
    uv export --no-hashes --no-dev --no-emit-project --output-file $REQS_PATH

    if ($LASTEXITCODE -ne 0) {
        Write-Host "uv export failed. Is uv installed? Run: pip install uv" -ForegroundColor Red
        exit 1
    }

    # Authenticate Docker to ECR
    aws ecr get-login-password --region $AWS_REGION |
        docker login --username AWS --password-stdin $ECR_URI

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ECR Docker login failed." -ForegroundColor Red
        Remove-Item $REQS_PATH -ErrorAction SilentlyContinue
        exit 1
    }

    # Build the image for Linux/amd64 — Lambda runs on x86_64 Amazon Linux 2023.
    # --platform ensures correct binary wheels even when building on ARM or Windows/WSL.
    # --provenance=false prevents BuildKit from wrapping the image in an OCI manifest list
    # (which happens by default in recent Docker Desktop). Lambda only accepts single-platform
    # image manifests and rejects manifest lists with a 400 "media type not supported" error.
    docker build `
        --platform linux/amd64 `
        --provenance=false `
        -t $IMAGE_URI `
        $PROJECT_ROOT

    if ($LASTEXITCODE -ne 0) {
        Write-Host "docker build failed. See output above." -ForegroundColor Red
        Remove-Item $REQS_PATH -ErrorAction SilentlyContinue
        exit 1
    }

    docker push $IMAGE_URI

    if ($LASTEXITCODE -ne 0) {
        Write-Host "docker push failed. See output above." -ForegroundColor Red
        Remove-Item $REQS_PATH -ErrorAction SilentlyContinue
        exit 1
    }

    Remove-Item $REQS_PATH -ErrorAction SilentlyContinue
    Write-Host "    Image pushed: $IMAGE_URI" -ForegroundColor Green
}


# -----------------------------------------------------------------------------
# STEP 3 — Deploy CloudFormation stack
# -----------------------------------------------------------------------------
Write-Host "[3/4] Deploying CloudFormation stack..." -ForegroundColor Yellow

$TEMPLATE_PATH = Join-Path $PROJECT_ROOT "cft\stack.yaml"

aws cloudformation deploy `
    --template-file $TEMPLATE_PATH `
    --stack-name "$STACK_NAME-$ENVIRONMENT" `
    --region $AWS_REGION `
    --capabilities CAPABILITY_NAMED_IAM `
    --parameter-overrides `
        "SmtpSenderEmail=$SENDER_EMAIL" `
        "SmtpRecipientEmail=$RECIPIENT_EMAIL" `
        "SmtpPassword=$SMTP_PASSWORD" `
        "ScheduleExpression=$SCHEDULE" `
        "Environment=$ENVIRONMENT" `
        "ImageUri=$IMAGE_URI" `
        "PersonalSiteBucketName=$PERSONAL_SITE_BUCKET" `
        "PersonalSiteDistributionId=$PERSONAL_SITE_CF_DIST" `
        "PersonalSiteDomain=$PERSONAL_SITE_DOMAIN" `
        "PersonalSiteAuthorName=$PERSONAL_SITE_AUTHOR" `
        "GeminiApiKey=$GEMINI_API_KEY" `
    --no-fail-on-empty-changeset

if ($LASTEXITCODE -ne 0) {
    Write-Host "CloudFormation deploy failed. Check the AWS console for details." -ForegroundColor Red
    Write-Host "Stack events:" -ForegroundColor Yellow
    aws cloudformation describe-stack-events `
        --stack-name "$STACK_NAME-$ENVIRONMENT" `
        --region $AWS_REGION `
        --query "StackEvents[?ResourceStatus=='CREATE_FAILED'||ResourceStatus=='UPDATE_FAILED'].{Resource:LogicalResourceId,Reason:ResourceStatusReason}" `
        --output table
    exit 1
}

Write-Host "    Stack deployed." -ForegroundColor Green


# -----------------------------------------------------------------------------
# STEP 4 — Print outputs
# -----------------------------------------------------------------------------
Write-Host "[4/4] Stack outputs:" -ForegroundColor Yellow

aws cloudformation describe-stacks `
    --stack-name "$STACK_NAME-$ENVIRONMENT" `
    --region $AWS_REGION `
    --query "Stacks[0].Outputs" `
    --output table

Write-Host ""
Write-Host "=== Deploy complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "To trigger a test run immediately:" -ForegroundColor Cyan
Write-Host "  aws lambda invoke --function-name digest-agent-$ENVIRONMENT --region $AWS_REGION response.json"
Write-Host "  cat response.json"
Write-Host ""
Write-Host "To tail logs:" -ForegroundColor Cyan
Write-Host "  aws logs tail /aws/lambda/digest-agent-$ENVIRONMENT --follow --region $AWS_REGION"
Write-Host ""
