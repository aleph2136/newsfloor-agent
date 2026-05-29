
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
# CONFIG — set these environment variables before running
# =============================================================================
# Required:
#   $env:NEWSFLOOR_SENDER_EMAIL        Gmail address used to send the digest
#   $env:NEWSFLOOR_RECIPIENT_EMAIL     address the digest is delivered to
#   $env:NEWSFLOOR_SMTP_PASSWORD       Gmail App Password (not your account password)
#                                      Generate at: https://myaccount.google.com/apppasswords
#
# Optional (defaults shown):
#   $env:NEWSFLOOR_AWS_REGION          default: "us-east-1"
#   $env:NEWSFLOOR_STACK_NAME          default: "newsroom-agent"
#   $env:NEWSFLOOR_ENVIRONMENT         default: "prod"
#   $env:NEWSFLOOR_SCHEDULE            default: "cron(0 12 * * ? *)"  (7am Eastern)
# =============================================================================

$STACK_NAME      = if ($env:NEWSFLOOR_STACK_NAME)    { $env:NEWSFLOOR_STACK_NAME }    else { "newsroom-agent" }
$ENVIRONMENT     = if ($env:NEWSFLOOR_ENVIRONMENT)   { $env:NEWSFLOOR_ENVIRONMENT }   else { "prod" }
$AWS_REGION      = if ($env:NEWSFLOOR_AWS_REGION)    { $env:NEWSFLOOR_AWS_REGION }    else { "us-east-1" }
$SENDER_EMAIL    = $env:NEWSFLOOR_SENDER_EMAIL
$RECIPIENT_EMAIL = $env:NEWSFLOOR_RECIPIENT_EMAIL
$SMTP_PASSWORD   = $env:NEWSFLOOR_SMTP_PASSWORD
$SCHEDULE        = if ($env:NEWSFLOOR_SCHEDULE)      { $env:NEWSFLOOR_SCHEDULE }      else { "cron(0 12 * * ? *)" }

# ECR repository name is derived from the environment — no env var needed
$ECR_REPO_NAME = "digest-agent-$ENVIRONMENT"

# Validate required variables
$missing = @()
if (-not $SENDER_EMAIL)    { $missing += "NEWSFLOOR_SENDER_EMAIL" }
if (-not $RECIPIENT_EMAIL) { $missing += "NEWSFLOOR_RECIPIENT_EMAIL" }
if (-not $SMTP_PASSWORD)   { $missing += "NEWSFLOOR_SMTP_PASSWORD" }

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "ERROR: The following required environment variables are not set:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    Write-Host ""
    Write-Host "Set them before running this script. Example:" -ForegroundColor Yellow
    Write-Host '  $env:NEWSFLOOR_SENDER_EMAIL      = "you@gmail.com"' -ForegroundColor Yellow
    Write-Host '  $env:NEWSFLOOR_RECIPIENT_EMAIL   = "you@gmail.com"' -ForegroundColor Yellow
    Write-Host '  $env:NEWSFLOOR_SMTP_PASSWORD     = "xxxx xxxx xxxx xxxx"' -ForegroundColor Yellow
    Write-Host ""
    Write-Host "SMTP_PASSWORD must be a Gmail App Password, not your regular account password." -ForegroundColor Yellow
    Write-Host "Generate one at: https://myaccount.google.com/apppasswords" -ForegroundColor Yellow
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
    Write-Host "[0/4] Running tests (unit, integration, tier3)..." -ForegroundColor Yellow

    uv run pytest tests/unit/ tests/integration/ tests/tier3/ -q

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
