
# =============================================================================
# deploy.ps1
#
# Full deployment script for the AI Agent Morning Digest.
# Run from the project root directory.
#
# First-time setup:
#   1. Set the required environment variables listed in the CONFIG section below
#   2. Run: .\deploy.ps1 -FirstRun
#      This creates the S3 bucket and deploys the stack.
#
# Subsequent deploys (code changes only):
#   .\deploy.ps1
#
# Deploy infrastructure changes only (no code repackage):
#   .\deploy.ps1 -InfraOnly
# =============================================================================

param(
    [switch]$FirstRun,   # creates S3 bucket on first deploy
    [switch]$InfraOnly   # skips packaging, just updates the CFT
)

# =============================================================================
# CONFIG — set these environment variables before running
# =============================================================================
# Required:
#   $env:NEWSFLOOR_DEPLOYMENT_BUCKET   globally-unique S3 bucket name
#                                      e.g. "newsroom-agent-deploy-123456789012"
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

$STACK_NAME        = if ($env:NEWSFLOOR_STACK_NAME)        { $env:NEWSFLOOR_STACK_NAME }        else { "newsroom-agent" }
$ENVIRONMENT       = if ($env:NEWSFLOOR_ENVIRONMENT)       { $env:NEWSFLOOR_ENVIRONMENT }       else { "prod" }
$AWS_REGION        = if ($env:NEWSFLOOR_AWS_REGION)        { $env:NEWSFLOOR_AWS_REGION }        else { "us-east-1" }
$DEPLOYMENT_BUCKET = $env:NEWSFLOOR_DEPLOYMENT_BUCKET
$SENDER_EMAIL      = $env:NEWSFLOOR_SENDER_EMAIL
$RECIPIENT_EMAIL   = $env:NEWSFLOOR_RECIPIENT_EMAIL
$SMTP_PASSWORD     = $env:NEWSFLOOR_SMTP_PASSWORD
$SCHEDULE          = if ($env:NEWSFLOOR_SCHEDULE)          { $env:NEWSFLOOR_SCHEDULE }          else { "cron(0 12 * * ? *)" }

# Validate required variables
$missing = @()
if (-not $DEPLOYMENT_BUCKET) { $missing += "NEWSFLOOR_DEPLOYMENT_BUCKET" }
if (-not $SENDER_EMAIL)      { $missing += "NEWSFLOOR_SENDER_EMAIL" }
if (-not $RECIPIENT_EMAIL)   { $missing += "NEWSFLOOR_RECIPIENT_EMAIL" }
if (-not $SMTP_PASSWORD)     { $missing += "NEWSFLOOR_SMTP_PASSWORD" }

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "ERROR: The following required environment variables are not set:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    Write-Host ""
    Write-Host "Set them before running this script. Example:" -ForegroundColor Yellow
    Write-Host '  $env:NEWSFLOOR_DEPLOYMENT_BUCKET = "newsroom-agent-deploy-123456789012"' -ForegroundColor Yellow
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

Write-Host ""
Write-Host "=== Digest Agent Deploy ===" -ForegroundColor Cyan
Write-Host "Stack:       $STACK_NAME-$ENVIRONMENT"
Write-Host "Region:      $AWS_REGION"
Write-Host "Bucket:      $DEPLOYMENT_BUCKET"
Write-Host ""


# -----------------------------------------------------------------------------
# STEP 1 — First-run setup (S3 bucket creation)
# -----------------------------------------------------------------------------
if ($FirstRun) {
    Write-Host "[1/5] Creating S3 deployment bucket..." -ForegroundColor Yellow

    # us-east-1 does not accept a LocationConstraint — other regions require it
    if ($AWS_REGION -eq "us-east-1") {
        aws s3api create-bucket `
            --bucket $DEPLOYMENT_BUCKET `
            --region $AWS_REGION
    } else {
        aws s3api create-bucket `
            --bucket $DEPLOYMENT_BUCKET `
            --region $AWS_REGION `
            --create-bucket-configuration LocationConstraint=$AWS_REGION
    }

    # Block all public access — this bucket should never be public
    aws s3api put-public-access-block `
        --bucket $DEPLOYMENT_BUCKET `
        --public-access-block-configuration `
            "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

    Write-Host "    Bucket created." -ForegroundColor Green
}


# -----------------------------------------------------------------------------
# STEP 2 — Build the deployment package
# -----------------------------------------------------------------------------
if (-not $InfraOnly) {
    Write-Host "[2/5] Building deployment package..." -ForegroundColor Yellow

    $PACKAGE_DIR = Join-Path $PROJECT_ROOT "package"
    $ZIP_PATH    = Join-Path $PROJECT_ROOT "deployment.zip"
    $TEMP_REQS   = Join-Path $PROJECT_ROOT "requirements-deploy.txt"

    # Clean previous build artifacts
    if (Test-Path $PACKAGE_DIR) { Remove-Item -Recurse -Force $PACKAGE_DIR }
    if (Test-Path $ZIP_PATH)    { Remove-Item -Force $ZIP_PATH }

    New-Item -ItemType Directory -Path $PACKAGE_DIR | Out-Null

    # Export production dependencies from pyproject.toml via uv
    # --no-hashes: pip-compatible format  --no-dev: exclude test/lint tools
    uv export --no-hashes --no-dev --output-file $TEMP_REQS

    if ($LASTEXITCODE -ne 0) {
        Write-Host "uv export failed. Is uv installed? Run: pip install uv" -ForegroundColor Red
        exit 1
    }

    # Install Linux-compatible wheels into ./package
    # --python-platform manylinux_2_28_x86_64: Lambda Python 3.12 runs on Amazon Linux 2023
    #   which requires glibc >= 2.28. manylinux_2_17 (AL2) is too old for recent packages.
    # --python-version 3.12: match Lambda runtime version
    # Cross-compilation is safe here — uv resolves the correct platform wheels on Windows
    uv pip install `
        --requirements $TEMP_REQS `
        --target $PACKAGE_DIR `
        --python-platform manylinux_2_28_x86_64 `
        --python-version 3.12 `
        --quiet

    if ($LASTEXITCODE -ne 0) {
        Write-Host "uv pip install failed. See output above." -ForegroundColor Red
        Remove-Item -Force $TEMP_REQS -ErrorAction SilentlyContinue
        exit 1
    }

    Remove-Item -Force $TEMP_REQS

    # Copy application source into the package directory
    # newsfloor/ — main package (handler, graph, node_definitions, data, contracts)
    Copy-Item -Recurse (Join-Path $PROJECT_ROOT "newsfloor") (Join-Path $PACKAGE_DIR "newsfloor")

    # config.py — root-level settings module imported by graph and node_definitions
    Copy-Item (Join-Path $PROJECT_ROOT "config.py") (Join-Path $PACKAGE_DIR "config.py")

    # Zip the package directory contents (not the folder itself)
    Compress-Archive -Path "$PACKAGE_DIR\*" -DestinationPath $ZIP_PATH

    $zipSize = [math]::Round((Get-Item $ZIP_PATH).Length / 1MB, 1)
    Write-Host "    Package built: deployment.zip ($zipSize MB)" -ForegroundColor Green

    # Warn if approaching Lambda's 50MB zipped limit
    if ($zipSize -gt 45) {
        Write-Host "    WARNING: zip is approaching Lambda's 50MB limit." -ForegroundColor Magenta
        Write-Host "    Consider moving to a Lambda Layer for dependencies." -ForegroundColor Magenta
    }
}


# -----------------------------------------------------------------------------
# STEP 3 — Upload zip to S3
# -----------------------------------------------------------------------------
if (-not $InfraOnly) {
    Write-Host "[3/5] Uploading deployment.zip to S3..." -ForegroundColor Yellow

    aws s3 cp `
        (Join-Path $PROJECT_ROOT "deployment.zip") `
        "s3://$DEPLOYMENT_BUCKET/digest-agent/deployment.zip" `
        --region $AWS_REGION

    if ($LASTEXITCODE -ne 0) {
        Write-Host "S3 upload failed." -ForegroundColor Red
        exit 1
    }

    Write-Host "    Uploaded." -ForegroundColor Green
}


# -----------------------------------------------------------------------------
# STEP 4 — Deploy CloudFormation stack
# -----------------------------------------------------------------------------
Write-Host "[4/5] Deploying CloudFormation stack..." -ForegroundColor Yellow

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
        "DeploymentBucket=$DEPLOYMENT_BUCKET" `
        "Environment=$ENVIRONMENT" `
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
# STEP 5 — Print outputs
# -----------------------------------------------------------------------------
Write-Host "[5/5] Stack outputs:" -ForegroundColor Yellow

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
