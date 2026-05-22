
Copy

# =============================================================================
# deploy.ps1
#
# Full deployment script for the AI Agent Morning Digest.
# Run from the project root directory.
#
# First-time setup:
#   1. Fill in the CONFIG section below
#   2. Run: .\infra\deploy.ps1 -FirstRun
#      This creates the S3 bucket and verifies your SES email addresses.
#
# Subsequent deploys (code changes only):
#   .\infra\deploy.ps1
#
# Deploy infrastructure changes only (no code repackage):
#   .\infra\deploy.ps1 -InfraOnly
# =============================================================================
 
param(
    [switch]$FirstRun,   # creates S3 bucket + verifies SES on first deploy
    [switch]$InfraOnly   # skips packaging, just updates the CFT
)
 
# =============================================================================
# CONFIG — fill these in before first deploy
# =============================================================================
$STACK_NAME       = "newsroom-agent"
$ENVIRONMENT      = "prod"
$AWS_REGION       = "us-east-2"
$DEPLOYMENT_BUCKET = "newsroom-agent-deploy-YOUR-ACCOUNT-ID"  # must be globally unique
$SENDER_EMAIL     = "sgriffith812@gmail.com"
$RECIPIENT_EMAIL  = "sgriffith812@gmail.com"
$SCHEDULE         = "cron(0 12 * * ? *)"   # 7am Eastern = 12:00 UTC
# =============================================================================
 
$ErrorActionPreference = "Stop"
$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
 
Write-Host ""
Write-Host "=== Digest Agent Deploy ===" -ForegroundColor Cyan
Write-Host "Stack:       $STACK_NAME-$ENVIRONMENT"
Write-Host "Region:      $AWS_REGION"
Write-Host "Bucket:      $DEPLOYMENT_BUCKET"
Write-Host ""
 
 
# -----------------------------------------------------------------------------
# STEP 1 — First-run setup (bucket + SES verification)
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
 
    Write-Host "[1/5] Requesting SES email verification..." -ForegroundColor Yellow
    aws ses verify-email-identity --email-address $SENDER_EMAIL --region $AWS_REGION
    aws ses verify-email-identity --email-address $RECIPIENT_EMAIL --region $AWS_REGION
 
    Write-Host ""
    Write-Host "    IMPORTANT: Check both inboxes and click the verification links" -ForegroundColor Magenta
    Write-Host "    before running your first deploy. SES will reject sends from" -ForegroundColor Magenta
    Write-Host "    unverified addresses." -ForegroundColor Magenta
    Write-Host ""
 
    $confirm = Read-Host "Have you verified both email addresses? (yes/no)"
    if ($confirm -ne "yes") {
        Write-Host "Exiting. Re-run without -FirstRun once emails are verified." -ForegroundColor Red
        exit 1
    }
}
 
 
# -----------------------------------------------------------------------------
# STEP 2 — Build the deployment package
# -----------------------------------------------------------------------------
if (-not $InfraOnly) {
    Write-Host "[2/5] Building deployment package..." -ForegroundColor Yellow
 
    $PACKAGE_DIR = Join-Path $PROJECT_ROOT "package"
    $ZIP_PATH    = Join-Path $PROJECT_ROOT "deployment.zip"
 
    # Clean previous build
    if (Test-Path $PACKAGE_DIR) { Remove-Item -Recurse -Force $PACKAGE_DIR }
    if (Test-Path $ZIP_PATH)    { Remove-Item -Force $ZIP_PATH }
 
    New-Item -ItemType Directory -Path $PACKAGE_DIR | Out-Null
 
    # Install dependencies into ./package as Linux-compatible wheels
    # --platform and --only-binary are critical on Windows — see requirements.txt
    pip install -r (Join-Path $PROJECT_ROOT "requirements.txt") `
        --target $PACKAGE_DIR `
        --platform manylinux2014_x86_64 `
        --implementation cp `
        --python-version 3.12 `
        --only-binary=:all: `
        --upgrade `
        --quiet
 
    if ($LASTEXITCODE -ne 0) {
        Write-Host "pip install failed. See output above." -ForegroundColor Red
        exit 1
    }
 
    # Add source files to the package directory before zipping
    # Add all .py files from project root
    Get-ChildItem -Path $PROJECT_ROOT/newsfloor/ | ForEach-Object {
        Copy-Item $_.FullName -Destination $PACKAGE_DIR
    }
 
    # Add the contracts package
    $contractsSrc  = Join-Path $PROJECT_ROOT "contracts"
    $contractsDest = Join-Path $PACKAGE_DIR "contracts"
    Copy-Item -Recurse $contractsSrc $contractsDest
 
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
 
$TEMPLATE_PATH = Join-Path $PSScriptRoot "template.yaml"
 
aws cloudformation deploy `
    --template-file $TEMPLATE_PATH `
    --stack-name "$STACK_NAME-$ENVIRONMENT" `
    --region $AWS_REGION `
    --capabilities CAPABILITY_NAMED_IAM `
    --parameter-overrides `
        SenderEmail=$SENDER_EMAIL `
        RecipientEmail=$RECIPIENT_EMAIL `
        ScheduleExpression=$SCHEDULE `
        DeploymentBucket=$DEPLOYMENT_BUCKET `
        Environment=$ENVIRONMENT `
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