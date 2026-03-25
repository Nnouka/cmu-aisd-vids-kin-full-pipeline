param(
    [Parameter(Mandatory = $true)][string]$Bucket,
    [Parameter(Mandatory = $true)][string]$DistributionId,
    [string]$BuildDir = "dist"
)

if (!(Test-Path $BuildDir)) {
    throw "Build folder '$BuildDir' not found. Run npm run build first."
}

Write-Host "Uploading static assets to s3://$Bucket"
aws s3 sync "$BuildDir" "s3://$Bucket" --delete

Write-Host "Invalidating CloudFront distribution $DistributionId"
aws cloudfront create-invalidation --distribution-id $DistributionId --paths "/*"

Write-Host "Frontend deployment complete."
