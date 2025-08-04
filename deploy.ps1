# deploy.ps1 - PowerShell deployment script

Write-Host "🚀 Starting deployment process..." -ForegroundColor Green

# Check if required tools are installed
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Git is required but not installed" -ForegroundColor Red
    exit 1
}

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "❌ npm is required but not installed" -ForegroundColor Red
    exit 1
}

# 1. Install dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
npm install

# 2. Test build locally
Write-Host "🔨 Testing build locally..." -ForegroundColor Yellow
npm run build

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed. Please fix errors before deploying." -ForegroundColor Red
    exit 1
}

# 3. Commit changes
Write-Host "📝 Committing changes..." -ForegroundColor Yellow
git add .
git commit -m "Production deployment updates - $(Get-Date)"

# 4. Push to GitHub
Write-Host "⬆️  Pushing to GitHub..." -ForegroundColor Yellow
git push origin main

Write-Host "✅ Deployment preparation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Deploy backend to Hugging Face Spaces" -ForegroundColor White
Write-Host "2. Deploy frontend to Vercel" -ForegroundColor White
Write-Host "3. Update environment variables" -ForegroundColor White
Write-Host "4. Test production deployment" -ForegroundColor White
