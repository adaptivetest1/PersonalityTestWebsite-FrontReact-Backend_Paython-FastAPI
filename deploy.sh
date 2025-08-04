#!/bin/bash
# deploy.sh - Automated deployment script

echo "🚀 Starting deployment process..."

# Check if required tools are installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is required but not installed"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "❌ npm is required but not installed"
    exit 1
fi

# 1. Install dependencies
echo "📦 Installing dependencies..."
npm install

# 2. Test build locally
echo "🔨 Testing build locally..."
npm run build

if [ $? -ne 0 ]; then
    echo "❌ Build failed. Please fix errors before deploying."
    exit 1
fi

# 3. Commit changes
echo "📝 Committing changes..."
git add .
git commit -m "Production deployment updates - $(date)"

# 4. Push to GitHub
echo "⬆️  Pushing to GitHub..."
git push origin main

echo "✅ Deployment preparation complete!"
echo ""
echo "Next steps:"
echo "1. Deploy backend to Hugging Face Spaces"
echo "2. Deploy frontend to Vercel"
echo "3. Update environment variables"
echo "4. Test production deployment"
