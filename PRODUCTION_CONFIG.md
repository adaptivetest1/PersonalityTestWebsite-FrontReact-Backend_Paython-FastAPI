# Production Configuration Guide

## Environment Variables Setup

### For Hugging Face Spaces (Backend):
1. Go to your Space settings
2. Add these variables in the "Variables" section:

GROQ_API_KEY=your_actual_groq_api_key_here
PORT=7860
STORAGE_TYPE=file
HF_SPACE=true

### For Vercel (Frontend):
1. Go to your Vercel project settings
2. Add these environment variables:

REACT_APP_API_URL=https://your-hf-space-name.hf.space/api
NODE_ENV=production
GENERATE_SOURCEMAP=false

## Security Notes:
- Never commit API keys to version control
- Use environment variables for all sensitive data
- Update CORS origins with your actual domain names
- Enable HTTPS in production

## Data Persistence:
- Development: Uses local JSON files
- Production: Ready for MongoDB/PostgreSQL integration
- Hugging Face: File-based storage in /app/data directory
