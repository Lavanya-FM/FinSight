# FinSight UI - Deployment Configuration

## Environment Variables

### For Local Development
```bash
# Copy .env.example to .env and update the values
cp .env.example .env

# Update .env with your local configuration
NODE_ENV=development
REACT_APP_API_URL=http://localhost:8000
```

### For Cloud Deployment

#### Environment Variables to Set:
- `NODE_ENV=production`
- `REACT_APP_API_URL=https://your-backend-api-url.com`
- `REACT_APP_SUPABASE_URL=your-supabase-url`
- `REACT_APP_SUPABASE_ANON_KEY=your-supabase-anon-key`

## Deployment Platforms

### 1. Vercel (Recommended)
1. Connect your GitHub repository to Vercel
2. Set environment variables in Vercel dashboard:
   - `REACT_APP_API_URL` = Your backend API URL
   - `REACT_APP_SUPABASE_URL` = Your Supabase URL
   - `REACT_APP_SUPABASE_ANON_KEY` = Your Supabase anon key
3. Deploy automatically on git push

### 2. Netlify
1. Connect your GitHub repository to Netlify
2. Set environment variables in Netlify dashboard:
   - `REACT_APP_API_URL` = Your backend API URL
   - `REACT_APP_SUPABASE_URL` = Your Supabase URL
   - `REACT_APP_SUPABASE_ANON_KEY` = Your Supabase anon key
3. Configure build settings:
   - Build command: `npm run build`
   - Publish directory: `build`

### 3. Heroku
1. Create a new app on Heroku
2. Connect your GitHub repository
3. Set environment variables in Heroku dashboard
4. Enable automatic deploys

### 4. AWS S3 + CloudFront
1. Build the app: `npm run build`
2. Upload to S3 bucket
3. Configure CloudFront distribution
4. Set environment variables in your backend API

## Build Commands

```bash
# Development build
npm run build

# Production build
npm run build:production

# Staging build
npm run build:staging
```

## API Configuration

The app automatically detects the environment and configures API endpoints:

- **Local Development**: Uses `http://localhost:8000` with proxy
- **Cloud Deployment**: Uses the `REACT_APP_API_URL` environment variable

## Proxy Configuration

For local development, the app uses a proxy to forward API requests:
- `/v1/*` → `http://localhost:8000/api/v1/*`

For cloud deployment, set `REACT_APP_API_URL` to your backend API URL.

## Supabase Configuration

Supabase is configured to work in both environments:
- Uses environment variables for URL and keys
- Automatically handles authentication
- Works with both local and cloud deployments

## Troubleshooting

### CORS Issues
- Ensure your backend API allows requests from your frontend domain
- Check that `REACT_APP_API_URL` is set correctly for production

### Build Issues
- Make sure all environment variables are set
- Check that Node.js version is >= 16.0.0
- Verify npm version is >= 8.0.0

### Runtime Issues
- Check browser console for errors
- Verify API endpoints are accessible
- Ensure Supabase configuration is correct

## Production Checklist

- [ ] Set all required environment variables
- [ ] Test API connectivity
- [ ] Verify Supabase authentication works
- [ ] Check that all routes work correctly
- [ ] Test PDF generation functionality
- [ ] Verify responsive design on mobile devices
- [ ] Check loading states and error handling
