# FinSight UI - Financial Analysis Platform

A modern, responsive React application for comprehensive loan eligibility analysis with PDF report generation and real-time data visualization.

## 🚀 Features

- **AI-Powered Analysis**: Advanced financial document analysis using machine learning
- **Interactive Dashboards**: Real-time charts and data visualizations
- **PDF Report Generation**: Professional reports with charts and insights
- **User Authentication**: Secure login with Supabase
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- **Cloud-Ready**: Deployable on multiple cloud platforms

## 🛠️ Technology Stack

- **Frontend**: React 18, TypeScript
- **Styling**: Tailwind CSS, Custom CSS
- **Charts**: Recharts, Chart.js
- **PDF Generation**: jsPDF, html2canvas
- **Authentication**: Supabase Auth
- **State Management**: React Hooks
- **Build Tool**: Create React App

## 📋 Prerequisites

- Node.js >= 16.0.0
- npm >= 8.0.0
- Git
- Supabase account
- Backend API server

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd finsight-ui
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start the development server**
   ```bash
   npm start
   ```

5. **Open your browser**
   ```
   http://localhost:3000
   ```

### Production Build

```bash
# Production build
npm run build

# Staging build
npm run build:staging

# Preview production build
npm run build && npx serve -s build
```

## 🌍 Cloud Deployment

### Vercel (Recommended)

1. **Connect to Vercel**
   ```bash
   npm i -g vercel
   vercel
   ```

2. **Set Environment Variables** in Vercel dashboard:
   - `REACT_APP_API_URL` = Your backend API URL
   - `REACT_APP_SUPABASE_URL` = Your Supabase URL
   - `REACT_APP_SUPABASE_ANON_KEY` = Your Supabase anon key

3. **Deploy**
   ```bash
   vercel --prod
   ```

### Netlify

1. **Build settings**:
   - Build command: `npm run build`
   - Publish directory: `build`

2. **Environment variables** in Netlify dashboard:
   - `REACT_APP_API_URL`
   - `REACT_APP_SUPABASE_URL`
   - `REACT_APP_SUPABASE_ANON_KEY`

### Other Platforms

- **Heroku**: Use Heroku's Node.js buildpack
- **AWS S3**: Upload `build/` folder contents
- **Railway**: Connect GitHub repo and set environment variables

## 🔧 Environment Configuration

### Required Environment Variables

```bash
# Supabase Configuration
REACT_APP_SUPABASE_URL=your-supabase-url
REACT_APP_SUPABASE_ANON_KEY=your-supabase-anon-key

# API Configuration
REACT_APP_API_URL=your-backend-api-url

# Environment
NODE_ENV=development|production|staging
```

### Environment-Specific URLs

- **Development**: `http://localhost:8000`
- **Production**: `https://your-api-domain.com`
- **Staging**: `https://your-staging-api.com`

## 📁 Project Structure

```
src/
├── components/          # Reusable UI components
├── pages/              # Page components
│   └── Dashboard/      # Main dashboard components
├── utils/              # Utility functions
├── supabase.js         # Supabase configuration
├── setupProxy.js       # Development proxy configuration
├── App.js              # Main app component
└── index.js           # App entry point
```

## 🔒 Security Features

- Environment variable validation
- API endpoint protection
- Secure authentication flow
- Input validation and sanitization
- Error boundary handling

## 🧪 Testing

```bash
# Run tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage
```

## 📊 Performance Optimization

- Code splitting and lazy loading
- Image optimization
- Bundle size optimization
- Caching strategies
- CDN integration support

## 🐛 Troubleshooting

### Common Issues

1. **CORS Errors**: Check API URL configuration
2. **Build Failures**: Verify environment variables
3. **Authentication Issues**: Check Supabase configuration
4. **API Connection Issues**: Verify backend server status

### Debug Mode

Enable debug logging by setting:
```bash
DEBUG=true
```

## 📝 API Integration

The app integrates with a FastAPI backend for:
- Document analysis
- Report generation
- User management
- Data processing

### API Endpoints Used

- `POST /v1/analyze-document` - Document analysis
- `GET /v1/reports` - Fetch user reports
- `POST /v1/save-report` - Save analysis results
- `DELETE /v1/reports/{id}` - Delete report

## 🎨 Customization

### Theming

Modify colors and styles in:
- `src/App.css` - Global styles
- Component-specific CSS files
- Tailwind configuration

### Branding

Update branding elements:
- Logo and favicon
- Company name and colors
- Custom fonts and typography

## 📱 Mobile Support

- Responsive design for all screen sizes
- Touch-optimized interactions
- Mobile-friendly navigation
- Optimized performance on mobile devices

## 🔄 Updates and Maintenance

### Regular Updates

- Keep dependencies updated
- Monitor security vulnerabilities
- Update environment configurations
- Test across different devices

### Monitoring

- Error tracking and logging
- Performance monitoring
- User analytics integration
- API usage monitoring

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting guide
- Review the deployment documentation

---

**Built with ❤️ for modern financial analysis**

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can't go back!**

If you aren't satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you're on your own.

You don't have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn't feel obligated to use this feature. However we understand that this tool wouldn't be useful if you couldn't customize it when you are ready for it.

## Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).

### Code Splitting

This section has moved here: [https://facebook.github.io/create-react-app/docs/code-splitting](https://facebook.github.io/create-react-app/docs/code-splitting)

### Analyzing the Bundle Size

This section has moved here: [https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size](https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size)

### Making a Progressive Web App

This section has moved here: [https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app](https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app)

### Advanced Configuration

This section has moved here: [https://facebook.github.io/create-react-app/docs/advanced-configuration](https://facebook.github.io/create-react-app/docs/advanced-configuration)

### Deployment

This section has moved here: [https://facebook.github.io/create-react-app/docs/deployment](https://facebook.github.io/create-react-app/docs/deployment)

### `npm run build` fails to minify

This section has moved here: [https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify](https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify)
