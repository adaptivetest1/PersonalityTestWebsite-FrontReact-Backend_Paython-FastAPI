# 🧠 AI-Powered Adaptive Personality Test

## 🌟 Project Overview
Advanced personality testing application that uses AI to generate personalized questions based on user demographics, implementing sophisticated IRT (Item Response Theory) and CAT (Computerized Adaptive Testing) algorithms.

## 🔥 Key Features

### 🤖 AI-Powered Question Generation
- **50 adaptive questions** generated in real-time using AI
- Questions personalized based on:
  - **Age group** (teen, young adult, middle age, senior)
  - **Gender** (with proper Arabic grammar adaptation)
  - **Education level** (high school → PhD)
  - **Marital status** (single, married, divorced, widowed)

### 🎯 Advanced Psychometric Algorithms
- **IRT (Item Response Theory)** with 2PL model
- **CAT (Computerized Adaptive Testing)** for optimal question selection
- **Maximum Likelihood Estimation** for ability assessment
- **Information-based stopping criteria**
- **Minimum/Maximum question constraints** (25-50 questions)

### 📊 Big Five Personality Assessment
- **Openness** (الانفتاح على التجربة)
- **Conscientiousness** (الضمير) 
- **Extraversion** (الانبساط)
- **Agreeableness** (المقبولية)
- **Neuroticism** (العصابية)

### 🚀 Technical Stack
- **Backend**: FastAPI with Python
- **Frontend**: React with Arabic RTL support
- **AI Engine**: Groq Cloud (LLama-3-8B-8192)
- **Algorithms**: IRT, CAT, MLE
- **Storage**: File-based persistence (production-ready)

---

## 📋 Installation & Setup

### 1. **Clone Repository**
```bash
git clone https://github.com/AmiraSayedMohamed/PersonalityTestWebsite-FrontReact-Backend_Paython-FastAPI.git
cd PersonalityTestWebsite-FrontReact-Backend_Paython-FastAPI
```

### 2. **Backend Dependencies**
```bash
pip install -r requirements_irt.txt
```

### 3. **Frontend Dependencies**
```bash
npm install
```

### 4. **Environment Configuration**
Copy `.env.template` to `.env` and add your API keys:
```env
PORT=3001
GROQ_API_KEY=your_actual_api_key_here
DANGEROUSLY_DISABLE_HOST_CHECK=true
GENERATE_SOURCEMAP=false
```

### 5. **Run the Application**
```bash
# Start Backend (Terminal 1)
python simple_backend.py
# Backend runs on: http://localhost:8889

# Start Frontend (Terminal 2)
npm start
# Frontend runs on: http://localhost:3000
```

---

## 🚀 Deployment

### **Production Deployment**
1. **Frontend → Vercel**: Automated deployment from GitHub
2. **Backend → Hugging Face Spaces**: Docker-based deployment

### **Environment Variables**
- See `.env.template` for development setup
- See `.env.hf.template` for Hugging Face deployment
- See `FINAL_DEPLOYMENT_GUIDE.md` for complete deployment instructions

---

## 🔧 Configuration

### **IRT Parameters**
```python
max_questions = 50         # Maximum questions per test
min_questions = 25         # Minimum questions
target_se = 0.3           # Target standard error for stopping
min_theta = -3.0          # Minimum ability level
max_theta = 3.0           # Maximum ability level
```

### **Question Generation**
- **10 questions per dimension** (50 total)
- **Adaptive difficulty** based on user responses
- **Demographic personalization** for cultural relevance
- **Arabic grammar adaptation** for gender-specific language

---

## 🎯 How It Works

### **1. User Registration**
User provides demographic information for personalized questions

### **2. AI Question Generation**
AI generates culturally relevant questions based on user demographics

### **3. Adaptive Testing (CAT)**
System selects most informative questions based on user responses

### **4. Progress Tracking**
- Real-time progress bar (0-100%)
- Question counter (1-50)
- Dimension-specific progress
- Standard error monitoring

---

## 📊 API Endpoints

### **User Endpoints**
```http
POST /api/sessions              # Create new test session
GET  /api/sessions/{id}/question # Get adaptive question
POST /api/answers               # Submit answer & update θ
GET  /api/sessions/{id}/report  # Get personality report
```

### **Admin Endpoints**
```http
POST /api/admin/login          # Admin authentication
GET  /api/admin/dashboard      # Statistics dashboard
GET  /api/admin/participants   # Participant management
GET  /api/test                 # Connection testing
```

---

## 🎨 Frontend Components

### **Main Application** (`App.js`)
- Multi-step test flow
- Real-time progress tracking
- Arabic RTL interface
- Demographic data collection

### **Admin Dashboard** (`AdminDashboard.js`)
- Real-time statistics
- Participant management
- Chart.js visualizations
- Connection monitoring

---

## 📈 Advanced Features

### **AI Integration**
- Model: LLama-3-8B-8192
- Temperature: 0.7 for creativity
- Context: Demographic-aware prompts
- Fallback: Built-in questions if AI unavailable

### **IRT Implementation**
- 2PL Model: P(θ) = 1/(1 + e^(-a(θ-b)))
- MLE Estimation: Newton-Raphson method
- Information Function: I(θ) = a²P(θ)(1-P(θ))
- Standard Error: SE = 1/√I(θ)

---

## 🔐 Security & Privacy

- **Admin Authentication**: Username/password protection
- **Session Management**: UUID-based session tracking
- **Data Persistence**: Production-ready file storage
- **API Key Security**: Environment variable protection

---

## 📋 Documentation

- `FINAL_DEPLOYMENT_GUIDE.md` - Complete deployment instructions
- `DEPLOYMENT_CHECKLIST.md` - Pre-deployment checklist
- `PRODUCTION_CONFIG.md` - Production configuration guide
- `UPDATE_SUMMARY.md` - Recent updates and changes

---

## 📊 Sample Output

### **Question Generation**
```json
{
  "question_id": "extraversion_young_adult_female_bachelor_single_1",
  "text": "هل تستمتعين بالحديث مع أشخاص جدد في المناسبات الاجتماعية؟",
  "difficulty": -0.5,
  "discrimination": 1.8,
  "reverse_scored": false,
  "dimension": "extraversion"
}
```

### **Progress Response**
```json
{
  "message": "Answer submitted successfully",
  "status": "active",
  "total_answered": 45,
  "progress_percentage": 90.0,
  "current_theta": 0.73,
  "current_se": 0.42
}
```

---

## ⚡ Performance Metrics

- **Question Generation**: ~2-3 seconds per dimension
- **Response Processing**: <100ms per answer
- **Progress Calculation**: Real-time updates
- **Total Test Time**: 10-15 minutes (adaptive, 50 questions)

---

## 🤝 Contributing

1. **Question Templates**: Add more demographic-specific templates
2. **Algorithm Optimization**: Improve IRT parameter estimation
3. **UI/UX Enhancement**: Better Arabic typography and animations
4. **Analytics Integration**: Add detailed reporting features

---

## 📞 Support

For technical support or questions:
- **GitHub Issues**: Use the repository issue tracker
- **Documentation**: Check the deployment guides
- **AI Integration**: Refer to AI service documentation

---

🎯 **Ready to revolutionize personality testing with AI!** 🚀
