import React, { useState, useEffect } from 'react';
import AdminDashboard from './AdminDashboard';
import './App.css';

// Use environment variable for backend URL or fallback to localhost
const BACKEND_URL = process.env.NODE_ENV === 'production' 
  ? (process.env.REACT_APP_API_URL || 'https://personalitytest-personality-test-backend.hf.space')
  : 'http://localhost:8889';

function App() {
  const [currentStep, setCurrentStep] = useState('welcome'); // welcome, test, report, admin
  const [session, setSession] = useState(null);
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [report, setReport] = useState(null);
  const [language, setLanguage] = useState('ar'); // 'ar' for Arabic, 'en' for English
  
  // New state for additional user info
  const [userName, setUserName] = useState('');
  const [gender, setGender] = useState('');
  const [birthYear, setBirthYear] = useState('');
  const [educationLevel, setEducationLevel] = useState('');
  const [maritalStatus, setMaritalStatus] = useState('');

  // Translation object
  const translations = {
    ar: {
      title: "اكتشف شخصيتك",
      subtitle: "اختبار تحليل الشخصية",
      welcomeText: "مرحباً بك في اختبار تحليل الشخصية المتقدم.",
      description: "سيستغرق الاختبار حوالي 15-20 دقيقة ويتكون من 50 سؤال",
      startButton: "ابدأ الاختبار",
      nameLabel: "الاسم:",
      namePlaceholder: "أدخل اسمك",
      genderLabel: "الجنس:",
      selectGender: "اختر الجنس",
      male: "ذكر",
      female: "أنثى",
      birthYearLabel: "سنة الميلاد:",
      birthYearPlaceholder: "أدخل سنة الميلاد",
      educationLabel: "المستوى التعليمي:",
      selectEducation: "اختر المستوى التعليمي",
      primary: "ابتدائي",
      middle: "إعدادي",
      secondary: "ثانوي",
      university: "جامعي",
      postgrad: "دراسات عليا",
      maritalLabel: "الحالة الاجتماعية:",
      selectMarital: "اختر الحالة الاجتماعية",
      single: "أعزب",
      married: "متزوج/متزوجه",
      divorced: "مطلق/مطلقة",
      widowed: "أرمل/أرملة",
      nextButton: "التالي",
      backButton: "رجوع",
      submitButton: "إرسال الإجابة",
      loadingQuestions: "جاري تحميل الأسئلة...",
      loadingReport: "جاري إنشاء التقرير...",
      finishButton: "إنهاء الاختبار",
      personalityReport: "تقرير تحليل الشخصية",
      pleaseFillAll: "يرجى ملء جميع الحقول المطلوبة",
      languageToggle: "English"
    },
    en: {
      title: "Discover Your Personality",
      subtitle: "Personality Analysis Test",
      welcomeText: "Welcome to the Advanced Personality Analysis Test.",
      description: "The test will take about 15-20 minutes and consists of 50 questions",
      startButton: "Start Test",
      nameLabel: "Name:",
      namePlaceholder: "Enter your name",
      genderLabel: "Gender:",
      selectGender: "Select Gender",
      male: "Male",
      female: "Female",
      birthYearLabel: "Birth Year:",
      birthYearPlaceholder: "Enter birth year",
      educationLabel: "Education Level:",
      selectEducation: "Select Education Level",
      primary: "Primary",
      middle: "Middle School",
      secondary: "High School",
      university: "University",
      postgrad: "Postgraduate",
      maritalLabel: "Marital Status:",
      selectMarital: "Select Marital Status",
      single: "Single",
      married: "Married",
      divorced: "Divorced",
      widowed: "Widowed",
      nextButton: "Next",
      backButton: "Back",
      submitButton: "Submit Answer",
      loadingQuestions: "Loading questions...",
      loadingReport: "Generating report...",
      finishButton: "Finish Test",
      personalityReport: "Personality Analysis Report",
      pleaseFillAll: "Please fill all required fields",
      languageToggle: "عربي"
    }
  };

  const t = translations[language];

  // Function to toggle language
  const toggleLanguage = () => {
    setLanguage(language === 'ar' ? 'en' : 'ar');
  };

  // Check for admin URL parameter
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('admin') === 'true') {
      setCurrentStep('admin');
    }
  }, []);

  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [progress, setProgress] = useState(0);

  // Helper function to get dimension index for progress calculation
  const getCurrentDimensionIndex = (dimension) => {
    const dimensionOrder = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'];
    const index = dimensionOrder.indexOf(dimension);
    return index >= 0 ? index : 0; // Return 0 if not found instead of -1
  };

  // Helper function to get Arabic name for dimension
  const getBigFiveName = (dimension) => {
    const dimensionNames = {
      'openness': 'الانفتاح على التجارب',
      'conscientiousness': 'الضمير الحي', 
      'extraversion': 'الانبساط',
      'agreeableness': 'المقبولية',
      'neuroticism': 'العصابية'
    };
    return dimensionNames[dimension] || dimension;
  };

  // Start new test session
  const startTest = async () => {
    if (!userName.trim()) {
      setError('الرجاء إدخال اسمك');
      return;
    }

    setLoading(true);
    setError('');
    
    console.log('Starting test with data:', {
      name: userName,
      gender: gender,
      birthYear: birthYear ? parseInt(birthYear) : null,
      educationLevel: educationLevel,
      maritalStatus: maritalStatus
    });

    try {
      console.log('Sending request to:', `${BACKEND_URL}/sessions`);
      const response = await fetch(`${BACKEND_URL}/sessions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: userName,
          gender: gender,
          birthYear: birthYear ? parseInt(birthYear) : null,
          educationLevel: educationLevel,
          maritalStatus: maritalStatus
        })
      });

      console.log('Response status:', response.status);
      console.log('Response headers:', response.headers);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.log('Error response text:', errorText);
        console.log('Response status text:', response.statusText);
        throw new Error(`فشل في إنشاء جلسة الاختبار: ${response.status} ${response.statusText}`);
      }

      const sessionData = await response.json();
      console.log('Session Data:', sessionData);  // للتحقق من البيانات
      setSession(sessionData);
      setCurrentStep('test');
      await loadCurrentQuestion(sessionData.session_id);
    } catch (err) {
      console.error('Full error object:', err);
      setError(err.message || 'خطأ غير معروف في إنشاء الجلسة');
    } finally {
      setLoading(false);
    }
  };

  // Load current question
  const loadCurrentQuestion = async (sessionId) => {
    setLoading(true);
    console.log('Loading question for session:', sessionId); // للتحقق
    try {
      const response = await fetch(`${BACKEND_URL}/sessions/${sessionId}/question`);
      console.log('Response status:', response.status); // للتحقق
      
      if (!response.ok) {
        const errorText = await response.text();
        console.log('Error response:', errorText); // للتحقق
        throw new Error('فشل في تحميل السؤال');
      }

      const questionData = await response.json();
      console.log('Question Data:', questionData); // للتحقق من البيانات
      
      // Check if test is completed
      if (questionData.status === 'completed') {
        console.log('Test completed, loading report...');
        await loadReport();
        return;
      }
      
      setCurrentQuestion(questionData);
      setSelectedAnswer(null);
      
      // Use progress information from backend
      if (questionData.progress_percentage !== undefined) {
        setProgress(questionData.progress_percentage);
      } else {
        // Fallback calculation if backend doesn't provide progress
        const totalQuestionsAnswered = questionData && questionData.total_answered !== undefined ? 
          questionData.total_answered : 0;
        setProgress((totalQuestionsAnswered / 50) * 100);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Submit answer
  const submitAnswer = async () => {
    if (selectedAnswer === null) {
      setError('الرجاء اختيار إجابة');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch(`${BACKEND_URL}/answers`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: session.session_id,
          question_id: currentQuestion.question_id,
          response: selectedAnswer
        })
      });

      if (!response.ok) {
        throw new Error('فشل في حفظ الإجابة');
      }

      const result = await response.json();
      
      // Update progress from answer submission response
      if (result.progress_percentage !== undefined) {
        setProgress(result.progress_percentage);
      }
      
      if (result.status === 'completed') {
        // Test completed, load report
        await loadReport();
      } else {
        // Continue to next question
        await loadCurrentQuestion(session.session_id);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Load personality report
  const loadReport = async () => {
    setLoading(true);
    try {
      console.log('Loading report for session:', session.session_id);
      const response = await fetch(`${BACKEND_URL}/sessions/${session.session_id}/report`);
      
      console.log('Report response status:', response.status);
      if (!response.ok) {
        const errorText = await response.text();
        console.log('Report error response:', errorText);
        throw new Error('فشل في تحميل التقرير');
      }

      const reportData = await response.json();
      console.log('Report data received:', reportData);
      setReport(reportData);
      setCurrentStep('report');
    } catch (err) {
      console.error('Error loading report:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Reset test
  const resetTest = () => {
    setCurrentStep('welcome');
    setSession(null);
    setCurrentQuestion(null);
    setReport(null);
    setUserName('');
    setGender('');
    setBirthYear('');
    setEducationLevel('');
    setMaritalStatus('');
    setSelectedAnswer(null);
    setProgress(0);
    setError('');
  };

  // Answer scale labels for both languages
  const answerLabels = {
    ar: [
      'غير صحيح تماماً',
      'غير صحيح نوعاً ما',
      'محايد',
      'صحيح نوعاً ما',
      'صحيح تماماً'
    ],
    en: [
      'Completely False',
      'Somewhat False',
      'Neutral',
      'Somewhat True',
      'Completely True'
    ]
  };

  return (
    <div className="app" dir={language === 'ar' ? 'rtl' : 'ltr'}>
      {/* Language Toggle Button */}
      <div style={{
        position: 'absolute',
        top: '20px',
        [language === 'ar' ? 'left' : 'right']: '20px',
        zIndex: 1000
      }}>
        <button
          onClick={toggleLanguage}
          style={{
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '25px',
            padding: '10px 20px',
            fontSize: '14px',
            fontWeight: 'bold',
            cursor: 'pointer',
            boxShadow: '0 2px 6px rgba(0,123,255,0.3)',
            transition: 'all 0.3s ease',
            fontFamily: language === 'ar' ? 'Arial, sans-serif' : 'Arial, sans-serif'
          }}
          onMouseOver={(e) => {
            e.target.style.backgroundColor = '#0056b3';
            e.target.style.transform = 'translateY(-1px)';
          }}
          onMouseOut={(e) => {
            e.target.style.backgroundColor = '#007bff';
            e.target.style.transform = 'translateY(0)';
          }}
        >
          🌍 {t.languageToggle}
        </button>
      </div>
      
      <div className="container">
        {/* Header */}
        <header className="header">
          <h1 className="title">{t.subtitle}</h1>
          <p className="subtitle">{t.title}</p>
        </header>

        {/* Error Message */}
        {error && (
          <div className="error-message">
            <div className="error-content">
              <span className="error-icon">⚠️</span>
              <span>{error}</span>
            </div>
          </div>
        )}

        {/* Welcome Step */}
        {currentStep === 'welcome' && (
          <div className="step-container">
            <div className="welcome-card">
              <div className="welcome-icon">🧠</div>
              <h2 className="welcome-title">{t.welcomeText}</h2>
              <p className="welcome-description">
                {t.description}
              </p>
              
              <div className="form-group">
                <label className="form-label">{t.nameLabel} *</label>
                <input
                  type="text"
                  className="form-input"
                  placeholder={t.namePlaceholder}
                  value={userName}
                  onChange={(e) => setUserName(e.target.value)}
                  required
                />
              </div>

              <div className="form-group">
                <label className="form-label">{t.birthYearLabel}</label>
                <input
                  type="number"
                  className="form-input"
                  placeholder={t.birthYearPlaceholder}
                  value={birthYear}
                  onChange={(e) => setBirthYear(e.target.value)}
                />
              </div>

              <div className="form-group">
                <label className="form-label">{t.genderLabel}</label>
                <select 
                  className="form-input" 
                  value={gender} 
                  onChange={(e) => setGender(e.target.value)}
                >
                  <option value="">{t.selectGender}</option>
                  <option value="male">{t.male}</option>
                  <option value="female">{t.female}</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">{t.educationLabel}</label>
                <select 
                  className="form-input" 
                  value={educationLevel} 
                  onChange={(e) => setEducationLevel(e.target.value)}
                >
                  <option value="">{t.selectEducation}</option>
                  <option value="high_school">{t.secondary}</option>
                  <option value="diploma">{language === 'ar' ? 'دبلوم' : 'Diploma'}</option>
                  <option value="bachelor">{t.university}</option>
                  <option value="master">{t.postgrad}</option>
                  <option value="phd">{language === 'ar' ? 'دكتوراه' : 'PhD'}</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">{t.maritalLabel}</label>
                <select 
                  className="form-input" 
                  value={maritalStatus} 
                  onChange={(e) => setMaritalStatus(e.target.value)}
                >
                  <option value="">{t.selectMarital}</option>
                  <option value="single">{t.single}</option>
                  <option value="married">{t.married}</option>
                  <option value="divorced">{t.divorced}</option>
                  <option value="widowed">{t.widowed}</option>
                </select>
              </div>

              <button
                className="primary-button"
                onClick={startTest}
                disabled={loading}
              >
                {loading ? (
                  <span className="loading-spinner">{t.loadingQuestions}</span>
                ) : (
                  t.startButton
                )}
              </button>

              <div className="info-cards">
                <div className="info-card">
                  <h3>🎯 دقيق علمياً</h3>
                  <p>مبني على نموذج الشخصية الخماسي المعترف به عالمياً</p>
                </div>
                <div className="info-card">
                  <h3>🤖 تكيفي بالذكاء الاصطناعي</h3>
                  <p>أسئلة مولدة بالذكاء الاصطناعي تتكيف مع إجاباتك</p>
                </div>
                <div className="info-card">
                  <h3>📊 تقرير شامل</h3>
                  <p>تحليل مفصل لشخصيتك مع توصيات للتطوير</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Test Step */}
        {currentStep === 'test' && currentQuestion && (
          <div className="step-container">
            <div className="test-card">
              {/* Progress Bar */}
              <div className="progress-container">
                <div className="progress-info">
                  <span>السؤال {currentQuestion && currentQuestion.total_answered !== undefined ? 
                    currentQuestion.total_answered + 1 : 
                    (currentQuestion && currentQuestion.dimension && currentQuestion.questionNumber ? 
                      (getCurrentDimensionIndex(currentQuestion.dimension) * 10) + currentQuestion.questionNumber : 
                      1)
                  } من 50</span>
                  <span>{Math.round(progress)}%</span>
                </div>
                <div className="progress-bar">
                  <div 
                    className="progress-fill" 
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
              </div>

              {/* Question */}
              <div className="question-container">
                <h2 className="question-text">{currentQuestion.text}</h2>
              </div>

              {/* Answer Options */}
              <div className="answers-container">
                <p className="answers-instruction">{language === 'ar' ? 'اختر الإجابة التي تصف شخصيتك بشكل أفضل:' : 'Choose the answer that best describes your personality:'}</p>
                <div className="answer-options">
                  {answerLabels[language].map((label, index) => (
                    <button
                      key={index + 1}
                      className={`answer-option ${selectedAnswer === index + 1 ? 'selected' : ''}`}
                      onClick={() => setSelectedAnswer(index + 1)}
                    >
                      <div className="answer-number">{index + 1}</div>
                      <div className="answer-label">{label}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Navigation */}
              <div className="navigation-container">
                <button
                  className="primary-button"
                  onClick={submitAnswer}
                  disabled={loading || selectedAnswer === null}
                >
                  {loading ? (
                    <span className="loading-spinner">{language === 'ar' ? 'جاري الحفظ...' : 'Saving...'}</span>
                  ) : (
                    language === 'ar' ? 'السؤال التالي' : 'Next Question'
                  )}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Report Step */}
        {currentStep === 'report' && report && (
          <div className="step-container">
            <div className="report-card">
              <div className="report-header">
                <h2 className="report-title">تقرير شخصيتك - {report.name}</h2>
                <p className="report-date">تاريخ الإكمال: {new Date(report.completionDate).toLocaleDateString('ar-SA')}</p>
              </div>

              {/* Scores */}
              <div className="scores-section">
                <h3 className="section-title">{language === 'ar' ? 'نتائج أبعاد الشخصية الخمسة' : 'Big Five Personality Dimensions Results'}</h3>
                <div className="scores-grid">
                  {Object.entries(report.scores).map(([dimension, data]) => (
                    <div key={dimension} className="score-card">
                      <h4 className="score-dimension">{data.name}</h4>
                      <div className="score-visual">
                        <div className="score-bar">
                          <div 
                            className="score-fill"
                            style={{ width: `${(data.score / 5) * 100}%` }}
                          ></div>
                        </div>
                        <div className="score-info">
                          <span className="score-value">{data.score.toFixed(1)}/5</span>
                          <span className={`score-level ${data.level.toLowerCase()}`}>{data.level}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Detailed Analysis */}
              <div className="analysis-section">
                <h3 className="section-title">التحليل المفصل</h3>
                <div className="analysis-content">
                  {report.detailed_analysis ? report.detailed_analysis.split('\n').map((paragraph, index) => (
                    paragraph.trim() && <p key={index} className="analysis-paragraph">{paragraph}</p>
                  )) : <p>لا يوجد تحليل مفصل متاح</p>}
                </div>
              </div>

              {/* Recommendations */}
              <div className="recommendations-section">
                <h3 className="section-title">{language === 'ar' ? 'توصيات للتطوير' : 'Development Recommendations'}</h3>
                <div className="recommendations-list">
                  {report.recommendations && report.recommendations.length > 0 ? report.recommendations.map((recommendation, index) => (
                    <div key={index} className="recommendation-item">
                      <span className="recommendation-icon">💡</span>
                      <span className="recommendation-text">{recommendation}</span>
                    </div>
                  )) : <p>{language === 'ar' ? 'لا توجد توصيات متاحة' : 'No recommendations available'}</p>}
                </div>
              </div>

              {/* Actions */}
              <div className="report-actions">
                <button className="secondary-button" onClick={resetTest}>
                  {language === 'ar' ? 'إجراء اختبار جديد' : 'Take New Test'}
                </button>
                <button 
                  className="primary-button"
                  onClick={() => window.print()}
                >
                  {language === 'ar' ? 'طباعة التقرير' : 'Print Report'}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Loading Overlay */}
        {loading && currentStep !== 'welcome' && (
          <div className="loading-overlay">
            <div className="loading-content">
              <div className="loading-spinner-large"></div>
              <p>جاري المعالجة...</p>
            </div>
          </div>
        )}

        {/* Admin Dashboard */}
        {currentStep === 'admin' && <AdminDashboard />}
      </div>
    </div>
  );
}

export default App;