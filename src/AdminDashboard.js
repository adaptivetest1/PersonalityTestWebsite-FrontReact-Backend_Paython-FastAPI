import React, { useState, useEffect } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement,
} from 'chart.js';
import { Bar, Pie, Line } from 'react-chartjs-2';
import './AdminDashboard.css';

// Use the same backend URL as the main app
const BACKEND_URL = process.env.NODE_ENV === 'production' 
  ? (process.env.REACT_APP_API_URL || 'https://personalitytest-personality-test-backend.hf.space/api')
  : 'http://localhost:8889/api';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement
);

const AdminDashboard = () => {
  const [dashboardData, setDashboardData] = useState(null);
  const [participants, setParticipants] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [searchTerm, setSearchTerm] = useState('');
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loginForm, setLoginForm] = useState({ username: '', password: '' });
  const [showPassword, setShowPassword] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem('adminToken');
    if (token) {
      setIsAuthenticated(true);
      // Add delay to ensure backend is ready
      setTimeout(() => {
        fetchDashboardData();
        fetchParticipants();
      }, 1000);
    }
  }, [currentPage, searchTerm]);

  const handleLogin = async (e) => {
    e.preventDefault();
    setError(''); // Clear previous errors
    try {
      console.log('Attempting login with:', loginForm.username);
      const response = await fetch(`${BACKEND_URL}/admin/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(loginForm),
      });

      console.log('Login response status:', response.status);
      const data = await response.json();
      console.log('Login response data:', data);
      
      if (data.success) {
        localStorage.setItem('adminToken', data.token);
        setIsAuthenticated(true);
        fetchDashboardData();
        fetchParticipants();
      } else {
        setError(data.message || 'فشل في تسجيل الدخول');
      }
    } catch (err) {
      console.error('Login error:', err);
      setError('خطأ في الاتصال بالسيرفر');
    }
  };

  const fetchDashboardData = async () => {
    try {
      const token = localStorage.getItem('adminToken');
      if (!token) {
        console.log('No admin token found');
        return;
      }
      
      console.log('Fetching dashboard data...');
      console.log('Backend URL:', BACKEND_URL);
      
      const response = await fetch(`${BACKEND_URL}/admin/dashboard`, {
        headers: {
          'Admin-Token': token,
        },
      });

      console.log('Dashboard response status:', response.status);
      
      if (response.ok) {
        const data = await response.json();
        console.log('Dashboard data received:', data);
        setDashboardData(data);
      } else {
        const errorText = await response.text();
        console.error('Dashboard error response:', errorText);
        // Check if response is HTML (which means wrong endpoint)
        if (errorText.includes('<!doctype') || errorText.includes('<html')) {
          setError('خطأ في الاتصال: تأكد من عنوان الـ backend');
        } else {
          setError('خطأ في جلب بيانات الداشبورد: ' + response.status);
        }
      }
    } catch (err) {
      console.error('Dashboard fetch error:', err);
      setError('خطأ في الاتصال بالسيرفر: ' + err.message);
    }
  };

  const fetchParticipants = async () => {
    try {
      const token = localStorage.getItem('adminToken');
      const response = await fetch(`${BACKEND_URL}/admin/participants?page=${currentPage}&search=${searchTerm}`, {
        headers: {
          'Admin-Token': token,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setParticipants(data.participants);
        setTotalPages(data.totalPages);
      } else {
        setError('خطأ في جلب بيانات المشاركين');
      }
    } catch (err) {
      setError('خطأ في الاتصال بالسيرفر');
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async () => {
    try {
      const token = localStorage.getItem('adminToken');
      const response = await fetch(`${BACKEND_URL}/admin/export`, {
        headers: {
          'Admin-Token': token,
        },
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `personality_test_data_${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
      }
    } catch (err) {
      setError('خطأ في تصدير البيانات');
    }
  };

  const testConnection = async () => {
    try {
      const token = localStorage.getItem('adminToken');
      const response = await fetch(`${BACKEND_URL}/admin/test`, {
        headers: {
          'Admin-Token': token,
        },
      });

      if (response.ok) {
        const data = await response.json();
        alert(`اختبار الاتصال نجح!\nالجلسات: ${data.sessionsCount}\nالإجابات: ${data.answersCount}`);
      } else {
        const errorText = await response.text();
        alert('فشل اختبار الاتصال: ' + errorText);
      }
    } catch (err) {
      alert('خطأ في اختبار الاتصال: ' + err.message);
    }
  };

  const logout = () => {
    localStorage.removeItem('adminToken');
    setIsAuthenticated(false);
    setDashboardData(null);
    setParticipants([]);
  };

  // Chart configurations
  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
      },
    },
  };

  const getAgeDistributionChart = () => {
    if (!dashboardData?.ageDistribution) return null;
    
    return {
      labels: dashboardData.ageDistribution.map(item => item.label),
      datasets: [
        {
          label: 'عدد المشاركين',
          data: dashboardData.ageDistribution.map(item => item.value),
          backgroundColor: [
            'rgba(255, 99, 132, 0.8)',
            'rgba(54, 162, 235, 0.8)',
            'rgba(255, 205, 86, 0.8)',
            'rgba(75, 192, 192, 0.8)',
            'rgba(153, 102, 255, 0.8)',
            'rgba(255, 159, 64, 0.8)',
          ],
          borderColor: [
            'rgba(255, 99, 132, 1)',
            'rgba(54, 162, 235, 1)',
            'rgba(255, 205, 86, 1)',
            'rgba(75, 192, 192, 1)',
            'rgba(153, 102, 255, 1)',
            'rgba(255, 159, 64, 1)',
          ],
          borderWidth: 1,
        },
      ],
    };
  };

  const getGenderDistributionChart = () => {
    if (!dashboardData?.genderDistribution) return null;
    
    return {
      labels: dashboardData.genderDistribution.map(item => item.label),
      datasets: [
        {
          data: dashboardData.genderDistribution.map(item => item.value),
          backgroundColor: [
            'rgba(54, 162, 235, 0.8)',
            'rgba(255, 99, 132, 0.8)',
          ],
          borderColor: [
            'rgba(54, 162, 235, 1)',
            'rgba(255, 99, 132, 1)',
          ],
          borderWidth: 2,
        },
      ],
    };
  };

  const getEducationDistributionChart = () => {
    if (!dashboardData?.educationDistribution) return null;
    
    return {
      labels: dashboardData.educationDistribution.map(item => item.label),
      datasets: [
        {
          label: 'عدد المشاركين',
          data: dashboardData.educationDistribution.map(item => item.value),
          backgroundColor: 'rgba(75, 192, 192, 0.8)',
          borderColor: 'rgba(75, 192, 192, 1)',
          borderWidth: 1,
        },
      ],
    };
  };

  const getDailyStatsChart = () => {
    if (!dashboardData?.dailyStats) return null;
    
    return {
      labels: dashboardData.dailyStats.map(item => 
        new Date(item.date).toLocaleDateString('ar-EG')
      ),
      datasets: [
        {
          label: 'مشاركين جدد',
          data: dashboardData.dailyStats.map(item => item.newParticipants),
          borderColor: 'rgba(255, 99, 132, 1)',
          backgroundColor: 'rgba(255, 99, 132, 0.2)',
          tension: 0.1,
        },
        {
          label: 'اختبارات مكتملة',
          data: dashboardData.dailyStats.map(item => item.completedTests),
          borderColor: 'rgba(54, 162, 235, 1)',
          backgroundColor: 'rgba(54, 162, 235, 0.2)',
          tension: 0.1,
        },
      ],
    };
  };

  // Login form
  if (!isAuthenticated) {
    return (
      <div className="admin-login">
        <div className="login-container">
          <div className="login-card">
            <h2>🔐 تسجيل دخول الإدارة</h2>
            <form onSubmit={handleLogin}>
              <div className="form-group">
                <label>اسم المستخدم:</label>
                <input
                  type="text"
                  value={loginForm.username}
                  onChange={(e) => setLoginForm({...loginForm, username: e.target.value})}
                  required
                />
              </div>
              <div className="form-group">
                <label>كلمة المرور:</label>
                <div className="password-input-container">
                  <input
                    type={showPassword ? "text" : "password"}
                    value={loginForm.password}
                    onChange={(e) => setLoginForm({...loginForm, password: e.target.value})}
                    required
                  />
                  <button
                    type="button"
                    className="password-toggle"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? '🙈' : '👁️'}
                  </button>
                </div>
              </div>
              <button type="submit" className="login-button">
                تسجيل الدخول
              </button>
              {error && <div className="error-message">{error}</div>}
            </form>
          </div>
        </div>
      </div>
    );
  }

  if (loading && !dashboardData) {
    return (
      <div className="admin-loading">
        <div className="loading-spinner">جاري تحميل البيانات...</div>
      </div>
    );
  }

  return (
    <div className="admin-dashboard" dir="rtl">
      {/* Header */}
      <header className="admin-header">
        <div className="header-content">
          <h1>📊 لوحة تحكم اختبار الشخصية</h1>
          <div style={{ display: 'flex', gap: '10px' }}>
            <button onClick={testConnection} className="logout-button" style={{ backgroundColor: '#28a745' }}>
              اختبار الاتصال
            </button>
            <button onClick={logout} className="logout-button">
              تسجيل الخروج
            </button>
          </div>
        </div>
      </header>

      {/* Statistics Cards */}
      <div className="stats-cards">
        <div className="stat-card">
          <div className="stat-icon">👥</div>
          <div className="stat-content">
            <h3>إجمالي المشاركين</h3>
            <div className="stat-number">{dashboardData?.totalParticipants || 0}</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">✅</div>
          <div className="stat-content">
            <h3>الاختبارات المكتملة</h3>
            <div className="stat-number">{dashboardData?.completedTests || 0}</div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon">📈</div>
          <div className="stat-content">
            <h3>معدل الإكمال</h3>
            <div className="stat-number">{dashboardData?.completionRate?.toFixed(1) || 0}%</div>
          </div>
        </div>
      </div>

      {/* Charts Section */}
      <div className="charts-section">
        <div className="chart-row">
          <div className="chart-container">
            <h3>📊 توزيع الأعمار</h3>
            {getAgeDistributionChart() && (
              <Bar data={getAgeDistributionChart()} options={chartOptions} />
            )}
          </div>
          <div className="chart-container">
            <h3>👫 توزيع الجنس</h3>
            {getGenderDistributionChart() && (
              <Pie data={getGenderDistributionChart()} options={chartOptions} />
            )}
          </div>
        </div>
        
        <div className="chart-row">
          <div className="chart-container">
            <h3>🎓 توزيع المستوى التعليمي</h3>
            {getEducationDistributionChart() && (
              <Bar data={getEducationDistributionChart()} options={chartOptions} />
            )}
          </div>
          <div className="chart-container">
            <h3>📈 الإحصائيات اليومية (آخر 30 يوم)</h3>
            {getDailyStatsChart() && (
              <Line data={getDailyStatsChart()} options={chartOptions} />
            )}
          </div>
        </div>
      </div>

      {/* Participants Table */}
      <div className="participants-section">
        <div className="section-header">
          <h3>📋 بيانات المشاركين</h3>
          <div className="section-actions">
            <input
              type="text"
              placeholder="البحث في البيانات..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
            />
            <button onClick={handleExport} className="export-button">
              📥 تصدير البيانات
            </button>
          </div>
        </div>

        <div className="table-container">
          <table className="participants-table">
            <thead>
              <tr>
                <th>الاسم</th>
                <th>الجنس</th>
                <th>العمر</th>
                <th>المستوى التعليمي</th>
                <th>الحالة الاجتماعية</th>
                <th>الحالة</th>
                <th>تاريخ البدء</th>
                <th>عدد الإجابات</th>
              </tr>
            </thead>
            <tbody>
              {participants.map((participant) => (
                <tr key={participant.sessionId}>
                  <td>{participant.name}</td>
                  <td>{participant.gender}</td>
                  <td>{participant.age || '-'}</td>
                  <td>{participant.educationLevel || '-'}</td>
                  <td>{participant.maritalStatus || '-'}</td>
                  <td>
                    <span className={`status ${participant.status === 'مكتمل' ? 'completed' : 'incomplete'}`}>
                      {participant.status}
                    </span>
                  </td>
                  <td>{participant.completionDate || 'لم يكتمل بعد'}</td>
                  <td>{participant.questionsAnswered}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="pagination">
          <button
            onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
            disabled={currentPage === 1}
            className="pagination-button"
          >
            السابق
          </button>
          <span className="pagination-info">
            صفحة {currentPage} من {totalPages}
          </span>
          <button
            onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
            disabled={currentPage === totalPages}
            className="pagination-button"
          >
            التالي
          </button>
        </div>
      </div>

      {error && (
        <div className="error-notification">
          {error}
        </div>
      )}
    </div>
  );
};

export default AdminDashboard;
