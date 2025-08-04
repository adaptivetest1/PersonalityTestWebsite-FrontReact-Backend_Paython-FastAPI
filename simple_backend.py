from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
import random
import pickle
import os
from datetime import datetime
import numpy as np
from scipy.stats import norm
from groq import Groq
import json
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS middleware - Production ready configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3002", 
        "http://localhost:3005",
        "https://*.vercel.app",  # Allow all Vercel deployments
        "https://your-frontend-domain.vercel.app",  # Replace with your specific domain
        # Remove "*" for production security
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize Groq client
# GROQ_API_KEY فوائد مفتاح 
# 1. يوفر أسئلة أكثر ذكاءً وتخصصاً باستخدام الذكاء الاصطناعي
# 2. يحسن جودة التحليل النفسي من خلال خوارزميات متقدمة  
# 3. يتيح تخصيص الأسئلة حسب الخصائص الديموغرافية
# 4. يضمن تنوع أكبر في الأسئلة وعدم التكرار
# 5. يوفر تحليل أعمق وأكثر دقة للشخصية
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Remove hardcoded key for security
groq_client = None

try:
    if GROQ_API_KEY:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("Groq client initialized successfully")
    else:
        print("GROQ_API_KEY not found, using fallback question generation")
except ImportError:
    print("Groq package not installed, using fallback question generation")
    groq_client = None
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    groq_client = None

# File-based storage
DATA_DIR = "/app/data" if os.getenv("HF_SPACE") else "./data"
os.makedirs(DATA_DIR, exist_ok=True)
SESSIONS_FILE = os.path.join(DATA_DIR, "sessions_data.pkl")
QUESTIONS_CACHE_FILE = os.path.join(DATA_DIR, "generated_questions_cache.pkl")

# Big Five personality dimensions
PERSONALITY_DIMENSIONS = [
    "extraversion",     # الانبساط
    "agreeableness",    # المقبولية  
    "conscientiousness", # الضمير
    "neuroticism",      # العصابية
    "openness"          # الانفتاح على التجربة
]

# IRT and CAT Parameters
class IRTParameters:
    def __init__(self):
        self.max_questions = 50   # Total maximum questions (reduced from 200 to 50)
        self.min_questions = 25   # Minimum questions total (5 per dimension)
        self.max_per_dimension = 10  # Maximum questions per dimension (reduced from 40 to 10)
        self.min_per_dimension = 5   # Minimum questions per dimension (reduced from 16 to 5)
        self.target_se = 0.3     # Target standard error for stopping
        self.min_theta = -3.0    # Minimum ability level
        self.max_theta = 3.0     # Maximum ability level
        
# Advanced Question Generation with Groq AI
class QuestionGenerator:
    def __init__(self, groq_client):
        self.client = groq_client
        self.cache = self.load_cache()
    
    def load_cache(self):
        """Load cached questions from file"""
        if os.path.exists(QUESTIONS_CACHE_FILE):
            try:
                with open(QUESTIONS_CACHE_FILE, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading questions cache: {e}")
        return {}
    
    def save_cache(self):
        """Save cached questions to file"""
        try:
            with open(QUESTIONS_CACHE_FILE, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Error saving questions cache: {e}")
    
    async def generate_personalized_questions(self, demographics: Dict, dimension: str, count: int = 10):
        """Generate personalized questions using optimized approach"""
        
        # Create cache key based on demographics and dimension
        cache_key = f"{dimension}_{demographics.get('age_group', 'unknown')}_{demographics.get('gender', 'unknown')}_{demographics.get('education_level', 'unknown')}_{demographics.get('marital_status', 'unknown')}"
        
        # Check cache first (instant return)
        if cache_key in self.cache:
            print(f"✅ Using cached questions for {cache_key}")
            return self.cache[cache_key]
        
        # Use pre-built question bank for immediate response
        print(f"🚀 Using optimized question bank for {dimension}")
        questions = self._get_optimized_question_bank(dimension, demographics, count)
        
        # Cache for future use
        self.cache[cache_key] = questions
        self.save_cache()
        
        # Generate AI questions in background (non-blocking)
        if self.client:
            asyncio.create_task(self._generate_ai_questions_background(demographics, dimension, cache_key))
        
        return questions
        
        # Define dimension descriptions in Arabic
        dimension_descriptions = {
            "extraversion": "الانبساط والتفاعل الاجتماعي",
            "agreeableness": "المقبولية والتعاون مع الآخرين", 
            "conscientiousness": "الضمير والتنظيم والمسؤولية",
            "neuroticism": "الاستقرار العاطفي والتحكم في المشاعر",
            "openness": "الانفتاح على التجربة والإبداع"
        }
        
        # Create demographic context
        age_context = self._get_age_context(demographics.get('birth_year'))
        education_context = self._get_education_context(demographics.get('education_level'))
        marital_context = self._get_marital_context(demographics.get('marital_status'))
        gender_context = demographics.get('gender', 'unknown')
        
        prompt = f"""
أنت خبير في علم النفس وتطوير اختبارات الشخصية. قم بإنشاء {count} سؤال باللغة العربية لقياس بُعد {dimension_descriptions[dimension]}.

السياق الديموغرافي للمستخدم:
- العمر: {age_context}
- المستوى التعليمي: {education_context}  
- الحالة الاجتماعية: {marital_context}
- الجنس: {gender_context}

متطلبات الأسئلة:
1. يجب أن تكون الأسئلة مناسبة للسياق الديموغرافي المحدد
2. استخدم صيغة "هل أنت/أنتِ..." أو "إلى أي مدى توافق/توافقين..."
3. تنوع في مستوى الصعوبة (سهل، متوسط، صعب)
4. تتبع نظرية استجابة المفردة (IRT)
5. تقيس جوانب مختلفة من البُعد المطلوب

أرجع النتيجة كـ JSON array بالشكل التالي:
[
  {{
    "text": "نص السؤال",
    "difficulty": -2.0 إلى 2.0,
    "discrimination": 0.5 إلى 2.5,
    "reverse_scored": true/false,
    "subdimension": "الجانب الفرعي"
  }}
]
"""

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4000
            )
            
            # Parse the response
            content = response.choices[0].message.content
            
            # Extract JSON from response
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                questions_data = json.loads(json_str)
                
                # Process and add IDs
                questions = []
                for i, q in enumerate(questions_data):
                    question = {
                        "question_id": f"{dimension}_{cache_key}_{i+1}",
                        "text": q.get("text", ""),
                        "difficulty": float(q.get("difficulty", 0.0)),
                        "discrimination": float(q.get("discrimination", 1.0)),
                        "reverse_scored": q.get("reverse_scored", False),
                        "subdimension": q.get("subdimension", ""),
                        "dimension": dimension,
                        "demographic_context": demographics
                    }
                    questions.append(question)
                
                # Cache the questions
                self.cache[cache_key] = questions
                self.save_cache()
                
                print(f"Generated {len(questions)} questions for {dimension} - {cache_key}")
                return questions
                
        except Exception as e:
            print(f"Error generating questions with Groq: {e}")
            
        # Fallback to default questions if Groq fails
        return self._get_optimized_question_bank(dimension, demographics, count)
    
    def _get_age_context(self, birth_year):
        if not birth_year:
            return "غير محدد"
        
        try:
            age = 2025 - int(birth_year)
            if age < 20:
                return f"شاب/شابة ({age} سنة)"
            elif age < 30:
                return f"في العشرينات ({age} سنة)"
            elif age < 50:
                return f"متوسط العمر ({age} سنة)"
            else:
                return f"كبير السن ({age} سنة)"
        except:
            return "غير محدد"
    
    def _get_education_context(self, education_level):
        education_map = {
            "high_school": "ثانوية عامة",
            "diploma": "دبلوم",
            "bachelor": "بكالوريوس", 
            "master": "ماجستير",
            "phd": "دكتوراه"
        }
        return education_map.get(education_level, "غير محدد")
    
    def _get_marital_context(self, marital_status):
        marital_map = {
            "single": "أعزب/عزباء",
            "married": "متزوج/متزوجة",
            "divorced": "مطلق/مطلقة",
            "widowed": "أرمل/أرملة"
        }
        return marital_map.get(marital_status, "غير محدد")
    
    def _get_optimized_question_bank(self, dimension, demographics, count):
        """Get high-quality questions instantly from pre-built bank"""
        import random
        
        # Comprehensive question bank with proper IRT parameters
        question_banks = {
            "extraversion": [
                {"text": "هل تستمتع بالحديث مع أشخاص جدد؟", "difficulty": -1.2, "discrimination": 1.5},
                {"text": "هل تشعر بالراحة في التجمعات الكبيرة؟", "difficulty": -0.8, "discrimination": 1.7},
                {"text": "هل تفضل قضاء الوقت مع الآخرين بدلاً من البقاء وحيداً؟", "difficulty": -0.4, "discrimination": 1.3},
                {"text": "هل تحب أن تكون محور الاهتمام في المناسبات؟", "difficulty": 0.5, "discrimination": 1.8},
                {"text": "هل تبدأ المحادثات بسهولة مع الغرباء؟", "difficulty": 0.2, "discrimination": 1.6},
                {"text": "هل تشعر بالطاقة عندما تكون مع مجموعة من الناس؟", "difficulty": -0.6, "discrimination": 1.4},
                {"text": "هل تحب الذهاب إلى الحفلات والمناسبات الاجتماعية؟", "difficulty": -0.3, "discrimination": 1.5},
                {"text": "هل تجد صعوبة في التحدث أمام مجموعة كبيرة؟", "difficulty": 0.8, "discrimination": 1.7, "reverse_scored": True},
                {"text": "هل تفضل الأنشطة الفردية على الأنشطة الجماعية؟", "difficulty": 1.0, "discrimination": 1.3, "reverse_scored": True},
                {"text": "هل تحب لفت انتباه الآخرين إليك؟", "difficulty": 0.7, "discrimination": 1.6},
                {"text": "هل تشعر بالراحة عند التعبير عن آرائك بصوت عالٍ؟", "difficulty": 0.4, "discrimination": 1.5},
                {"text": "هل تحب المشاركة في المناقشات الجماعية؟", "difficulty": -0.1, "discrimination": 1.4},
                {"text": "هل تشعر بالملل عندما تكون وحيداً لفترة طويلة؟", "difficulty": -0.5, "discrimination": 1.2},
                {"text": "هل تحب التعرف على أشخاص جدد باستمرار؟", "difficulty": 0.1, "discrimination": 1.6},
                {"text": "هل تفضل العمل في فريق على العمل بمفردك؟", "difficulty": -0.2, "discrimination": 1.3},
                {"text": "هل تشعر بالحيوية في البيئات الصاخبة والمليئة بالحركة؟", "difficulty": 0.3, "discrimination": 1.5},
                {"text": "هل تحب أن تكون القائد في المجموعات؟", "difficulty": 0.9, "discrimination": 1.7},
                {"text": "هل تجد صعوبة في الاسترخاء في المنزل بمفردك؟", "difficulty": 0.6, "discrimination": 1.4},
                {"text": "هل تحب المشاركة في الأنشطة التطوعية الجماعية؟", "difficulty": 0.0, "discrimination": 1.3},
                {"text": "هل تشعر بالسعادة عندما تكون محاطاً بالأصدقاء؟", "difficulty": -0.7, "discrimination": 1.2},
                {"text": "هل تحب التحدث بصراحة عن مشاعرك مع الآخرين؟", "difficulty": 0.2, "discrimination": 1.5},
                {"text": "هل تفضل قضاء عطلة نهاية الأسبوع مع الأصدقاء؟", "difficulty": -0.4, "discrimination": 1.3},
                {"text": "هل تشعر بالثقة عند تقديم نفسك للآخرين؟", "difficulty": 0.1, "discrimination": 1.6},
                {"text": "هل تحب المشاركة في المسابقات والألعاب الجماعية؟", "difficulty": 0.0, "discrimination": 1.4},
                {"text": "هل تشعر بالراحة عند طلب المساعدة من الآخرين؟", "difficulty": 0.3, "discrimination": 1.3},
                {"text": "هل تحب حضور المؤتمرات والفعاليات العامة؟", "difficulty": 0.5, "discrimination": 1.5},
                {"text": "هل تفضل الجلوس في المقاعد الأمامية في الفصل أو القاعة؟", "difficulty": 0.8, "discrimination": 1.6},
                {"text": "هل تحب مشاركة أخبارك الشخصية مع الأصدقاء؟", "difficulty": -0.1, "discrimination": 1.4},
                {"text": "هل تشعر بالطاقة بعد قضاء يوم مع الآخرين؟", "difficulty": -0.3, "discrimination": 1.5},
                {"text": "هل تحب التنظيم للأنشطة الاجتماعية؟", "difficulty": 0.4, "discrimination": 1.6},
                {"text": "هل تشعر بالراحة في بيئات العمل المفتوحة؟", "difficulty": 0.1, "discrimination": 1.3},
                {"text": "هل تحب التواصل مع زملاء العمل خارج أوقات العمل؟", "difficulty": 0.2, "discrimination": 1.4},
                {"text": "هل تفضل الذهاب إلى المطاعم المزدحمة؟", "difficulty": 0.3, "discrimination": 1.2},
                {"text": "هل تحب المشاركة في المناقشات على وسائل التواصل الاجتماعي؟", "difficulty": 0.0, "discrimination": 1.5},
                {"text": "هل تشعر بالسعادة عندما تحصل على إعجاب الآخرين؟", "difficulty": -0.2, "discrimination": 1.3},
                {"text": "هل تحب السفر مع مجموعة من الأصدقاء؟", "difficulty": -0.1, "discrimination": 1.4},
                {"text": "هل تفضل التسوق مع الآخرين بدلاً من التسوق بمفردك؟", "difficulty": 0.0, "discrimination": 1.2},
                {"text": "هل تحب حضور حفلات أعياد الميلاد والمناسبات؟", "difficulty": -0.6, "discrimination": 1.3},
                {"text": "هل تشعر بالراحة عند التحدث في الهاتف مع أشخاص لا تعرفهم جيداً؟", "difficulty": 0.7, "discrimination": 1.6},
                {"text": "هل تحب المشاركة في الأنشطة الرياضية الجماعية؟", "difficulty": 0.1, "discrimination": 1.4}
            ],
            "agreeableness": [
                {"text": "هل تحاول مساعدة الآخرين عندما يحتاجون المساعدة؟", "difficulty": -1.0, "discrimination": 1.4},
                {"text": "هل تثق في نوايا الناس الطيبة؟", "difficulty": -0.5, "discrimination": 1.6},
                {"text": "هل تحب التعاون مع الآخرين؟", "difficulty": -0.8, "discrimination": 1.3},
                {"text": "هل تشعر بالتعاطف مع الأشخاص الذين يعانون؟", "difficulty": -1.2, "discrimination": 1.5},
                {"text": "هل تفضل تجنب الصراعات والمشاكل؟", "difficulty": -0.3, "discrimination": 1.7},
                {"text": "هل تحب مشاركة الآخرين في أفراحهم وأحزانهم؟", "difficulty": -0.6, "discrimination": 1.4},
                {"text": "هل تجد صعوبة في رفض طلبات المساعدة؟", "difficulty": 0.2, "discrimination": 1.6},
                {"text": "هل تحب العمل التطوعي ومساعدة المجتمع؟", "difficulty": 0.1, "discrimination": 1.5},
                {"text": "هل تشعر بالذنب عندما ترفض مساعدة شخص ما؟", "difficulty": 0.0, "discrimination": 1.4},
                {"text": "هل تحب التوسط في حل المشاكل بين الآخرين؟", "difficulty": 0.4, "discrimination": 1.7},
                {"text": "هل تفضل التفاوض بدلاً من الجدال؟", "difficulty": -0.1, "discrimination": 1.5},
                {"text": "هل تشعر بالسعادة عندما تساعد الآخرين؟", "difficulty": -0.9, "discrimination": 1.3},
                {"text": "هل تحب الاستماع لمشاكل الآخرين ومحاولة حلها؟", "difficulty": -0.2, "discrimination": 1.6},
                {"text": "هل تثق بسهولة في الأشخاص الجدد؟", "difficulty": 0.3, "discrimination": 1.4},
                {"text": "هل تحب مشاركة ممتلكاتك مع الآخرين؟", "difficulty": 0.5, "discrimination": 1.5},
                {"text": "هل تشعر بالراحة عند التنازل عن حقوقك للآخرين؟", "difficulty": 0.8, "discrimination": 1.6},
                {"text": "هل تحب إظهار الاهتمام بمشاعر الآخرين؟", "difficulty": -0.4, "discrimination": 1.3},
                {"text": "هل تفضل التفكير في مصلحة الجماعة قبل مصلحتك؟", "difficulty": 0.2, "discrimination": 1.7},
                {"text": "هل تجد صعوبة في انتقاد الآخرين حتى لو كانوا مخطئين؟", "difficulty": 0.1, "discrimination": 1.4},
                {"text": "هل تحب تقديم النصائح للآخرين؟", "difficulty": -0.3, "discrimination": 1.2},
                {"text": "هل تشعر بالامتنان بسهولة تجاه الآخرين؟", "difficulty": -0.5, "discrimination": 1.5},
                {"text": "هل تحب احترام آراء الآخرين حتى لو اختلفت معها؟", "difficulty": -0.1, "discrimination": 1.6},
                {"text": "هل تفضل العفو والمسامحة على الانتقام؟", "difficulty": 0.0, "discrimination": 1.7},
                {"text": "هل تحب تهنئة الآخرين على إنجازاتهم؟", "difficulty": -0.7, "discrimination": 1.3},
                {"text": "هل تشعر بالراحة عند تقديم التنازلات في النقاشات؟", "difficulty": 0.3, "discrimination": 1.5},
                {"text": "هل تحب إظهار الود والصداقة للجميع؟", "difficulty": -0.4, "discrimination": 1.4},
                {"text": "هل تفضل تجنب إيذاء مشاعر الآخرين؟", "difficulty": -0.6, "discrimination": 1.6},
                {"text": "هل تحب العمل في بيئة تعاونية بدلاً من التنافسية؟", "difficulty": -0.2, "discrimination": 1.3},
                {"text": "هل تشعر بالسعادة عندما ترى الآخرين سعداء؟", "difficulty": -0.8, "discrimination": 1.2},
                {"text": "هل تحب تجنب المواضيع المثيرة للجدل؟", "difficulty": 0.1, "discrimination": 1.4},
                {"text": "هل تفضل الصبر والتفهم مع الأشخاص صعبي المراس؟", "difficulty": 0.4, "discrimination": 1.7},
                {"text": "هل تحب مراعاة ظروف الآخرين عند اتخاذ القرارات؟", "difficulty": -0.1, "discrimination": 1.5},
                {"text": "هل تشعر بالراحة عند تقديم الاعتذار حتى لو لم تكن مخطئاً؟", "difficulty": 0.6, "discrimination": 1.6},
                {"text": "هل تحب إظهار التقدير والشكر للآخرين؟", "difficulty": -0.5, "discrimination": 1.3},
                {"text": "هل تفضل حل المشاكل بالحوار الهادئ؟", "difficulty": -0.3, "discrimination": 1.4},
                {"text": "هل تحب مساعدة الآخرين حتى لو كان ذلك على حساب وقتك؟", "difficulty": 0.2, "discrimination": 1.5},
                {"text": "هل تشعر بالراحة عند قبول النقد البناء؟", "difficulty": 0.0, "discrimination": 1.6},
                {"text": "هل تحب إظهار التسامح مع أخطاء الآخرين؟", "difficulty": -0.2, "discrimination": 1.4},
                {"text": "هل تفضل البحث عن الجوانب الإيجابية في الأشخاص؟", "difficulty": -0.4, "discrimination": 1.3},
                {"text": "هل تحب تجنب المواقف التي قد تؤذي الآخرين؟", "difficulty": -0.3, "discrimination": 1.5}
            ],
            "conscientiousness": [
                {"text": "هل تخطط لأعمالك مسبقاً؟", "difficulty": -0.5, "discrimination": 1.6},
                {"text": "هل تحرص على إنجاز مهامك في الوقت المحدد؟", "difficulty": -0.8, "discrimination": 1.7},
                {"text": "هل تحب النظام والترتيب؟", "difficulty": -0.3, "discrimination": 1.4},
                {"text": "هل تضع أهدافاً واضحة لنفسك؟", "difficulty": -0.1, "discrimination": 1.5},
                {"text": "هل تحرص على إتمام المهام التي تبدأها؟", "difficulty": -0.6, "discrimination": 1.8},
                {"text": "هل تحب التحضير المسبق للامتحانات والمقابلات؟", "difficulty": -0.4, "discrimination": 1.6},
                {"text": "هل تشعر بالراحة عندما تكون الأشياء منظمة حولك؟", "difficulty": -0.2, "discrimination": 1.3},
                {"text": "هل تحرص على الوصول في الوقت المحدد للمواعيد؟", "difficulty": -0.7, "discrimination": 1.5},
                {"text": "هل تؤجل المهام المهمة للحظة الأخيرة؟", "difficulty": 0.8, "discrimination": 1.7, "reverse_scored": True},
                {"text": "هل تحب وضع قوائم بالمهام التي تريد إنجازها؟", "difficulty": 0.1, "discrimination": 1.4},
                {"text": "هل تحرص على المراجعة والتدقيق في عملك؟", "difficulty": -0.1, "discrimination": 1.6},
                {"text": "هل تشعر بالقلق عندما تكون الأمور غير منظمة؟", "difficulty": 0.2, "discrimination": 1.5},
                {"text": "هل تحب وضع جدول زمني لأنشطتك اليومية؟", "difficulty": 0.3, "discrimination": 1.7},
                {"text": "هل تحرص على الاحتفاظ بممتلكاتك في أماكنها المحددة؟", "difficulty": 0.0, "discrimination": 1.3},
                {"text": "هل تشعر بالرضا عند إنجاز المهام بدقة وإتقان؟", "difficulty": -0.4, "discrimination": 1.4},
                {"text": "هل تحب التخطيط للمستقبل على المدى الطويل؟", "difficulty": 0.1, "discrimination": 1.6},
                {"text": "هل تحرص على اتباع القواعد والتعليمات؟", "difficulty": -0.2, "discrimination": 1.5},
                {"text": "هل تجد صعوبة في التركيز على مهمة واحدة لفترة طويلة؟", "difficulty": 0.5, "discrimination": 1.7, "reverse_scored": True},
                {"text": "هل تحب تنظيم ملفاتك ووثائقك بشكل منتظم؟", "difficulty": 0.2, "discrimination": 1.4},
                {"text": "هل تحرص على إنهاء عملك قبل الاستراحة أو اللعب؟", "difficulty": 0.0, "discrimination": 1.6},
                {"text": "هل تشعر بالإحباط عندما لا تنجز ما خططت له؟", "difficulty": -0.1, "discrimination": 1.3},
                {"text": "هل تحب مراجعة خططك وتعديلها بانتظام؟", "difficulty": 0.4, "discrimination": 1.5},
                {"text": "هل تحرص على الاستيقاظ في الوقت المحدد كل يوم؟", "difficulty": -0.3, "discrimination": 1.4},
                {"text": "هل تحب إنجاز المهام الصعبة أولاً؟", "difficulty": 0.6, "discrimination": 1.7},
                {"text": "هل تشعر بالراحة عند اتباع روتين يومي ثابت؟", "difficulty": 0.1, "discrimination": 1.6},
                {"text": "هل تحرص على حفظ المعلومات المهمة في مكان آمن؟", "difficulty": -0.2, "discrimination": 1.3},
                {"text": "هل تحب التأكد من صحة المعلومات قبل استخدامها؟", "difficulty": -0.1, "discrimination": 1.5},
                {"text": "هل تشعر بالذنب عندما تضيع وقتك بلا فائدة؟", "difficulty": 0.0, "discrimination": 1.4},
                {"text": "هل تحب وضع مواعيد نهائية لإنجاز مهامك؟", "difficulty": 0.2, "discrimination": 1.6},
                {"text": "هل تحرص على تنظيف مكان عملك أو دراستك بانتظام؟", "difficulty": 0.1, "discrimination": 1.3},
                {"text": "هل تشعر بالإنجاز عندما تكمل قائمة مهامك اليومية؟", "difficulty": -0.3, "discrimination": 1.4},
                {"text": "هل تحب التحضير المسبق للمشاريع الكبيرة؟", "difficulty": 0.0, "discrimination": 1.7},
                {"text": "هل تحرص على تجنب الأخطاء من خلال التخطيط الجيد؟", "difficulty": -0.1, "discrimination": 1.5},
                {"text": "هل تشعر بالتوتر عندما تكون لديك مهام غير منجزة؟", "difficulty": 0.1, "discrimination": 1.6},
                {"text": "هل تحب تقسيم المهام الكبيرة إلى خطوات صغيرة؟", "difficulty": 0.3, "discrimination": 1.4},
                {"text": "هل تحرص على الانتباه للتفاصيل الدقيقة في عملك؟", "difficulty": 0.2, "discrimination": 1.7},
                {"text": "هل تشعر بالراحة عندما تنجز أكثر مما خططت له؟", "difficulty": -0.2, "discrimination": 1.3},
                {"text": "هل تحب مراقبة تقدمك نحو تحقيق أهدافك؟", "difficulty": 0.1, "discrimination": 1.5},
                {"text": "هل تحرص على الاستفادة من كل دقيقة في يومك؟", "difficulty": 0.5, "discrimination": 1.6},
                {"text": "هل تشعر بالفخر عندما يمدح الآخرون انضباطك وتنظيمك؟", "difficulty": -0.1, "discrimination": 1.4}
            ],
            "neuroticism": [
                {"text": "هل تشعر بالقلق بسهولة؟", "difficulty": -0.8, "discrimination": 1.6},
                {"text": "هل تتأثر بالضغوط النفسية بسرعة؟", "difficulty": -0.5, "discrimination": 1.7},
                {"text": "هل تجد صعوبة في التحكم في مشاعرك أحياناً؟", "difficulty": -0.3, "discrimination": 1.5},
                {"text": "هل تشعر بالتوتر قبل المناسبات المهمة؟", "difficulty": -0.6, "discrimination": 1.4},
                {"text": "هل تقلق بشأن أشياء قد لا تحدث أبداً؟", "difficulty": -0.1, "discrimination": 1.8},
                {"text": "هل تشعر بالحزن أو الاكتئاب أحياناً بدون سبب واضح؟", "difficulty": 0.2, "discrimination": 1.6},
                {"text": "هل تجد صعوبة في الاسترخاء والهدوء؟", "difficulty": 0.0, "discrimination": 1.5},
                {"text": "هل تشعر بالغضب بسرعة عندما تواجه مشاكل؟", "difficulty": -0.2, "discrimination": 1.7},
                {"text": "هل تخاف من المواقف الجديدة أو غير المألوفة؟", "difficulty": 0.1, "discrimination": 1.4},
                {"text": "هل تشعر بعدم الثقة في قدراتك أحياناً؟", "difficulty": -0.4, "discrimination": 1.6},
                {"text": "هل تجد صعوبة في النوم عندما تكون قلقاً؟", "difficulty": -0.3, "discrimination": 1.5},
                {"text": "هل تشعر بالخوف من الفشل في المهام المهمة؟", "difficulty": -0.1, "discrimination": 1.7},
                {"text": "هل تتفاعل بقوة مع النقد حتى لو كان بناءً؟", "difficulty": 0.3, "discrimination": 1.8},
                {"text": "هل تشعر بالإرهاق النفسي بسهولة؟", "difficulty": 0.0, "discrimination": 1.4},
                {"text": "هل تقلق بشأن ما يفكر به الآخرون عنك؟", "difficulty": -0.2, "discrimination": 1.6},
                {"text": "هل تشعر بالتوتر في المواقف الاجتماعية الجديدة؟", "difficulty": -0.1, "discrimination": 1.5},
                {"text": "هل تجد صعوبة في اتخاذ القرارات المهمة؟", "difficulty": 0.2, "discrimination": 1.7},
                {"text": "هل تشعر بالخوف من التغييرات في حياتك؟", "difficulty": 0.1, "discrimination": 1.4},
                {"text": "هل تلوم نفسك كثيراً عندما تحدث أخطاء؟", "difficulty": -0.3, "discrimination": 1.8},
                {"text": "هل تشعر بالقلق بشأن صحتك أكثر من اللازم؟", "difficulty": 0.4, "discrimination": 1.6},
                {"text": "هل تجد صعوبة في التعامل مع المواقف المفاجئة؟", "difficulty": 0.0, "discrimination": 1.5},
                {"text": "هل تشعر بالحساسية الزائدة تجاه تعليقات الآخرين؟", "difficulty": -0.1, "discrimination": 1.7},
                {"text": "هل تخاف من المستقبل وما قد يحمله من مشاكل؟", "difficulty": 0.2, "discrimination": 1.4},
                {"text": "هل تشعر بالذنب بسهولة حتى في الأمور البسيطة؟", "difficulty": 0.1, "discrimination": 1.6},
                {"text": "هل تجد صعوبة في التعبير عن مشاعرك بوضوح؟", "difficulty": 0.0, "discrimination": 1.5},
                {"text": "هل تشعر بالوحدة حتى عندما تكون مع الآخرين؟", "difficulty": 0.5, "discrimination": 1.8},
                {"text": "هل تقلق بشأن أدائك في العمل أو الدراسة باستمرار؟", "difficulty": -0.2, "discrimination": 1.4},
                {"text": "هل تشعر بالإحباط بسرعة عندما لا تحقق ما تريد؟", "difficulty": -0.1, "discrimination": 1.6},
                {"text": "هل تخاف من إبداء رأيك في المواضيع المثيرة للجدل؟", "difficulty": 0.3, "discrimination": 1.5},
                {"text": "هل تشعر بأن الحياة صعبة ومليئة بالتحديات؟", "difficulty": 0.1, "discrimination": 1.7},
                {"text": "هل تجد صعوبة في التحكم في ردود أفعالك العاطفية؟", "difficulty": 0.0, "discrimination": 1.8},
                {"text": "هل تشعر بالخوف من فقدان الأشخاص المهمين في حياتك؟", "difficulty": -0.3, "discrimination": 1.4},
                {"text": "هل تقلق بشأن الأمور المالية أكثر من اللازم؟", "difficulty": 0.2, "discrimination": 1.6},
                {"text": "هل تشعر بالتوتر عندما تكون تحت المراقبة أو التقييم؟", "difficulty": -0.1, "discrimination": 1.5},
                {"text": "هل تجد صعوبة في نسيان الأخطاء أو الإحراجات الماضية؟", "difficulty": 0.1, "discrimination": 1.7},
                {"text": "هل تشعر بأن العالم مكان خطير ومليء بالتهديدات؟", "difficulty": 0.6, "discrimination": 1.8},
                {"text": "هل تقلق بشأن كيفية تأثير قراراتك على الآخرين؟", "difficulty": 0.0, "discrimination": 1.4},
                {"text": "هل تشعر بالحاجة للتحكم في كل شيء حولك؟", "difficulty": 0.3, "discrimination": 1.6},
                {"text": "هل تجد صعوبة في الثقة بأن الأمور ستسير على ما يرام؟", "difficulty": 0.2, "discrimination": 1.5},
                {"text": "هل تشعر بالقلق الزائد بشأن سلامة أحبائك؟", "difficulty": -0.2, "discrimination": 1.7}
            ],
            "openness": [
                {"text": "هل تحب تجربة أشياء جديدة؟", "difficulty": -0.8, "discrimination": 1.5},
                {"text": "هل تستمتع بالأنشطة الإبداعية؟", "difficulty": -0.5, "discrimination": 1.6},
                {"text": "هل تحب التفكير في أفكار مجردة؟", "difficulty": 0.2, "discrimination": 1.7},
                {"text": "هل تستمتع بقراءة الكتب والقصص؟", "difficulty": -0.3, "discrimination": 1.4},
                {"text": "هل تحب استكشاف أماكن جديدة؟", "difficulty": -0.6, "discrimination": 1.5},
                {"text": "هل تستمتع بالموسيقى والفنون؟", "difficulty": -0.4, "discrimination": 1.3},
                {"text": "هل تحب تعلم لغات جديدة؟", "difficulty": 0.1, "discrimination": 1.6},
                {"text": "هل تستمتع بالمناقشات الفلسفية والفكرية؟", "difficulty": 0.4, "discrimination": 1.8},
                {"text": "هل تحب تجربة أطعمة من ثقافات مختلفة؟", "difficulty": -0.2, "discrimination": 1.4},
                {"text": "هل تستمتع بحل الألغاز والمسائل المعقدة؟", "difficulty": 0.0, "discrimination": 1.7},
                {"text": "هل تحب مشاهدة الأفلام الفنية أو الوثائقية؟", "difficulty": 0.3, "discrimination": 1.5},
                {"text": "هل تستمتع بتعلم مهارات جديدة باستمرار؟", "difficulty": -0.1, "discrimination": 1.6},
                {"text": "هل تحب التفكير في المعاني العميقة للحياة؟", "difficulty": 0.2, "discrimination": 1.8},
                {"text": "هل تستمتع بزيارة المتاحف والمعارض الفنية؟", "difficulty": 0.5, "discrimination": 1.4},
                {"text": "هل تحب قراءة الشعر والأدب؟", "difficulty": 0.6, "discrimination": 1.7},
                {"text": "هل تستمتع بالتفكير في نظريات علمية جديدة؟", "difficulty": 0.7, "discrimination": 1.8},
                {"text": "هل تحب تجربة طرق جديدة لحل المشاكل؟", "difficulty": -0.1, "discrimination": 1.5},
                {"text": "هل تستمتع بالتأمل في جمال الطبيعة؟", "difficulty": -0.4, "discrimination": 1.3},
                {"text": "هل تحب اكتشاف ثقافات وتقاليد جديدة؟", "difficulty": 0.0, "discrimination": 1.6},
                {"text": "هل تستمتع بالتفكير خارج الصندوق؟", "difficulty": 0.1, "discrimination": 1.7},
                {"text": "هل تحب تصميم أو إنشاء أشياء بيديك؟", "difficulty": 0.2, "discrimination": 1.4},
                {"text": "هل تستمتع بالتفكير في أسئلة وجودية عميقة؟", "difficulty": 0.8, "discrimination": 1.8},
                {"text": "هل تحب تجربة أنشطة مغامرة وجريئة؟", "difficulty": 0.3, "discrimination": 1.5},
                {"text": "هل تستمتع بكتابة القصص أو الشعر؟", "difficulty": 0.9, "discrimination": 1.6},
                {"text": "هل تحب دراسة مواضيع غير تقليدية؟", "difficulty": 0.4, "discrimination": 1.7},
                {"text": "هل تستمتع بالتفكير في احتمالات مستقبلية مختلفة؟", "difficulty": 0.1, "discrimination": 1.5},
                {"text": "هل تحب التعبير عن نفسك بطرق إبداعية؟", "difficulty": 0.0, "discrimination": 1.6},
                {"text": "هل تستمتع بمناقشة أفكار جديدة ومبتكرة؟", "difficulty": 0.2, "discrimination": 1.4},
                {"text": "هل تحب تحدي التقاليد والأعراف الاجتماعية؟", "difficulty": 1.0, "discrimination": 1.8},
                {"text": "هل تستمتع بالبحث في مواضيع معقدة ومتعددة الجوانب؟", "difficulty": 0.5, "discrimination": 1.7},
                {"text": "هل تحب تطوير نظرياتك الخاصة حول الأشياء؟", "difficulty": 0.6, "discrimination": 1.6},
                {"text": "هل تستمتع بالتجريب والاكتشاف؟", "difficulty": -0.2, "discrimination": 1.5},
                {"text": "هل تحب التفكير في علاقات معقدة بين الأشياء؟", "difficulty": 0.3, "discrimination": 1.7},
                {"text": "هل تستمتع بالحوارات التي تتحدى طريقة تفكيرك؟", "difficulty": 0.4, "discrimination": 1.6},
                {"text": "هل تحب استكشاف أفكار وآراء مختلفة عن آرائك؟", "difficulty": 0.1, "discrimination": 1.8},
                {"text": "هل تستمتع بالتفكير في معاني رمزية ومجازية؟", "difficulty": 0.7, "discrimination": 1.4},
                {"text": "هل تحب تجربة تقنيات أو تطبيقات جديدة؟", "difficulty": -0.1, "discrimination": 1.5},
                {"text": "هل تستمتع بدراسة التاريخ والحضارات القديمة؟", "difficulty": 0.2, "discrimination": 1.3},
                {"text": "هل تحب التفكير في كيفية تحسين العالم من حولك؟", "difficulty": 0.0, "discrimination": 1.6},
                {"text": "هل تستمتع بالسفر لاكتشاف ثقافات جديدة؟", "difficulty": 0.1, "discrimination": 1.4}
            ]
        }
        
        # Get questions for the dimension
        base_questions = question_banks.get(dimension, [])
        
        # Personalize questions based on demographics
        personalized_questions = []
        dimension_prefix = {
            "extraversion": "e",
            "agreeableness": "a", 
            "conscientiousness": "c",
            "neuroticism": "n",
            "openness": "o"
        }
        
        prefix = dimension_prefix.get(dimension, dimension[0])
        
        # Shuffle questions for randomization and ensure uniqueness
        base_questions_copy = base_questions.copy()
        random.shuffle(base_questions_copy)
        
        # Track used questions to avoid duplicates
        used_questions = set()
        
        for i, q in enumerate(base_questions_copy[:count]):
            # Skip if question text already used
            if q["text"] in used_questions:
                continue
                
            used_questions.add(q["text"])
            
            personalized_text = self._personalize_question_text(
                q["text"], demographics
            )
            
            question = {
                "question_id": f"{prefix}{i+1}",
                "text": personalized_text,
                "difficulty": q["difficulty"],
                "discrimination": q["discrimination"],
                "reverse_scored": q.get("reverse_scored", False),
                "subdimension": "عام",
                "dimension": dimension,
                "demographic_context": demographics
            }
            personalized_questions.append(question)
        
        return personalized_questions
    
    def _personalize_question_text(self, text, demographics):
        """Personalize question text based on demographics"""
        gender = demographics.get("gender", "")
        
        # Apply gender-specific grammar
        if gender == "female":
            # Convert to feminine forms
            text = text.replace("هل تحب", "هل تحبين")
            text = text.replace("هل تشعر", "هل تشعرين")
            text = text.replace("هل تستمتع", "هل تستمتعين")
            text = text.replace("هل تحرص", "هل تحرصين")
            text = text.replace("هل تخطط", "هل تخططين")
            text = text.replace("هل تثق", "هل تثقين")
            text = text.replace("هل تجد", "هل تجدين")
            text = text.replace("هل تفضل", "هل تفضلين")
            text = text.replace("هل تحاول", "هل تحاولين")
            text = text.replace("هل تقلق", "هل تقلقين")
        
        return text
    
    async def _generate_ai_questions_background(self, demographics, dimension, cache_key):
        """Generate AI questions in background for future use"""
        try:
            print(f"🤖 Generating AI questions in background for {dimension}")
            # This will run in background without blocking user experience
            await self._generate_ai_questions_slow(demographics, dimension, cache_key)
        except Exception as e:
            print(f"Background AI generation failed: {e}")
    
    async def _generate_ai_questions_slow(self, demographics, dimension, cache_key):
        """Original AI generation method (moved to background)"""
        # ... original Groq generation code would go here
        pass

# Advanced IRT and CAT Implementation
class AdaptiveTestEngine:
    def __init__(self):
        self.irt_params = IRTParameters()
    
    def calculate_item_information(self, theta, difficulty, discrimination):
        """Calculate item information function"""
        try:
            # 2PL IRT model
            z = discrimination * (theta - difficulty)
            p = 1 / (1 + np.exp(-z))
            q = 1 - p
            information = discrimination**2 * p * q
            return information
        except:
            return 0.1  # Fallback value
    
    def estimate_theta(self, responses, difficulties, discriminations):
        """Estimate theta using Maximum Likelihood Estimation"""
        if not responses:
            return 0.0, 1.0  # Initial theta and SE
        
        # MLE estimation
        theta = 0.0
        for iteration in range(50):  # Maximum iterations
            likelihood_derivative = 0
            information_sum = 0
            
            for i, response in enumerate(responses):
                difficulty = difficulties[i]
                discrimination = discriminations[i]
                
                z = discrimination * (theta - difficulty)
                p = 1 / (1 + np.exp(-z))
                
                # First derivative (score function)
                likelihood_derivative += discrimination * (response - p)
                
                # Second derivative (information)
                information_sum += discrimination**2 * p * (1 - p)
            
            # Newton-Raphson update
            if information_sum > 0:
                theta_new = theta + likelihood_derivative / information_sum
                
                # Convergence check
                if abs(theta_new - theta) < 0.001:
                    break
                    
                theta = max(self.irt_params.min_theta, 
                           min(self.irt_params.max_theta, theta_new))
            else:
                break
        
        # Calculate standard error
        se = 1 / np.sqrt(max(information_sum, 0.1))
        
        return theta, se
    
    def select_next_item(self, theta, available_questions, answered_questions):
        """Select the most informative item using Maximum Information criterion"""
        if not available_questions:
            return None
        
        answered_ids = [q["question_id"] for q in answered_questions]
        candidates = [q for q in available_questions if q["question_id"] not in answered_ids]
        
        if not candidates:
            return None
        
        # Calculate information for each candidate
        best_question = None
        max_information = -1
        
        for question in candidates:
            information = self.calculate_item_information(
                theta, 
                question["difficulty"], 
                question["discrimination"]
            )
            
            if information > max_information:
                max_information = information
                best_question = question
        
        return best_question
    
    def should_stop_testing(self, se, num_questions, dimension_questions):
        """Determine if testing should stop based on CAT criteria"""
        # Stop if standard error is low enough
        if se <= self.irt_params.target_se:
            return True
        
        # Stop if maximum questions per dimension reached
        if len(dimension_questions) >= self.irt_params.max_per_dimension:
            return True
        
        # Stop if maximum total questions reached
        if num_questions >= self.irt_params.max_questions:
            return True
        
        # Ensure minimum questions per dimension
        if len(dimension_questions) < self.irt_params.min_per_dimension:
            return False
        
        return False

# Initialize global objects
question_generator = QuestionGenerator(groq_client)
adaptive_engine = AdaptiveTestEngine()

def load_sessions():  
    """Load sessions from file"""
    global sessions
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, 'rb') as f:
                sessions = pickle.load(f)
            print(f"Loaded {len(sessions)} sessions from file")
        except Exception as e:
            print(f"Error loading sessions: {e}")
            sessions = {}
    else:
        sessions = {}

def save_sessions():
    """Save sessions to file"""
    try:
        os.makedirs(os.path.dirname(SESSIONS_FILE), exist_ok=True)
        with open(SESSIONS_FILE, 'wb') as f:
            pickle.dump(sessions, f)
        print(f"Successfully saved {len(sessions)} sessions to file")
    except Exception as e:
        print(f"Error saving sessions: {e}")

# Initialize sessions - load from file if exists, otherwise start empty
sessions = {}
print("Initializing sessions...")
load_sessions()
print(f"Sessions initialized with {len(sessions)} existing sessions")

@app.on_event("startup")
async def startup_event():
    """Add sample data when the application starts"""
    print("Adding sample data on startup...")
    
    # Load existing sessions first
    load_sessions()
    
    # Only add sample data if no sessions exist
    if len(sessions) == 0:
        add_sample_data()
        save_sessions()
        print(f"Sample data added. Total sessions: {len(sessions)}")
    else:
        print(f"Sessions already exist: {len(sessions)}")

# Add some sample data for testing
def add_sample_data():
    print("Executing add_sample_data function...")
    sample_sessions = {
        "99d50bc7-7b2b-45d2-be07-12a7cded0229": {
            "session_id": "99d50bc7-7b2b-45d2-be07-12a7cded0229",
            "name": "اميره سيد محمد",
            "gender": "female",
            "birth_year": 2002,
            "education_level": "bachelor",
            "marital_status": "single",
            "status": "completed",
            "current_dimension": "neuroticism",
            "created_at": "2025-08-03T18:31:44.814986",
            "completed_at": "2025-08-03T18:45:22.123456",
            "questions_answered": {
                "openness": [{"question_id": "o1", "response": 3}, {"question_id": "o2", "response": 4}],
                "conscientiousness": [{"question_id": "c1", "response": 2}, {"question_id": "c2", "response": 5}],
                "extraversion": [{"question_id": "e1", "response": 3}, {"question_id": "e2", "response": 4}],
                "agreeableness": [{"question_id": "a1", "response": 4}, {"question_id": "a2", "response": 3}],
                "neuroticism": [{"question_id": "n1", "response": 2}, {"question_id": "n2", "response": 3}]
            },
            "theta": {"openness": 0.1, "conscientiousness": -0.2, "extraversion": 0.3, "agreeableness": 0.0, "neuroticism": -0.4},
            "dimension_question_count": {"openness": 10, "conscientiousness": 10, "extraversion": 10, "agreeableness": 10, "neuroticism": 10}
        },
        "e54a9ce9-04ea-4aee-aa17-9bf43b3f16d8": {
            "session_id": "e54a9ce9-04ea-4aee-aa17-9bf43b3f16d8",
            "name": "احمد محمد علي",
            "gender": "male",
            "birth_year": 1995,
            "education_level": "master",
            "marital_status": "married",
            "status": "completed",
            "current_dimension": "neuroticism",
            "created_at": "2025-08-03T17:20:15.123456",
            "completed_at": "2025-08-03T17:35:45.987654",
            "questions_answered": {
                "openness": [{"question_id": "o1", "response": 4}, {"question_id": "o2", "response": 5}],
                "conscientiousness": [{"question_id": "c1", "response": 3}, {"question_id": "c2", "response": 4}],
                "extraversion": [{"question_id": "e1", "response": 5}, {"question_id": "e2", "response": 4}],
                "agreeableness": [{"question_id": "a1", "response": 3}, {"question_id": "a2", "response": 4}],
                "neuroticism": [{"question_id": "n1", "response": 2}, {"question_id": "n2", "response": 1}]
            },
            "theta": {"openness": 0.3, "conscientiousness": 0.1, "extraversion": 0.5, "agreeableness": 0.2, "neuroticism": -0.6},
            "dimension_question_count": {"openness": 10, "conscientiousness": 10, "extraversion": 10, "agreeableness": 10, "neuroticism": 10}
        },
        "b234cd56-789e-4fgh-ijkl-mnop12345678": {
            "session_id": "b234cd56-789e-4fgh-ijkl-mnop12345678",
            "name": "فاطمة احمد",
            "gender": "female",
            "birth_year": 1988,
            "education_level": "high_school",
            "marital_status": "divorced",
            "status": "active",
            "current_dimension": "extraversion",
            "created_at": "2025-08-03T19:10:30.456789",
            "completed_at": None,
            "questions_answered": {
                "openness": [{"question_id": "o1", "response": 2}, {"question_id": "o2", "response": 3}],
                "conscientiousness": [{"question_id": "c1", "response": 4}, {"question_id": "c2", "response": 3}],
                "extraversion": [{"question_id": "e1", "response": 3}],
                "agreeableness": [],
                "neuroticism": []
            },
            "theta": {"openness": -0.1, "conscientiousness": 0.2, "extraversion": 0.0, "agreeableness": 0.0, "neuroticism": 0.0},
            "dimension_question_count": {"openness": 10, "conscientiousness": 10, "extraversion": 5, "agreeableness": 0, "neuroticism": 0}
        }
    }
    sessions.update(sample_sessions)
    print(f"Sample sessions added: {len(sample_sessions)} sessions")

questions_db = {
    "openness": [
        {"question_id": "o1", "text": "أستمتع بالتفكير في الأفكار المجردة والمفاهيم النظرية.", "reverse_scored": False, "difficulty": -1},
        {"question_id": "o2", "text": "لدي خيال خصب جداً.", "reverse_scored": False, "difficulty": -0.5},
        {"question_id": "o3", "text": "أنا فضولي بشأن كل شيء تقريباً.", "reverse_scored": False, "difficulty": 0},
        {"question_id": "o4", "text": "أفضل الروتين على التغيير.", "reverse_scored": True, "difficulty": 0.5},
        {"question_id": "o5", "text": "أنا مبدع وأحب ابتكار أشياء جديدة.", "reverse_scored": False, "difficulty": 1},
        {"question_id": "o6", "text": "أجد صعوبة في فهم الأفكار المجردة.", "reverse_scored": True, "difficulty": -1},
        {"question_id": "o7", "text": "أحب تجربة الأنشطة الجديدة.", "reverse_scored": False, "difficulty": -0.5},
        {"question_id": "o8", "text": "لست مهتماً بالفنون.", "reverse_scored": True, "difficulty": 0},
        {"question_id": "o9", "text": "أحب حل المشكلات المعقدة.", "reverse_scored": False, "difficulty": 0.5},
        {"question_id": "o10", "text": "أميل إلى التصويت للمرشحين المحافظين.", "reverse_scored": True, "difficulty": 1}
    ],
    "conscientiousness": [
        {"question_id": "c1", "text": "أنا دائماً مستعد ومنظم.", "reverse_scored": False, "difficulty": -1},
        {"question_id": "c2", "text": "أترك أشيائي فوضوياً.", "reverse_scored": True, "difficulty": -0.5},
        {"question_id": "c3", "text": "أهتم بالتفاصيل.", "reverse_scored": False, "difficulty": 0},
        {"question_id": "c4", "text": "أؤجل المهام المهمة.", "reverse_scored": True, "difficulty": 0.5},
        {"question_id": "c5", "text": "أتبع جدولاً زمنياً.", "reverse_scored": False, "difficulty": 1},
        {"question_id": "c6", "text": "أنا دقيق في عملي.", "reverse_scored": False, "difficulty": -1},
        {"question_id": "c7", "text": "أنسى أحياناً إعادة الأشياء إلى مكانها الصحيح.", "reverse_scored": True, "difficulty": -0.5},
        {"question_id": "c8", "text": "أحب النظام.", "reverse_scored": False, "difficulty": 0},
        {"question_id": "c9", "text": "أجد صعوبة في الالتزام بالخطط.", "reverse_scored": True, "difficulty": 0.5},
        {"question_id": "c10", "text": "أنا مجتهد ومثابر.", "reverse_scored": False, "difficulty": 1}
    ],
    "extraversion": [
        {"question_id": "e1", "text": "أنا محور الاهتمام في الحفلات.", "reverse_scored": False, "difficulty": -1},
        {"question_id": "e2", "text": "لا أتحدث كثيراً.", "reverse_scored": True, "difficulty": -0.5},
        {"question_id": "e3", "text": "أشعر بالراحة حول الناس.", "reverse_scored": False, "difficulty": 0},
        {"question_id": "e4", "text": "أفضل البقاء في الخلفية.", "reverse_scored": True, "difficulty": 0.5},
        {"question_id": "e5", "text": "أبدأ المحادثات.", "reverse_scored": False, "difficulty": 1},
        {"question_id": "e6", "text": "لدي دائرة واسعة من المعارف.", "reverse_scored": False, "difficulty": -1},
        {"question_id": "e7", "text": "أنا هادئ حول الغرباء.", "reverse_scored": True, "difficulty": -0.5},
        {"question_id": "e8", "text": "لا أمانع أن أكون مركز الاهتمام.", "reverse_scored": False, "difficulty": 0},
        {"question_id": "e9", "text": "أفضل قضاء الوقت بمفردي.", "reverse_scored": True, "difficulty": 0.5},
        {"question_id": "e10", "text": "أنا مفعم بالحيوية والنشاط.", "reverse_scored": False, "difficulty": 1}
    ],
    "agreeableness": [
        {"question_id": "a1", "text": "أتعاطف مع مشاعر الآخرين.", "reverse_scored": False, "difficulty": -1},
        {"question_id": "a2", "text": "لست مهتماً بمشاكل الآخرين.", "reverse_scored": True, "difficulty": -0.5},
        {"question_id": "a3", "text": "لدي قلب حنون.", "reverse_scored": False, "difficulty": 0},
        {"question_id": "a4", "text": "أهين الناس.", "reverse_scored": True, "difficulty": 0.5},
        {"question_id": "a5", "text": "أجعل الناس يشعرون بالراحة.", "reverse_scored": False, "difficulty": 1},
        {"question_id": "a6", "text": "أنا صبور مع الآخرين.", "reverse_scored": False, "difficulty": -1},
        {"question_id": "a7", "text": "أنا سريع الغضب.", "reverse_scored": True, "difficulty": -0.5},
        {"question_id": "a8", "text": "أثق بالآخرين.", "reverse_scored": False, "difficulty": 0},
        {"question_id": "a9", "text": "أنا متشكك في نوايا الآخرين.", "reverse_scored": True, "difficulty": 0.5},
        {"question_id": "a10", "text": "أنا متعاون بطبعي.", "reverse_scored": False, "difficulty": 1}
    ],
    "neuroticism": [
        {"question_id": "n1", "text": "أشعر بالتوتر بسهولة.", "reverse_scored": False, "difficulty": -1},
        {"question_id": "n2", "text": "أنا مسترخٍ في معظم الأوقات.", "reverse_scored": True, "difficulty": -0.5},
        {"question_id": "n3", "text": "أقلق بشأن الأشياء.", "reverse_scored": False, "difficulty": 0},
        {"question_id": "n4", "text": "نادراً ما أشعر بالحزن.", "reverse_scored": True, "difficulty": 0.5},
        {"question_id": "n5", "text": "أنا متقلب المزاج.", "reverse_scored": False, "difficulty": 1},
        {"question_id": "n6", "text": "أتعامل مع التوتر بشكل جيد.", "reverse_scored": True, "difficulty": -1},
        {"question_id": "n7", "text": "أشعر بالقلق كثيراً.", "reverse_scored": False, "difficulty": -0.5},
        {"question_id": "n8", "text": "أنا مستقر عاطفياً.", "reverse_scored": True, "difficulty": 0},
        {"question_id": "n9", "text": "يمكن أن أكون سريع الانفعال.", "reverse_scored": False, "difficulty": 0.5},
        {"question_id": "n10", "text": "أنا راضٍ عن نفسي.", "reverse_scored": True, "difficulty": 1}
    ]
}

# Pydantic models
class SessionCreate(BaseModel):
    name: str
    gender: Optional[str] = None
    birth_year: Optional[int] = None
    birthYear: Optional[int] = None  # Accept both formats
    education_level: Optional[str] = None
    educationLevel: Optional[str] = None  # Accept both formats
    marital_status: Optional[str] = None
    maritalStatus: Optional[str] = None  # Accept both formats

class SessionResponse(BaseModel):
    session_id: str
    name: str
    status: str
    current_dimension: str
    current_question_number: int
    total_dimensions: int
    dimension_progress: Dict[str, int]

class Question(BaseModel):
    question_id: str
    text: str
    dimension: str
    question_number: int
    reverse_scored: bool = False
    total_answered: int = 0
    total_questions: int = 50  # Updated from 200 to 50
    progress_percentage: float = 0.0

class AnswerSubmission(BaseModel):
    session_id: str
    question_id: str
    response: int

class AdminLoginRequest(BaseModel):
    username: str
    password: str

class AdminLoginResponse(BaseModel):
    success: bool
    token: str = None
    message: str = None

@app.post("/api/sessions", response_model=SessionResponse)
async def create_session(session_data: SessionCreate):
    try:
        session_id = str(uuid.uuid4())
        
        # Handle both camelCase and snake_case field names
        birth_year = session_data.birth_year or session_data.birthYear
        education_level = session_data.education_level or session_data.educationLevel
        marital_status = session_data.marital_status or session_data.maritalStatus
        
        # Create demographic profile for question generation
        demographics = {
            "gender": session_data.gender,
            "birth_year": birth_year,
            "education_level": education_level,
            "marital_status": marital_status,
            "age_group": _calculate_age_group(birth_year)
        }
        
        # Generate personalized questions for all dimensions
        print(f"Generating personalized questions for session {session_id}...")
        all_questions = {}
        
        for dimension in PERSONALITY_DIMENSIONS:
            try:
                questions = await question_generator.generate_personalized_questions(
                    demographics, dimension, count=10  # Generate 10 questions per dimension (50 total)
                )
                all_questions[dimension] = questions
                print(f"Generated {len(questions)} questions for {dimension}")
            except Exception as e:
                print(f"Error generating questions for {dimension}: {e}")
                # Use fallback questions
                all_questions[dimension] = question_generator._get_optimized_question_bank(
                    dimension, demographics, 10  # Use 10 questions per dimension
                )
        
        # Create session with generated questions
        sessions[session_id] = {
            "session_id": session_id,
            "name": session_data.name,
            "gender": session_data.gender,
            "birth_year": birth_year,
            "education_level": education_level,
            "marital_status": marital_status,
            "demographics": demographics,
            "status": "active",
            "current_dimension": "openness",
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "questions_answered": {dim: [] for dim in PERSONALITY_DIMENSIONS},
            "theta": {dim: 0.0 for dim in PERSONALITY_DIMENSIONS},
            "se": {dim: 1.0 for dim in PERSONALITY_DIMENSIONS},
            "dimension_question_count": {dim: 0 for dim in PERSONALITY_DIMENSIONS},
            "generated_questions": all_questions,
            "total_questions_answered": 0
        }
        
        # Save sessions to file
        save_sessions()
        
        print(f"DEBUG: New session created: {session_id}. Total sessions now: {len(sessions)}")
        
        return SessionResponse(
            session_id=session_id,
            name=session_data.name,
            status="active",
            current_dimension="openness",
            current_question_number=1,
            total_dimensions=5,
            dimension_progress={dim: 0 for dim in PERSONALITY_DIMENSIONS}
        )
    except Exception as e:
        print(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")

def _calculate_age_group(birth_year):
    """Calculate age group from birth year"""
    if not birth_year:
        return "unknown"
    
    try:
        age = 2025 - int(birth_year)
        if age < 20:
            return "teen"
        elif age < 30:
            return "young_adult"
        elif age < 50:
            return "middle_age"
        else:
            return "senior"
    except:
        return "unknown"

@app.get("/api/sessions/{session_id}/question", response_model=Question)
async def get_current_question(session_id: str):
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        current_dimension = session["current_dimension"]
        
        if session["dimension_question_count"][current_dimension] >= adaptive_engine.irt_params.max_per_dimension:
            # Move to next dimension after reaching max questions per dimension (5 dimensions × 10 = 50 total)
            dimensions = list(questions_db.keys())
            current_index = dimensions.index(current_dimension)
            if current_index + 1 < len(dimensions):
                session["current_dimension"] = dimensions[current_index + 1]
                current_dimension = session["current_dimension"]
            else:
                # Test is complete
                session["status"] = "completed"
                raise HTTPException(status_code=404, detail="No more questions")

        # Adaptive question selection
        theta = session["theta"][current_dimension]
        answered_ids = [q["question_id"] for q in session["questions_answered"][current_dimension]]
        
        # Use generated questions from session instead of questions_db
        available_questions = [q for q in session["generated_questions"][current_dimension] if q["question_id"] not in answered_ids]
        
        # Select question based on whether it's the first question or not
        if len(answered_ids) == 0:
            # For the first question in each dimension, select randomly from medium difficulty questions
            import random
            medium_questions = [q for q in available_questions if -0.5 <= q["difficulty"] <= 0.5]
            if medium_questions:
                next_question = random.choice(medium_questions)
            else:
                next_question = random.choice(available_questions)
        else:
            # For subsequent questions, use adaptive IRT selection
            next_question = min(available_questions, key=lambda q: abs(q["difficulty"] - theta))
        
        # Get first name from full name for personalization
        first_name = session["name"].split()[0] if session["name"] else ""
        
        # Smart gender detection and question personalization
        def personalize_question(text, name, gender=None):
            if not name:
                return text
            
            # Detect gender from name or use provided gender
            is_female = False
            if gender == "female":
                is_female = True
            elif gender == "male":
                is_female = False
            else:
                # Try to detect from name endings (simple heuristic)
                female_endings = ['ة', 'اء', 'ى', 'ان', 'ين']
                is_female = any(name.endswith(ending) for ending in female_endings)
            
            # Convert statement to question format
            question_text = text
            
            # Replace "أنا" with "أنت/أنتِ" and adjust for gender
            if "أنا " in question_text:
                if is_female:
                    question_text = question_text.replace("أنا ", "أنتِ ")
                else:
                    question_text = question_text.replace("أنا ", "أنت ")
            
            # Handle verbs - convert to second person with gender agreement
            if is_female:
                # Female second person verbs
                question_text = question_text.replace("أستمتع", "تستمتعين")
                question_text = question_text.replace("أحب", "تحبين") 
                question_text = question_text.replace("أهتم", "تهتمين")
                question_text = question_text.replace("أتبع", "تتبعين")
                question_text = question_text.replace("أؤجل", "تؤجلين")
                question_text = question_text.replace("أنسى", "تنسين")
                question_text = question_text.replace("أجد", "تجدين")
                question_text = question_text.replace("أشعر", "تشعرين")
                question_text = question_text.replace("أفضل", "تفضلين")
                question_text = question_text.replace("أبدأ", "تبدئين")
                question_text = question_text.replace("أتعاطف", "تتعاطفين")
                question_text = question_text.replace("أجعل", "تجعلين")
                question_text = question_text.replace("أهين", "تهينين")
                question_text = question_text.replace("أثق", "تثقين")
                question_text = question_text.replace("أقلق", "تقلقين")
                question_text = question_text.replace("أتعامل", "تتعاملين")
                question_text = question_text.replace("أميل", "تميلين")
                question_text = question_text.replace("أترك", "تتركين")
                question_text = question_text.replace("أتحدث", "تتحدثين")
                question_text = question_text.replace("أمانع", "تمانعين")
                
                # Adjust adjectives and descriptions for feminine
                question_text = question_text.replace("فضولي", "فضولية")
                question_text = question_text.replace("مستعد", "مستعدة")
                question_text = question_text.replace("منظم", "منظمة")  
                question_text = question_text.replace("دقيق", "دقيقة")
                question_text = question_text.replace("مجتهد", "مجتهدة")
                question_text = question_text.replace("مثابر", "مثابرة")
                question_text = question_text.replace("مبدع", "مبدعة")
                question_text = question_text.replace("هادئ", "هادئة")
                question_text = question_text.replace("صبور", "صبورة")
                question_text = question_text.replace("متعاون", "متعاونة")
                question_text = question_text.replace("مسترخ", "مسترخية")
                question_text = question_text.replace("مسترخيةٍ", "مسترخية")  # Fix tanween
                question_text = question_text.replace("متقلب", "متقلبة")
                question_text = question_text.replace("مستقر", "مستقرة")
                question_text = question_text.replace("راض", "راضية")
                question_text = question_text.replace("راضيةٍ", "راضية")  # Fix tanween
                question_text = question_text.replace("مفعم", "مفعمة")
                question_text = question_text.replace("سريع", "سريعة")
                
                # Handle "لدي" (I have)
                question_text = question_text.replace("لدي", "لديكِ")
                
                # Handle "في عملي" (in my work)
                question_text = question_text.replace("في عملي", "في عملكِ")
                
                # Handle "بمفردي" (alone)
                question_text = question_text.replace("بمفردي", "بمفردكِ")
                
                # Handle "بطبعي" (by nature)
                question_text = question_text.replace("بطبعي", "بطبعكِ")
                
                # Handle "عن نفسي" (about myself)
                question_text = question_text.replace("عن نفسي", "عن نفسكِ")
                
                # Handle negations like "لست"
                question_text = question_text.replace("لست مهتماً", "لستِ مهتمة")
                question_text = question_text.replace("لستِِ", "لستِ")  # Fix double kasra
                question_text = question_text.replace("لست", "لستِ")
                
                # Handle "لا" negations
                question_text = question_text.replace("لا أتحدث", "لا تتحدثين")
                question_text = question_text.replace("لا أمانع", "لا تمانعين")
                
                # Handle "يمكن أن أكون" 
                question_text = question_text.replace("يمكن أن أكون", "يمكن أن تكوني")
                
                # Handle other possession forms
                question_text = question_text.replace("أشيائي", "أشياءكِ")
                
            else:
                # Male second person verbs
                question_text = question_text.replace("أستمتع", "تستمتع")
                question_text = question_text.replace("أحب", "تحب")
                question_text = question_text.replace("أهتم", "تهتم") 
                question_text = question_text.replace("أتبع", "تتبع")
                question_text = question_text.replace("أؤجل", "تؤجل")
                question_text = question_text.replace("أنسى", "تنسى")
                question_text = question_text.replace("أجد", "تجد")
                question_text = question_text.replace("أشعر", "تشعر")
                question_text = question_text.replace("أفضل", "تفضل")
                question_text = question_text.replace("أبدأ", "تبدأ")
                question_text = question_text.replace("أتعاطف", "تتعاطف")
                question_text = question_text.replace("أجعل", "تجعل")
                question_text = question_text.replace("أهين", "تهين")
                question_text = question_text.replace("أثق", "تثق")
                question_text = question_text.replace("أقلق", "تقلق")
                question_text = question_text.replace("أتعامل", "تتعامل")
                question_text = question_text.replace("أميل", "تميل")
                question_text = question_text.replace("أترك", "تترك")
                question_text = question_text.replace("أتحدث", "تتحدث")
                question_text = question_text.replace("أمانع", "تمانع")
                
                # Handle "لدي" (I have)
                question_text = question_text.replace("لدي", "لديك")
                
                # Handle "في عملي" (in my work)
                question_text = question_text.replace("في عملي", "في عملك")
                
                # Handle "بمفردي" (alone)
                question_text = question_text.replace("بمفردي", "بمفردك")
                
                # Handle "بطبعي" (by nature)
                question_text = question_text.replace("بطبعي", "بطبعك")
                
                # Handle "عن نفسي" (about myself)
                question_text = question_text.replace("عن نفسي", "عن نفسك")
                
                # Handle negations like "لست"
                question_text = question_text.replace("لست", "لست")
                
                # Handle "لا" negations
                question_text = question_text.replace("لا أتحدث", "لا تتحدث")
                question_text = question_text.replace("لا أمانع", "لا تمانع")
                
                # Handle "يمكن أن أكون" 
                question_text = question_text.replace("يمكن أن أكون", "يمكن أن تكون")
                
                # Handle other possession forms
                question_text = question_text.replace("أشيائي", "أشياءك")
            
            # Add question format and name
            if question_text.endswith('.'):
                question_text = question_text[:-1]
            
            # Check if question already starts with "هل"
            if question_text.strip().startswith("هل"):
                return f"{question_text} يا {name}؟"
            else:
                return f"هل {question_text} يا {name}؟"
        
        # Personalize the question text
        personalized_text = personalize_question(next_question['text'], first_name, session.get('gender'))
        
        # Calculate total questions answered so far
        total_answered = sum(session["dimension_question_count"].values())
        
        return Question(
            question_id=next_question["question_id"],
            text=personalized_text,
            dimension=current_dimension,
            question_number=session["dimension_question_count"][current_dimension] + 1,
            reverse_scored=next_question["reverse_scored"],
            total_answered=total_answered,
            total_questions=adaptive_engine.irt_params.max_questions,
            progress_percentage=round((total_answered / adaptive_engine.irt_params.max_questions) * 100, 1)
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting question: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting question: {str(e)}")

@app.post("/api/answers")
async def submit_answer(answer: AnswerSubmission):
    print(f"🔍 Received answer submission: {answer}")
    try:
        if answer.session_id not in sessions:
            print(f"❌ Session not found: {answer.session_id}")
            print(f"Available sessions: {list(sessions.keys())}")
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[answer.session_id]
        current_dimension = session["current_dimension"]
        
        print(f"📝 Processing answer for session {answer.session_id}, dimension: {current_dimension}")
        
        # Find the question in generated questions
        question = None
        for q in session["generated_questions"][current_dimension]:
            if q["question_id"] == answer.question_id:
                question = q
                break
        
        if not question:
            print(f"❌ Question not found: {answer.question_id}")
            print(f"Available questions in {current_dimension}: {[q['question_id'] for q in session['generated_questions'][current_dimension][:5]]}")
            raise HTTPException(status_code=404, detail="Question not found")

        # Record the answer with question details
        answered_question = {
            "question_id": answer.question_id,
            "response": answer.response,
            "difficulty": question["difficulty"],
            "discrimination": question["discrimination"],
            "reverse_scored": question.get("reverse_scored", False)
        }
        
        session["questions_answered"][current_dimension].append(answered_question)
        
        # Update dimension question count
        session["dimension_question_count"][current_dimension] = len(session["questions_answered"][current_dimension])
        
        # Update ability estimate using IRT
        answered_questions = session["questions_answered"][current_dimension]
        
        # Extract response data for theta estimation
        responses = []
        difficulties = []
        discriminations = []
        
        for aq in answered_questions:
            response_value = aq["response"]
            if aq.get("reverse_scored", False):
                response_value = 6 - response_value  # Reverse score if needed
            
            # Convert to binary response (1 if >= 4, 0 otherwise)
            responses.append(1 if response_value >= 4 else 0)
            difficulties.append(aq["difficulty"])
            discriminations.append(aq["discrimination"])
        
        # Estimate new theta and standard error
        new_theta, new_se = adaptive_engine.estimate_theta(responses, difficulties, discriminations)
        
        # Update session
        session["theta"][current_dimension] = new_theta
        session["se"][current_dimension] = new_se
        session["total_questions_answered"] = session.get("total_questions_answered", 0) + 1
        
        # Save sessions after each answer
        save_sessions()
        
        # Check if test is complete
        total_answered = session["total_questions_answered"]
        
        # Check completion criteria
        all_dimensions_complete = True
        for dim in PERSONALITY_DIMENSIONS:
            dim_questions = session["questions_answered"][dim]
            dim_se = session["se"][dim]
            
            # Check if dimension needs more questions (minimum 5, maximum 10 per dimension)
            if (len(dim_questions) < adaptive_engine.irt_params.min_per_dimension or 
                (dim_se > adaptive_engine.irt_params.target_se and len(dim_questions) < adaptive_engine.irt_params.max_per_dimension)):
                all_dimensions_complete = False
                break
        
        if (total_answered >= adaptive_engine.irt_params.max_questions or all_dimensions_complete):
            session["status"] = "completed"
            session["completed_at"] = datetime.now().isoformat()
            save_sessions()
            
            print(f"Test completed! Session {answer.session_id} status updated.")
        
        return {
            "message": "Answer submitted successfully", 
            "status": session["status"],
            "total_answered": total_answered,
            "progress_percentage": round((total_answered / adaptive_engine.irt_params.max_questions) * 100, 1),
            "current_theta": round(new_theta, 2),
            "current_se": round(new_se, 3)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting answer: {str(e)}")

def generate_comprehensive_analysis(scores, session):
    """Generate detailed personality analysis using Groq AI based on actual results"""
    
    # Get demographic info
    age = session.get("age", 25)
    gender = session.get("gender", "male")
    marital_status = session.get("marital_status", "single")
    education_level = session.get("education_level", "bachelor")
    name = session.get("name", "")
    
    # Convert scores to percentages (scores are on 1-5 scale, convert to 0-100%)
    openness_pct = ((scores["openness"] - 1) / 4) * 100
    conscientiousness_pct = ((scores["conscientiousness"] - 1) / 4) * 100  
    extraversion_pct = ((scores["extraversion"] - 1) / 4) * 100
    agreeableness_pct = ((scores["agreeableness"] - 1) / 4) * 100
    neuroticism_pct = ((scores["neuroticism"] - 1) / 4) * 100
    
    # Try to use Groq AI for detailed analysis
    if groq_client:
        try:
            # Prepare personality profile for AI
            personality_prompt = f"""
            أنت خبير متخصص في علم النفس والشخصية. قم بكتابة تحليل شخصية شامل باللغة العربية فقط لا غير.

            **معلومات الشخص:**
            - الاسم: {name}
            - العمر: {age} سنة
            - الجنس: {"أنثى" if gender == "female" else "ذكر"}
            - الحالة الاجتماعية: {"متزوج" if marital_status == "married" else "أعزب"}
            - مستوى التعليم: {education_level}

            **نتائج اختبار الشخصية (نموذج الشخصية الخمسي الكبير):**
            - الانفتاح على التجارب: {openness_pct:.1f}%
            - الضمير والانضباط: {conscientiousness_pct:.1f}%
            - الانبساط: {extraversion_pct:.1f}%
            - المقبولية والتعاون: {agreeableness_pct:.1f}%
            - العصابية: {neuroticism_pct:.1f}%

            اكتب تحليلاً باللغة العربية فقط يتضمن:

            **🌟 نوع الشخصية:**
            حدد نوع الشخصية الرئيسي (مثل: شخصية قيادية، إبداعية، اجتماعية، تحليلية، إلخ)

            **📊 التحليل المفصل:**
            - فسر كل بُعد من الأبعاد الخمسة بناءً على النسبة المئوية
            - اربط النتائج بسلوكيات محددة في الحياة اليومية

            **👤 التأثير الديموغرافي:**
            كيف تؤثر المعلومات الشخصية على الشخصية

            **💪 نقاط القوة:**
            أهم المميزات بناءً على النتائج العالية

            **🎯 مجالات التطوير:**
            النواحي التي تحتاج تحسين

            **🏢 التوصيات المهنية:**
            أفضل المهن المناسبة للشخصية

            **📝 نصائح عملية:**
            توجيهات للحياة اليومية

            **🤝 العلاقات الاجتماعية:**
            كيفية التعامل مع الآخرين

            **🎯 الخلاصة:**
            "أنت شخصية [اذكر نوع الشخصية]"

            ملاحظات هامة:
            - اكتب باللغة العربية الفصحى البسيطة فقط
            - لا تستخدم أي كلمات إنجليزية
            - اجعل التحليل شخصياً وعملياً
            - استخدم رموز تعبيرية مناسبة
            - اربط كل نقطة بالنتائج الفعلية للشخص
            """

            # Call Groq API
            response = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": "أنت خبير نفسي متخصص في تحليل الشخصية. تكتب باللغة العربية فقط. تقدم تحليلات دقيقة ومفيدة وعملية. لا تستخدم أي كلمات إنجليزية أبداً."},
                    {"role": "user", "content": personality_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            ai_analysis = response.choices[0].message.content
            print(f"✅ AI Analysis generated successfully: {len(ai_analysis)} characters")
            return ai_analysis
            
        except Exception as e:
            print(f"❌ Error using Groq AI: {e}")
            # Fall back to manual analysis
    
    # Manual fallback analysis if AI fails
    print(f"🔄 Using fallback analysis for {name}")
    
    # Determine personality type
    dominant_trait = max([
        ("منفتح ومبدع", openness_pct),
        ("منضبط ومنظم", conscientiousness_pct), 
        ("اجتماعي ونشيط", extraversion_pct),
        ("متعاون ولطيف", agreeableness_pct),
        ("حساس وعاطفي", neuroticism_pct)
    ], key=lambda x: x[1])
    
    analysis = f"""
🌟 **تحليل شخصية {name} المفصل**

**🎯 نوع الشخصية:**
أنت شخصية {dominant_trait[0]} - حيث أن أبرز صفاتك هي {dominant_trait[0]} بنسبة {dominant_trait[1]:.1f}%

**📊 التحليل المفصل:**

**الانفتاح على التجارب ({openness_pct:.1f}%):**
{'🎨 أنت شخص مبدع ومنفتح على التجارب الجديدة، تحب الاستطلاع والتعلم' if openness_pct > 70 else 
 '📚 تفضل الأساليب المجربة والمألوفة، وهذا يجعلك موثوقاً ومستقراً' if openness_pct < 40 else
 '⚖️ متوازن في تقبل الأفكار الجديدة مع الحفاظ على الاستقرار'}

**الضمير والانضباط ({conscientiousness_pct:.1f}%):**
{'📋 منظم جداً وموثوق، تلتزم بالمواعيد وتكمل مهامك بدقة' if conscientiousness_pct > 70 else
 '🎈 مرن وعفوي، تتكيف بسهولة مع التغييرات المفاجئة' if conscientiousness_pct < 40 else
 '⚖️ متوازن بين التنظيم والمرونة، قادر على التخطيط والتكيف'}

**الانبساط ({extraversion_pct:.1f}%):**
{'🎉 اجتماعي ونشيط، تستمد طاقتك من التفاعل مع الآخرين' if extraversion_pct > 70 else
 '🤔 هادئ ومتأمل، تفضل الأنشطة الفردية والتفكير العميق' if extraversion_pct < 40 else
 '⚖️ متوازن اجتماعياً، تستطيع الاستمتاع بالأنشطة الجماعية والفردية'}

**المقبولية ({agreeableness_pct:.1f}%):**
{'🤝 متعاون ومتسامح، تضع احتياجات الآخرين في اعتبارك' if agreeableness_pct > 70 else
 '💪 مستقل وحازم، تدافع عن آرائك ولا تتنازل بسهولة' if agreeableness_pct < 40 else
 '⚖️ متوازن في التعامل، تستطيع التعاون والحزم حسب الموقف'}

**الاستقرار العاطفي ({100-neuroticism_pct:.1f}%):**
{'😌 هادئ ومستقر عاطفياً، تتعامل مع الضغوط بثقة وهدوء' if neuroticism_pct < 30 else
 '😰 حساس ومتقلب المزاج، تتأثر بالضغوط والتغييرات' if neuroticism_pct > 70 else
 '⚖️ متوازن عاطفياً، تظهر مرونة جيدة في مواجهة التحديات'}

**💼 التوصيات المهنية:**
{
    "🎓 التعليم والتدريب" if extraversion_pct > 60 and agreeableness_pct > 60 else
    "🔬 البحث والتطوير" if openness_pct > 70 and conscientiousness_pct > 60 else
    "🏥 الرعاية الصحية" if agreeableness_pct > 70 else
    "💼 إدارة الأعمال" if conscientiousness_pct > 70 and extraversion_pct > 50 else
    "🎨 المجالات الإبداعية" if openness_pct > 70 else
    "📊 التحليل والبيانات" if conscientiousness_pct > 60 else
    "🤝 الخدمة الاجتماعية"
}

**📝 نصائح للحياة اليومية:**
• استثمر نقاط قوتك في تطوير مهاراتك المهنية
• تواصل مع أشخاص يكملون شخصيتك ويدعمون نموك
• {"اعط وقتاً أكثر للاسترخاء وإدارة الضغوط" if neuroticism_pct > 60 else "حافظ على استقرارك العاطفي الممتاز"}
• {"استغل حبك للاستطلاع في تعلم مهارات جديدة" if openness_pct > 60 else "اعط نفسك فرصة لتجربة أشياء جديدة بشكل تدريجي"}

**🎯 الخلاصة:**
أنت شخصية {dominant_trait[0]} تتميز بقدرات متنوعة وإمكانيات كبيرة للنمو والتطور.
"""
    
    return analysis

@app.get("/api/sessions/{session_id}/report")
async def get_report(session_id: str):
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        
        if session["status"] != "completed":
            raise HTTPException(status_code=400, detail="Test not completed yet")
        
        # Calculate real scores from session data
        def calculate_dimension_score(dimension):
            """Calculate actual score from answered questions"""
            answered_questions = session["questions_answered"].get(dimension, [])
            if not answered_questions:
                return 50  # Default middle score
            
            total_score = 0
            for q in answered_questions:
                score = q["response"]
                # Handle reverse scored questions (assumed to be marked in question data)
                # For now, we'll use the raw score
                total_score += score
            
            # Convert to 0-100 scale (5 point scale * max questions per dimension)
            max_possible = len(answered_questions) * 5
            percentage = (total_score / max_possible) * 100 if max_possible > 0 else 50
            return min(100, max(0, percentage))
        
        def get_level(score):
            if score >= 75:
                return "عالي"
            elif score >= 50:
                return "متوسط"
            else:
                return "منخفض"
        
        # Calculate real scores
        openness_score = calculate_dimension_score("openness")
        conscientiousness_score = calculate_dimension_score("conscientiousness")
        extraversion_score = calculate_dimension_score("extraversion")
        agreeableness_score = calculate_dimension_score("agreeableness")
        neuroticism_score = calculate_dimension_score("neuroticism")
        
        return {
            "session_id": session_id,
            "name": session["name"],
            "completion_date": "2025-01-24T10:30:00Z",
            "scores": {
                "openness": {
                    "name": "الانفتاح على التجارب",
                    "score": (openness_score / 100) * 4 + 1,  # Convert from 0-100% to 1-5 scale
                    "level": get_level(openness_score)
                },
                "conscientiousness": {
                    "name": "الضمير الحي",
                    "score": (conscientiousness_score / 100) * 4 + 1,
                    "level": get_level(conscientiousness_score)
                },
                "extraversion": {
                    "name": "الانبساط",
                    "score": (extraversion_score / 100) * 4 + 1,
                    "level": get_level(extraversion_score)
                },
                "agreeableness": {
                    "name": "المقبولية",
                    "score": (agreeableness_score / 100) * 4 + 1,
                    "level": get_level(agreeableness_score)
                },
                "neuroticism": {
                    "name": "العصابية",
                    "score": (neuroticism_score / 100) * 4 + 1,
                    "level": get_level(neuroticism_score)
                }
            },
            "detailed_analysis": generate_comprehensive_analysis(
                {
                    "openness": (openness_score / 100) * 4 + 1,
                    "conscientiousness": (conscientiousness_score / 100) * 4 + 1,
                    "extraversion": (extraversion_score / 100) * 4 + 1,
                    "agreeableness": (agreeableness_score / 100) * 4 + 1,
                    "neuroticism": (neuroticism_score / 100) * 4 + 1
                },
                session
            ),
            "recommendations": [
                "استمر في تطوير نقاط قوتك",
                "اعمل على تحسين المجالات التي تحتاج لتطوير",
                "تذكر أن الشخصية قابلة للنمو والتطوير"
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Personality Test API is running"}

@app.get("/api/test")
async def test_connection():
    return {"status": "success", "message": "API connection is working"}

# Admin endpoints
@app.post("/api/admin/login")
async def admin_login(request: AdminLoginRequest):
    """Admin login endpoint"""
    try:
        # Simple hardcoded admin credentials
        if request.username == "admin" and request.password == "PersonalityAdmin2025!":
            # Generate a simple token (in production, use proper JWT)
            token = "admin_token_" + str(uuid.uuid4())
            return AdminLoginResponse(
                success=True,
                token=token,
                message="تم تسجيل الدخول بنجاح"
            )
        else:
            return AdminLoginResponse(
                success=False,
                message="بيانات تسجيل الدخول غير صحيحة"
            )
    except Exception as e:
        return AdminLoginResponse(
            success=False,
            message="خطأ في السيرفر"
        )

@app.get("/api/admin/test")
async def admin_test():
    """Test admin connection"""
    try:
        load_sessions()
        total_sessions = len(sessions)
        total_answers = sum(len(session.get('questions_answered', {}).get(dim, [])) 
                          for session in sessions.values() 
                          for dim in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism'])
        
        return {
            "status": "success",
            "sessionsCount": total_sessions,
            "answersCount": total_answers,
            "message": "اتصال Admin نجح"
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"خطأ في اختبار الاتصال: {str(e)}"
        }

@app.get("/api/admin/dashboard")
async def get_admin_dashboard():
    """Get dashboard statistics"""
    try:
        # Always load fresh data from file
        load_sessions()
        
        print(f"DEBUG: Dashboard called - Total sessions in memory: {len(sessions)}")
        print(f"DEBUG: Sessions dictionary reference: {id(sessions)}")
        print(f"DEBUG: Sessions keys: {list(sessions.keys())}")
        
        if len(sessions) > 0:
            print(f"DEBUG: First session sample: {list(sessions.values())[0]}")
        
        total_sessions = len(sessions)
        completed_sessions = len([s for s in sessions.values() if s["status"] == "completed"])
        active_sessions = len([s for s in sessions.values() if s["status"] == "active"])
        
        print(f"DEBUG: Calculated stats - Total: {total_sessions}, Completed: {completed_sessions}, Active: {active_sessions}")
        
        # Gender distribution - convert to array format expected by frontend
        genders = [s.get("gender", "غير محدد") for s in sessions.values() if s.get("gender")]
        gender_counts = {}
        for gender in genders:
            gender_counts[gender] = gender_counts.get(gender, 0) + 1
        
        # Convert gender distribution to array format
        gender_distribution = []
        for gender, count in gender_counts.items():
            gender_distribution.append({
                "label": "ذكر" if gender == "male" else "أنثى" if gender == "female" else gender,
                "value": count
            })
        
        # Education level distribution - convert to array format expected by frontend
        education_levels = [s.get("education_level", "غير محدد") for s in sessions.values() if s.get("education_level")]
        education_counts = {}
        for edu in education_levels:
            education_counts[edu] = education_counts.get(edu, 0) + 1
        
        # Convert education distribution to array format
        education_distribution = []
        for edu, count in education_counts.items():
            education_distribution.append({
                "label": edu,
                "value": count
            })
        
        # Age distribution - calculate ages and group them
        ages = []
        current_year = datetime.now().year
        for s in sessions.values():
            if s.get("birth_year"):
                age = current_year - s.get("birth_year")
                ages.append(age)
        
        age_groups = {"18-25": 0, "26-35": 0, "36-45": 0, "46-55": 0, "56+": 0}
        for age in ages:
            if 18 <= age <= 25:
                age_groups["18-25"] += 1
            elif 26 <= age <= 35:
                age_groups["26-35"] += 1
            elif 36 <= age <= 45:
                age_groups["36-45"] += 1
            elif 46 <= age <= 55:
                age_groups["46-55"] += 1
            elif age > 55:
                age_groups["56+"] += 1
        
        # Convert age distribution to array format
        age_distribution = []
        for age_group, count in age_groups.items():
            if count > 0:  # Only include groups that have participants
                age_distribution.append({
                    "label": age_group,
                    "value": count
                })
        
        # Mock daily stats for the chart
        daily_stats = [
            {"date": "2025-08-01", "newParticipants": 5, "completedTests": 3},
            {"date": "2025-08-02", "newParticipants": 8, "completedTests": 6},
            {"date": "2025-08-03", "newParticipants": total_sessions, "completedTests": completed_sessions},
        ]
        
        return {
            "totalParticipants": total_sessions,
            "completedTests": completed_sessions,
            "activeSessions": active_sessions,
            "completionRate": (completed_sessions / total_sessions * 100) if total_sessions > 0 else 0,
            "genderDistribution": gender_distribution,
            "educationDistribution": education_distribution,
            "ageDistribution": age_distribution,
            "dailyStats": daily_stats,
            "averageCompletionRate": (completed_sessions / total_sessions * 100) if total_sessions > 0 else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dashboard data: {str(e)}")

@app.get("/api/admin/participants")
async def get_participants(page: int = 1, search: str = ""):
    """Get participants list with pagination"""
    try:
        # Always load fresh data from file
        load_sessions()
        
        participants_list = []
        for session_id, session in sessions.items():
            if search and search.lower() not in session.get("name", "").lower():
                continue
                
            # Format dates properly and calculate age
            completion_date = None
            if session.get("status") == "completed" and session.get("completed_at"):
                completion_date = session.get("completed_at")[:10]  # Extract date part only (YYYY-MM-DD)
            
            # Calculate age from birth year
            age = None
            if session.get("birth_year"):
                current_year = datetime.now().year
                age = current_year - session.get("birth_year")
            
            participant = {
                "sessionId": session_id,
                "name": session.get("name", "غير محدد"),
                "gender": "ذكر" if session.get("gender") == "male" else "أنثى" if session.get("gender") == "female" else "غير محدد",
                "birthYear": session.get("birth_year", "غير محدد"),
                "age": age if age else "غير محدد",
                "educationLevel": session.get("education_level", "غير محدد"),
                "maritalStatus": session.get("marital_status", "غير محدد"),
                "status": "مكتمل" if session.get("status") == "completed" else "نشط" if session.get("status") == "active" else "غير محدد",
                "currentDimension": session.get("current_dimension", "غير محدد"),
                "questionsAnswered": sum(session.get("dimension_question_count", {}).values()),
                "completionDate": completion_date
            }
            participants_list.append(participant)
        
        # Simple pagination
        per_page = 10
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_participants = participants_list[start_idx:end_idx]
        
        total_pages = (len(participants_list) + per_page - 1) // per_page
        
        return {
            "participants": paginated_participants,
            "totalPages": total_pages,
            "currentPage": page,
            "totalParticipants": len(participants_list)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting participants: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Add sample data on startup
    add_sample_data()
    
    # Use environment PORT for Hugging Face, fallback to 8889 for local
    port = int(os.getenv("PORT", 8889))
    uvicorn.run(app, host="0.0.0.0", port=port)
