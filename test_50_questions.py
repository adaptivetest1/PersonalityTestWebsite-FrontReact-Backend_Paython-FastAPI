#!/usr/bin/env python3
"""
Test script to verify the 50-question configuration works correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_backend import IRTParameters, adaptive_engine

def test_configuration():
    """Test that the configuration is set correctly for 50 questions"""
    print("🧪 Testing 50-question configuration...")
    
    # Test IRT parameters
    params = IRTParameters()
    
    print(f"✅ Maximum questions: {params.max_questions} (should be 50)")
    print(f"✅ Minimum questions: {params.min_questions} (should be 25)")
    print(f"✅ Max per dimension: {params.max_per_dimension} (should be 10)")
    print(f"✅ Min per dimension: {params.min_per_dimension} (should be 5)")
    
    # Verify calculations
    total_max = params.max_per_dimension * 5  # 5 dimensions
    print(f"✅ Total max questions (5 dimensions × {params.max_per_dimension}): {total_max}")
    
    total_min = params.min_per_dimension * 5  # 5 dimensions
    print(f"✅ Total min questions (5 dimensions × {params.min_per_dimension}): {total_min}")
    
    # Verify adaptive engine uses the same parameters
    print(f"✅ Adaptive engine max questions: {adaptive_engine.irt_params.max_questions}")
    
    # Check if configuration is consistent
    if params.max_questions == 50 and total_max == 50:
        print("🎉 Configuration is correct for 50 questions!")
        return True
    else:
        print("❌ Configuration mismatch!")
        return False

if __name__ == "__main__":
    success = test_configuration()
    if success:
        print("\n🎯 الاختبار سيحتوي على 50 سؤال كحد أقصى (10 أسئلة لكل بُعد)")
        print("📊 شريط التقدم سيعرض النسبة المئوية من 50 سؤال")
        print("⚡ هذا سيجعل الاختبار أسرع وأكثر فعالية!")
    else:
        print("\n❌ يوجد خطأ في التكوين - يرجى مراجعة الإعدادات")
        sys.exit(1)
