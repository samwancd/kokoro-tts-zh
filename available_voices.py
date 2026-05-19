#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
获取Kokoro TTS可用语音模型列表
"""

import kokoro
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

def resolve_voice_path(voice: str) -> str:
    if voice.endswith('.pt'):
        return voice
    local_voice = Path(__file__).parent / 'voices' / f'{voice}.pt'
    if local_voice.exists():
        return str(local_voice)
    return voice

def get_available_voices():
    """获取可用的语音模型列表"""
    print("正在初始化Kokoro TTS...")
    pipeline = kokoro.KPipeline('z', repo_id='hexgrad/Kokoro-82M-v1.1-zh', device='cpu')
    
    # 尝试一些常见的语音模型
    test_voices = [
        # 英文女声
        'af_alloy', 'af_echo', 'af_fable', 'af_onyx', 'af_nova', 'af_shimmer',
        'af_heart', 'af_maple', 'af_sky', 'af_nicole', 'af_sarah',
        # 英文男声
        'am_adam', 'am_liam', 'am_ryan', 'am_jason', 'am_michael',
        # 中文女声（常见的）
        'zf_001', 'zf_002', 'zf_003', 'zf_004', 'zf_005',
        # 中文男声（常见的）
        'zm_001', 'zm_002', 'zm_003', 'zm_009', 'zm_012',
    ]
    
    available_voices = []
    failed_voices = []
    
    test_text = "测试"
    
    print(f"\n测试 {len(test_voices)} 个语音模型...")
    
    for voice in test_voices:
        try:
            print(f"测试 {voice}... ", end="")
            resolved = resolve_voice_path(voice)
            results = list(pipeline(test_text, voice=resolved))
            if results:
                available_voices.append(voice)
                print("✅ 可用")
            else:
                failed_voices.append(voice)
                print("❌ 无结果")
        except Exception as e:
            failed_voices.append(voice)
            print(f"❌ 失败: {str(e)[:50]}...")
    
    return available_voices, failed_voices

def main():
    print("🎤 Kokoro TTS 可用语音模型检测")
    print("="*50)
    
    available, failed = get_available_voices()
    
    print("\n" + "="*50)
    print("📊 检测结果")
    print("="*50)
    
    print(f"\n✅ 可用语音模型 ({len(available)} 个):")
    if available:
        # 按类型分组
        english_female = [v for v in available if v.startswith('af_')]
        english_male = [v for v in available if v.startswith('am_')]
        chinese_female = [v for v in available if v.startswith('zf_')]
        chinese_male = [v for v in available if v.startswith('zm_')]
        
        if english_female:
            print(f"  英文女声: {', '.join(english_female)}")
        if english_male:
            print(f"  英文男声: {', '.join(english_male)}")
        if chinese_female:
            print(f"  中文女声: {', '.join(chinese_female)}")
        if chinese_male:
            print(f"  中文男声: {', '.join(chinese_male)}")
    else:
        print("  无可用语音模型")
    
    print(f"\n❌ 不可用语音模型 ({len(failed)} 个):")
    if failed:
        for voice in failed:
            print(f"  {voice}")
    
    # 推荐配置
    if available:
        print(f"\n🌟 推荐使用的语音模型:")
        # 优先推荐成功的语音
        recommendations = []
        for voice in ['af_heart', 'af_maple', 'af_alloy']:
            if voice in available:
                recommendations.append(voice)
                break
        
        if recommendations:
            print(f"  推荐: {recommendations[0]} (测试成功)")
        else:
            print(f"  推荐: {available[0]} (第一个可用)")
    
    print(f"\n💡 提示: 将可用的语音模型更新到测试脚本中")

if __name__ == "__main__":
    main()