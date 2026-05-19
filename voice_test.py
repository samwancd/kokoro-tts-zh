#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kokoro TTS 语音模型性能测试脚本
测试不同语音模型的合成效果和性能
"""

import kokoro
import torchaudio
import soundfile as sf
import os
import time
from pathlib import Path
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

class VoiceModelTester:
    def __init__(self, device='cpu'):
        """初始化测试器"""
        print("正在初始化Kokoro TTS测试器...")
        self.pipeline = kokoro.KPipeline('z', repo_id='hexgrad/Kokoro-82M-v1.1-zh', device=device)
        self.sample_rate = 24000
        self.device = device
        print(f"初始化完成，使用设备: {device}")
    
    def _resolve_voice_path(self, voice: str) -> str:
        """优先使用项目本地音色文件，避免按音色名回退到HF下载。"""
        if voice.endswith('.pt'):
            return voice
        
        local_voice = Path(__file__).parent / 'voices' / f'{voice}.pt'
        if local_voice.exists():
            return str(local_voice)
            
        return voice
    
    def test_voice_model(self, text, voice, output_dir='voice_tests'):
        """测试单个语音模型"""
        resolved_voice = self._resolve_voice_path(voice)
        try:
            start_time = time.time()
            
            # 生成语音
            results = list(self.pipeline(text, voice=resolved_voice))
            
            if not results:
                return {
                    'voice': voice,
                    'success': False,
                    'error': '没有生成结果',
                    'duration': 0,
                    'synthesis_time': 0
                }
            
            result = results[0]
            audio = result.audio
            synthesis_time = time.time() - start_time
            
            # 计算音频时长
            audio_duration = len(audio) / self.sample_rate
            
            # 保存音频文件
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"test_{voice}.wav")
            sf.write(output_path, audio.numpy(), self.sample_rate)
            
            return {
                'voice': voice,
                'success': True,
                'audio_duration': audio_duration,
                'synthesis_time': synthesis_time,
                'speed_ratio': audio_duration / synthesis_time,
                'output_path': output_path,
                'phonemes_count': len(result.phonemes),
                'audio_samples': len(audio)
            }
            
        except Exception as e:
            return {
                'voice': voice,
                'success': False,
                'error': str(e),
                'synthesis_time': time.time() - start_time
            }
    
    def benchmark_voices(self, text="这是一个语音合成测试，用来评估不同语音模型的性能和质量。"):
        """批量测试推荐的语音模型"""
        # 推荐的语音模型列表
        recommended_voices = [
            'af_maple',    # 女声，温暖
            'zf_001',      # 女声，标准
            'zf_004',      # 女声，清晰
            'zm_009',      # 男声，沉稳
            'zm_012',      # 男声，自然
        ]
        
        print(f"开始测试 {len(recommended_voices)} 个推荐语音模型...")
        print(f"测试文本: {text}")
        print("="*60)
        
        results = []
        for voice in recommended_voices:
            print(f"正在测试语音模型: {voice}")
            result = self.test_voice_model(text, voice)
            results.append(result)
            
            if result['success']:
                print(f"  ✅ 成功 - 音频时长: {result['audio_duration']:.2f}s, "
                      f"合成耗时: {result['synthesis_time']:.2f}s, "
                      f"速度比: {result['speed_ratio']:.1f}x")
            else:
                print(f"  ❌ 失败 - {result['error']}")
            print()
        
        return results
    
    def generate_report(self, results):
        """生成测试报告"""
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            print("❌ 所有测试都失败了")
            return
        
        print("\n" + "="*60)
        print("🎤 语音模型性能测试报告")
        print("="*60)
        
        # 统计信息
        total_tests = len(results)
        successful_tests = len(successful_results)
        success_rate = (successful_tests / total_tests) * 100
        
        print(f"📊 测试统计:")
        print(f"  总测试数: {total_tests}")
        print(f"  成功数: {successful_tests}")
        print(f"  成功率: {success_rate:.1f}%")
        print()
        
        # 性能排名
        print(f"⚡ 合成速度排名 (越高越快):")
        speed_sorted = sorted(successful_results, key=lambda x: x['speed_ratio'], reverse=True)
        for i, result in enumerate(speed_sorted, 1):
            print(f"  {i}. {result['voice']}: {result['speed_ratio']:.1f}x 实时")
        print()
        
        # 详细结果
        print(f"📋 详细测试结果:")
        for result in successful_results:
            print(f"\n🎵 {result['voice']}:")
            print(f"  音频时长: {result['audio_duration']:.2f} 秒")
            print(f"  合成耗时: {result['synthesis_time']:.2f} 秒")
            print(f"  速度比例: {result['speed_ratio']:.1f}x 实时")
            print(f"  音频采样: {result['audio_samples']:,} samples")
            print(f"  音素数量: {result['phonemes_count']}")
            print(f"  输出文件: {result['output_path']}")
        
        # 推荐
        print(f"\n🌟 推荐使用:")
        if speed_sorted:
            fastest = speed_sorted[0]
            print(f"  最快合成: {fastest['voice']} ({fastest['speed_ratio']:.1f}x 实时)")
        
        # 平均性能
        avg_speed = sum(r['speed_ratio'] for r in successful_results) / len(successful_results)
        avg_duration = sum(r['audio_duration'] for r in successful_results) / len(successful_results)
        avg_synthesis = sum(r['synthesis_time'] for r in successful_results) / len(successful_results)
        
        print(f"\n📈 平均性能:")
        print(f"  平均音频时长: {avg_duration:.2f} 秒")
        print(f"  平均合成耗时: {avg_synthesis:.2f} 秒")
        print(f"  平均速度比例: {avg_speed:.1f}x 实时")

def main():
    print("🎤 Kokoro TTS 语音模型性能测试")
    print("="*60)
    
    # 初始化测试器
    tester = VoiceModelTester(device='cpu')
    
    # 运行基准测试
    results = tester.benchmark_voices()
    
    # 生成报告
    tester.generate_report(results)
    
    print("\n✅ 测试完成！查看 voice_tests/ 目录获取生成的音频文件。")

if __name__ == "__main__":
    main()