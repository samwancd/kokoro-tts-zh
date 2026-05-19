#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kokoro TTS 中文版 - 本地Web测试应用
提供Web界面测试TTS功能，支持音色选择和实时生成
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_file, url_for
from flask_cors import CORS
import torch
import numpy as np
import soundfile as sf
import tempfile
import base64
import io

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

# 尝试导入Kokoro
try:
    from kokoro import KModel, KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    print("⚠️  警告: kokoro包未安装，部分功能将不可用")
    print("请运行: pip install kokoro>=0.8.2 'misaki[zh]>=0.8.2'")

app = Flask(__name__)
CORS(app)

# 配置
REPO_ID = 'hexgrad/Kokoro-82M-v1.1-zh'
SAMPLE_RATE = 24000
try:
    MAX_TEXT_LENGTH = max(1, int(os.getenv('MAX_TEXT_LENGTH', '500')))
except ValueError:
    MAX_TEXT_LENGTH = 500
OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

# 全局变量
model = None
zh_pipeline = None
en_pipelines = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def resolve_voice_path(voice: str) -> str:
    """优先使用项目本地音色文件，避免按音色名回退到HF下载。"""
    if voice.endswith('.pt'):
        return voice

    local_voice = Path(__file__).parent / 'voices' / f'{voice}.pt'
    if local_voice.exists():
        return str(local_voice)

    return voice

def get_available_voices():
    """获取可用的音色列表"""
    voices_dir = Path(__file__).parent / 'voices'
    if not voices_dir.exists():
        return {'female': [], 'male': [], 'english': []}
    
    voice_files = list(voices_dir.glob('*.pt'))
    
    voices = {
        'female': [],
        'male': [],
        'english': []
    }
    
    for voice_file in voice_files:
        name = voice_file.stem
        if name.startswith('zf_'):
            voices['female'].append(name)
        elif name.startswith('zm_'):
            voices['male'].append(name)
        elif name.startswith(('af_', 'bf_')):
            voices['english'].append(name)
    
    # 排序
    for category in voices:
        voices[category].sort()
    
    return voices

def init_model():
    """初始化模型"""
    global model, zh_pipeline, en_pipelines
    
    if not KOKORO_AVAILABLE:
        return False
    
    try:
        print(f"🚀 正在初始化模型... (设备: {device})")
        
        # 加载模型
        model = KModel(repo_id=REPO_ID).to(device).eval()
        
        # 初始化中文pipeline
        en_pipeline = KPipeline(lang_code='a', repo_id=REPO_ID, model=False)
        def en_callable(text):
            if text == 'Kokoro':
                return 'kˈOkəɹO'
            elif text == 'Sol':
                return 'sˈOl'
            return next(en_pipeline(text)).phonemes
        
        zh_pipeline = KPipeline(lang_code='z', repo_id=REPO_ID, model=model, en_callable=en_callable)
        
        # 初始化英文pipelines
        en_pipelines = [KPipeline(lang_code='b' if british else 'a', repo_id=REPO_ID, model=model) 
                       for british in (False, True)]
        
        print("✅ 模型初始化完成")
        return True
        
    except Exception as e:
        print(f"❌ 模型初始化失败: {str(e)}")
        return False

def speed_callable(len_ps):
    """动态调整语速"""
    speed = 0.8
    if len_ps <= 83:
        speed = 1
    elif len_ps < 183:
        speed = 1 - (len_ps - 83) / 500
    return speed * 1.1

def synthesize_speech(text: str, voice: str, language: str):
    """执行语音生成并返回统一结果，供JSON接口和二进制接口复用。"""
    if not text:
        raise ValueError('文本不能为空')

    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f'文本长度不能超过{MAX_TEXT_LENGTH}字符')

    start_time = time.time()
    resolved_voice = resolve_voice_path(voice)

    if language == 'zh' or voice.startswith(('zf_', 'zm_')):
        generator = zh_pipeline(text, voice=resolved_voice, speed=speed_callable)
        result = next(generator)
        wav = result.audio
    else:
        british = voice.startswith('bf_')
        generator = en_pipelines[british](text, voice=resolved_voice)
        result = next(generator)
        wav = result.audio

    generation_time = time.time() - start_time
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    filename = f'tts_{voice}_{timestamp}.wav'
    filepath = OUTPUT_DIR / filename
    sf.write(filepath, wav, SAMPLE_RATE)

    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, wav, SAMPLE_RATE, format='wav')
    audio_bytes = audio_buffer.getvalue()

    return {
        'audio_bytes': audio_bytes,
        'filename': filename,
        'generation_time': round(generation_time, 3),
        'audio_length': round(len(wav) / SAMPLE_RATE, 2),
        'sample_rate': SAMPLE_RATE,
        'voice': voice,
        'text_length': len(text),
        'download_url': url_for('api_download', filename=filename)
    }

@app.route('/')
def index():
    """主页"""
    voices = get_available_voices()
    return render_template('index.html', 
                         voices=voices, 
                         kokoro_available=KOKORO_AVAILABLE,
                         device=device,
                         max_text_length=MAX_TEXT_LENGTH)

@app.route('/api/voices')
def api_voices():
    """API: 获取可用音色"""
    return jsonify(get_available_voices())

@app.route('/api/generate', methods=['POST'])
def api_generate():
    """API: 生成语音"""
    if not KOKORO_AVAILABLE:
        return jsonify({
            'success': False, 
            'error': 'Kokoro模块未安装'
        }), 500
    
    if model is None:
        return jsonify({
            'success': False, 
            'error': '模型未初始化'
        }), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        voice = data.get('voice', 'zf_001')
        language = data.get('language', 'zh')
        result = synthesize_speech(text, voice, language)
        audio_base64 = base64.b64encode(result['audio_bytes']).decode('utf-8')
        
        return jsonify({
            'success': True,
            'audio_data': f'data:audio/wav;base64,{audio_base64}',
            'filename': result['filename'],
            'generation_time': result['generation_time'],
            'audio_length': result['audio_length'],
            'sample_rate': result['sample_rate'],
            'voice': result['voice'],
            'text_length': result['text_length'],
            'download_url': result['download_url']
        })
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': f'生成失败: {str(e)}'
        }), 500

@app.route('/api/generate-file', methods=['POST'])
def api_generate_file():
    """API: 直接返回WAV二进制流，适合curl等外部客户端。"""
    if not KOKORO_AVAILABLE:
        return jsonify({'success': False, 'error': 'Kokoro模块未安装'}), 500

    if model is None:
        return jsonify({'success': False, 'error': '模型未初始化'}), 500

    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        voice = data.get('voice', 'zf_001')
        language = data.get('language', 'zh')
        result = synthesize_speech(text, voice, language)

        response = send_file(
            io.BytesIO(result['audio_bytes']),
            mimetype='audio/wav',
            as_attachment=True,
            download_name=result['filename']
        )
        response.headers['X-TTS-Filename'] = result['filename']
        response.headers['X-TTS-Voice'] = result['voice']
        response.headers['X-TTS-Text-Length'] = str(result['text_length'])
        response.headers['X-TTS-Audio-Length'] = str(result['audio_length'])
        response.headers['X-TTS-Generation-Time'] = str(result['generation_time'])
        response.headers['X-TTS-Sample-Rate'] = str(result['sample_rate'])
        response.headers['X-TTS-Download-URL'] = result['download_url']
        response.headers['Access-Control-Expose-Headers'] = (
            'Content-Disposition, X-TTS-Filename, X-TTS-Voice, X-TTS-Text-Length, '
            'X-TTS-Audio-Length, X-TTS-Generation-Time, X-TTS-Sample-Rate, X-TTS-Download-URL'
        )
        return response
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': f'生成失败: {str(e)}'}), 500

@app.route('/api/download/<filename>')
def api_download(filename):
    """API: 下载生成的音频文件"""
    filepath = OUTPUT_DIR / filename
    if filepath.exists():
        return send_file(filepath, as_attachment=True)
    else:
        return jsonify({'error': '文件不存在'}), 404

@app.route('/api/status')
def api_status():
    """API: 获取系统状态"""
    voices = get_available_voices()
    total_voices = sum(len(v) for v in voices.values())
    
    return jsonify({
        'kokoro_available': KOKORO_AVAILABLE,
        'model_loaded': model is not None,
        'device': device,
        'total_voices': total_voices,
        'voices_by_category': {k: len(v) for k, v in voices.items()},
        'max_text_length': MAX_TEXT_LENGTH,
        'sample_rate': SAMPLE_RATE
    })

if __name__ == '__main__':
    print("🎵 Kokoro TTS 中文版 - 本地测试应用")
    print("=" * 50)
    
    # 检查声音库
    voices = get_available_voices()
    total_voices = sum(len(v) for v in voices.values())
    print(f"📊 发现音色: {total_voices} 个")
    print(f"   - 女声: {len(voices['female'])} 个")
    print(f"   - 男声: {len(voices['male'])} 个")
    print(f"   - 英文: {len(voices['english'])} 个")
    
    # 初始化模型
    if KOKORO_AVAILABLE:
        if init_model():
            print("✅ 模型准备就绪")
        else:
            print("❌ 模型初始化失败")
    
    print(f"\n🌐 启动Web服务器...")
    print(f"📱 访问地址: http://localhost:5001")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5001)
