#!/usr/bin/env python3
"""
ONNX Inference Module for Vocal Remover

This module provides ONNX runtime support for faster inference
with optimized models.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import torch
import torchaudio

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX Runtime not available. Install with: pip install onnxruntime")

from .utils import audio_to_spectrogram, spectrogram_to_audio


class ONNXVocalRemover:
    """ONNX-based vocal remover for optimized inference"""
    
    def __init__(self, model_path: str, providers: Optional[list] = None):
        """
        Initialize ONNX vocal remover
        
        Args:
            model_path: Path to ONNX model file
            providers: List of execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime is required. Install with: pip install onnxruntime")
        
        self.model_path = model_path
        
        # Set default providers
        if providers is None:
            providers = ['CPUExecutionProvider']
            # Try to use CUDA if available
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Create ONNX Runtime session
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get model input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        print(f"ONNX model loaded: {model_path}")
        print(f"Providers: {self.session.get_providers()}")
        print(f"Input shape: {self.input_shape}")
    
    def separate_vocals(self, audio: torch.Tensor, audio_config: dict) -> torch.Tensor:
        """
        Separate vocals using ONNX model
        
        Args:
            audio: Input audio tensor [channels, samples]
            audio_config: Audio configuration dictionary
            
        Returns:
            Separated audio tensor [channels, samples]
        """
        # Convert to spectrogram
        spectrogram = audio_to_spectrogram(audio, audio_config)
        
        # Prepare input for ONNX (add batch dimension)
        input_data = spectrogram.unsqueeze(0).numpy().astype(np.float32)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        
        # Convert output back to torch tensor
        output_spec = torch.from_numpy(outputs[0]).squeeze(0)
        
        # Convert back to audio
        separated_audio = spectrogram_to_audio(output_spec, audio_config)
        
        return separated_audio
    
    def get_model_info(self) -> dict:
        """Get ONNX model information"""
        return {
            'model_path': self.model_path,
            'providers': self.session.get_providers(),
            'input_shape': self.input_shape,
            'input_name': self.input_name,
            'output_name': self.output_name
        }


def convert_pytorch_to_onnx(pytorch_model_path: str, 
                           onnx_model_path: str,
                           config: dict,
                           opset_version: int = 11) -> str:
    """
    Convert PyTorch model to ONNX format
    
    Args:
        pytorch_model_path: Path to PyTorch model
        onnx_model_path: Path to save ONNX model
        config: Model configuration
        opset_version: ONNX opset version
        
    Returns:
        Path to converted ONNX model
    """
    from .model import create_model
    
    # Load PyTorch model
    model = create_model(config)
    checkpoint = torch.load(pytorch_model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    audio_config = config['audio']
    n_fft = audio_config['n_fft']
    chunk_samples = int(audio_config['chunk_duration'] * audio_config['sample_rate'])
    
    # Dummy spectrogram input [batch, channels, freq_bins, time_frames]
    freq_bins = n_fft // 2 + 1
    time_frames = chunk_samples // audio_config['hop_length'] + 1
    dummy_input = torch.randn(1, 2, freq_bins, time_frames, dtype=torch.complex64)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 3: 'time_frames'},
            'output': {0: 'batch_size', 3: 'time_frames'}
        }
    )
    
    print(f"Model converted to ONNX: {onnx_model_path}")
    return onnx_model_path


def optimize_onnx_model(model_path: str, optimized_path: str) -> str:
    """
    Optimize ONNX model for better performance
    
    Args:
        model_path: Path to original ONNX model
        optimized_path: Path to save optimized model
        
    Returns:
        Path to optimized model
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX Runtime is required")
    
    # Load and optimize model
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.optimized_model_filepath = optimized_path
    
    # Create session to trigger optimization
    session = ort.InferenceSession(model_path, session_options)
    
    print(f"Model optimized: {optimized_path}")
    return optimized_path


class ModelConverter:
    """Utility class for model format conversion"""
    
    @staticmethod
    def pytorch_to_onnx(pytorch_path: str, onnx_path: str, config: dict) -> str:
        """Convert PyTorch model to ONNX"""
        return convert_pytorch_to_onnx(pytorch_path, onnx_path, config)
    
    @staticmethod
    def optimize_onnx(model_path: str, optimized_path: str) -> str:
        """Optimize ONNX model"""
        return optimize_onnx_model(model_path, optimized_path)
    
    @staticmethod
    def get_model_format(model_path: str) -> str:
        """Detect model format from file extension"""
        suffix = Path(model_path).suffix.lower()
        
        format_map = {
            '.pth': 'pytorch',
            '.pt': 'pytorch',
            '.ckpt': 'checkpoint',
            '.onnx': 'onnx',
            '.pkl': 'pickle'
        }
        
        return format_map.get(suffix, 'unknown')
    
    @staticmethod
    def is_onnx_available() -> bool:
        """Check if ONNX Runtime is available"""
        return ONNX_AVAILABLE