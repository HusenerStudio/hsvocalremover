#!/usr/bin/env python3
"""
Simple ONNX Model Handler

This module provides a workaround for ONNX Runtime compatibility issues
by converting ONNX models to PyTorch format for inference.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import json

try:
    import onnx
    from onnx2torch import convert
    ONNX2TORCH_AVAILABLE = True
except ImportError:
    ONNX2TORCH_AVAILABLE = False


class SimpleONNXInference:
    """Simple ONNX model inference using onnx2torch conversion"""
    
    def __init__(self, model_path: str):
        """
        Initialize simple ONNX inference
        
        Args:
            model_path: Path to ONNX model file
        """
        self.model_path = model_path
        self.model = None
        self.input_shape = None
        self.output_shape = None
        
        if not ONNX2TORCH_AVAILABLE:
            raise ImportError("onnx2torch is required. Install with: pip install onnx2torch")
        
        self._load_model()
    
    def _load_model(self):
        """Load and convert ONNX model to PyTorch"""
        try:
            # Load ONNX model
            onnx_model = onnx.load(self.model_path)
            
            # Convert to PyTorch
            self.model = convert(onnx_model)
            self.model.eval()
            
            # Get input/output shapes from ONNX model
            self.input_shape = [dim.dim_value for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim]
            self.output_shape = [dim.dim_value for dim in onnx_model.graph.output[0].type.tensor_type.shape.dim]
            
            print(f"ONNX model converted to PyTorch successfully")
            print(f"Input shape: {self.input_shape}")
            print(f"Output shape: {self.output_shape}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to convert ONNX model: {e}")
    
    def separate_vocals(self, audio: torch.Tensor, audio_config: dict) -> torch.Tensor:
        """
        Separate vocals using converted PyTorch model
        
        Args:
            audio: Input audio tensor [channels, samples]
            audio_config: Audio configuration dictionary
            
        Returns:
            Separated audio tensor [channels, samples]
        """
        from .utils import audio_to_spectrogram, spectrogram_to_audio
        
        # Convert to spectrogram
        spectrogram = audio_to_spectrogram(audio, audio_config)
        
        # Prepare input (add batch dimension)
        input_tensor = spectrogram.unsqueeze(0)
        
        # Handle complex numbers - convert to real representation
        if input_tensor.dtype == torch.complex64 or input_tensor.dtype == torch.complex128:
            # Stack real and imaginary parts
            real_part = input_tensor.real
            imag_part = input_tensor.imag
            input_tensor = torch.stack([real_part, imag_part], dim=-1)
            input_tensor = input_tensor.view(input_tensor.shape[0], -1, input_tensor.shape[-2], input_tensor.shape[-1])
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Handle output format
        if len(output.shape) == 5:  # [batch, channels*2, freq, time, 2] (real/imag)
            # Reshape back to complex
            batch_size, channels_x2, freq, time, _ = output.shape
            channels = channels_x2 // 2
            output = output.view(batch_size, channels, 2, freq, time)
            output_spec = torch.complex(output[:, :, 0], output[:, :, 1])
        else:
            # Assume it's already in the right format
            output_spec = output
        
        # Remove batch dimension
        output_spec = output_spec.squeeze(0)
        
        # Convert back to audio
        separated_audio = spectrogram_to_audio(output_spec, audio_config)
        
        return separated_audio
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            'model_path': self.model_path,
            'model_type': 'ONNX (converted to PyTorch)',
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'conversion_method': 'onnx2torch'
        }


def is_simple_onnx_available() -> bool:
    """Check if simple ONNX conversion is available"""
    return ONNX2TORCH_AVAILABLE


def install_onnx2torch_instructions() -> str:
    """Get installation instructions for onnx2torch"""
    return """
To enable ONNX model support without ONNX Runtime, install onnx2torch:

pip install onnx2torch

This will allow converting ONNX models to PyTorch format for inference.
"""


class FallbackONNXHandler:
    """Fallback handler when onnx2torch is not available"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        raise ImportError(
            "ONNX model support requires onnx2torch. " + 
            install_onnx2torch_instructions()
        )
    
    @staticmethod
    def is_available() -> bool:
        return False


# Factory function
def create_simple_onnx_inference(model_path: str):
    """Create simple ONNX inference handler"""
    if ONNX2TORCH_AVAILABLE:
        return SimpleONNXInference(model_path)
    else:
        return FallbackONNXHandler(model_path)