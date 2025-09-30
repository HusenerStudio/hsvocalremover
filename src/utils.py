import torch
import torchaudio
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import soundfile as sf


def create_directories(paths: List[Path]):
    """Create directories if they don't exist"""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, config_path: str):
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def spectrogram_to_audio(spectrogram: torch.Tensor, audio_config: dict) -> torch.Tensor:
    """Convert complex spectrogram back to audio waveform"""
    
    # Get audio parameters
    n_fft = audio_config['n_fft']
    hop_length = audio_config['hop_length']
    win_length = audio_config['win_length']
    
    # Convert each channel
    audio_channels = []
    
    for channel in range(spectrogram.shape[0]):
        # Apply inverse STFT
        audio = torch.istft(
            spectrogram[channel],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length),
            return_complex=False
        )
        audio_channels.append(audio)
    
    # Stack channels
    audio = torch.stack(audio_channels, dim=0)
    
    return audio


def audio_to_spectrogram(audio: torch.Tensor, audio_config: dict) -> torch.Tensor:
    """Convert audio waveform to complex spectrogram"""
    
    # Get audio parameters
    n_fft = audio_config['n_fft']
    hop_length = audio_config['hop_length']
    win_length = audio_config['win_length']
    
    # Convert each channel
    spectrograms = []
    
    for channel in range(audio.shape[0]):
        spec = torch.stft(
            audio[channel],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length),
            return_complex=True
        )
        spectrograms.append(spec)
    
    # Stack channels: [channels, freq_bins, time_frames]
    spectrogram = torch.stack(spectrograms, dim=0)
    
    return spectrogram


def normalize_audio(audio: torch.Tensor, target_db: float = -20.0) -> torch.Tensor:
    """Normalize audio to target dB level"""
    
    # Calculate RMS
    rms = torch.sqrt(torch.mean(audio ** 2))
    
    # Convert target dB to linear scale
    target_rms = 10 ** (target_db / 20.0)
    
    # Normalize
    if rms > 0:
        audio = audio * (target_rms / rms)
    
    # Clip to prevent clipping
    audio = torch.clamp(audio, -1.0, 1.0)
    
    return audio


def compute_snr(clean: torch.Tensor, noisy: torch.Tensor) -> float:
    """Compute Signal-to-Noise Ratio"""
    
    # Compute power of clean and noise signals
    clean_power = torch.mean(clean ** 2)
    noise_power = torch.mean((noisy - clean) ** 2)
    
    # Avoid division by zero
    if noise_power == 0:
        return float('inf')
    
    # Compute SNR in dB
    snr = 10 * torch.log10(clean_power / noise_power)
    
    return snr.item()


def compute_sdr(reference: torch.Tensor, estimation: torch.Tensor) -> float:
    """Compute Source-to-Distortion Ratio"""
    
    # Flatten tensors
    reference = reference.flatten()
    estimation = estimation.flatten()
    
    # Compute optimal scaling factor
    alpha = torch.sum(reference * estimation) / torch.sum(reference ** 2)
    
    # Compute scaled reference
    scaled_reference = alpha * reference
    
    # Compute distortion
    distortion = estimation - scaled_reference
    
    # Compute SDR
    reference_power = torch.sum(scaled_reference ** 2)
    distortion_power = torch.sum(distortion ** 2)
    
    if distortion_power == 0:
        return float('inf')
    
    sdr = 10 * torch.log10(reference_power / distortion_power)
    
    return sdr.item()


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load training checkpoint"""
    checkpoint = torch.load(path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']


def get_audio_info(audio_path: str) -> Dict:
    """Get audio file information"""
    try:
        info = torchaudio.info(audio_path)
        return {
            'sample_rate': info.sample_rate,
            'num_channels': info.num_channels,
            'num_frames': info.num_frames,
            'duration': info.num_frames / info.sample_rate,
            'bits_per_sample': info.bits_per_sample,
            'encoding': str(info.encoding)
        }
    except Exception as e:
        print(f"Error getting info for {audio_path}: {e}")
        return {}


def validate_audio_file(audio_path: str, min_duration: float = 1.0) -> bool:
    """Validate if audio file is suitable for processing"""
    try:
        info = get_audio_info(audio_path)
        
        # Check if file exists and has basic properties
        if not info:
            return False
        
        # Check minimum duration
        if info['duration'] < min_duration:
            return False
        
        # Check if it's stereo or mono (we can handle both)
        if info['num_channels'] not in [1, 2]:
            return False
        
        return True
        
    except Exception:
        return False


def convert_to_stereo(audio: torch.Tensor) -> torch.Tensor:
    """Convert mono audio to stereo"""
    if audio.shape[0] == 1:
        # Duplicate mono channel
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        # Take first two channels
        audio = audio[:2]
    
    return audio


def apply_fade(audio: torch.Tensor, fade_samples: int = 1024) -> torch.Tensor:
    """Apply fade in/out to audio to prevent clicks"""
    if audio.shape[1] <= 2 * fade_samples:
        return audio
    
    # Create fade curves
    fade_in = torch.linspace(0, 1, fade_samples)
    fade_out = torch.linspace(1, 0, fade_samples)
    
    # Apply fade in
    audio[:, :fade_samples] *= fade_in
    
    # Apply fade out
    audio[:, -fade_samples:] *= fade_out
    
    return audio


def chunk_audio(audio: torch.Tensor, chunk_size: int, overlap: int = 0) -> List[torch.Tensor]:
    """Split audio into overlapping chunks"""
    chunks = []
    step = chunk_size - overlap
    
    for start in range(0, audio.shape[1] - chunk_size + 1, step):
        end = start + chunk_size
        chunk = audio[:, start:end]
        chunks.append(chunk)
    
    return chunks


def reconstruct_audio(chunks: List[torch.Tensor], overlap: int = 0) -> torch.Tensor:
    """Reconstruct audio from overlapping chunks"""
    if not chunks:
        return torch.empty(0)
    
    chunk_size = chunks[0].shape[1]
    step = chunk_size - overlap
    total_length = (len(chunks) - 1) * step + chunk_size
    
    # Initialize output
    audio = torch.zeros(chunks[0].shape[0], total_length)
    weight = torch.zeros(total_length)
    
    # Add chunks with overlap handling
    for i, chunk in enumerate(chunks):
        start = i * step
        end = start + chunk_size
        
        audio[:, start:end] += chunk
        weight[start:end] += 1
    
    # Normalize by weight to handle overlaps
    weight = torch.clamp(weight, min=1)
    audio = audio / weight
    
    return audio


def calculate_model_size(model: torch.nn.Module) -> Dict[str, int]:
    """Calculate model size statistics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': model_size_mb
    }


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


def create_mel_spectrogram(audio: torch.Tensor, sample_rate: int, n_mels: int = 128) -> torch.Tensor:
    """Create mel spectrogram for visualization"""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=2048,
        hop_length=512
    )
    
    mel_spec = mel_transform(audio)
    
    # Convert to dB
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    
    return mel_spec_db