import os
import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from typing import List, Tuple, Optional, Union
import soundfile as sf
from tqdm import tqdm


class AudioDataset(Dataset):
    """Dataset class for audio files with vocal/instrumental separation"""
    
    def __init__(self, 
                 data_paths: List[str], 
                 config: dict,
                 mode: str = 'train',
                 transform=None):
        """
        Args:
            data_paths: List of paths to audio files or directories
            config: Configuration dictionary
            mode: 'train', 'val', or 'test'
            transform: Optional audio transformations
        """
        self.data_paths = data_paths
        self.config = config
        self.mode = mode
        self.transform = transform
        
        self.audio_config = config['audio']
        self.sample_rate = self.audio_config['sample_rate']
        self.n_fft = self.audio_config['n_fft']
        self.hop_length = self.audio_config['hop_length']
        self.win_length = self.audio_config['win_length']
        self.chunk_duration = self.audio_config['chunk_duration']
        
        # Collect all audio files
        self.audio_files = self._collect_audio_files()
        
        # Pre-compute chunks for efficient training
        self.chunks = self._prepare_chunks()
        
    def _collect_audio_files(self) -> List[str]:
        """Collect all audio files from given paths"""
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
        audio_files = []
        
        for path in self.data_paths:
            path = Path(path)
            
            if path.is_file() and path.suffix.lower() in audio_extensions:
                audio_files.append(str(path))
            elif path.is_dir():
                for ext in audio_extensions:
                    audio_files.extend([str(f) for f in path.rglob(f'*{ext}')])
        
        print(f"Found {len(audio_files)} audio files")
        return audio_files
    
    def _prepare_chunks(self) -> List[Tuple[str, float, float]]:
        """Prepare audio chunks for training"""
        chunks = []
        chunk_samples = int(self.chunk_duration * self.sample_rate)
        overlap_samples = int(chunk_samples * self.audio_config.get('overlap', 0.25))
        step_samples = chunk_samples - overlap_samples
        
        for audio_file in tqdm(self.audio_files, desc="Preparing chunks"):
            try:
                # Get audio duration without loading the full file
                info = torchaudio.info(audio_file)
                duration = info.num_frames / info.sample_rate
                
                # Create chunks
                start_time = 0
                while start_time + self.chunk_duration <= duration:
                    end_time = start_time + self.chunk_duration
                    chunks.append((audio_file, start_time, end_time))
                    start_time += step_samples / self.sample_rate
                    
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
        
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        """Get a training sample"""
        audio_file, start_time, end_time = self.chunks[idx]
        
        # Load audio chunk
        audio, sr = self._load_audio_chunk(audio_file, start_time, end_time)
        
        # Apply transformations if specified
        if self.transform and self.mode == 'train':
            audio = self.transform(audio)
        
        # Convert to spectrogram
        mixture_spec = self._audio_to_spectrogram(audio)
        
        # For training, we need to create target (instrumental) from mixture
        # This is a simplified approach - in practice, you'd have separate vocal/instrumental tracks
        target_spec = self._create_target_spectrogram(mixture_spec, audio)
        
        return {
            'mixture': mixture_spec,
            'target': target_spec,
            'audio_file': audio_file,
            'start_time': start_time
        }
    
    def _load_audio_chunk(self, audio_file: str, start_time: float, end_time: float) -> torch.Tensor:
        """Load a specific chunk of audio"""
        try:
            # Calculate frame indices
            start_frame = int(start_time * self.sample_rate)
            num_frames = int((end_time - start_time) * self.sample_rate)
            
            # Load audio chunk
            audio, sr = torchaudio.load(
                audio_file, 
                frame_offset=start_frame, 
                num_frames=num_frames
            )
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(audio)
            
            # Ensure stereo
            if audio.shape[0] == 1:
                audio = audio.repeat(2, 1)
            elif audio.shape[0] > 2:
                audio = audio[:2]
            
            return audio
            
        except Exception as e:
            print(f"Error loading {audio_file}: {e}")
            # Return silence as fallback
            return torch.zeros(2, int(self.chunk_duration * self.sample_rate))
    
    def _audio_to_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to complex spectrogram"""
        # Apply STFT to each channel
        spectrograms = []
        
        for channel in range(audio.shape[0]):
            spec = torch.stft(
                audio[channel],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=torch.hann_window(self.win_length),
                return_complex=True
            )
            spectrograms.append(spec)
        
        # Stack channels: [channels, freq_bins, time_frames]
        spectrogram = torch.stack(spectrograms, dim=0)
        
        return spectrogram
    
    def _create_target_spectrogram(self, mixture_spec: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """
        Create target spectrogram for training.
        This is a simplified approach - ideally you'd have ground truth instrumental tracks.
        """
        # For demonstration, we'll use a simple vocal removal technique
        # In practice, you'd want to have actual separated tracks for training
        
        # Convert to magnitude and phase
        magnitude = torch.abs(mixture_spec)
        phase = torch.angle(mixture_spec)
        
        # Simple vocal removal: subtract center channel from sides
        if mixture_spec.shape[0] == 2:  # stereo
            # Center extraction (vocals are typically centered)
            center = (mixture_spec[0] + mixture_spec[1]) / 2
            sides = (mixture_spec[0] - mixture_spec[1]) / 2
            
            # Target is the sides (instrumental)
            target_spec = torch.stack([sides, sides], dim=0)
        else:
            # Fallback: use original (this won't work well for training)
            target_spec = mixture_spec
        
        return target_spec


class AudioTransforms:
    """Audio augmentation transforms"""
    
    def __init__(self, config: dict):
        self.config = config
        self.aug_config = config.get('augmentation', {})
        self.enabled = self.aug_config.get('enabled', False)
        
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return audio
        
        # Apply random transformations
        if random.random() < 0.5:
            audio = self._pitch_shift(audio)
        
        if random.random() < 0.3:
            audio = self._time_stretch(audio)
        
        if random.random() < 0.2:
            audio = self._add_noise(audio)
        
        return audio
    
    def _pitch_shift(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply random pitch shift"""
        shift_range = self.aug_config.get('pitch_shift_range', [-2, 2])
        shift = random.uniform(shift_range[0], shift_range[1])
        
        # Convert to numpy for librosa processing
        audio_np = audio.numpy()
        shifted_audio = []
        
        for channel in range(audio_np.shape[0]):
            shifted = librosa.effects.pitch_shift(
                audio_np[channel], 
                sr=self.config['audio']['sample_rate'], 
                n_steps=shift
            )
            shifted_audio.append(shifted)
        
        return torch.from_numpy(np.stack(shifted_audio))
    
    def _time_stretch(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply random time stretching"""
        stretch_range = self.aug_config.get('time_stretch_range', [0.9, 1.1])
        rate = random.uniform(stretch_range[0], stretch_range[1])
        
        audio_np = audio.numpy()
        stretched_audio = []
        
        for channel in range(audio_np.shape[0]):
            stretched = librosa.effects.time_stretch(audio_np[channel], rate=rate)
            # Ensure same length
            if len(stretched) > audio_np.shape[1]:
                stretched = stretched[:audio_np.shape[1]]
            elif len(stretched) < audio_np.shape[1]:
                stretched = np.pad(stretched, (0, audio_np.shape[1] - len(stretched)))
            stretched_audio.append(stretched)
        
        return torch.from_numpy(np.stack(stretched_audio))
    
    def _add_noise(self, audio: torch.Tensor) -> torch.Tensor:
        """Add random noise"""
        noise_factor = self.aug_config.get('noise_factor', 0.01)
        noise = torch.randn_like(audio) * noise_factor
        return audio + noise


def create_data_loaders(data_paths: List[str], config: dict) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    
    # Create transforms
    transforms = AudioTransforms(config)
    
    # Create full dataset
    full_dataset = AudioDataset(data_paths, config, mode='train', transform=transforms)
    
    # Split into train/validation
    val_split = config['training']['validation_split']
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_inference_dataset(audio_path: str, config: dict) -> AudioDataset:
    """Create dataset for inference on a single audio file"""
    return AudioDataset([audio_path], config, mode='test')