import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import argparse
import time

from .model import create_model
from .utils import (
    load_config, 
    audio_to_spectrogram, 
    spectrogram_to_audio,
    normalize_audio,
    apply_fade,
    chunk_audio,
    reconstruct_audio,
    get_audio_info,
    validate_audio_file
)


class VocalRemover:
    """Vocal removal inference pipeline"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize vocal remover
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file (optional if config is in checkpoint)
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and configuration
        self.model, self.config = self._load_model_and_config(model_path, config_path)
        self.audio_config = self.config['audio']
        
        print(f"Vocal remover initialized on {self.device}")
        print(f"Model sample rate: {self.audio_config['sample_rate']} Hz")
    
    def _load_model_and_config(self, model_path: str, config_path: Optional[str]) -> Tuple[torch.nn.Module, dict]:
        """Load model and configuration"""
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get configuration
        if 'config' in checkpoint:
            config = checkpoint['config']
        elif config_path:
            config = load_config(config_path)
        else:
            raise ValueError("Configuration not found in checkpoint and no config_path provided")
        
        # Create and load model
        model = create_model(config).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, config
    
    def separate_vocals(self, 
                       input_path: str, 
                       output_path: str,
                       chunk_duration: Optional[float] = None,
                       overlap_ratio: float = 0.25,
                       normalize_output: bool = True,
                       apply_fade_in_out: bool = True) -> dict:
        """
        Separate vocals from audio file
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save separated audio
            chunk_duration: Duration of chunks in seconds (None for auto)
            overlap_ratio: Overlap ratio between chunks
            normalize_output: Whether to normalize output audio
            apply_fade_in_out: Whether to apply fade in/out
            
        Returns:
            Dictionary with processing information
        """
        
        # Validate input file
        if not validate_audio_file(input_path):
            raise ValueError(f"Invalid audio file: {input_path}")
        
        print(f"Processing: {input_path}")
        start_time = time.time()
        
        # Load audio
        audio, original_sr = torchaudio.load(input_path)
        original_duration = audio.shape[1] / original_sr
        
        print(f"Original audio: {audio.shape[0]} channels, {original_sr} Hz, {original_duration:.2f}s")
        
        # Resample if necessary
        target_sr = self.audio_config['sample_rate']
        if original_sr != target_sr:
            resampler = torchaudio.transforms.Resample(original_sr, target_sr)
            audio = resampler(audio)
            print(f"Resampled to {target_sr} Hz")
        
        # Ensure stereo
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
            print("Converted mono to stereo")
        elif audio.shape[0] > 2:
            audio = audio[:2]
            print("Reduced to stereo from multi-channel")
        
        # Determine chunk size
        if chunk_duration is None:
            chunk_duration = self.audio_config.get('chunk_duration', 10.0)
        
        chunk_samples = int(chunk_duration * target_sr)
        overlap_samples = int(chunk_samples * overlap_ratio)
        
        # Process audio
        if audio.shape[1] <= chunk_samples:
            # Process entire audio at once
            separated_audio = self._process_chunk(audio)
        else:
            # Process in chunks
            separated_audio = self._process_in_chunks(
                audio, chunk_samples, overlap_samples
            )
        
        # Post-processing
        if normalize_output:
            separated_audio = normalize_audio(separated_audio)
        
        if apply_fade_in_out:
            separated_audio = apply_fade(separated_audio)
        
        # Save output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torchaudio.save(str(output_path), separated_audio, target_sr)
        
        # Processing info
        processing_time = time.time() - start_time
        processing_info = {
            'input_path': input_path,
            'output_path': str(output_path),
            'original_duration': original_duration,
            'processing_time': processing_time,
            'real_time_factor': original_duration / processing_time,
            'original_sample_rate': original_sr,
            'output_sample_rate': target_sr,
            'channels': separated_audio.shape[0],
            'chunk_duration': chunk_duration,
            'overlap_ratio': overlap_ratio
        }
        
        print(f"Vocal separation completed in {processing_time:.2f}s")
        print(f"Real-time factor: {processing_info['real_time_factor']:.2f}x")
        print(f"Output saved to: {output_path}")
        
        return processing_info
    
    def _process_chunk(self, audio_chunk: torch.Tensor) -> torch.Tensor:
        """Process a single audio chunk"""
        
        with torch.no_grad():
            # Convert to spectrogram
            spectrogram = audio_to_spectrogram(audio_chunk, self.audio_config)
            
            # Add batch dimension and move to device
            spectrogram_batch = spectrogram.unsqueeze(0).to(self.device)
            
            # Run inference
            separated_spec = self.model(spectrogram_batch)
            
            # Remove batch dimension and move to CPU
            separated_spec = separated_spec.squeeze(0).cpu()
            
            # Convert back to audio
            separated_audio = spectrogram_to_audio(separated_spec, self.audio_config)
            
            return separated_audio
    
    def _process_in_chunks(self, 
                          audio: torch.Tensor, 
                          chunk_samples: int, 
                          overlap_samples: int) -> torch.Tensor:
        """Process audio in overlapping chunks"""
        
        # Split into chunks
        chunks = chunk_audio(audio, chunk_samples, overlap_samples)
        
        print(f"Processing {len(chunks)} chunks...")
        
        # Process each chunk
        separated_chunks = []
        for i, chunk in enumerate(chunks):
            if i % 10 == 0:
                print(f"Processing chunk {i+1}/{len(chunks)}")
            
            separated_chunk = self._process_chunk(chunk)
            separated_chunks.append(separated_chunk)
        
        # Reconstruct audio from chunks
        separated_audio = reconstruct_audio(separated_chunks, overlap_samples)
        
        return separated_audio
    
    def batch_process(self, 
                     input_dir: str, 
                     output_dir: str,
                     file_extensions: List[str] = None,
                     **kwargs) -> List[dict]:
        """
        Process multiple audio files in a directory
        
        Args:
            input_dir: Directory containing input audio files
            output_dir: Directory to save processed files
            file_extensions: List of file extensions to process
            **kwargs: Additional arguments for separate_vocals
            
        Returns:
            List of processing information dictionaries
        """
        
        if file_extensions is None:
            file_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg']
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all audio files
        audio_files = []
        for ext in file_extensions:
            audio_files.extend(input_path.rglob(f'*{ext}'))
        
        print(f"Found {len(audio_files)} audio files to process")
        
        # Process each file
        results = []
        for i, audio_file in enumerate(audio_files):
            print(f"\n--- Processing file {i+1}/{len(audio_files)} ---")
            
            # Create output path
            relative_path = audio_file.relative_to(input_path)
            output_file = output_path / relative_path.with_suffix('.wav')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                result = self.separate_vocals(
                    str(audio_file), 
                    str(output_file),
                    **kwargs
                )
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                results.append({
                    'input_path': str(audio_file),
                    'error': str(e)
                })
        
        print(f"\nBatch processing completed. Processed {len(results)} files.")
        return results
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': type(self.model).__name__,
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'audio_config': self.audio_config,
            'model_config': self.config.get('model', {})
        }


def main():
    """Command line interface for vocal removal"""
    
    parser = argparse.ArgumentParser(description="Vocal Removal Inference")
    parser.add_argument("--model", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--config", help="Path to configuration file (optional)")
    parser.add_argument("--input", required=True, help="Input audio file or directory")
    parser.add_argument("--output", required=True, help="Output audio file or directory")
    parser.add_argument("--device", choices=['cpu', 'cuda'], help="Device to use for inference")
    parser.add_argument("--chunk-duration", type=float, default=10.0, 
                       help="Chunk duration in seconds")
    parser.add_argument("--overlap-ratio", type=float, default=0.25,
                       help="Overlap ratio between chunks")
    parser.add_argument("--no-normalize", action='store_true',
                       help="Skip output normalization")
    parser.add_argument("--no-fade", action='store_true',
                       help="Skip fade in/out")
    parser.add_argument("--batch", action='store_true',
                       help="Process directory in batch mode")
    
    args = parser.parse_args()
    
    # Initialize vocal remover
    vocal_remover = VocalRemover(
        model_path=args.model,
        config_path=args.config,
        device=args.device
    )
    
    # Print model info
    model_info = vocal_remover.get_model_info()
    print(f"Model: {model_info['model_type']}")
    print(f"Parameters: {model_info['total_parameters']:,}")
    print(f"Device: {model_info['device']}")
    
    # Process audio
    if args.batch or Path(args.input).is_dir():
        # Batch processing
        results = vocal_remover.batch_process(
            input_dir=args.input,
            output_dir=args.output,
            chunk_duration=args.chunk_duration,
            overlap_ratio=args.overlap_ratio,
            normalize_output=not args.no_normalize,
            apply_fade_in_out=not args.no_fade
        )
        
        # Print summary
        successful = sum(1 for r in results if 'error' not in r)
        print(f"\nSummary: {successful}/{len(results)} files processed successfully")
        
    else:
        # Single file processing
        result = vocal_remover.separate_vocals(
            input_path=args.input,
            output_path=args.output,
            chunk_duration=args.chunk_duration,
            overlap_ratio=args.overlap_ratio,
            normalize_output=not args.no_normalize,
            apply_fade_in_out=not args.no_fade
        )
        
        print(f"\nProcessing completed:")
        print(f"Real-time factor: {result['real_time_factor']:.2f}x")


if __name__ == "__main__":
    main()