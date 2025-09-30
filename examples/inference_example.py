#!/usr/bin/env python3
"""
Example inference script for vocal removal

This script demonstrates how to:
1. Load a trained vocal removal model
2. Process single audio files
3. Batch process multiple files
4. Use different processing options
5. Download and use pre-trained models

Usage:
    python examples/inference_example.py --model path/to/model.pth --input song.wav --output instrumental.wav
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference import VocalRemover
from src.model_hub import ModelHub
from src.utils import get_audio_info, format_time


def download_pretrained_model(repo_id, local_dir):
    """Download a pre-trained model from Hugging Face"""
    
    print(f"Downloading model from Hugging Face: {repo_id}")
    
    try:
        hub = ModelHub({})
        model_dir = hub.download_from_huggingface(
            repo_id=repo_id,
            local_dir=local_dir
        )
        
        model_path = Path(model_dir) / "model.pth"
        if model_path.exists():
            print(f"Model downloaded successfully: {model_path}")
            return str(model_path)
        else:
            print("Error: model.pth not found in downloaded package")
            return None
            
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None


def analyze_audio_file(file_path):
    """Analyze audio file and print information"""
    
    print(f"\n--- Audio File Analysis ---")
    print(f"File: {file_path}")
    
    info = get_audio_info(file_path)
    if info:
        print(f"Duration: {format_time(info['duration'])}")
        print(f"Sample Rate: {info['sample_rate']} Hz")
        print(f"Channels: {info['num_channels']}")
        print(f"Bits per Sample: {info.get('bits_per_sample', 'Unknown')}")
        print(f"Encoding: {info.get('encoding', 'Unknown')}")
    else:
        print("Could not analyze audio file")
    
    return info


def process_single_file(vocal_remover, input_path, output_path, args):
    """Process a single audio file"""
    
    print(f"\n--- Processing Single File ---")
    
    # Analyze input file
    analyze_audio_file(input_path)
    
    # Process with custom options
    start_time = time.time()
    
    try:
        result = vocal_remover.separate_vocals(
            input_path=input_path,
            output_path=output_path,
            chunk_duration=args.chunk_duration,
            overlap_ratio=args.overlap_ratio,
            normalize_output=not args.no_normalize,
            apply_fade_in_out=not args.no_fade
        )
        
        # Print results
        print(f"\n--- Processing Results ---")
        print(f"Input: {result['input_path']}")
        print(f"Output: {result['output_path']}")
        print(f"Duration: {format_time(result['original_duration'])}")
        print(f"Processing Time: {format_time(result['processing_time'])}")
        print(f"Real-time Factor: {result['real_time_factor']:.2f}x")
        print(f"Sample Rate: {result['original_sample_rate']} → {result['output_sample_rate']} Hz")
        
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False


def process_batch(vocal_remover, input_dir, output_dir, args):
    """Process multiple files in batch"""
    
    print(f"\n--- Batch Processing ---")
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")
    
    start_time = time.time()
    
    try:
        results = vocal_remover.batch_process(
            input_dir=input_dir,
            output_dir=output_dir,
            file_extensions=args.extensions,
            chunk_duration=args.chunk_duration,
            overlap_ratio=args.overlap_ratio,
            normalize_output=not args.no_normalize,
            apply_fade_in_out=not args.no_fade
        )
        
        # Analyze results
        successful = [r for r in results if 'error' not in r]
        failed = [r for r in results if 'error' in r]
        
        total_time = time.time() - start_time
        total_duration = sum(r.get('original_duration', 0) for r in successful)
        
        print(f"\n--- Batch Results ---")
        print(f"Total Files: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Total Audio Duration: {format_time(total_duration)}")
        print(f"Total Processing Time: {format_time(total_time)}")
        
        if total_duration > 0:
            print(f"Average Real-time Factor: {total_duration / total_time:.2f}x")
        
        # Print failed files
        if failed:
            print(f"\n--- Failed Files ---")
            for result in failed:
                print(f"  {result['input_path']}: {result['error']}")
        
        return len(successful) > 0
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return False


def compare_models(model_paths, test_file, output_dir):
    """Compare multiple models on the same test file"""
    
    print(f"\n--- Model Comparison ---")
    print(f"Test File: {test_file}")
    
    results = []
    
    for i, model_path in enumerate(model_paths):
        print(f"\nTesting model {i+1}/{len(model_paths)}: {model_path}")
        
        try:
            # Load model
            vocal_remover = VocalRemover(model_path)
            
            # Process file
            output_path = Path(output_dir) / f"model_{i+1}_output.wav"
            result = vocal_remover.separate_vocals(str(test_file), str(output_path))
            
            results.append({
                'model_path': model_path,
                'output_path': str(output_path),
                'processing_time': result['processing_time'],
                'real_time_factor': result['real_time_factor']
            })
            
            print(f"  Processing time: {format_time(result['processing_time'])}")
            print(f"  Real-time factor: {result['real_time_factor']:.2f}x")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'model_path': model_path,
                'error': str(e)
            })
    
    # Print comparison summary
    print(f"\n--- Comparison Summary ---")
    for i, result in enumerate(results):
        print(f"Model {i+1}: {result['model_path']}")
        if 'error' in result:
            print(f"  Status: Failed ({result['error']})")
        else:
            print(f"  Status: Success")
            print(f"  Speed: {result['real_time_factor']:.2f}x real-time")
            print(f"  Output: {result['output_path']}")


def main():
    parser = argparse.ArgumentParser(description="Vocal removal inference example")
    
    # Model arguments
    parser.add_argument("--model", 
                       help="Path to trained model checkpoint")
    parser.add_argument("--download_model",
                       help="Download model from Hugging Face (repo_id)")
    parser.add_argument("--model_dir", default="downloaded_models",
                       help="Directory to download models to")
    
    # Input/Output arguments
    parser.add_argument("--input", required=True,
                       help="Input audio file or directory")
    parser.add_argument("--output", required=True,
                       help="Output audio file or directory")
    
    # Processing options
    parser.add_argument("--batch", action="store_true",
                       help="Process directory in batch mode")
    parser.add_argument("--chunk_duration", type=float, default=10.0,
                       help="Chunk duration in seconds")
    parser.add_argument("--overlap_ratio", type=float, default=0.25,
                       help="Overlap ratio between chunks")
    parser.add_argument("--no_normalize", action="store_true",
                       help="Skip output normalization")
    parser.add_argument("--no_fade", action="store_true",
                       help="Skip fade in/out")
    parser.add_argument("--extensions", nargs="+", 
                       default=['.wav', '.mp3', '.flac', '.m4a'],
                       help="File extensions to process in batch mode")
    
    # Advanced options
    parser.add_argument("--compare_models", nargs="+",
                       help="Compare multiple models on the same file")
    parser.add_argument("--analyze_only", action="store_true",
                       help="Only analyze input file, don't process")
    
    args = parser.parse_args()
    
    print("=== Vocal Remover Inference Example ===")
    
    try:
        # Handle model loading
        model_path = args.model
        
        if args.download_model:
            print(f"Downloading model: {args.download_model}")
            model_path = download_pretrained_model(args.download_model, args.model_dir)
            if not model_path:
                return 1
        
        if not model_path:
            print("Error: No model specified. Use --model or --download_model")
            return 1
        
        if not Path(model_path).exists():
            print(f"Error: Model file not found: {model_path}")
            return 1
        
        # Handle analysis only mode
        if args.analyze_only:
            if Path(args.input).is_file():
                analyze_audio_file(args.input)
            else:
                print("Analysis mode only works with single files")
            return 0
        
        # Handle model comparison
        if args.compare_models:
            if not Path(args.input).is_file():
                print("Model comparison requires a single input file")
                return 1
            
            Path(args.output).mkdir(parents=True, exist_ok=True)
            compare_models(args.compare_models, args.input, args.output)
            return 0
        
        # Load vocal remover
        print(f"Loading model: {model_path}")
        vocal_remover = VocalRemover(model_path)
        
        # Print model info
        model_info = vocal_remover.get_model_info()
        print(f"\n--- Model Information ---")
        print(f"Model Type: {model_info['model_type']}")
        print(f"Parameters: {model_info['total_parameters']:,}")
        print(f"Device: {model_info['device']}")
        print(f"Sample Rate: {model_info['audio_config']['sample_rate']} Hz")
        
        # Process audio
        input_path = Path(args.input)
        output_path = Path(args.output)
        
        if args.batch or input_path.is_dir():
            # Batch processing
            if not input_path.is_dir():
                print("Error: Batch mode requires input directory")
                return 1
            
            output_path.mkdir(parents=True, exist_ok=True)
            success = process_batch(vocal_remover, str(input_path), str(output_path), args)
            
        else:
            # Single file processing
            if not input_path.is_file():
                print(f"Error: Input file not found: {input_path}")
                return 1
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            success = process_single_file(vocal_remover, str(input_path), str(output_path), args)
        
        if success:
            print("\n=== Processing Completed Successfully ===")
            return 0
        else:
            print("\n=== Processing Failed ===")
            return 1
    
    except KeyboardInterrupt:
        print("\n=== Processing Interrupted by User ===")
        return 1
    
    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    exit(main())