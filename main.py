#!/usr/bin/env python3
"""
Main entry point for the Vocal Remover system

This script provides a unified command-line interface for:
- Training vocal removal models
- Running inference on audio files
- Deploying models to repositories
- Managing model packages

Usage:
    python main.py train --data_dir path/to/audio --output_dir models/
    python main.py infer --model models/best_model.pth --input song.wav --output instrumental.wav
    python main.py deploy --model models/best_model.pth --hf_repo username/model-name
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.trainer import train_model
from src.inference import VocalRemover
from src.model_hub import deploy_model
from src.utils import load_config


def train_command(args):
    """Handle training command"""
    
    print("=== Training Vocal Removal Model ===")
    
    # Validate arguments
    if not Path(args.data_dir).exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        return 1
    
    # Use default config if not provided
    config_path = args.config or "config.yaml"
    if not Path(config_path).exists():
        print(f"Error: Configuration file not found: {config_path}")
        print("Make sure config.yaml exists or provide --config argument")
        return 1
    
    try:
        train_model(
            config_path=config_path,
            data_paths=[args.data_dir],
            resume_from=args.resume_from
        )
        
        print("Training completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Training failed: {e}")
        return 1


def infer_command(args):
    """Handle inference command"""
    
    print("=== Vocal Removal Inference ===")
    
    # Validate arguments
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    if not Path(args.input).exists():
        print(f"Error: Input file/directory not found: {args.input}")
        return 1
    
    try:
        # Initialize vocal remover
        vocal_remover = VocalRemover(args.model, args.config)
        
        # Print model info
        model_info = vocal_remover.get_model_info()
        print(f"Model: {model_info['model_type']}")
        print(f"Parameters: {model_info['total_parameters']:,}")
        print(f"Device: {model_info['device']}")
        
        # Process audio
        input_path = Path(args.input)
        output_path = Path(args.output)
        
        if args.batch or input_path.is_dir():
            # Batch processing
            output_path.mkdir(parents=True, exist_ok=True)
            results = vocal_remover.batch_process(
                input_dir=str(input_path),
                output_dir=str(output_path)
            )
            
            successful = sum(1 for r in results if 'error' not in r)
            print(f"Processed {successful}/{len(results)} files successfully")
            
        else:
            # Single file processing
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result = vocal_remover.separate_vocals(
                str(input_path), 
                str(output_path)
            )
            
            print(f"Processing completed in {result['processing_time']:.2f}s")
            print(f"Real-time factor: {result['real_time_factor']:.2f}x")
        
        print("Inference completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Inference failed: {e}")
        return 1


def deploy_command(args):
    """Handle deployment command"""
    
    print("=== Model Deployment ===")
    
    # Validate arguments
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    # Find config file if not provided
    config_path = args.config
    if not config_path:
        model_dir = Path(args.model).parent
        for config_name in ['config.yaml', 'config.yml']:
            potential_config = model_dir / config_name
            if potential_config.exists():
                config_path = str(potential_config)
                break
    
    if not config_path or not Path(config_path).exists():
        print("Error: Configuration file not found")
        print("Provide --config or place config.yaml next to the model file")
        return 1
    
    try:
        results = deploy_model(
            model_path=args.model,
            config_path=config_path,
            model_name=args.model_name,
            description=args.description or f"Vocal removal model: {args.model_name}",
            github_repo=args.github_repo,
            hf_repo=args.hf_repo,
            output_dir=args.output_dir
        )
        
        print("Deployment results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        
        print("Deployment completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Deployment failed: {e}")
        return 1


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Vocal Remover - Deep Learning Audio Separation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python main.py train --data_dir music_dataset/ --output_dir models/

  # Remove vocals from a song
  python main.py infer --model models/best_model.pth --input song.wav --output instrumental.wav

  # Batch process multiple files
  python main.py infer --model models/best_model.pth --input songs/ --output instrumentals/ --batch

  # Deploy model to Hugging Face
  python main.py deploy --model models/best_model.pth --hf_repo username/my-vocal-remover

  # Deploy model to GitHub
  python main.py deploy --model models/best_model.pth --github_repo https://github.com/user/repo.git
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train a vocal removal model')
    train_parser.add_argument('--data_dir', required=True,
                             help='Directory containing training audio files')
    train_parser.add_argument('--config', 
                             help='Configuration file (default: config.yaml)')
    train_parser.add_argument('--resume_from',
                             help='Checkpoint to resume training from')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Remove vocals from audio')
    infer_parser.add_argument('--model', required=True,
                             help='Path to trained model')
    infer_parser.add_argument('--input', required=True,
                             help='Input audio file or directory')
    infer_parser.add_argument('--output', required=True,
                             help='Output audio file or directory')
    infer_parser.add_argument('--config',
                             help='Configuration file (optional)')
    infer_parser.add_argument('--batch', action='store_true',
                             help='Process directory in batch mode')
    
    # Deployment command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy model to repositories')
    deploy_parser.add_argument('--model', required=True,
                              help='Path to trained model')
    deploy_parser.add_argument('--config',
                              help='Configuration file (optional)')
    deploy_parser.add_argument('--model_name', default='vocal_remover',
                              help='Name for the deployed model')
    deploy_parser.add_argument('--description',
                              help='Model description')
    deploy_parser.add_argument('--hf_repo',
                              help='Hugging Face repository (username/repo-name)')
    deploy_parser.add_argument('--github_repo',
                              help='GitHub repository URL')
    deploy_parser.add_argument('--output_dir', default='deployments',
                              help='Output directory for packages')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate command handler
    if args.command == 'train':
        return train_command(args)
    elif args.command == 'infer':
        return infer_command(args)
    elif args.command == 'deploy':
        return deploy_command(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)