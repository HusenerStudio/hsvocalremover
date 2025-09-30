#!/usr/bin/env python3
"""
Example deployment script for vocal removal models

This script demonstrates how to:
1. Package trained models for deployment
2. Upload models to Hugging Face Hub
3. Upload models to GitHub repositories
4. Download and use deployed models
5. Manage model versions and metadata

Usage:
    python examples/deployment_example.py --model path/to/model.pth --deploy-hf username/repo-name
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model_hub import ModelHub, deploy_model
from src.utils import load_config


def create_model_package_example(model_path, config_path, output_dir):
    """Example of creating a model package"""
    
    print("=== Creating Model Package ===")
    
    # Load configuration
    config = load_config(config_path)
    
    # Create model hub
    hub = ModelHub(config)
    
    # Create package
    package_dir = hub.save_model_package(
        model_path=model_path,
        output_dir=output_dir,
        model_name="example_vocal_remover",
        description="Example vocal removal model for demonstration",
        tags=["vocal-removal", "audio-separation", "example", "demo"]
    )
    
    print(f"Model package created at: {package_dir}")
    
    # Show package contents
    package_path = Path(package_dir)
    print(f"\nPackage contents:")
    for file_path in package_path.rglob('*'):
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  {file_path.name}: {size_mb:.2f} MB")
    
    return package_dir


def deploy_to_huggingface_example(package_dir, repo_id, token=None):
    """Example of deploying to Hugging Face"""
    
    print(f"\n=== Deploying to Hugging Face: {repo_id} ===")
    
    # Get token from environment if not provided
    if not token:
        token = os.getenv('HF_TOKEN')
    
    if not token:
        print("Warning: No Hugging Face token provided.")
        print("Set HF_TOKEN environment variable or pass --hf-token argument")
        print("You can get a token from: https://huggingface.co/settings/tokens")
        return None
    
    try:
        # Load configuration to create hub
        config_path = Path(package_dir) / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        hub = ModelHub({'model': config['architecture'], 'audio': config['audio_config']})
        
        # Upload to Hugging Face
        repo_url = hub.upload_to_huggingface(
            package_dir=package_dir,
            repo_id=repo_id,
            token=token,
            private=False  # Make it public
        )
        
        print(f"Successfully deployed to: {repo_url}")
        print(f"\nYou can now use your model with:")
        print(f"from src.model_hub import ModelHub")
        print(f"hub = ModelHub(config)")
        print(f"model_dir = hub.download_from_huggingface('{repo_id}', 'local_models')")
        
        return repo_url
        
    except Exception as e:
        print(f"Deployment to Hugging Face failed: {e}")
        return None


def deploy_to_github_example(package_dir, repo_url, branch="main"):
    """Example of deploying to GitHub"""
    
    print(f"\n=== Deploying to GitHub: {repo_url} ===")
    
    try:
        # Load configuration to create hub
        config_path = Path(package_dir) / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        hub = ModelHub({'model': config['architecture'], 'audio': config['audio_config']})
        
        # Upload to GitHub
        result_url = hub.upload_to_github(
            package_dir=package_dir,
            repo_url=repo_url,
            branch=branch,
            commit_message="Add vocal removal model package"
        )
        
        print(f"Successfully deployed to: {result_url}")
        print(f"\nYou can now clone and use your model with:")
        print(f"git clone {repo_url}")
        print(f"# Then load the model from the cloned directory")
        
        return result_url
        
    except Exception as e:
        print(f"Deployment to GitHub failed: {e}")
        print("Make sure you have:")
        print("1. Created the repository on GitHub")
        print("2. Set up Git authentication (SSH keys or personal access token)")
        print("3. Have write access to the repository")
        return None


def download_model_example(repo_id, local_dir, source="huggingface"):
    """Example of downloading a deployed model"""
    
    print(f"\n=== Downloading Model from {source.title()}: {repo_id} ===")
    
    try:
        hub = ModelHub({})
        
        if source == "huggingface":
            model_dir = hub.download_from_huggingface(
                repo_id=repo_id,
                local_dir=local_dir
            )
        elif source == "github":
            model_dir = hub.download_from_github(
                repo_url=repo_id,  # In this case, repo_id is actually the URL
                local_dir=local_dir
            )
        else:
            raise ValueError(f"Unknown source: {source}")
        
        print(f"Model downloaded to: {model_dir}")
        
        # Try to load the model
        try:
            model = hub.load_model_from_package(model_dir)
            print("Model loaded successfully!")
            
            # Show model info
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Model parameters: {total_params:,}")
            
        except Exception as e:
            print(f"Could not load model: {e}")
        
        return model_dir
        
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def list_model_files(directory):
    """List all model files in a directory"""
    
    print(f"\n=== Model Files in {directory} ===")
    
    model_dir = Path(directory)
    if not model_dir.exists():
        print("Directory does not exist")
        return
    
    # Look for model files
    model_files = []
    for pattern in ['*.pth', '*.pt', '*.ckpt']:
        model_files.extend(model_dir.rglob(pattern))
    
    if not model_files:
        print("No model files found")
        return
    
    print(f"Found {len(model_files)} model files:")
    for model_file in model_files:
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"  {model_file.relative_to(model_dir)}: {size_mb:.2f} MB")
    
    # Look for config files
    config_files = []
    for pattern in ['config.json', 'config.yaml', '*.yaml', '*.yml']:
        config_files.extend(model_dir.rglob(pattern))
    
    if config_files:
        print(f"\nFound {len(config_files)} config files:")
        for config_file in config_files:
            print(f"  {config_file.relative_to(model_dir)}")


def main():
    parser = argparse.ArgumentParser(description="Model deployment example")
    
    # Model arguments
    parser.add_argument("--model", 
                       help="Path to trained model checkpoint")
    parser.add_argument("--config", 
                       help="Path to configuration file")
    
    # Deployment arguments
    parser.add_argument("--deploy_hf", 
                       help="Deploy to Hugging Face (provide repo_id like 'username/model-name')")
    parser.add_argument("--deploy_github",
                       help="Deploy to GitHub (provide repo URL)")
    parser.add_argument("--hf_token",
                       help="Hugging Face token (or set HF_TOKEN env var)")
    parser.add_argument("--github_branch", default="main",
                       help="GitHub branch to deploy to")
    
    # Download arguments
    parser.add_argument("--download_hf",
                       help="Download model from Hugging Face (repo_id)")
    parser.add_argument("--download_github",
                       help="Download model from GitHub (repo URL)")
    parser.add_argument("--download_dir", default="downloaded_models",
                       help="Directory to download models to")
    
    # Package arguments
    parser.add_argument("--package_only", action="store_true",
                       help="Only create package, don't deploy")
    parser.add_argument("--output_dir", default="deployments",
                       help="Output directory for packages")
    parser.add_argument("--model_name", default="vocal_remover",
                       help="Name for the model package")
    
    # Utility arguments
    parser.add_argument("--list_models",
                       help="List model files in directory")
    
    args = parser.parse_args()
    
    print("=== Vocal Remover Deployment Example ===")
    
    try:
        # List models mode
        if args.list_models:
            list_model_files(args.list_models)
            return 0
        
        # Download mode
        if args.download_hf:
            download_model_example(args.download_hf, args.download_dir, "huggingface")
            return 0
        
        if args.download_github:
            download_model_example(args.download_github, args.download_dir, "github")
            return 0
        
        # Deployment mode - requires model and config
        if not args.model:
            print("Error: --model is required for deployment")
            return 1
        
        if not Path(args.model).exists():
            print(f"Error: Model file not found: {args.model}")
            return 1
        
        # Try to find config file if not provided
        config_path = args.config
        if not config_path:
            # Look for config.yaml in the same directory as the model
            model_dir = Path(args.model).parent
            for config_name in ['config.yaml', 'config.yml']:
                potential_config = model_dir / config_name
                if potential_config.exists():
                    config_path = str(potential_config)
                    print(f"Found config file: {config_path}")
                    break
        
        if not config_path or not Path(config_path).exists():
            print("Error: Configuration file not found")
            print("Provide --config or place config.yaml next to the model file")
            return 1
        
        # Create model package
        package_dir = create_model_package_example(
            args.model, 
            config_path, 
            args.output_dir
        )
        
        if args.package_only:
            print(f"\nPackage created successfully at: {package_dir}")
            return 0
        
        # Deploy to platforms
        deployed = False
        
        if args.deploy_hf:
            result = deploy_to_huggingface_example(
                package_dir, 
                args.deploy_hf, 
                args.hf_token
            )
            if result:
                deployed = True
        
        if args.deploy_github:
            result = deploy_to_github_example(
                package_dir, 
                args.deploy_github, 
                args.github_branch
            )
            if result:
                deployed = True
        
        if not args.deploy_hf and not args.deploy_github:
            print("\nNo deployment targets specified.")
            print("Use --deploy_hf or --deploy_github to deploy the model")
            print(f"Package is ready at: {package_dir}")
        
        if deployed:
            print(f"\n=== Deployment Completed Successfully ===")
            print(f"Package directory: {package_dir}")
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())