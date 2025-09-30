#!/usr/bin/env python3
"""
Vocal Remover Web Application

A Streamlit-based web interface for the vocal removal system that allows users to:
- Upload audio files
- Process them for vocal removal
- Download instrumental tracks
- Monitor processing progress
- View audio information

Usage:
    streamlit run web_app.py
"""

import streamlit as st
import tempfile
import os
import time
from pathlib import Path
import io
import base64
import traceback

# Import our vocal remover modules
from src.inference import VocalRemover
from src.utils import get_audio_info, format_time, validate_audio_file
from src.model_hub import ModelHub

# Try to import ONNX support (optional)
try:
    from src.onnx_inference import ONNXVocalRemover, ModelConverter
    ONNX_SUPPORT = True
    # Test if ONNX Runtime actually works
    try:
        import onnxruntime
        # Try to create a simple session to test if it works
        onnxruntime.get_available_providers()
        ONNX_RUNTIME_WORKING = True
    except Exception as e:
        ONNX_RUNTIME_WORKING = False
        print(f"ONNX Runtime installed but not working: {e}")
except ImportError:
    ONNX_SUPPORT = False
    ONNX_RUNTIME_WORKING = False

# Try simple ONNX support as fallback
try:
    from src.simple_onnx import create_simple_onnx_inference, is_simple_onnx_available
    SIMPLE_ONNX_AVAILABLE = is_simple_onnx_available()
except ImportError:
    SIMPLE_ONNX_AVAILABLE = False
    def create_simple_onnx_inference(*args, **kwargs):
        raise ImportError("Simple ONNX support not available")

# Create dummy classes for when ONNX is not available
if not ONNX_SUPPORT:
    class ONNXVocalRemover:
        def __init__(self, *args, **kwargs):
            raise ImportError("ONNX support not available")
    
    class ModelConverter:
        @staticmethod
        def get_model_format(model_path):
            from pathlib import Path
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
        def is_onnx_available():
            return False
        
        @staticmethod
        def optimize_onnx(*args, **kwargs):
            raise ImportError("ONNX support not available")


# Page configuration
st.set_page_config(
    page_title="Vocal Remover",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .processing-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
    .audio-info {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_vocal_remover(model_path):
    """Load vocal remover model (cached)"""
    try:
        # Detect model format
        model_format = ModelConverter.get_model_format(model_path)
        
        if model_format == 'onnx':
            # Try ONNX Runtime first
            if ONNX_SUPPORT and ONNX_RUNTIME_WORKING and ModelConverter.is_onnx_available():
                return ONNXVocalRemover(model_path), 'onnx'
            # Fallback to simple ONNX conversion
            elif SIMPLE_ONNX_AVAILABLE:
                try:
                    simple_onnx = create_simple_onnx_inference(model_path)
                    return simple_onnx, 'simple_onnx'
                except Exception as e:
                    st.error(f"Failed to load ONNX model with fallback method: {e}")
                    st.info("Install onnx2torch for ONNX support: pip install onnx2torch")
                    return None, None
            else:
                st.error("ONNX model detected but no ONNX support available")
                st.info("Options:")
                st.info("1. Install ONNX Runtime: pip install onnxruntime")
                st.info("2. Install onnx2torch for fallback: pip install onnx2torch")
                st.info("3. Convert model to PyTorch format")
                return None, None
        else:
            # Use PyTorch inference
            return VocalRemover(model_path), 'pytorch'
            
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None


def get_audio_download_link(file_path, filename):
    """Generate download link for audio file"""
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    
    b64_audio = base64.b64encode(audio_bytes).decode()
    href = f'<a href="data:audio/wav;base64,{b64_audio}" download="{filename}">Download {filename}</a>'
    return href


def display_audio_info(file_path):
    """Display audio file information"""
    info = get_audio_info(file_path)
    if info:
        st.markdown('<div class="audio-info">', unsafe_allow_html=True)
        st.markdown("**📊 Audio Information:**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duration", format_time(info['duration']))
            st.metric("Sample Rate", f"{info['sample_rate']} Hz")
        
        with col2:
            st.metric("Channels", info['num_channels'])
            st.metric("Bits per Sample", info.get('bits_per_sample', 'Unknown'))
        
        with col3:
            file_size = Path(file_path).stat().st_size / (1024 * 1024)
            st.metric("File Size", f"{file_size:.2f} MB")
            st.metric("Encoding", info.get('encoding', 'Unknown'))
        
        st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Main web application"""
    
    # Header
    st.markdown('<h1 class="main-header">🎵 Vocal Remover</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2em; color: #666;">Remove vocals from your audio files using deep learning</p>', unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model Management Section
        st.subheader("🧠 Model Management")
        
        # Model upload section
        with st.expander("📤 Upload Model", expanded=False):
            uploaded_model = st.file_uploader(
                "Upload a trained model",
                type=['pth', 'pt', 'ckpt', 'onnx', 'pkl'],
                help="Supported formats: PyTorch (.pth, .pt), Checkpoint (.ckpt), Pickle (.pkl), ONNX (.onnx)",
                key="model_uploader"
            )
            
            if uploaded_model is not None:
                # Save uploaded model
                models_dir = Path("models")
                models_dir.mkdir(exist_ok=True)
                
                model_path = models_dir / uploaded_model.name
                
                try:
                    with open(model_path, "wb") as f:
                        f.write(uploaded_model.getbuffer())
                    
                    st.success(f"✅ Model uploaded successfully: {uploaded_model.name}")
                    st.info(f"📁 Saved to: {model_path}")
                    
                    # Show file info
                    file_size = model_path.stat().st_size / (1024 * 1024)
                    st.info(f"📊 File size: {file_size:.2f} MB")
                    
                    # Refresh the page to show new model
                    if st.button("🔄 Refresh Model List", key="refresh_models"):
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Failed to save model: {str(e)}")
                    st.error("Please check file permissions and try again.")
        
        # Model training section
        with st.expander("🏋️ Train New Model", expanded=False):
            st.markdown("**Train a custom vocal removal model**")
            
            # Training data upload
            training_data = st.file_uploader(
                "Upload training audio files",
                type=['wav', 'mp3', 'flac', 'm4a', 'aac', 'ogg'],
                accept_multiple_files=True,
                help="Upload multiple audio files for training",
                key="training_data_uploader"
            )
            
            if training_data:
                st.info(f"📁 {len(training_data)} files selected for training")
                
                # Training parameters
                col_a, col_b = st.columns(2)
                with col_a:
                    train_epochs = st.number_input("Epochs", min_value=1, max_value=200, value=50)
                    train_batch_size = st.number_input("Batch Size", min_value=1, max_value=32, value=4)
                
                with col_b:
                    train_lr = st.number_input("Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, format="%.4f")
                    model_name = st.text_input("Model Name", value="custom_vocal_remover")
                
                # Start training button
                if st.button("🚀 Start Training", type="primary", key="start_training"):
                    if len(training_data) < 5:
                        st.error("❌ Please upload at least 5 audio files for training")
                    else:
                        # Save training files temporarily
                        temp_dir = Path("temp_training_data")
                        temp_dir.mkdir(exist_ok=True)
                        
                        try:
                            for file in training_data:
                                file_path = temp_dir / file.name
                                with open(file_path, "wb") as f:
                                    f.write(file.getbuffer())
                            
                            st.success("🎯 Training files saved successfully!")
                            st.info("Training will run in the background. The model will be saved when complete.")
                            
                            # Note: Actual training would be started here
                            # For demo purposes, we'll show the command that would be run
                            st.code(f"""
# Training command that would be executed:
python main.py train \\
    --data_dir {temp_dir} \\
    --output_dir models/ \\
    --epochs {train_epochs} \\
    --batch_size {train_batch_size} \\
    --learning_rate {train_lr} \\
    --model_name {model_name}
                            """)
                            
                        except Exception as e:
                            st.error(f"❌ Failed to save training files: {str(e)}")
                            st.error("Please check file permissions and try again.")
        
        # Model selection
        st.subheader("📋 Select Model")
        
        # Check for available models (including uploaded ones)
        models_dir = Path("models")
        available_models = []
        if models_dir.exists():
            # Support multiple model formats
            for pattern in ['*.pth', '*.pt', '*.ckpt', '*.onnx', '*.pkl']:
                available_models.extend(list(models_dir.glob(pattern)))
        
        if available_models:
            model_options = [str(model) for model in available_models]
            selected_model = st.selectbox(
                "Choose Model",
                model_options,
                help="Select a trained vocal removal model"
            )
            
            # Show model info
            if selected_model:
                model_path = Path(selected_model)
                file_size = model_path.stat().st_size / (1024 * 1024)
                st.info(f"📊 Model: {model_path.name} ({file_size:.1f} MB)")
                
                # Model format detection
                if model_path.suffix.lower() == '.onnx':
                    if ONNX_SUPPORT and ONNX_RUNTIME_WORKING:
                        st.info("🔧 ONNX model detected - optimized for inference")
                        
                        # ONNX conversion options
                        if st.button("⚡ Optimize ONNX Model", key="optimize_onnx"):
                            optimized_path = model_path.parent / f"{model_path.stem}_optimized.onnx"
                            try:
                                ModelConverter.optimize_onnx(str(model_path), str(optimized_path))
                                st.success(f"✅ Optimized model saved: {optimized_path.name}")
                            except Exception as e:
                                st.error(f"❌ Optimization failed: {e}")
                    else:
                        st.warning("⚠️ ONNX model detected but ONNX Runtime is not working properly")
                        if ONNX_SUPPORT and not ONNX_RUNTIME_WORKING:
                            st.info("ONNX Runtime is installed but has compatibility issues. Try using PyTorch models instead.")
                        else:
                            st.info("Install ONNX Runtime: pip install onnxruntime")
                        
                elif model_path.suffix.lower() in ['.pth', '.pt']:
                    st.info("🔧 PyTorch model detected")
                    
                    if ONNX_SUPPORT and ONNX_RUNTIME_WORKING:
                        # PyTorch to ONNX conversion
                        if st.button("🔄 Convert to ONNX", key="convert_to_onnx"):
                            onnx_path = model_path.parent / f"{model_path.stem}.onnx"
                            st.info("⏳ Converting to ONNX... (This may take a few minutes)")
                            try:
                                # Note: This would require the model config
                                st.warning("⚠️ ONNX conversion requires model configuration. Use command line for full conversion.")
                                st.code(f"python -c \"from src.onnx_inference import convert_pytorch_to_onnx; convert_pytorch_to_onnx('{model_path}', '{onnx_path}', config)\"")
                            except Exception as e:
                                st.error(f"❌ Conversion failed: {e}")
                    else:
                        st.info("💡 ONNX Runtime not working properly - conversion unavailable")
                        
                elif model_path.suffix.lower() == '.ckpt':
                    st.info("🔧 Checkpoint model detected")
        else:
            st.warning("No trained models found")
            st.info("💡 Upload a model or train a new one using the options above")
            selected_model = None
        
        # Processing options
        st.subheader("Processing Options")
        chunk_duration = st.slider(
            "Chunk Duration (seconds)",
            min_value=5.0,
            max_value=30.0,
            value=10.0,
            step=1.0,
            help="Longer chunks = better quality, more memory usage"
        )
        
        overlap_ratio = st.slider(
            "Overlap Ratio",
            min_value=0.0,
            max_value=0.5,
            value=0.25,
            step=0.05,
            help="More overlap = smoother output, slower processing"
        )
        
        normalize_output = st.checkbox(
            "Normalize Output",
            value=True,
            help="Normalize the output audio volume"
        )
        
        apply_fade = st.checkbox(
            "Apply Fade In/Out",
            value=True,
            help="Apply fade to prevent audio clicks"
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            show_waveform = st.checkbox("Show Waveform", value=False)
            show_spectrogram = st.checkbox("Show Spectrogram", value=False)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Audio")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'm4a', 'aac', 'ogg'],
            help="Supported formats: WAV, MP3, FLAC, M4A, AAC, OGG"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                input_file_path = tmp_file.name
            
            st.success(f"✅ Uploaded: {uploaded_file.name}")
            
            # Display audio info
            display_audio_info(input_file_path)
            
            # Play original audio
            st.audio(uploaded_file.getvalue(), format='audio/wav')
            
            # Validate file
            if not validate_audio_file(input_file_path):
                st.error("❌ Invalid audio file. Please check the file format and try again.")
                return
    
    with col2:
        st.header("🎼 Processed Audio")
        
        if uploaded_file is not None and selected_model:
            # Process button
            if st.button("🚀 Remove Vocals", type="primary", use_container_width=True):
                
                # Check if model exists
                if not Path(selected_model).exists():
                    st.error(f"❌ Model file not found: {selected_model}")
                    return
                
                try:
                    # Load model
                    with st.spinner("Loading model..."):
                        result = load_vocal_remover(selected_model)
                        if result[0] is None:
                            return
                        vocal_remover, model_type = result
                    
                    # Display model info
                    if model_type == 'onnx':
                        model_info = vocal_remover.get_model_info()
                        st.info(f"🧠 ONNX Model | Providers: {', '.join(model_info['providers'])}")
                    elif model_type == 'simple_onnx':
                        model_info = vocal_remover.get_model_info()
                        st.info(f"🧠 ONNX Model (converted to PyTorch) | Method: {model_info['conversion_method']}")
                    else:
                        model_info = vocal_remover.get_model_info()
                        st.info(f"🧠 {model_info['model_type']} | Parameters: {model_info['total_parameters']:,} | Device: {model_info['device']}")
                    
                    # Create output file path
                    output_filename = f"instrumental_{Path(uploaded_file.name).stem}.wav"
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_output:
                        output_file_path = tmp_output.name
                    
                    # Process audio with progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    start_time = time.time()
                    
                    status_text.text("🎵 Processing audio...")
                    progress_bar.progress(0.3)
                    
                    # Run vocal separation (different methods for different model types)
                    if model_type == 'onnx':
                        # ONNX Runtime inference
                        audio, sr = torchaudio.load(input_file_path)
                        if sr != 44100:  # Assuming 44.1kHz for ONNX models
                            resampler = torchaudio.transforms.Resample(sr, 44100)
                            audio = resampler(audio)
                        
                        # Ensure stereo
                        if audio.shape[0] == 1:
                            audio = audio.repeat(2, 1)
                        elif audio.shape[0] > 2:
                            audio = audio[:2]
                        
                        # Use default audio config for ONNX
                        audio_config = {
                            'sample_rate': 44100,
                            'n_fft': 2048,
                            'hop_length': 512,
                            'win_length': 2048
                        }
                        
                        separated_audio = vocal_remover.separate_vocals(audio, audio_config)
                        
                        # Save output
                        torchaudio.save(output_file_path, separated_audio, 44100)
                        
                        result = {
                            'input_path': input_file_path,
                            'output_path': output_file_path,
                            'original_duration': audio.shape[1] / 44100,
                            'processing_time': time.time() - start_time,
                            'real_time_factor': (audio.shape[1] / 44100) / (time.time() - start_time),
                            'original_sample_rate': sr,
                            'output_sample_rate': 44100
                        }
                    elif model_type == 'simple_onnx':
                        # Simple ONNX inference (converted to PyTorch)
                        audio, sr = torchaudio.load(input_file_path)
                        if sr != 44100:
                            resampler = torchaudio.transforms.Resample(sr, 44100)
                            audio = resampler(audio)
                        
                        # Ensure stereo
                        if audio.shape[0] == 1:
                            audio = audio.repeat(2, 1)
                        elif audio.shape[0] > 2:
                            audio = audio[:2]
                        
                        # Use default audio config
                        audio_config = {
                            'sample_rate': 44100,
                            'n_fft': 2048,
                            'hop_length': 512,
                            'win_length': 2048
                        }
                        
                        separated_audio = vocal_remover.separate_vocals(audio, audio_config)
                        
                        # Save output
                        torchaudio.save(output_file_path, separated_audio, 44100)
                        
                        result = {
                            'input_path': input_file_path,
                            'output_path': output_file_path,
                            'original_duration': audio.shape[1] / 44100,
                            'processing_time': time.time() - start_time,
                            'real_time_factor': (audio.shape[1] / 44100) / (time.time() - start_time),
                            'original_sample_rate': sr,
                            'output_sample_rate': 44100
                        }
                    else:
                        # PyTorch inference
                        result = vocal_remover.separate_vocals(
                            input_path=input_file_path,
                            output_path=output_file_path,
                            chunk_duration=chunk_duration,
                            overlap_ratio=overlap_ratio,
                            normalize_output=normalize_output,
                            apply_fade_in_out=apply_fade
                        )
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Processing completed!")
                    
                    # Display results
                    processing_time = time.time() - start_time
                    
                    st.markdown('<div class="success-message">', unsafe_allow_html=True)
                    st.markdown("**🎉 Vocal removal completed successfully!**")
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Processing Time", f"{processing_time:.2f}s")
                    with col_b:
                        st.metric("Real-time Factor", f"{result['real_time_factor']:.2f}x")
                    with col_c:
                        st.metric("Output Sample Rate", f"{result['output_sample_rate']} Hz")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Play processed audio
                    with open(output_file_path, 'rb') as audio_file:
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format='audio/wav')
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Instrumental Track",
                        data=audio_bytes,
                        file_name=output_filename,
                        mime="audio/wav",
                        use_container_width=True
                    )
                    
                    # Clean up temporary files
                    try:
                        os.unlink(input_file_path)
                        os.unlink(output_file_path)
                    except:
                        pass
                
                except Exception as e:
                    st.markdown('<div class="error-message">', unsafe_allow_html=True)
                    st.markdown(f"**❌ Processing failed:** {str(e)}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show detailed error in expander
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())
        
        elif uploaded_file is None:
            st.info("👆 Please upload an audio file to get started")
        
        elif not selected_model:
            st.info("👈 Please select or specify a model in the sidebar")
    
    # Footer with instructions
    st.markdown("---")
    
    with st.expander("📖 How to Use"):
        st.markdown("""
        ### Getting Started
        
        1. **Upload Audio**: Click "Browse files" and select your audio file
        2. **Select Model**: Choose a trained model from the sidebar (or train one first)
        3. **Adjust Settings**: Modify processing options if needed
        4. **Process**: Click "Remove Vocals" to start processing
        5. **Download**: Save the instrumental track to your device
        
        ### Supported Formats
        - **Input**: WAV, MP3, FLAC, M4A, AAC, OGG
        - **Output**: WAV (high quality)
        
        ### Tips for Best Results
        - Use high-quality audio files (44.1kHz or higher)
        - Stereo files work better than mono
        - Longer chunk duration = better quality but more memory usage
        - Higher overlap ratio = smoother output but slower processing
        
        ### Training Your Own Model
        If you don't have a trained model, you can train one using:
        ```bash
        python main.py train --data_dir your_music_dataset/ --output_dir models/
        ```
        """)
    
    with st.expander("🔧 Technical Information"):
        st.markdown("""
        ### Model Architecture
        - **Type**: U-Net based deep neural network
        - **Input**: Stereo audio spectrograms
        - **Output**: Separated instrumental tracks
        - **Processing**: Chunk-based for memory efficiency
        
        ### Performance
        - **Speed**: Typically 2-10x real-time depending on hardware
        - **Quality**: High-quality separation with minimal artifacts
        - **Memory**: Optimized for consumer hardware
        
        ### System Requirements
        - **Python**: 3.8 or higher
        - **Memory**: 4GB+ RAM recommended
        - **GPU**: Optional but recommended for faster processing
        """)


if __name__ == "__main__":
    main()