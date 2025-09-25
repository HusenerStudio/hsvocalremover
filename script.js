document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const startButton = document.getElementById('start-button');
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const audioControls = document.getElementById('audio-controls');
    const audioPlayer = document.getElementById('audio-player');
    const removeVocalsButton = document.getElementById('remove-vocals');
    const downloadButton = document.getElementById('download-instrumental');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const statusMessage = document.getElementById('status-message');
    
    // Audio Context and nodes
    let audioContext;
    let sourceNode;
    let audioBuffer;
    let processedBuffer;
    
    // Scroll to tool section when start button is clicked
    startButton.addEventListener('click', () => {
        document.getElementById('vocal-remover').scrollIntoView({ behavior: 'smooth' });
    });
    
    // File Upload Handling
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and Drop Handling
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropArea.classList.add('dragover');
    }
    
    function unhighlight() {
        dropArea.classList.remove('dragover');
    }
    
    dropArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length) {
            handleFiles(files);
        }
    }
    
    function handleFiles(files) {
        const file = files[0];
        if (file.type.startsWith('audio/')) {
            fileInfo.textContent = `Selected file: ${file.name}`;
            loadAudioFile(file);
        } else {
            fileInfo.textContent = 'Please select an audio file.';
        }
    }
    
    function handleFileSelect(e) {
        const files = e.target.files;
        if (files.length) {
            handleFiles(files);
        }
    }
    
    function loadAudioFile(file) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            const arrayBuffer = e.target.result;
            
            // Initialize AudioContext if not already done
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            
            // Decode the audio file
            audioContext.decodeAudioData(arrayBuffer)
                .then(buffer => {
                    audioBuffer = buffer;
                    
                    // Create a blob URL for the audio player
                    const blob = new Blob([arrayBuffer], { type: file.type });
                    const url = URL.createObjectURL(blob);
                    
                    // Set the audio player source
                    audioPlayer.src = url;
                    
                    // Show audio controls
                    audioControls.classList.remove('hidden');
                    
                    // Reset UI elements
                    downloadButton.disabled = true;
                    progressContainer.classList.add('hidden');
                    statusMessage.textContent = '';
                })
                .catch(error => {
                    console.error('Error decoding audio data', error);
                    statusMessage.textContent = 'Error loading audio file. Please try another file.';
                });
        };
        
        reader.readAsArrayBuffer(file);
    }
    
    // Process audio to remove vocals
    removeVocalsButton.addEventListener('click', removeVocals);
    
    function removeVocals() {
        if (!audioBuffer) {
            statusMessage.textContent = 'Please upload an audio file first.';
            return;
        }
        
        // Show progress
        progressContainer.classList.remove('hidden');
        progressBar.style.width = '0%';
        statusMessage.textContent = 'Processing... Please wait.';
        
        // Disable buttons during processing
        removeVocalsButton.disabled = true;
        
        // Simulate progress (in a real app, this would be based on actual processing)
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += 5;
            progressBar.style.width = `${progress}%`;
            
            if (progress >= 100) {
                clearInterval(progressInterval);
                finishProcessing();
            }
        }, 200);
        
        // Process the audio (vocal removal algorithm)
        setTimeout(() => {
            processedBuffer = removeVocalsFromBuffer(audioBuffer);
        }, 500);
    }
    
    function removeVocalsFromBuffer(buffer) {
        // Get the audio data
        const leftChannel = buffer.getChannelData(0);
        const rightChannel = buffer.getChannelData(1);
        
        // Create a new buffer for the processed audio
        const processedBuffer = audioContext.createBuffer(
            buffer.numberOfChannels,
            buffer.length,
            buffer.sampleRate
        );
        
        // Get the processed buffer channels
        const processedLeftChannel = processedBuffer.getChannelData(0);
        const processedRightChannel = processedBuffer.getChannelData(1);
        
        // Apply center channel cancellation (basic vocal removal technique)
        // This works because vocals are usually centered in the stereo field
        for (let i = 0; i < buffer.length; i++) {
            // Invert one channel and mix with the other to cancel out center content
            processedLeftChannel[i] = leftChannel[i] - rightChannel[i];
            processedRightChannel[i] = rightChannel[i] - leftChannel[i];
        }
        
        return processedBuffer;
    }
    
    function finishProcessing() {
        statusMessage.textContent = 'Vocal removal complete!';
        
        // Enable download button
        downloadButton.disabled = false;
        removeVocalsButton.disabled = false;
        
        // Create a new audio source from the processed buffer
        const audioData = bufferToWave(processedBuffer, processedBuffer.length);
        const blob = new Blob([audioData], { type: 'audio/wav' });
        const url = URL.createObjectURL(blob);
        
        // Update audio player with processed audio
        audioPlayer.src = url;
        
        // Set up download button
        downloadButton.addEventListener('click', () => {
            const a = document.createElement('a');
            a.href = url;
            a.download = 'instrumental.wav';
            a.click();
        });
    }
    
    // Convert AudioBuffer to WAV format
    function bufferToWave(buffer, len) {
        const numOfChan = buffer.numberOfChannels;
        const length = len * numOfChan * 2 + 44;
        const data = new Uint8Array(length);
        
        // Write WAV header
        writeString(data, 0, 'RIFF');
        data[4] = (length & 0xff);
        data[5] = ((length >> 8) & 0xff);
        data[6] = ((length >> 16) & 0xff);
        data[7] = ((length >> 24) & 0xff);
        writeString(data, 8, 'WAVE');
        writeString(data, 12, 'fmt ');
        data[16] = 16; // PCM format
        data[17] = 0;
        data[18] = 0;
        data[19] = 0;
        data[20] = 1; // No compression
        data[21] = 0;
        data[22] = numOfChan;
        data[23] = 0;
        data[24] = (buffer.sampleRate & 0xff);
        data[25] = ((buffer.sampleRate >> 8) & 0xff);
        data[26] = ((buffer.sampleRate >> 16) & 0xff);
        data[27] = ((buffer.sampleRate >> 24) & 0xff);
        const bytesPerSample = 2;
        const blockAlign = numOfChan * bytesPerSample;
        const byteRate = buffer.sampleRate * blockAlign;
        data[28] = (byteRate & 0xff);
        data[29] = ((byteRate >> 8) & 0xff);
        data[30] = ((byteRate >> 16) & 0xff);
        data[31] = ((byteRate >> 24) & 0xff);
        data[32] = blockAlign;
        data[33] = 0;
        data[34] = bytesPerSample * 8;
        data[35] = 0;
        writeString(data, 36, 'data');
        data[40] = ((length - 44) & 0xff);
        data[41] = (((length - 44) >> 8) & 0xff);
        data[42] = (((length - 44) >> 16) & 0xff);
        data[43] = (((length - 44) >> 24) & 0xff);
        
        // Write audio data
        let offset = 44;
        for (let i = 0; i < buffer.length; i++) {
            for (let channel = 0; channel < numOfChan; channel++) {
                const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
                const int16 = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
                data[offset++] = int16 & 0xFF;
                data[offset++] = (int16 >> 8) & 0xFF;
            }
        }
        
        return data;
    }
    
    function writeString(data, offset, string) {
        for (let i = 0; i < string.length; i++) {
            data[offset + i] = string.charCodeAt(i);
        }
    }
});