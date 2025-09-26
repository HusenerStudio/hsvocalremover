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
    
    // Model Training Elements
    const modelSelect = document.getElementById('model-select');
    const trainModelBtn = document.getElementById('train-model-btn');
    const modelTrainingSection = document.getElementById('model-training');
    const addTrainingPairBtn = document.getElementById('add-training-pair');
    const startTrainingBtn = document.getElementById('start-training');
    const cancelTrainingBtn = document.getElementById('cancel-training');
    const trainingProgress = document.getElementById('training-progress');
    const trainingProgressBar = document.getElementById('training-progress-bar');
    const currentEpochSpan = document.getElementById('current-epoch');
    const totalEpochsSpan = document.getElementById('total-epochs');
    const currentLossSpan = document.getElementById('current-loss');
    const timeRemainingSpan = document.getElementById('time-remaining');
    const trainingLogContent = document.getElementById('training-log-content');
    
    // Audio Context and nodes
    let audioContext;
    let sourceNode;
    let audioBuffer;
    let processedBuffer;
    
    // AI Model variables
    let trainedModel = null;
    let trainingData = [];
    let isTraining = false;
    let trainingStartTime = null;
    
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

    // ===== AI MODEL TRAINING FUNCTIONALITY =====
    
    // Show/hide training section
    trainModelBtn.addEventListener('click', () => {
        modelTrainingSection.classList.toggle('hidden');
        if (!modelTrainingSection.classList.contains('hidden')) {
            modelTrainingSection.scrollIntoView({ behavior: 'smooth' });
        }
    });
    
    // Add training pair functionality
    let pairCounter = 1;
    addTrainingPairBtn.addEventListener('click', () => {
        pairCounter++;
        const pairContainer = document.getElementById('training-pairs-container');
        const newPair = createTrainingPairElement(pairCounter);
        pairContainer.appendChild(newPair);
        updateTrainingButtonState();
    });
    
    function createTrainingPairElement(pairNumber) {
        const pairDiv = document.createElement('div');
        pairDiv.className = 'training-pair-item';
        pairDiv.innerHTML = `
            <div class="pair-header">
                <h4>Training Pair ${pairNumber}</h4>
                <button class="remove-pair-btn" onclick="removePair(this)">×</button>
            </div>
            <div class="file-pair">
                <div class="file-upload-section">
                    <label>Original Song (with vocals):</label>
                    <input type="file" class="original-file" accept="audio/*">
                    <span class="file-name">No file selected</span>
                </div>
                <div class="file-upload-section">
                    <label>Instrumental Version:</label>
                    <input type="file" class="instrumental-file" accept="audio/*">
                    <span class="file-name">No file selected</span>
                </div>
            </div>
        `;
        
        // Add event listeners for file inputs
        const originalInput = pairDiv.querySelector('.original-file');
        const instrumentalInput = pairDiv.querySelector('.instrumental-file');
        
        originalInput.addEventListener('change', (e) => {
            const fileName = e.target.files[0]?.name || 'No file selected';
            pairDiv.querySelector('.file-upload-section:first-child .file-name').textContent = fileName;
            updateTrainingButtonState();
        });
        
        instrumentalInput.addEventListener('change', (e) => {
            const fileName = e.target.files[0]?.name || 'No file selected';
            pairDiv.querySelector('.file-upload-section:last-child .file-name').textContent = fileName;
            updateTrainingButtonState();
        });
        
        return pairDiv;
    }
    
    // Remove training pair
    window.removePair = function(button) {
        const pairItem = button.closest('.training-pair-item');
        if (document.querySelectorAll('.training-pair-item').length > 1) {
            pairItem.remove();
            updateTrainingButtonState();
        }
    };
    
    // Update training button state based on uploaded files
    function updateTrainingButtonState() {
        const pairs = document.querySelectorAll('.training-pair-item');
        let validPairs = 0;
        
        pairs.forEach(pair => {
            const originalFile = pair.querySelector('.original-file').files[0];
            const instrumentalFile = pair.querySelector('.instrumental-file').files[0];
            if (originalFile && instrumentalFile) {
                validPairs++;
            }
        });
        
        startTrainingBtn.disabled = validPairs === 0;
    }
    
    // Add event listeners to initial training pair
    document.addEventListener('DOMContentLoaded', () => {
        const initialPair = document.querySelector('.training-pair-item');
        if (initialPair) {
            const originalInput = initialPair.querySelector('.original-file');
            const instrumentalInput = initialPair.querySelector('.instrumental-file');
            
            originalInput.addEventListener('change', (e) => {
                const fileName = e.target.files[0]?.name || 'No file selected';
                initialPair.querySelector('.file-upload-section:first-child .file-name').textContent = fileName;
                updateTrainingButtonState();
            });
            
            instrumentalInput.addEventListener('change', (e) => {
                const fileName = e.target.files[0]?.name || 'No file selected';
                initialPair.querySelector('.file-upload-section:last-child .file-name').textContent = fileName;
                updateTrainingButtonState();
            });
        }
    });
    
    // Start training process
    startTrainingBtn.addEventListener('click', async () => {
        if (isTraining) return;
        
        try {
            await startModelTraining();
        } catch (error) {
            console.error('Training error:', error);
            addTrainingLog('Error: ' + error.message, 'error');
        }
    });
    
    // Cancel training
    cancelTrainingBtn.addEventListener('click', () => {
        if (isTraining) {
            isTraining = false;
            addTrainingLog('Training cancelled by user', 'warning');
            resetTrainingUI();
        }
    });
    
    async function startModelTraining() {
        isTraining = true;
        trainingStartTime = Date.now();
        
        // Show training progress
        trainingProgress.classList.remove('hidden');
        startTrainingBtn.disabled = true;
        cancelTrainingBtn.classList.remove('hidden');
        
        // Clear previous logs
        trainingLogContent.innerHTML = '';
        addTrainingLog('Initializing training process...', 'info');
        
        // Collect training data
        const pairs = document.querySelectorAll('.training-pair-item');
        const trainingPairs = [];
        
        addTrainingLog(`Found ${pairs.length} training pairs`, 'info');
        
        // Process each training pair
        for (let i = 0; i < pairs.length; i++) {
            if (!isTraining) break;
            
            const pair = pairs[i];
            const originalFile = pair.querySelector('.original-file').files[0];
            const instrumentalFile = pair.querySelector('.instrumental-file').files[0];
            
            if (originalFile && instrumentalFile) {
                addTrainingLog(`Processing training pair ${i + 1}...`, 'info');
                
                try {
                    const originalBuffer = await loadAudioFileForTraining(originalFile);
                    const instrumentalBuffer = await loadAudioFileForTraining(instrumentalFile);
                    
                    trainingPairs.push({
                        original: originalBuffer,
                        instrumental: instrumentalBuffer
                    });
                    
                    addTrainingLog(`Training pair ${i + 1} processed successfully`, 'success');
                } catch (error) {
                    addTrainingLog(`Error processing pair ${i + 1}: ${error.message}`, 'error');
                }
            }
        }
        
        if (trainingPairs.length === 0) {
            addTrainingLog('No valid training pairs found', 'error');
            resetTrainingUI();
            return;
        }
        
        // Create and train the model
        addTrainingLog('Creating neural network model...', 'info');
        const model = await createVocalRemovalModel();
        
        addTrainingLog('Starting training process...', 'info');
        await trainModel(model, trainingPairs);
        
        if (isTraining) {
            // Save the trained model
            trainedModel = model;
            localStorage.setItem('hasTrainedModel', 'true');
            localStorage.setItem('modelName', document.getElementById('model-name').value);
            
            // Enable the trained model option
            const trainedOption = modelSelect.querySelector('option[value="trained"]');
            trainedOption.disabled = false;
            trainedOption.textContent = `${document.getElementById('model-name').value}`;
            
            addTrainingLog('Training completed successfully!', 'success');
            addTrainingLog('Model saved and ready to use', 'success');
        }
        
        resetTrainingUI();
    }
    
    async function loadAudioFileForTraining(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = async (e) => {
                try {
                    if (!audioContext) {
                        audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    }
                    const buffer = await audioContext.decodeAudioData(e.target.result);
                    resolve(buffer);
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsArrayBuffer(file);
        });
    }
    
    async function createVocalRemovalModel() {
        // Create a simple neural network for vocal removal
        const model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [1024], // FFT size
                    units: 512,
                    activation: 'relu'
                }),
                tf.layers.dropout({ rate: 0.3 }),
                tf.layers.dense({
                    units: 256,
                    activation: 'relu'
                }),
                tf.layers.dropout({ rate: 0.3 }),
                tf.layers.dense({
                    units: 128,
                    activation: 'relu'
                }),
                tf.layers.dense({
                    units: 1024, // Output FFT size
                    activation: 'sigmoid'
                })
            ]
        });
        
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });
        
        return model;
    }
    
    async function trainModel(model, trainingPairs) {
        const epochs = parseInt(document.getElementById('training-epochs').value);
        totalEpochsSpan.textContent = epochs;
        
        // Prepare training data
        const { inputs, outputs } = prepareTrainingData(trainingPairs);
        
        // Training callbacks
        const callbacks = {
            onEpochEnd: (epoch, logs) => {
                if (!isTraining) return;
                
                const progress = ((epoch + 1) / epochs) * 100;
                trainingProgressBar.style.width = `${progress}%`;
                currentEpochSpan.textContent = epoch + 1;
                currentLossSpan.textContent = logs.loss.toFixed(6);
                
                // Estimate remaining time
                const elapsed = Date.now() - trainingStartTime;
                const avgTimePerEpoch = elapsed / (epoch + 1);
                const remainingEpochs = epochs - (epoch + 1);
                const estimatedRemaining = avgTimePerEpoch * remainingEpochs;
                
                timeRemainingSpan.textContent = formatTime(estimatedRemaining);
                
                addTrainingLog(`Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(6)}`, 'info');
            }
        };
        
        try {
            await model.fit(inputs, outputs, {
                epochs: epochs,
                batchSize: 32,
                validationSplit: 0.2,
                callbacks: callbacks
            });
        } catch (error) {
            if (isTraining) {
                throw error;
            }
        }
    }
    
    function prepareTrainingData(trainingPairs) {
        const inputs = [];
        const outputs = [];
        
        trainingPairs.forEach(pair => {
            // Convert audio buffers to spectrograms
            const originalSpectrum = audioBufferToSpectrum(pair.original);
            const instrumentalSpectrum = audioBufferToSpectrum(pair.instrumental);
            
            // Create training samples
            for (let i = 0; i < originalSpectrum.length - 1024; i += 512) {
                const input = originalSpectrum.slice(i, i + 1024);
                const output = instrumentalSpectrum.slice(i, i + 1024);
                
                inputs.push(input);
                outputs.push(output);
            }
        });
        
        return {
            inputs: tf.tensor2d(inputs),
            outputs: tf.tensor2d(outputs)
        };
    }
    
    function audioBufferToSpectrum(buffer) {
        // Simple FFT-like conversion (simplified for demo)
        const channelData = buffer.getChannelData(0);
        const spectrum = [];
        
        for (let i = 0; i < channelData.length; i += 1024) {
            const chunk = channelData.slice(i, i + 1024);
            // Normalize and add to spectrum
            const normalized = chunk.map(sample => (sample + 1) / 2);
            spectrum.push(...normalized);
        }
        
        return spectrum;
    }
    
    function resetTrainingUI() {
        isTraining = false;
        startTrainingBtn.disabled = false;
        cancelTrainingBtn.classList.add('hidden');
        trainingProgressBar.style.width = '0%';
        currentEpochSpan.textContent = '0';
        currentLossSpan.textContent = '-';
        timeRemainingSpan.textContent = '-';
    }
    
    function addTrainingLog(message, type = 'info') {
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry log-${type}`;
        logEntry.innerHTML = `<span class="timestamp">[${new Date().toLocaleTimeString()}]</span> ${message}`;
        trainingLogContent.appendChild(logEntry);
        trainingLogContent.scrollTop = trainingLogContent.scrollHeight;
    }
    
    function formatTime(milliseconds) {
        const seconds = Math.floor(milliseconds / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        
        if (hours > 0) {
            return `${hours}h ${minutes % 60}m`;
        } else if (minutes > 0) {
            return `${minutes}m ${seconds % 60}s`;
        } else {
            return `${seconds}s`;
        }
    }
    
    // Load saved model on page load
    if (localStorage.getItem('hasTrainedModel') === 'true') {
        const trainedOption = modelSelect.querySelector('option[value="trained"]');
        trainedOption.disabled = false;
        trainedOption.textContent = localStorage.getItem('modelName') || 'Custom Trained Model';
    }
    
    // Update vocal removal to use selected model
    const originalRemoveVocals = removeVocals;
    window.removeVocals = function() {
        const selectedModel = modelSelect.value;
        
        if (selectedModel === 'trained' && trainedModel) {
            removeVocalsWithAI();
        } else {
            originalRemoveVocals();
        }
    };
    
    async function removeVocalsWithAI() {
        if (!audioBuffer || !trainedModel) {
            statusMessage.textContent = 'Please upload an audio file and ensure model is loaded.';
            return;
        }
        
        // Show progress
        progressContainer.classList.remove('hidden');
        progressBar.style.width = '0%';
        statusMessage.textContent = 'Processing with AI model... Please wait.';
        
        // Disable buttons during processing
        removeVocalsButton.disabled = true;
        
        try {
            // Convert audio to spectrum
            const spectrum = audioBufferToSpectrum(audioBuffer);
            
            // Process in chunks
            const processedSpectrum = [];
            const chunkSize = 1024;
            
            for (let i = 0; i < spectrum.length - chunkSize; i += chunkSize) {
                const chunk = spectrum.slice(i, i + chunkSize);
                const input = tf.tensor2d([chunk]);
                const prediction = trainedModel.predict(input);
                const output = await prediction.data();
                
                processedSpectrum.push(...output);
                
                // Update progress
                const progress = (i / (spectrum.length - chunkSize)) * 100;
                progressBar.style.width = `${progress}%`;
                
                // Clean up tensors
                input.dispose();
                prediction.dispose();
            }
            
            // Convert back to audio buffer
            processedBuffer = spectrumToAudioBuffer(processedSpectrum, audioBuffer.sampleRate);
            
            finishProcessing();
        } catch (error) {
            console.error('AI processing error:', error);
            statusMessage.textContent = 'Error processing with AI model. Using default algorithm.';
            originalRemoveVocals();
        }
    }
    
    function spectrumToAudioBuffer(spectrum, sampleRate) {
        // Convert spectrum back to audio buffer (simplified)
        const length = spectrum.length;
        const buffer = audioContext.createBuffer(2, length, sampleRate);
        
        const leftChannel = buffer.getChannelData(0);
        const rightChannel = buffer.getChannelData(1);
        
        for (let i = 0; i < length; i++) {
            const sample = (spectrum[i] * 2) - 1; // Denormalize
            leftChannel[i] = sample;
            rightChannel[i] = sample;
        }
        
        return buffer;
    }
});