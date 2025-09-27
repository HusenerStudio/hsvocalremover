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
    const addTrainingPairBtn = document.getElementById('add-training-pair-btn');
    const startTrainingBtn = document.getElementById('start-training');
    const cancelTrainingBtn = document.getElementById('cancel-training');
    const trainingProgress = document.getElementById('training-progress');
    const trainingProgressBar = document.getElementById('training-progress-bar');
    const currentEpochSpan = document.getElementById('current-epoch');
    const totalEpochsSpan = document.getElementById('total-epochs');
    const currentLossSpan = document.getElementById('current-loss');
    const timeRemainingSpan = document.getElementById('time-remaining');
    const trainingLogContent = document.getElementById('training-log-content');
    
    // Tab Elements
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Local Files Tab Elements
    const modelUrlInput = document.getElementById('model-url-input');
    const modelFileInput = document.getElementById('model-file-input');
    const modelWeightsInput = document.getElementById('model-weights-input');
    const uploadedModelNameInput = document.getElementById('uploaded-model-name');
    const loadModelBtn = document.getElementById('load-model-btn');
    
    // GitHub Tab Elements
    const githubRepoInput = document.getElementById('github-repo-input');
    const githubBranchInput = document.getElementById('github-branch-input');
    const githubTokenInput = document.getElementById('github-token-input');
    const githubPathInput = document.getElementById('github-path-input');
    const githubModelNameInput = document.getElementById('github-model-name');
    const loadGithubModelBtn = document.getElementById('load-github-model-btn');
    
    // Hugging Face Tab Elements
    const hfModelIdInput = document.getElementById('hf-model-id-input');
    const hfRevisionInput = document.getElementById('hf-revision-input');
    const hfTokenInput = document.getElementById('hf-token-input');
    const hfFilenameInput = document.getElementById('hf-filename-input');
    const hfModelNameInput = document.getElementById('hf-model-name');
    const loadHfModelBtn = document.getElementById('load-hf-model-btn');

    // Tab Switching Functionality
    function initializeTabs() {
        tabButtons.forEach((button, index) => {
            button.addEventListener('click', () => {
                // Remove active class from all tabs and contents
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));
                
                // Add active class to clicked tab and corresponding content
                button.classList.add('active');
                tabContents[index].classList.add('active');
            });
        });
        
        // Set first tab as active by default
        if (tabButtons.length > 0) {
            tabButtons[0].classList.add('active');
            tabContents[0].classList.add('active');
        }
    }

    // GitHub Model Loading
    async function loadModelFromGitHub() {
        const repo = githubRepoInput.value.trim();
        const branch = githubBranchInput.value.trim() || 'main';
        const token = githubTokenInput.value.trim();
        const path = githubPathInput.value.trim();
        const modelName = githubModelNameInput.value.trim() || 'GitHub Model';
        
        if (!repo || !path) {
            showStatusMessage('github-status', 'Please enter repository URL and model path', 'error');
            return;
        }
        
        try {
            showStatusMessage('github-status', 'Downloading model from GitHub...', 'info');
            setButtonLoading(loadGithubModelBtn, true);
            
            // Extract owner and repo name from URL
            const repoMatch = repo.match(/github\.com\/([^\/]+)\/([^\/]+)/);
            if (!repoMatch) {
                throw new Error('Invalid GitHub repository URL');
            }
            
            const [, owner, repoName] = repoMatch;
            
            // Construct GitHub API URL
            const apiUrl = `https://api.github.com/repos/${owner}/${repoName}/contents/${path}?ref=${branch}`;
            
            const headers = {
                'Accept': 'application/vnd.github.v3+json'
            };
            
            if (token) {
                headers['Authorization'] = `token ${token}`;
            }
            
            const response = await fetch(apiUrl, { headers });
            
            if (!response.ok) {
                throw new Error(`GitHub API error: ${response.status} ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (data.type !== 'file') {
                throw new Error('Path does not point to a file');
            }
            
            // Download the file content
            const fileResponse = await fetch(data.download_url);
            if (!fileResponse.ok) {
                throw new Error('Failed to download model file');
            }
            
            // Load the model using TensorFlow.js
            const model = await tf.loadLayersModel(data.download_url);
            
            // Add the loaded model to our model management system
            trainedModel = model;
            saveModelToLocalStorage(modelName, model);
            
            showStatusMessage('github-status', `Successfully loaded model: ${modelName}`, 'success');
            updateModelSelect();
            
        } catch (error) {
            showStatusMessage('github-status', `Failed to load model: ${error.message}`, 'error');
            console.error('GitHub model loading error:', error);
        } finally {
            setButtonLoading(loadGithubModelBtn, false);
        }
    }

    // Hugging Face Model Loading
    async function loadModelFromHuggingFace() {
        const modelId = hfModelIdInput.value.trim();
        const revision = hfRevisionInput.value.trim() || 'main';
        const token = hfTokenInput.value.trim();
        const filename = hfFilenameInput.value.trim() || 'model.json';
        const modelName = hfModelNameInput.value.trim() || 'Hugging Face Model';
        
        if (!modelId) {
            showStatusMessage('hf-status', 'Please enter a model ID', 'error');
            return;
        }
        
        try {
            showStatusMessage('hf-status', 'Downloading model from Hugging Face...', 'info');
            setButtonLoading(loadHfModelBtn, true);
            
            // Construct Hugging Face model URL
            const modelUrl = `https://huggingface.co/${modelId}/resolve/${revision}/${filename}`;
            
            const headers = {};
            if (token) {
                headers['Authorization'] = `Bearer ${token}`;
            }
            
            // Test if the model file exists
            const testResponse = await fetch(modelUrl, { 
                method: 'HEAD',
                headers 
            });
            
            if (!testResponse.ok) {
                throw new Error(`Model file not found: ${testResponse.status} ${testResponse.statusText}`);
            }
            
            // Load the model using TensorFlow.js
            const model = await tf.loadLayersModel(modelUrl);
            
            // Add the loaded model to our model management system
            trainedModel = model;
            saveModelToLocalStorage(modelName, model);
            
            showStatusMessage('hf-status', `Successfully loaded model: ${modelName}`, 'success');
            updateModelSelect();
            
        } catch (error) {
            showStatusMessage('hf-status', `Failed to load model: ${error.message}`, 'error');
            console.error('Hugging Face model loading error:', error);
        } finally {
            setButtonLoading(loadHfModelBtn, false);
        }
    }

    // Utility Functions
    function showStatusMessage(containerId, message, type) {
        const container = document.getElementById(containerId);
        if (!container) {
            // Create status message container if it doesn't exist
            const statusDiv = document.createElement('div');
            statusDiv.id = containerId;
            statusDiv.className = 'status-message';
            
            // Find the appropriate parent to append to
            if (containerId.includes('github')) {
                document.getElementById('github-tab').appendChild(statusDiv);
            } else if (containerId.includes('hf')) {
                document.getElementById('huggingface-tab').appendChild(statusDiv);
            } else {
                document.getElementById('local-files-tab').appendChild(statusDiv);
            }
        }
        
        const statusElement = document.getElementById(containerId);
        statusElement.textContent = message;
        statusElement.className = `status-message ${type}`;
        statusElement.style.display = 'block';
        
        // Auto-hide success messages after 5 seconds
        if (type === 'success') {
            setTimeout(() => {
                statusElement.style.display = 'none';
            }, 5000);
        }
    }

    function setButtonLoading(button, isLoading) {
        if (isLoading) {
            button.classList.add('loading');
            button.disabled = true;
        } else {
            button.classList.remove('loading');
            button.disabled = false;
        }
    }

    // Initialize tabs
    initializeTabs();

    // Event listener for Train Model button
    trainModelBtn.addEventListener('click', () => {
        modelTrainingSection.classList.remove('hidden');
        modelTrainingSection.scrollIntoView({ behavior: 'smooth' });
    });

    // Event Listeners for Tab Actions
    if (loadGithubModelBtn) {
        loadGithubModelBtn.addEventListener('click', loadModelFromGitHub);
    }

    if (loadHfModelBtn) {
        loadHfModelBtn.addEventListener('click', loadModelFromHuggingFace);
    }

    // Placeholder for model management functions (will be fully implemented later)
    function saveModelToLocalStorage(modelName, model) {
        console.log(`Saving model ${modelName} to local storage`);
        // Actual implementation will go here
        const modelJson = model.toJSON();
        localStorage.setItem(`tfjs-model-${modelName}-json`, JSON.stringify(modelJson));
        // For weights, we might need to save them separately or rely on TF.js to handle it
        // For now, we'll just save the model JSON and assume weights are handled by TF.js during loading
        updateModelSelect();
    }

    function updateModelSelect() {
        console.log('Updating model select dropdown');
        // Actual implementation will go here
        const modelSelect = document.getElementById('model-select');
        // Clear existing options, except the default ones
        modelSelect.innerHTML = `
            <option value="default">Default Vocal Remover</option>
            <option value="trained" disabled>Custom Trained Model</option>
        `;

        // Add models from local storage
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key.startsWith('tfjs-model-') && key.endsWith('-json')) {
                const modelName = key.replace('tfjs-model-', '').replace('-json', '');
                const option = document.createElement('option');
                option.value = modelName;
                option.textContent = modelName;
                modelSelect.appendChild(option);
            }
        }

        // If a trained model exists, enable and update its option
        if (localStorage.getItem('hasTrainedModel') === 'true') {
            const trainedOption = modelSelect.querySelector('option[value="trained"]');
            if (trainedOption) {
                trainedOption.disabled = false;
                trainedOption.textContent = localStorage.getItem('modelName') || 'Custom Trained Model';
            }
        }
    }

    // Load model from URL (Local Files Tab)
    if (loadModelBtn) {
        loadModelBtn.addEventListener('click', async () => {
            const modelUrl = modelUrlInput.value.trim();
            const modelName = uploadedModelNameInput.value.trim() || 'Imported Model';
            
            if (!modelUrl) {
                showStatusMessage('local-status', 'Please enter a valid model URL', 'error');
                return;
            }
            
            try {
                showStatusMessage('local-status', `Loading model from URL: ${modelUrl}`, 'info');
                setButtonLoading(loadModelBtn, true);
                
                // Load the model using TensorFlow.js
                const model = await tf.loadLayersModel(modelUrl);
                
                // Add the loaded model to our model management system
                trainedModel = model;
                saveModelToLocalStorage(modelName, model);
                
                showStatusMessage('local-status', `Successfully loaded model: ${modelName}`, 'success');
                updateModelSelect();
            } catch (error) {
                showStatusMessage('local-status', `Failed to load model: ${error.message}`, 'error');
                console.error('Model loading error:', error);
            } finally {
                setButtonLoading(loadModelBtn, false);
            }
        });
    }

    // Audio Context and nodes
    let audioContext;
    let sourceNode;
    let audioBuffer;
    let processedBuffer;
    let currentFileName = null;
    
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
            currentFileName = file.name;
            fileInfo.textContent = `Selected file: ${file.name}`;
            loadAudioFile(file);
            
            // Save upload to history if user is logged in
            if (window.authSystem && window.authSystem.isLoggedIn()) {
                const historyItem = {
                    type: 'upload',
                    fileName: file.name,
                    status: 'completed',
                    description: `Uploaded audio file: ${file.name}`,
                    fileSize: file.size
                };
                window.authSystem.saveToHistory(historyItem);
            }
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
        
        // Save to history if user is logged in
        if (window.authSystem && window.authSystem.isLoggedIn() && currentFileName) {
            const historyItem = {
                type: 'separation',
                fileName: currentFileName,
                status: 'completed',
                description: 'Vocal separation completed successfully',
                playUrl: url,
                downloadUrl: url,
                modelName: modelSelect.options[modelSelect.selectedIndex].text
            };
            window.authSystem.saveToHistory(historyItem);
        }
        
        // Set up download button
        downloadButton.addEventListener('click', () => {
            const a = document.createElement('a');
            a.href = url;
            a.download = 'instrumental.wav';
            a.click();
            
            // Save download to history if user is logged in
            if (window.authSystem && window.authSystem.isLoggedIn() && currentFileName) {
                const historyItem = {
                    type: 'download',
                    fileName: 'instrumental.wav',
                    status: 'completed',
                    description: `Downloaded instrumental version of ${currentFileName}`,
                    downloadUrl: url
                };
                window.authSystem.saveToHistory(historyItem);
            }
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

    // Theme Toggle Functionality
    const themeToggleBtn = document.getElementById('theme-toggle');
    const body = document.body;

    // Load saved theme from localStorage
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        body.classList.add(savedTheme);
        if (savedTheme === 'dark-theme') {
            themeToggleBtn.querySelector('i').classList.remove('fa-moon');
            themeToggleBtn.querySelector('i').classList.add('fa-sun');
        }
    }

    themeToggleBtn.addEventListener('click', () => {
        body.classList.toggle('dark-theme');
        if (body.classList.contains('dark-theme')) {
            localStorage.setItem('theme', 'dark-theme');
            themeToggleBtn.querySelector('i').classList.remove('fa-moon');
            themeToggleBtn.querySelector('i').classList.add('fa-sun');
        } else {
            localStorage.setItem('theme', 'light-theme');
            themeToggleBtn.querySelector('i').classList.remove('fa-sun');
            themeToggleBtn.querySelector('i').classList.add('fa-moon');
        }
    });
});