// Generator Page JavaScript

const api = window.KamyiiAPI;
const utils = window.KamyiiUtils;

// State
let currentModelId = null;
let socket = null;
let currentTaskId = null;

// Initialize WebSocket connection for progress updates
try {
    if (typeof io !== 'undefined') {
        socket = io('/progress');

        socket.on('connect', function() {
            console.log('Connected to progress tracking');
        });

        socket.on('disconnect', function() {
            console.log('Disconnected from progress tracking');
        });

        socket.on('progress_update', function(data) {
            updateProgress(data);
        });

        console.log('WebSocket progress tracking initialized');
    } else {
        console.warn('Socket.IO not available - progress tracking disabled');
    }
} catch (error) {
    console.warn('Failed to initialize WebSocket:', error);
}

// UI Elements
const form = document.getElementById('generationForm');
const generateBtn = document.getElementById('generateBtn');

// State containers
const initialState = document.getElementById('initialState');
const loadingState = document.getElementById('loadingState');
const resultState = document.getElementById('resultState');
const errorState = document.getElementById('errorState');

// Form inputs
const promptInput = document.getElementById('prompt');
const negativePromptInput = document.getElementById('negativePrompt');
const styleSelect = document.getElementById('style');
const widthInput = document.getElementById('width');
const heightInput = document.getElementById('height');
const stepsInput = document.getElementById('steps');
const guidanceScaleInput = document.getElementById('guidanceScale');
const seedInput = document.getElementById('seed');
const includeRiggingInput = document.getElementById('includeRigging');

// Range value displays
const stepsValue = document.getElementById('stepsValue');
const guidanceScaleValue = document.getElementById('guidanceScaleValue');

// Result elements
const previewImage = document.getElementById('previewImage');
const modelIdSpan = document.getElementById('modelId');
const modelStatusSpan = document.getElementById('modelStatus');
const generationTimeSpan = document.getElementById('generationTime');
const assetCountSpan = document.getElementById('assetCount');

// Buttons
const downloadBtn = document.getElementById('downloadBtn');
const viewDetailsBtn = document.getElementById('viewDetailsBtn');
const generateAnotherBtn = document.getElementById('generateAnotherBtn');
const retryBtn = document.getElementById('retryBtn');

// Update range value displays
stepsInput.addEventListener('input', function() {
    stepsValue.textContent = this.value;
});

guidanceScaleInput.addEventListener('input', function() {
    guidanceScaleValue.textContent = this.value;
});

// Form submission
form.addEventListener('submit', async function(e) {
    e.preventDefault();
    await generateModel();
});

// Generate model
async function generateModel() {
    // Get form values
    const params = {
        prompt: promptInput.value.trim(),
        negative_prompt: negativePromptInput.value.trim(),
        style: styleSelect.value,
        width: parseInt(widthInput.value),
        height: parseInt(heightInput.value),
        steps: parseInt(stepsInput.value),
        guidance_scale: parseFloat(guidanceScaleInput.value),
        seed: seedInput.value ? parseInt(seedInput.value) : null,
        include_rigging: includeRiggingInput.checked
    };

    // Validate
    if (!params.prompt) {
        alert('Please enter a character description');
        return;
    }

    // Show loading state
    showLoadingState();

    try {
        // Call API
        const result = await api.generateModel(params);

        if (result.success) {
            currentModelId = result.model_id;
            // Store task_id for progress tracking
            if (result.task_id) {
                currentTaskId = result.task_id;
                console.log('Tracking progress for task:', currentTaskId);
            }
            showResultState(result.model);
        } else {
            throw new Error(result.error || 'Generation failed');
        }
    } catch (error) {
        showErrorState(error.toString());
        // Clear task ID on error
        currentTaskId = null;
    }
}

// Show loading state
function showLoadingState() {
    initialState.style.display = 'none';
    loadingState.style.display = 'block';
    resultState.style.display = 'none';
    errorState.style.display = 'none';

    generateBtn.disabled = true;

    // Reset progress bar
    const progressBar = document.getElementById('progressBar');
    const loadingMessage = document.getElementById('loadingMessage');
    progressBar.style.width = '0%';
    loadingMessage.textContent = 'Starting generation...';

    // Use real-time progress tracking if available, otherwise simulate
    simulateProgress();
}

// Update progress from WebSocket
function updateProgress(data) {
    // Check if this update is for the current task
    if (currentTaskId && data.task_id !== currentTaskId) {
        return; // Ignore updates for other tasks
    }

    const progressBar = document.getElementById('progressBar');
    const loadingMessage = document.getElementById('loadingMessage');

    // Update progress bar
    if (data.progress !== undefined) {
        progressBar.style.width = data.progress + '%';
        progressBar.setAttribute('aria-valuenow', data.progress);
    }

    // Update message
    if (data.message) {
        let displayMessage = data.message;

        // Add ETA if available
        if (data.eta && data.eta > 1) {
            const etaSeconds = Math.round(data.eta);
            const etaMinutes = Math.floor(etaSeconds / 60);
            const remainingSeconds = etaSeconds % 60;

            if (etaMinutes > 0) {
                displayMessage += ` (ETA: ${etaMinutes}m ${remainingSeconds}s)`;
            } else {
                displayMessage += ` (ETA: ${etaSeconds}s)`;
            }
        }

        loadingMessage.textContent = displayMessage;
    }

    // Handle completion states
    if (data.status === 'completed' || data.status === 'complete') {
        console.log('Generation completed');
        progressBar.style.width = '100%';
        loadingMessage.textContent = data.message || 'Generation complete!';
    } else if (data.status === 'failed') {
        console.error('Generation failed:', data.message);
        showErrorState(data.message || 'Generation failed');
    } else if (data.status === 'cancelled') {
        console.log('Generation cancelled');
        showErrorState('Generation was cancelled');
    }
}

// Fallback: Simulate progress if WebSocket is not available
function simulateProgress() {
    if (socket && socket.connected) {
        // WebSocket available, don't simulate
        console.log('Using real-time progress tracking');
        return;
    }

    console.log('WebSocket not available, simulating progress');

    const progressBar = document.getElementById('progressBar');
    const loadingMessage = document.getElementById('loadingMessage');

    const stages = [
        { message: 'Initializing...', progress: 10 },
        { message: 'Generating base image...', progress: 30 },
        { message: 'Separating assets...', progress: 60 },
        { message: 'Assembling model...', progress: 80 },
        { message: 'Finalizing...', progress: 95 }
    ];

    let currentStage = 0;

    const interval = setInterval(function() {
        if (currentStage < stages.length) {
            const stage = stages[currentStage];
            loadingMessage.textContent = stage.message;
            progressBar.style.width = stage.progress + '%';
            currentStage++;
        } else {
            clearInterval(interval);
        }
    }, 3000);
}

// Show result state
function showResultState(model) {
    initialState.style.display = 'none';
    loadingState.style.display = 'none';
    resultState.style.display = 'block';
    errorState.style.display = 'none';

    generateBtn.disabled = false;

    // Update UI with model data
    previewImage.src = api.getPreviewUrl(model.id);
    modelIdSpan.textContent = model.id;
    modelStatusSpan.textContent = utils.getStatusName(model.status);
    modelStatusSpan.className = 'badge ' + utils.getStatusBadge(model.status);
    generationTimeSpan.textContent = utils.formatDuration(model.generation_time);
    assetCountSpan.textContent = model.assets?.length || 0;

    // Set up download button
    downloadBtn.onclick = function() {
        window.location.href = api.getDownloadUrl(model.id);
    };

    // Set up edit expressions button
    const editExpressionsBtn = document.getElementById('editExpressionsBtn');
    if (editExpressionsBtn) {
        editExpressionsBtn.href = `/editor/${model.id}`;
        editExpressionsBtn.style.display = 'block';
    }

    // Set up view details button
    viewDetailsBtn.onclick = function() {
        showModelDetails(model);
    };
}

// Show error state
function showErrorState(errorMessage) {
    initialState.style.display = 'none';
    loadingState.style.display = 'none';
    resultState.style.display = 'none';
    errorState.style.display = 'block';

    generateBtn.disabled = false;

    document.getElementById('errorMessage').textContent = errorMessage;
}

// Show model details
function showModelDetails(model) {
    let details = `
Model Details:
--------------
ID: ${model.id}
Name: ${model.name || 'N/A'}
Status: ${model.status}
Created: ${model.created_at}
Generation Time: ${utils.formatDuration(model.generation_time)}

Assets (${model.assets?.length || 0}):
${model.assets?.map(a => `- ${a.layer_type}: ${a.file_path}`).join('\n') || 'None'}

Base Image: ${model.base_image_path || 'N/A'}
Final Model: ${model.final_model_path || 'N/A'}
    `.trim();

    alert(details);
}

// Generate another model
generateAnotherBtn.addEventListener('click', function() {
    initialState.style.display = 'block';
    loadingState.style.display = 'none';
    resultState.style.display = 'none';
    errorState.style.display = 'none';

    currentModelId = null;
    currentTaskId = null;  // Clear task ID
    form.reset();

    // Reset range displays
    stepsValue.textContent = stepsInput.value;
    guidanceScaleValue.textContent = guidanceScaleInput.value;
});

// Retry on error
retryBtn.addEventListener('click', function() {
    generateModel();
});

// Model Status Checking
async function checkModelStatus() {
    const statusAlert = document.getElementById('modelStatusAlert');
    const statusMessage = document.getElementById('modelStatusMessage');
    const preloadBtn = document.getElementById('preloadModelBtn');

    try {
        const response = await axios.get('/api/generation/model-status');

        if (response.data.success) {
            const { model_loaded, device, is_dummy, message } = response.data;

            statusAlert.style.display = 'block';

            if (is_dummy) {
                statusAlert.className = 'alert alert-warning';
                statusMessage.textContent = message;
                preloadBtn.style.display = 'none';
            } else if (model_loaded) {
                statusAlert.className = 'alert alert-success';
                statusMessage.textContent = `${message} (${device.toUpperCase()})`;
                preloadBtn.style.display = 'none';
            } else {
                statusAlert.className = 'alert alert-warning';
                statusMessage.textContent = message + '. Click the button to download it now (4-5 GB, takes 10-30 min).';
                preloadBtn.style.display = 'inline-block';
            }
        }
    } catch (error) {
        console.error('Failed to check model status:', error);
        statusAlert.style.display = 'block';
        statusAlert.className = 'alert alert-danger';
        statusMessage.textContent = 'Could not check model status';
    }
}

// Preload Model
async function preloadModel() {
    const preloadBtn = document.getElementById('preloadModelBtn');
    const preloadProgress = document.getElementById('preloadProgress');
    const statusMessage = document.getElementById('modelStatusMessage');

    preloadBtn.disabled = true;
    preloadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Downloading...';
    preloadProgress.style.display = 'block';

    try {
        statusMessage.textContent = 'Downloading Stable Diffusion model... Please wait (this may take 10-30 minutes)';

        const response = await axios.post('/api/generation/preload-model');

        if (response.data.success) {
            preloadProgress.style.display = 'none';
            statusMessage.textContent = 'Model loaded successfully! You can now generate models.';
            document.getElementById('modelStatusAlert').className = 'alert alert-success';
            preloadBtn.style.display = 'none';
        } else {
            throw new Error(response.data.message || 'Model loading failed');
        }
    } catch (error) {
        preloadProgress.style.display = 'none';
        statusMessage.textContent = 'Model download failed: ' + error.message;
        document.getElementById('modelStatusAlert').className = 'alert alert-danger';
        preloadBtn.disabled = false;
        preloadBtn.innerHTML = '<i class="fas fa-download"></i> Try Again';
        preloadBtn.style.display = 'inline-block';
    }
}

// Attach preload button click
document.getElementById('preloadModelBtn').addEventListener('click', preloadModel);

// Initialize
console.log('Generator page initialized');

// Check model status on page load
checkModelStatus();
