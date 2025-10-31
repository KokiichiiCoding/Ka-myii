// Generator Page JavaScript

const api = window.KamyiiAPI;
const utils = window.KamyiiUtils;

// State
let currentModelId = null;

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
            showResultState(result.model);
        } else {
            throw new Error(result.error || 'Generation failed');
        }
    } catch (error) {
        showErrorState(error.toString());
    }
}

// Show loading state
function showLoadingState() {
    initialState.style.display = 'none';
    loadingState.style.display = 'block';
    resultState.style.display = 'none';
    errorState.style.display = 'none';

    generateBtn.disabled = true;

    // Simulate progress (in real implementation, use WebSocket or polling)
    simulateProgress();
}

// Simulate progress
function simulateProgress() {
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
    form.reset();

    // Reset range displays
    stepsValue.textContent = stepsInput.value;
    guidanceScaleValue.textContent = guidanceScaleInput.value;
});

// Retry on error
retryBtn.addEventListener('click', function() {
    generateModel();
});

// Initialize
console.log('Generator page initialized');
