// Expression & Accessory Editor JavaScript

const api = window.KamyiiAPI;
const utils = window.KamyiiUtils;

// Get model ID from template
const modelId = window.MODEL_ID;

// UI Elements
const generateExpressionsBtn = document.getElementById('generateExpressionsBtn');
const generateAccessoriesBtn = document.getElementById('generateAccessoriesBtn');
const applyAdjustmentsBtn = document.getElementById('applyAdjustmentsBtn');
const previewImage = document.getElementById('previewImage');
const previewLoading = document.getElementById('previewLoading');
const expressionList = document.getElementById('expressionList');
const accessoryList = document.getElementById('accessoryList');
const expressionToggleBar = document.getElementById('expressionToggleBar');
const expressionToggles = document.getElementById('expressionToggles');

// Checkboxes
const checkAll = document.getElementById('checkAll');
const exprCheckboxes = document.querySelectorAll('.expr-checkbox');

// Adjustment sliders
const brightnessSlider = document.getElementById('brightness');
const contrastSlider = document.getElementById('contrast');
const saturationSlider = document.getElementById('saturation');

const brightnessValue = document.getElementById('brightnessValue');
const contrastValue = document.getElementById('contrastValue');
const saturationValue = document.getElementById('saturationValue');

// State
let currentExpressions = [];
let currentAccessories = [];
let currentExpression = null;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    loadExistingExpressions();
    loadExistingAccessories();
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    // Check all toggle
    checkAll.addEventListener('change', function() {
        exprCheckboxes.forEach(cb => cb.checked = this.checked);
    });

    // Slider updates
    brightnessSlider.addEventListener('input', function() {
        brightnessValue.textContent = this.value;
    });

    contrastSlider.addEventListener('input', function() {
        contrastValue.textContent = this.value;
    });

    saturationSlider.addEventListener('input', function() {
        saturationValue.textContent = this.value;
    });

    // Generate expressions
    generateExpressionsBtn.addEventListener('click', generateExpressions);

    // Generate accessories
    generateAccessoriesBtn.addEventListener('click', generateAccessories);

    // Apply adjustments
    applyAdjustmentsBtn.addEventListener('click', applyAdjustments);

    // Export buttons
    document.getElementById('exportAllBtn').addEventListener('click', exportAll);
    document.getElementById('exportLive2DBtn').addEventListener('click', exportForLive2D);
}

// Generate expressions
async function generateExpressions() {
    const selectedExpressions = Array.from(exprCheckboxes)
        .filter(cb => cb.checked)
        .map(cb => cb.value);

    if (selectedExpressions.length === 0) {
        showToast('Please select at least one expression', 'warning');
        return;
    }

    showLoading(true);
    generateExpressionsBtn.disabled = true;
    showToast(`Generating ${selectedExpressions.length} expressions...`, 'info');

    try {
        const response = await fetch('/api/expression/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_id: modelId,
                expressions: selectedExpressions
            })
        });

        const result = await response.json();

        if (result.success) {
            currentExpressions = result.expressions.expressions;
            showToast(`Successfully generated ${currentExpressions.length} expressions!`, 'success');
            displayExpressions(result.expressions);
            createExpressionToggles(currentExpressions);
        } else {
            throw new Error(result.error || 'Failed to generate expressions');
        }
    } catch (error) {
        console.error('Expression generation failed:', error);
        showToast('Failed to generate expressions: ' + error.message, 'error');
    } finally {
        showLoading(false);
        generateExpressionsBtn.disabled = false;
    }
}

// Generate accessories
async function generateAccessories() {
    const preset = document.getElementById('accessoryPreset').value;

    if (!preset) {
        showToast('Please select an accessory preset', 'warning');
        return;
    }

    showLoading(true);
    generateAccessoriesBtn.disabled = true;
    showToast('Generating accessories...', 'info');

    try {
        const response = await fetch('/api/expression/accessories/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_id: modelId,
                preset: preset
            })
        });

        const result = await response.json();

        if (result.success) {
            currentAccessories = result.accessories;
            showToast(`Successfully added ${result.count} accessories!`, 'success');
            displayAccessories(result.accessories);

            // Refresh preview
            previewImage.src = previewImage.src + '?t=' + new Date().getTime();
        } else {
            throw new Error(result.error || 'Failed to generate accessories');
        }
    } catch (error) {
        console.error('Accessory generation failed:', error);
        showToast('Failed to generate accessories: ' + error.message, 'error');
    } finally {
        showLoading(false);
        generateAccessoriesBtn.disabled = false;
    }
}

// Apply image adjustments
async function applyAdjustments() {
    const operations = [];

    const brightness = parseFloat(brightnessSlider.value);
    const contrast = parseFloat(contrastSlider.value);
    const saturation = parseFloat(saturationSlider.value);

    if (brightness !== 1.0) {
        operations.push({ type: 'brightness', factor: brightness });
    }
    if (contrast !== 1.0) {
        operations.push({ type: 'contrast', factor: contrast });
    }
    if (saturation !== 1.0) {
        operations.push({ type: 'saturation', factor: saturation });
    }

    if (operations.length === 0) {
        showToast('No adjustments to apply', 'info');
        return;
    }

    showLoading(true);
    applyAdjustmentsBtn.disabled = true;
    showToast('Applying adjustments...', 'info');

    try {
        const response = await fetch('/api/expression/edit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_id: modelId,
                image_path: 'base_image.png',
                operations: operations
            })
        });

        const result = await response.json();

        if (result.success) {
            showToast('Adjustments applied successfully!', 'success');

            // Update preview to show edited image
            const editedPath = `/outputs/${modelId}/${result.edited_image_path}`;
            previewImage.src = editedPath + '?t=' + new Date().getTime();
        } else {
            throw new Error(result.error || 'Failed to apply adjustments');
        }
    } catch (error) {
        console.error('Adjustment failed:', error);
        showToast('Failed to apply adjustments: ' + error.message, 'error');
    } finally {
        showLoading(false);
        applyAdjustmentsBtn.disabled = false;
    }
}

// Load existing expressions
async function loadExistingExpressions() {
    try {
        const response = await fetch(`/api/expression/model/${modelId}/expressions`);
        const result = await response.json();

        if (result.success && result.expressions.expressions) {
            currentExpressions = result.expressions.expressions;
            displayExpressions(result.expressions);
            createExpressionToggles(currentExpressions);
        }
    } catch (error) {
        console.error('Failed to load expressions:', error);
    }
}

// Load existing accessories
async function loadExistingAccessories() {
    try {
        const response = await fetch(`/api/expression/accessories/preview/${modelId}`);
        const result = await response.json();

        if (result.success && result.accessories.accessories) {
            currentAccessories = result.accessories.accessories;
            displayAccessories(currentAccessories);
        }
    } catch (error) {
        console.error('Failed to load accessories:', error);
    }
}

// Display expressions in list
function displayExpressions(expressionsData) {
    if (!expressionsData.expressions || expressionsData.expressions.length === 0) {
        expressionList.innerHTML = `
            <div class="text-center text-muted py-3">
                <i class="fas fa-info-circle"></i>
                <p class="small mb-0">No expressions generated yet</p>
            </div>
        `;
        return;
    }

    expressionList.innerHTML = '';

    expressionsData.expressions.forEach(expr => {
        const item = document.createElement('a');
        item.href = '#';
        item.className = 'list-group-item list-group-item-action';
        item.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <span>${capitalizeFirst(expr)}</span>
                <i class="fas fa-eye"></i>
            </div>
        `;

        item.addEventListener('click', (e) => {
            e.preventDefault();
            showExpressionPreview(expr);
        });

        expressionList.appendChild(item);
    });
}

// Display accessories in list
function displayAccessories(accessories) {
    if (!accessories || accessories.length === 0) {
        accessoryList.innerHTML = `
            <div class="text-center text-muted py-3">
                <i class="fas fa-info-circle"></i>
                <p class="small mb-0">No accessories added yet</p>
            </div>
        `;
        return;
    }

    accessoryList.innerHTML = '';

    accessories.forEach(acc => {
        const item = document.createElement('div');
        item.className = 'list-group-item';
        item.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <span class="small">${acc.name}</span>
                <span class="badge bg-success">✓</span>
            </div>
        `;
        accessoryList.appendChild(item);
    });
}

// Create expression toggle buttons
function createExpressionToggles(expressions) {
    if (!expressions || expressions.length === 0) {
        expressionToggleBar.style.display = 'none';
        return;
    }

    expressionToggleBar.style.display = 'block';
    expressionToggles.innerHTML = '';

    // Add "Original" button
    const originalBtn = createToggleButton('original', 'Original');
    expressionToggles.appendChild(originalBtn);

    // Add expression buttons
    expressions.forEach(expr => {
        const btn = createToggleButton(expr, capitalizeFirst(expr));
        expressionToggles.appendChild(btn);
    });
}

// Create single toggle button
function createToggleButton(expression, label) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn btn-outline-primary expression-toggle-btn';
    btn.textContent = label;
    btn.dataset.expression = expression;

    if (expression === 'original') {
        btn.classList.add('active');
    }

    btn.addEventListener('click', function() {
        // Remove active from all buttons
        document.querySelectorAll('.expression-toggle-btn').forEach(b => {
            b.classList.remove('active');
        });

        // Add active to this button
        this.classList.add('active');

        // Show expression
        if (expression === 'original') {
            showOriginalPreview();
        } else {
            showExpressionPreview(expression);
        }
    });

    return btn;
}

// Show expression preview
function showExpressionPreview(expressionName) {
    currentExpression = expressionName;
    previewImage.style.opacity = '0.5';

    const previewUrl = `/api/expression/preview/${modelId}/${expressionName}?t=${new Date().getTime()}`;

    // Preload image
    const img = new Image();
    img.onload = function() {
        previewImage.src = previewUrl;
        previewImage.style.opacity = '1';
    };
    img.onerror = function() {
        showToast('Failed to load expression preview', 'error');
        previewImage.style.opacity = '1';
    };
    img.src = previewUrl;
}

// Show original preview
function showOriginalPreview() {
    currentExpression = null;
    previewImage.style.opacity = '0.5';

    setTimeout(() => {
        previewImage.src = `/api/generation/preview/${modelId}?t=${new Date().getTime()}`;
        previewImage.style.opacity = '1';
    }, 100);
}

// Export all
function exportAll() {
    const downloadUrl = `/api/generation/download/${modelId}`;
    window.location.href = downloadUrl;
}

// Export for Live2D
function exportForLive2D() {
    showToast('Live2D export will be available soon!', 'info');

    // In the future, this would package the model in Live2D format
    // For now, just download the full package
    exportAll();
}

// Show loading overlay
function showLoading(show) {
    previewLoading.style.display = show ? 'flex' : 'none';
}

// Show toast notification
function showToast(message, type = 'info') {
    const toastEl = document.getElementById('statusToast');
    const toastBody = document.getElementById('toastBody');

    // Set message
    toastBody.textContent = message;

    // Set color based on type
    const headerClass = {
        'success': 'bg-success text-white',
        'error': 'bg-danger text-white',
        'warning': 'bg-warning',
        'info': 'bg-info text-white'
    };

    const header = toastEl.querySelector('.toast-header');
    header.className = 'toast-header ' + (headerClass[type] || '');

    // Show toast
    const toast = new bootstrap.Toast(toastEl);
    toast.show();
}

// Helper function
function capitalizeFirst(str) {
    return str.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

// Initialize
console.log('Expression editor initialized for model:', modelId);
