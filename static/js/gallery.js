// Gallery Page JavaScript

const api = window.KamyiiAPI;
const utils = window.KamyiiUtils;

// UI Elements
const loadingGallery = document.getElementById('loadingGallery');
const emptyGallery = document.getElementById('emptyGallery');
const galleryGrid = document.getElementById('galleryGrid');
const searchInput = document.getElementById('searchInput');
const refreshBtn = document.getElementById('refreshBtn');

// Stats elements
const totalModelsSpan = document.getElementById('totalModels');
const completedModelsSpan = document.getElementById('completedModels');
const totalSizeSpan = document.getElementById('totalSize');
const systemStatusSpan = document.getElementById('systemStatus');

// State
let allModels = [];

// Load gallery on page load
document.addEventListener('DOMContentLoaded', function() {
    loadGallery();
    loadStats();
});

// Refresh button
refreshBtn.addEventListener('click', function() {
    loadGallery();
    loadStats();
});

// Search functionality
searchInput.addEventListener('input', function() {
    const query = this.value.toLowerCase();
    filterModels(query);
});

// Load gallery
async function loadGallery() {
    showLoading();

    try {
        const result = await api.listModels();

        if (result.success) {
            allModels = result.models || [];

            if (allModels.length === 0) {
                showEmpty();
            } else {
                displayModels(allModels);
            }
        } else {
            throw new Error(result.error || 'Failed to load models');
        }
    } catch (error) {
        console.error('Failed to load gallery:', error);
        alert('Failed to load gallery: ' + error);
        showEmpty();
    }
}

// Load stats
async function loadStats() {
    try {
        const result = await api.getStats();

        if (result.success) {
            const stats = result.stats;

            totalModelsSpan.textContent = stats.total_models || 0;
            completedModelsSpan.textContent = stats.status_distribution?.completed || 0;
            totalSizeSpan.textContent = utils.formatFileSize(stats.total_size_mb || 0);
            systemStatusSpan.innerHTML = '<span class="badge bg-success">Online</span>';
        }
    } catch (error) {
        console.error('Failed to load stats:', error);
        systemStatusSpan.innerHTML = '<span class="badge bg-danger">Error</span>';
    }
}

// Show loading state
function showLoading() {
    loadingGallery.style.display = 'block';
    emptyGallery.style.display = 'none';
    galleryGrid.style.display = 'none';
}

// Show empty state
function showEmpty() {
    loadingGallery.style.display = 'none';
    emptyGallery.style.display = 'block';
    galleryGrid.style.display = 'none';
}

// Display models
function displayModels(models) {
    loadingGallery.style.display = 'none';
    emptyGallery.style.display = 'none';
    galleryGrid.style.display = 'flex';

    galleryGrid.innerHTML = '';

    models.forEach(function(model) {
        const card = createModelCard(model);
        galleryGrid.appendChild(card);
    });
}

// Create model card
function createModelCard(model) {
    const col = document.createElement('div');
    col.className = 'col-md-4 fade-in';

    const previewUrl = api.getPreviewUrl(model.id);
    const downloadUrl = api.getDownloadUrl(model.id);

    col.innerHTML = `
        <div class="card h-100">
            <img src="${previewUrl}" class="card-img-top" alt="Model Preview"
                 style="height: 250px; object-fit: cover;"
                 onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 width=%27100%27 height=%27100%27%3E%3Crect fill=%27%23ddd%27 width=%27100%27 height=%27100%27/%3E%3Ctext fill=%27%23999%27 x=%2750%25%27 y=%2750%25%27 text-anchor=%27middle%27 dy=%27.3em%27%3ENo Preview%3C/text%3E%3C/svg%3E'">
            <div class="card-body">
                <h6 class="card-title">${model.name || 'Model ' + model.id.substring(0, 8)}</h6>
                <p class="card-text">
                    <small class="text-muted">
                        <i class="fas fa-clock"></i> ${new Date(model.created_at).toLocaleDateString()}
                    </small>
                </p>
                <p class="card-text">
                    <span class="badge ${utils.getStatusBadge(model.status)}">
                        ${utils.getStatusName(model.status)}
                    </span>
                </p>
                <p class="card-text">
                    <small>
                        <i class="fas fa-layer-group"></i> ${model.assets?.length || 0} assets
                        <br>
                        <i class="fas fa-stopwatch"></i> ${utils.formatDuration(model.generation_time)}
                    </small>
                </p>
            </div>
            <div class="card-footer bg-white border-0">
                <div class="btn-group w-100" role="group">
                    <button type="button" class="btn btn-sm btn-outline-primary" onclick="viewModel('${model.id}')">
                        <i class="fas fa-eye"></i> View
                    </button>
                    <a href="${downloadUrl}" class="btn btn-sm btn-outline-success">
                        <i class="fas fa-download"></i> Download
                    </a>
                    <button type="button" class="btn btn-sm btn-outline-danger" onclick="deleteModel('${model.id}')">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        </div>
    `;

    return col;
}

// Filter models
function filterModels(query) {
    if (!query) {
        displayModels(allModels);
        return;
    }

    const filtered = allModels.filter(function(model) {
        return (
            model.id.toLowerCase().includes(query) ||
            (model.name && model.name.toLowerCase().includes(query)) ||
            model.status.toLowerCase().includes(query)
        );
    });

    if (filtered.length === 0) {
        galleryGrid.innerHTML = `
            <div class="col-12 text-center py-5">
                <i class="fas fa-search fa-3x text-muted mb-3"></i>
                <p class="text-muted">No models found matching "${query}"</p>
            </div>
        `;
    } else {
        displayModels(filtered);
    }
}

// View model details
window.viewModel = async function(modelId) {
    try {
        const result = await api.getModelStatus(modelId);

        if (result.success) {
            const model = result.model;

            let details = `
Model Details
═════════════

ID: ${model.id}
Name: ${model.name || 'N/A'}
Status: ${utils.getStatusName(model.status)}
Created: ${new Date(model.created_at).toLocaleString()}
Generation Time: ${utils.formatDuration(model.generation_time)}

Assets: ${model.assets?.length || 0}
${model.assets?.map(a => `  • ${a.layer_type}`).join('\n') || ''}

Base Image: ${model.base_image_path ? '✓' : '✗'}
Final Model: ${model.final_model_path ? '✓' : '✗'}
            `.trim();

            alert(details);
        }
    } catch (error) {
        alert('Failed to load model details: ' + error);
    }
};

// Delete model
window.deleteModel = async function(modelId) {
    if (!confirm('Are you sure you want to delete this model? This cannot be undone.')) {
        return;
    }

    try {
        const result = await api.deleteModel(modelId);

        if (result.success) {
            alert('Model deleted successfully');
            loadGallery();
            loadStats();
        } else {
            throw new Error(result.error || 'Failed to delete model');
        }
    } catch (error) {
        alert('Failed to delete model: ' + error);
    }
};

// Initialize
console.log('Gallery page initialized');
