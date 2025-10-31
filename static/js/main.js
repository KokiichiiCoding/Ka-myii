// Ka-myii Main JavaScript

// API base URL
const API_BASE = '/api';

// Utility functions
const utils = {
    // Show toast notification
    showToast: function(message, type = 'info') {
        // Simple console log for now - can be enhanced with a toast library
        console.log(`[${type.toUpperCase()}] ${message}`);

        // You could integrate a toast library here like Toastify
        alert(`${type.toUpperCase()}: ${message}`);
    },

    // Format time duration
    formatDuration: function(seconds) {
        if (seconds < 60) {
            return `${seconds.toFixed(1)}s`;
        } else if (seconds < 3600) {
            const minutes = seconds / 60;
            return `${minutes.toFixed(1)}m`;
        } else {
            const hours = seconds / 3600;
            return `${hours.toFixed(1)}h`;
        }
    },

    // Format file size
    formatFileSize: function(mb) {
        if (mb < 1024) {
            return `${mb.toFixed(2)} MB`;
        } else {
            const gb = mb / 1024;
            return `${gb.toFixed(2)} GB`;
        }
    },

    // Format status badge
    getStatusBadge: function(status) {
        const badges = {
            'completed': 'bg-success',
            'pending': 'bg-warning',
            'image_generation': 'bg-info',
            'asset_separation': 'bg-info',
            'model_assembly': 'bg-info',
            'rigging': 'bg-info',
            'failed': 'bg-danger'
        };
        return badges[status] || 'bg-secondary';
    },

    // Get status display name
    getStatusName: function(status) {
        const names = {
            'completed': 'Completed',
            'pending': 'Pending',
            'image_generation': 'Generating Image',
            'asset_separation': 'Separating Assets',
            'model_assembly': 'Assembling Model',
            'rigging': 'Applying Rigging',
            'failed': 'Failed'
        };
        return names[status] || status;
    }
};

// API client
const api = {
    // Generate a model
    generateModel: async function(params) {
        try {
            const response = await axios.post(`${API_BASE}/generation/generate`, params);
            return response.data;
        } catch (error) {
            console.error('API Error:', error);
            throw error.response?.data?.error || 'Failed to generate model';
        }
    },

    // Get model status
    getModelStatus: async function(modelId) {
        try {
            const response = await axios.get(`${API_BASE}/generation/status/${modelId}`);
            return response.data;
        } catch (error) {
            console.error('API Error:', error);
            throw error.response?.data?.error || 'Failed to get model status';
        }
    },

    // List all models
    listModels: async function() {
        try {
            const response = await axios.get(`${API_BASE}/generation/list`);
            return response.data;
        } catch (error) {
            console.error('API Error:', error);
            throw error.response?.data?.error || 'Failed to list models';
        }
    },

    // Get model info
    getModelInfo: async function() {
        try {
            const response = await axios.get(`${API_BASE}/models/info`);
            return response.data;
        } catch (error) {
            console.error('API Error:', error);
            throw error.response?.data?.error || 'Failed to get model info';
        }
    },

    // Get stats
    getStats: async function() {
        try {
            const response = await axios.get(`${API_BASE}/models/stats`);
            return response.data;
        } catch (error) {
            console.error('API Error:', error);
            throw error.response?.data?.error || 'Failed to get stats';
        }
    },

    // Delete model
    deleteModel: async function(modelId) {
        try {
            const response = await axios.delete(`${API_BASE}/generation/delete/${modelId}`);
            return response.data;
        } catch (error) {
            console.error('API Error:', error);
            throw error.response?.data?.error || 'Failed to delete model';
        }
    },

    // Get preview image URL
    getPreviewUrl: function(modelId) {
        return `${API_BASE}/generation/preview/${modelId}`;
    },

    // Get download URL
    getDownloadUrl: function(modelId) {
        return `${API_BASE}/generation/download/${modelId}`;
    }
};

// Export for use in other scripts
window.KamyiiAPI = api;
window.KamyiiUtils = utils;

// Check health on load
document.addEventListener('DOMContentLoaded', async function() {
    try {
        const response = await axios.get('/health');
        console.log('Ka-myii Status:', response.data);
    } catch (error) {
        console.error('Health check failed:', error);
    }
});
