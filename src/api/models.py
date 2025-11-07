"""
API routes for model management
"""
from flask import Blueprint, request, jsonify
import logging

from src.utils.helpers import load_json
import config

logger = logging.getLogger(__name__)

# Create blueprint
models_bp = Blueprint('models', __name__, url_prefix='/api/models')


@models_bp.route('/info', methods=['GET'])
def get_info():
    """
    Get information about available model types and configurations

    Response:
    {
        "success": true,
        "info": {...}
    }
    """
    try:
        info = {
            "supported_styles": [
                "anime",
                "live2d",
                "vtuber",
                "realistic"
            ],
            "default_settings": config.IMAGE_GENERATION,
            "asset_layers": config.ASSET_SEPARATION["layers"],
            "pipeline_profiles": config.PIPELINE_PROFILES,
            "default_expressions": config.DEFAULT_EXPRESSIONS,
            "default_accessories": config.DEFAULT_ACCESSORIES,
            "max_concurrent": config.API["max_concurrent_generations"],
            "features": {
                "image_generation": True,
                "asset_separation": True,
                "model_assembly": True,
                "auto_rigging": False  # Not implemented yet
            }
        }

        return jsonify({
            'success': True,
            'info': info
        })

    except Exception as e:
        logger.error(f"Failed to get info: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@models_bp.route('/stats', methods=['GET'])
def get_stats():
    """
    Get statistics about generated models

    Response:
    {
        "success": true,
        "stats": {...}
    }
    """
    try:
        total_models = 0
        total_size_mb = 0
        status_counts = {}

        # Scan output directory
        for model_dir in config.OUTPUT_DIR.iterdir():
            if model_dir.is_dir():
                metadata_path = model_dir / "metadata.json"
                if metadata_path.exists():
                    total_models += 1

                    # Load metadata
                    model_data = load_json(metadata_path)
                    status = model_data.get('status', 'unknown')
                    status_counts[status] = status_counts.get(status, 0) + 1

                    # Calculate size
                    for file_path in model_dir.rglob('*'):
                        if file_path.is_file():
                            total_size_mb += file_path.stat().st_size / (1024 * 1024)

        stats = {
            "total_models": total_models,
            "total_size_mb": round(total_size_mb, 2),
            "status_distribution": status_counts,
            "output_directory": str(config.OUTPUT_DIR)
        }

        return jsonify({
            'success': True,
            'stats': stats
        })

    except Exception as e:
        logger.error(f"Failed to get stats: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
