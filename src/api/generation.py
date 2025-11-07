"""
API routes for model generation
"""
from flask import Blueprint, request, jsonify, send_file
from pathlib import Path
import logging
import uuid
from typing import Dict, Any

from src.models.vtuber_model import GenerationRequest, VTuberModel
from src.pipeline.assembly_line import AssemblyLine
from src.utils.helpers import save_json, load_json
import config

logger = logging.getLogger(__name__)

# Create blueprint
generation_bp = Blueprint('generation', __name__, url_prefix='/api/generation')

# Global assembly line instance (will be initialized in app.py)
assembly_line: AssemblyLine = None


def init_assembly_line(use_dummy: bool = False):
    """Initialize the assembly line"""
    global assembly_line
    assembly_line = AssemblyLine(
        output_base_dir=config.OUTPUT_DIR,
        use_dummy_generators=use_dummy
    )
    logger.info("Assembly line initialized")


@generation_bp.route('/generate', methods=['POST'])
def generate_model():
    """
    Generate a new VTuber model

    Request JSON:
    {
        "prompt": "anime girl with blue hair",
        "negative_prompt": "low quality",
        "style": "anime",
        "width": 512,
        "height": 512,
        "steps": 30,
        "guidance_scale": 7.5,
        "seed": null,
        "include_rigging": false,
        "custom_layers": null
    }

    Response:
    {
        "success": true,
        "model_id": "uuid",
        "message": "Model generation started"
    }
    """
    try:
        data = request.json

        # Normalise list inputs
        def _normalise_list(value):
            if value is None:
                return None
            if isinstance(value, str):
                return [item.strip() for item in value.split(',') if item.strip()]
            if isinstance(value, list):
                cleaned = []
                for item in value:
                    if isinstance(item, str):
                        stripped = item.strip()
                        if stripped:
                            cleaned.append(stripped)
                return cleaned or None
            return None

        expressions = _normalise_list(data.get('expressions'))
        accessories = _normalise_list(data.get('accessories'))

        # Create generation request
        gen_request = GenerationRequest(
            prompt=data.get('prompt', ''),
            negative_prompt=data.get('negative_prompt', ''),
            style=data.get('style', 'anime'),
            width=data.get('width', 512),
            height=data.get('height', 512),
            steps=data.get('steps', 30),
            guidance_scale=data.get('guidance_scale', 7.5),
            seed=data.get('seed'),
            include_rigging=data.get('include_rigging', False),
            custom_layers=data.get('custom_layers'),
            pipeline_profile=data.get('pipeline_profile', 'standard'),
            segmentation_mode=data.get('segmentation_mode', 'auto'),
            use_controlnet=data.get('use_controlnet'),
            generate_expressions=data.get('generate_expressions'),
            expression_list=expressions,
            expression_variations=int(data.get('expression_variations', 0) or 0),
            generate_accessories=data.get('generate_accessories'),
            accessory_list=accessories,
            enable_auto_physics=data.get('enable_auto_physics'),
        )

        if not gen_request.prompt:
            return jsonify({
                'success': False,
                'error': 'Prompt is required'
            }), 400

        logger.info(f"Generation request received: {gen_request.prompt}")

        # Generate task ID for progress tracking
        task_id = f"gen_{uuid.uuid4().hex[:8]}"
        logger.info(f"Created task: {task_id}")

        # Generate model with task ID for progress tracking
        model = assembly_line.generate_model(gen_request, task_id=task_id)

        # Save model metadata
        model_metadata_path = config.OUTPUT_DIR / model.id / "metadata.json"
        save_json(model.to_dict(), model_metadata_path)

        return jsonify({
            'success': True,
            'model_id': model.id,
            'task_id': task_id,
            'model': model.to_dict(),
            'message': 'Model generated successfully'
        })

    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@generation_bp.route('/status/<model_id>', methods=['GET'])
def get_status(model_id: str):
    """
    Get status of a model generation

    Response:
    {
        "success": true,
        "model": {...}
    }
    """
    try:
        model_metadata_path = config.OUTPUT_DIR / model_id / "metadata.json"

        if not model_metadata_path.exists():
            return jsonify({
                'success': False,
                'error': 'Model not found'
            }), 404

        model_data = load_json(model_metadata_path)

        return jsonify({
            'success': True,
            'model': model_data
        })

    except Exception as e:
        logger.error(f"Failed to get status: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@generation_bp.route('/list', methods=['GET'])
def list_models():
    """
    List all generated models

    Response:
    {
        "success": true,
        "models": [...]
    }
    """
    try:
        models = []

        # Scan output directory for models
        for model_dir in config.OUTPUT_DIR.iterdir():
            if model_dir.is_dir():
                metadata_path = model_dir / "metadata.json"
                if metadata_path.exists():
                    model_data = load_json(metadata_path)
                    models.append(model_data)

        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        return jsonify({
            'success': True,
            'count': len(models),
            'models': models
        })

    except Exception as e:
        logger.error(f"Failed to list models: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@generation_bp.route('/download/<model_id>', methods=['GET'])
def download_model(model_id: str):
    """Download a generated model as a zip file"""
    try:
        import zipfile
        import tempfile

        model_dir = config.OUTPUT_DIR / model_id

        if not model_dir.exists():
            return jsonify({
                'success': False,
                'error': 'Model not found'
            }), 404

        # Create temporary zip file
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')

        with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in model_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(model_dir)
                    zipf.write(file_path, arcname)

        return send_file(
            temp_zip.name,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'vtuber_model_{model_id}.zip'
        )

    except Exception as e:
        logger.error(f"Failed to download model: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@generation_bp.route('/preview/<model_id>', methods=['GET'])
def get_preview(model_id: str):
    """Get preview image for a model"""
    try:
        preview_path = config.OUTPUT_DIR / model_id / "preview.png"

        if not preview_path.exists():
            # Try base image as fallback
            preview_path = config.OUTPUT_DIR / model_id / "base_image.png"

        if not preview_path.exists():
            return jsonify({
                'success': False,
                'error': 'Preview not found'
            }), 404

        return send_file(preview_path, mimetype='image/png')

    except Exception as e:
        logger.error(f"Failed to get preview: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@generation_bp.route('/delete/<model_id>', methods=['DELETE'])
def delete_model(model_id: str):
    """Delete a generated model"""
    try:
        import shutil

        model_dir = config.OUTPUT_DIR / model_id

        if not model_dir.exists():
            return jsonify({
                'success': False,
                'error': 'Model not found'
            }), 404

        shutil.rmtree(model_dir)

        return jsonify({
            'success': True,
            'message': 'Model deleted successfully'
        })

    except Exception as e:
        logger.error(f"Failed to delete model: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@generation_bp.route('/model-status', methods=['GET'])
def get_model_status():
    """
    Check if the AI model is loaded and ready

    Response:
    {
        "success": true,
        "model_loaded": true/false,
        "device": "cuda"/"cpu",
        "model_name": "...",
        "is_dummy": false
    }
    """
    try:
        if assembly_line is None:
            return jsonify({
                'success': False,
                'error': 'Assembly line not initialized'
            }), 500

        # Check if it's a dummy generator
        from src.core.image_generator import DummyImageGenerator
        is_dummy = isinstance(assembly_line.image_generator, DummyImageGenerator)

        if is_dummy:
            return jsonify({
                'success': True,
                'model_loaded': True,
                'device': 'dummy',
                'model_name': 'Dummy Generator (Testing Mode)',
                'is_dummy': True,
                'message': 'Running in dummy mode - no actual AI generation'
            })

        # Check real model
        model_loaded = assembly_line.image_generator.pipeline is not None

        return jsonify({
            'success': True,
            'model_loaded': model_loaded,
            'device': assembly_line.image_generator.device,
            'model_name': assembly_line.image_generator.model_name,
            'is_dummy': False,
            'message': 'Model loaded and ready' if model_loaded else 'Model not loaded yet - will load on first generation'
        })

    except Exception as e:
        logger.error(f"Failed to get model status: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@generation_bp.route('/preload-model', methods=['POST'])
def preload_model():
    """
    Preload the AI model before generation
    This downloads and loads the Stable Diffusion model
    Can take 10-30 minutes on first run (downloads ~4-5GB)

    Response:
    {
        "success": true,
        "message": "Model loaded successfully"
    }
    """
    try:
        if assembly_line is None:
            return jsonify({
                'success': False,
                'error': 'Assembly line not initialized'
            }), 500

        # Check if dummy mode
        from src.core.image_generator import DummyImageGenerator
        if isinstance(assembly_line.image_generator, DummyImageGenerator):
            return jsonify({
                'success': True,
                'message': 'Running in dummy mode - no model to load',
                'is_dummy': True
            })

        logger.info("Preloading AI model...")

        # This will download the model if not cached
        assembly_line.image_generator.load_model()

        logger.info("Model preloaded successfully")

        return jsonify({
            'success': True,
            'message': 'Model loaded successfully',
            'device': assembly_line.image_generator.device,
            'model_name': assembly_line.image_generator.model_name
        })

    except Exception as e:
        logger.error(f"Failed to preload model: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Model loading failed. Check logs for details.'
        }), 500
