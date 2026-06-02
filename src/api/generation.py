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

        # Create generation request
        image_settings = config.IMAGE_GENERATION

        gen_request = GenerationRequest(
            prompt=data.get('prompt', ''),
            negative_prompt=data.get('negative_prompt', ''),
            style=data.get('style', 'vtuber'),
            width=int(data.get('width', image_settings.get('default_width', 832))),
            height=int(data.get('height', image_settings.get('default_height', 1216))),
            steps=int(data.get('steps', image_settings.get('default_steps', 28))),
            guidance_scale=float(data.get('guidance_scale', image_settings.get('default_guidance_scale', 6.5))),
            seed=data.get('seed'),
            model_id=data.get('model_id'),
            sampler=data.get('sampler', image_settings.get('default_sampler')),
            clip_skip=data.get('clip_skip'),
            loras=data.get('loras'),
            batch_size=int(data.get('batch_size', 1)),
            include_rigging=data.get('include_rigging', False),
            custom_layers=data.get('custom_layers')
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


@generation_bp.route('/options', methods=['GET'])
def get_options():
    """Return everything needed to populate the generation UI controls:
    checkpoints, LoRAs, VAEs, samplers, styles and defaults."""
    try:
        from src.utils import model_scanner

        igen = config.IMAGE_GENERATION
        return jsonify({
            'success': True,
            'checkpoints': model_scanner.list_checkpoints(),
            'loras': model_scanner.list_loras(),
            'vaes': model_scanner.list_vaes(),
            'samplers': model_scanner.list_samplers(),
            'styles': list(config.STYLE_PRESETS.keys()),
            'defaults': {
                'width': igen['default_width'],
                'height': igen['default_height'],
                'steps': igen['default_steps'],
                'guidance_scale': igen['default_guidance_scale'],
                'sampler': igen['default_sampler'],
                'model': igen['model_name'],
                'negative_prompt': config.DEFAULT_NEGATIVE_PROMPT,
                'clip_skip': igen.get('clip_skip', 2),
            },
            'resolutions': [
                {'label': 'Portrait 832×1216 (SDXL)', 'width': 832, 'height': 1216},
                {'label': 'Portrait 768×1152', 'width': 768, 'height': 1152},
                {'label': 'Square 1024×1024', 'width': 1024, 'height': 1024},
                {'label': 'Tall 896×1152', 'width': 896, 'height': 1152},
                {'label': 'Fast 512×768', 'width': 512, 'height': 768},
            ],
        })
    except Exception as e:
        logger.error(f"Failed to get options: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@generation_bp.route('/layers/<model_id>', methods=['GET'])
def list_layers(model_id: str):
    """List the decomposed layer images for a model (for the Layer Studio)."""
    try:
        assets_dir = config.OUTPUT_DIR / model_id / "assets"
        if not assets_dir.exists():
            return jsonify({'success': False, 'error': 'No layers found'}), 404
        layers = []
        for layer_file in sorted(assets_dir.glob("*.png")):
            if layer_file.stem == "character":
                continue
            # filenames are like "09_eye_L.png"
            parts = layer_file.stem.split("_", 1)
            name = parts[1] if len(parts) == 2 and parts[0].isdigit() else layer_file.stem
            layers.append({
                'name': name,
                'file': layer_file.name,
                'url': f'/api/generation/layer/{model_id}/{layer_file.name}',
            })
        return jsonify({'success': True, 'count': len(layers), 'layers': layers})
    except Exception as e:
        logger.error(f"Failed to list layers: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@generation_bp.route('/layer/<model_id>/<path:filename>', methods=['GET'])
def get_layer(model_id: str, filename: str):
    """Serve a single decomposed layer PNG."""
    try:
        # Prevent path traversal.
        safe_name = Path(filename).name
        layer_path = config.OUTPUT_DIR / model_id / "assets" / safe_name
        if not layer_path.exists():
            return jsonify({'success': False, 'error': 'Layer not found'}), 404
        return send_file(layer_path, mimetype='image/png')
    except Exception as e:
        logger.error(f"Failed to get layer: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@generation_bp.route('/artifact/<model_id>/<kind>', methods=['GET'])
def get_artifact(model_id: str, kind: str):
    """Download a rig-ready artifact: 'psd', 'ora', 'flat' or 'package' (zip)."""
    try:
        model_dir = config.OUTPUT_DIR / model_id
        if not model_dir.exists():
            return jsonify({'success': False, 'error': 'Model not found'}), 404

        metadata_path = model_dir / "metadata.json"
        artifacts = {}
        if metadata_path.exists():
            artifacts = load_json(metadata_path).get('metadata', {}).get('artifacts', {})

        if kind in ('psd', 'ora', 'flat') and kind in artifacts:
            artifact_path = Path(artifacts[kind])
            if artifact_path.exists():
                return send_file(artifact_path, as_attachment=True,
                                 download_name=artifact_path.name)

        if kind == 'package':
            # Zip the assembled Cubism project folder.
            import tempfile, zipfile
            model_subdir = next((p for p in model_dir.iterdir()
                                 if p.is_dir() and (p / 'kamyii_manifest.json').exists()), None)
            target = model_subdir or model_dir
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            with zipfile.ZipFile(tmp.name, 'w', zipfile.ZIP_DEFLATED) as zf:
                for fp in target.rglob('*'):
                    if fp.is_file():
                        zf.write(fp, fp.relative_to(target))
            return send_file(tmp.name, as_attachment=True,
                             download_name=f'{model_id}_live2d_package.zip',
                             mimetype='application/zip')

        return jsonify({'success': False, 'error': f"Artifact '{kind}' not available"}), 404
    except Exception as e:
        logger.error(f"Failed to get artifact: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500
