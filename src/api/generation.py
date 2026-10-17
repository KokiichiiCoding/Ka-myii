"""
API routes for model generation
"""
from flask import Blueprint, request, jsonify, send_file
from pathlib import Path
import logging
from typing import Dict, Any

from src.models.vtuber_model import GenerationRequest, VTuberModel
from src.pipeline.assembly_line import AssemblyLine
from src.pipeline.jobs import JobQueue
from src.utils.helpers import save_json, load_json
import config

logger = logging.getLogger(__name__)

# Create blueprint
generation_bp = Blueprint('generation', __name__, url_prefix='/api/generation')

# Global assembly line + job queue (initialized in app.py)
assembly_line: AssemblyLine = None
job_queue: JobQueue = None


def _run_job(gen_request: GenerationRequest, task_id: str) -> str:
    model = assembly_line.generate_model(gen_request, task_id=task_id)
    save_json(model.to_dict(), config.OUTPUT_DIR / model.id / "metadata.json")
    return model.id


def init_assembly_line(use_dummy: bool = False):
    """Initialize the assembly line"""
    global assembly_line, job_queue
    assembly_line = AssemblyLine(
        output_base_dir=config.OUTPUT_DIR,
        use_dummy_generators=use_dummy
    )
    job_queue = JobQueue(_run_job)
    logger.info("Assembly line initialized")


@generation_bp.route('/generate', methods=['POST'])
def generate_model():
    """
    Queue a new VTuber model; poll /jobs/<task_id> for progress and the finished model_id.

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

    Response (202):
    {
        "success": true,
        "task_id": "gen_1a2b3c4d",
        "job": {"id": ..., "state": "queued", "position": 0, ...}
    }
    """
    try:
        data = request.json

        if data.get('framing', 'full_body') not in ('full_body', 'upper_body'):
            return jsonify({'success': False, 'error': 'Choose full_body or upper_body framing'}), 400

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
            framing=data.get('framing', 'full_body'),
            include_rigging=data.get('include_rigging', False),
            custom_layers=data.get('custom_layers')
        )

        if not gen_request.prompt:
            return jsonify({
                'success': False,
                'error': 'Prompt is required'
            }), 400

        job = job_queue.submit(gen_request)
        return jsonify({
            'success': True,
            'task_id': job.id,
            'job': job_queue.describe(job),
            'message': 'Queued'
        }), 202

    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@generation_bp.route('/jobs', methods=['GET'])
def list_jobs():
    """Recent and active jobs, newest first."""
    jobs = sorted(job_queue.jobs(), key=lambda j: j.created, reverse=True)
    return jsonify({'success': True, 'jobs': [job_queue.describe(j) for j in jobs]})


@generation_bp.route('/jobs/<job_id>', methods=['GET'])
def get_job(job_id: str):
    job = job_queue.get(job_id)
    if job is None:
        return jsonify({'success': False, 'error': 'Job not found (the server may have restarted)'}), 404
    return jsonify({'success': True, 'job': job_queue.describe(job)})


@generation_bp.route('/jobs/<job_id>/cancel', methods=['POST'])
def cancel_job(job_id: str):
    job = job_queue.cancel(job_id)
    if job is None:
        return jsonify({'success': False, 'error': 'Job not found'}), 404
    return jsonify({'success': True, 'job': job_queue.describe(job)})


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
        model_dir = config.OUTPUT_DIR / model_id

        if not model_dir.exists():
            return jsonify({
                'success': False,
                'error': 'Model not found'
            }), 404

        return _download_zip(model_dir, f'vtuber_model_{model_id}.zip')

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
        checkpoints = model_scanner.list_checkpoints()
        # Prefer a checkpoint the user already has over the multi-GB auto-download default.
        local = [c for c in checkpoints if c.get('source') == 'local' and c.get('available', True)]
        return jsonify({
            'success': True,
            'checkpoints': checkpoints,
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
                'model': local[0]['id'] if local else igen['model_name'],
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
        if request.args.get('crop'):
            # Layers are full-canvas and mostly transparent; thumbnails want just the part.
            from io import BytesIO
            from PIL import Image

            img = Image.open(layer_path)
            bbox = img.getbbox()
            buf = BytesIO()
            (img.crop(bbox) if bbox else img).save(buf, format='PNG')
            buf.seek(0)
            return send_file(buf, mimetype='image/png')
        return send_file(layer_path, mimetype='image/png')
    except Exception as e:
        logger.error(f"Failed to get layer: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


def _download_zip(directory: Path, name: str):
    """Stream a temporary archive, closing/deleting it with the response.

    Using one file object avoids Windows locks and keeps big packages off RAM.
    """
    import tempfile
    import zipfile
    archive = tempfile.TemporaryFile(mode='w+b')
    try:
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as package:
            for file in directory.rglob('*'):
                if file.is_file():
                    package.write(file, file.relative_to(directory))
        archive.seek(0)
        return send_file(archive, mimetype='application/zip', as_attachment=True,
                         download_name=name)
    except Exception:
        archive.close()
        raise


def preview_model_dir(model_id: str):
    # A single model directory, including on Windows (drive/backslash aware).
    if not model_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in model_id):
        return None
    base = config.OUTPUT_DIR.resolve()
    candidate = (base / model_id).resolve()
    return candidate if candidate.parent == base and candidate.is_dir() else None


@generation_bp.route('/artifact/<model_id>/<kind>', methods=['GET'])
def get_artifact(model_id: str, kind: str):
    """Download a rig-ready artifact: 'psd', 'ora', 'flat' or 'package' (zip)."""
    try:
        model_dir = preview_model_dir(model_id)
        if not model_dir:
            return jsonify({'success': False, 'error': 'Model not found'}), 404

        if kind == 'preview_psd':
            from src.core.live_preview import ensure_preview_psd
            try:
                artifact_path = ensure_preview_psd(model_dir)
            except (FileNotFoundError, ValueError) as exc:
                return jsonify({'success': False, 'error': str(exc)}), 422
            return send_file(artifact_path, mimetype='image/vnd.adobe.photoshop',
                             download_name=f'{model_id}_preview.psd', conditional=True)

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
            # The manifest lives at model/<name>/, not directly under the output.
            manifest = next((model_dir / 'model').glob('*/kamyii_manifest.json'), None)
            target = manifest.parent.resolve() if manifest else model_dir
            if not target.is_relative_to(model_dir.resolve()):
                return jsonify({'success': False, 'error': 'Invalid package path'}), 404
            return _download_zip(target, f'{model_id}_live2d_package.zip')

        return jsonify({'success': False, 'error': f"Artifact '{kind}' not available"}), 404
    except Exception as e:
        logger.error(f"Failed to get artifact: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500
