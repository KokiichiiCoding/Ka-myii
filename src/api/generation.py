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
    # Real runs take ~11 min each, so a queued batch is kept across restarts (not in demo mode,
    # which must never pick up real jobs).
    job_queue = JobQueue(_run_job, state_path=None if use_dummy else config.OUTPUT_DIR / "_queue.json")
    logger.info("Assembly line initialized")


MAX_PROMPT_CHARS = 5000


def _number(data, key, default, kind, low, high):
    value = data.get(key)
    if value is None or value == '':
        return default
    try:
        number = kind(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be a number")
    if not low <= number <= high:
        raise ValueError(f"'{key}' must be between {low} and {high}")
    return number


def _parse_request(data) -> GenerationRequest:
    """Validate a /generate body. Raises ValueError with a message for the user."""
    from src.utils import model_scanner

    if not isinstance(data, dict):
        raise ValueError('Send the brief as a JSON object')
    prompt = str(data.get('prompt') or '').strip()
    negative = str(data.get('negative_prompt') or '')
    if not prompt:
        raise ValueError('Prompt is required')
    if max(len(prompt), len(negative)) > MAX_PROMPT_CHARS:
        # Every 75 tokens add a text-encoder pass; ~60 chunks is far beyond any real brief.
        raise ValueError(f'Keep prompts under {MAX_PROMPT_CHARS} characters')
    framing = data.get('framing') or 'full_body'
    if framing not in ('full_body', 'upper_body'):
        raise ValueError('Choose full_body or upper_body framing')
    style = data.get('style') or 'vtuber'
    if style not in config.STYLE_PRESETS:
        raise ValueError(f"Unknown style '{style}'")
    defaults = config.IMAGE_GENERATION
    sampler = data.get('sampler') or defaults.get('default_sampler')
    if sampler not in config.SAMPLERS:
        raise ValueError(f"Unknown sampler '{sampler}'")

    # Only checkpoints/LoRAs the Studio lists: a free-form path would load any file on disk.
    model_id = data.get('model_id') or None
    if model_id and model_id not in {c['id'] for c in model_scanner.list_checkpoints() if c.get('available', True)}:
        raise ValueError('Unknown checkpoint; pick one from the list (or drop it into models/checkpoints)')
    loras = data.get('loras') or None
    if loras is not None:
        known = {entry['id'] for entry in model_scanner.list_loras()}
        if not isinstance(loras, list) or not all(isinstance(l, dict) and l.get('id') in known for l in loras):
            raise ValueError('Unknown LoRA; pick one from the list (or drop it into models/loras)')
        loras = [{'id': l['id'], 'weight': _number(l, 'weight', 1.0, float, -4.0, 4.0)} for l in loras]

    # Generous bounds; the point is to refuse sizes that would only exhaust the GPU.
    w = _number(data, 'width', defaults.get('default_width', 832), int, 256, 2048)
    h = _number(data, 'height', defaults.get('default_height', 1216), int, 256, 2048)
    return GenerationRequest(
        prompt=prompt,
        negative_prompt=negative,
        style=style,
        width=w - w % 8,
        height=h - h % 8,
        steps=_number(data, 'steps', defaults.get('default_steps', 28), int, 1, 150),
        guidance_scale=_number(data, 'guidance_scale', defaults.get('default_guidance_scale', 6.5), float, 0.0, 30.0),
        seed=_number(data, 'seed', None, int, 0, 2**32 - 1),
        model_id=model_id,
        sampler=sampler,
        clip_skip=_number(data, 'clip_skip', None, int, 1, 12),
        loras=loras,
        framing=framing,
        sketch=bool(data.get('sketch')),
    )


@generation_bp.route('/generate', methods=['POST'])
def generate_model():
    """
    Queue a new VTuber model; poll /jobs/<task_id> for progress and the finished model_id.

    Request JSON (only "prompt" is required):
    {
        "prompt": "1girl, silver hair, witch hat",
        "negative_prompt": "lowres",
        "style": "vtuber",                  # a key of config.STYLE_PRESETS
        "framing": "upper_body",            # or "full_body"
        "model_id": "<id from /options>",   # checkpoint; default from config
        "sampler": "DPM++ 2M Karras",
        "width": 832, "height": 1216, "steps": 28, "guidance_scale": 6.5,
        "clip_skip": 2, "seed": null,
        "loras": [{"id": "<id from /options>", "weight": 0.8}]
    }

    Response (202):
    {
        "success": true,
        "task_id": "<job id>",
        "job": {"id": ..., "state": "queued", "position": 0, ...}
    }
    """
    try:
        gen_request = _parse_request(request.get_json(silent=True))
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400

    job = job_queue.submit(gen_request)
    return jsonify({
        'success': True,
        'task_id': job.id,
        'job': job_queue.describe(job),
        'message': 'Queued'
    }), 202


@generation_bp.route('/upload', methods=['POST'])
def upload_artwork():
    """Queue the decomposition of the user's own character art (multipart field "image").

    Same response as /generate; the image stage imports the file instead of running SDXL.
    """
    from PIL import Image, UnidentifiedImageError

    upload = request.files.get('image')
    if not upload or not upload.filename:
        return jsonify({'success': False, 'error': 'Choose an image to upload'}), 400
    try:
        with Image.open(upload.stream) as img:
            img.load()
            # Re-encoding as PNG keeps only pixels; transparency is flattened onto white like a sketch.
            if img.mode in ('RGBA', 'LA', 'PA') or 'transparency' in img.info:
                rgba = img.convert('RGBA')
                image = Image.new('RGB', rgba.size, 'white')
                image.paste(rgba, mask=rgba.getchannel('A'))
            else:
                image = img.convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return jsonify({'success': False, 'error': 'That file is not an image Ka-myii can read (PNG, JPG or WebP)'}), 400
    if min(image.size) < 256:
        return jsonify({'success': False, 'error': 'The image is too small; use at least 256 px on each side'}), 400
    image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)  # See-through works at 1024-1280 anyway
    title = Path(upload.filename).stem.strip()[:60] or 'artwork'
    return _queue_artwork(image, {'prompt': f'Uploaded artwork: {title}', 'style': 'anime',
                                  'framing': 'upper_body', 'source': 'upload'})


@generation_bp.route('/retry/<model_id>', methods=['POST'])
def retry_model(model_id: str):
    """Separate the layers of a saved result's artwork as a new character: a sketch, a failed run,
    or a finished one to redo. The new character keeps the original brief, seed and checkpoint."""
    from dataclasses import fields
    from PIL import Image

    model_dir = model_dir_for(model_id)
    base = model_dir / 'base_image.png' if model_dir else None
    if not base or not base.is_file():
        return jsonify({'success': False, 'error': 'Model not found'}), 404
    metadata_path = model_dir / 'metadata.json'
    saved = (load_json(metadata_path).get('request') or {}) if metadata_path.exists() else {}
    known = {f.name for f in fields(GenerationRequest)}
    brief = {k: v for k, v in saved.items() if k in known}
    brief['prompt'] = brief.get('prompt') or f'Artwork of {model_id[:8]}'
    brief['source'] = 'upload' if brief.get('source') == 'upload' else 'artwork'
    with Image.open(base) as img:
        image = img.convert('RGB')
    return _queue_artwork(image, brief)


def _queue_artwork(image, brief: dict):
    """Queue a job that separates ``image`` as is (its image stage imports instead of generating)."""
    import uuid

    uploads = config.OUTPUT_DIR / '_uploads'
    uploads.mkdir(parents=True, exist_ok=True)
    path = uploads / f'{uuid.uuid4().hex}.png'
    image.save(path)
    job = job_queue.submit(GenerationRequest(**{**brief, 'width': image.width, 'height': image.height,
                                                'reference_image': str(path), 'sketch': False}))
    return jsonify({'success': True, 'task_id': job.id, 'job': job_queue.describe(job), 'message': 'Queued'}), 202


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
        model_dir = model_dir_for(model_id)
        model_metadata_path = model_dir / "metadata.json" if model_dir else None

        if not model_metadata_path or not model_metadata_path.exists():
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
        model_dir = model_dir_for(model_id)

        if not model_dir:
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
        model_dir = model_dir_for(model_id)
        preview_path = model_dir / "preview.png" if model_dir else None

        if preview_path and not preview_path.exists():
            # Try base image as fallback
            preview_path = model_dir / "base_image.png"

        if not preview_path or not preview_path.exists():
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

        model_dir = model_dir_for(model_id)

        if not model_dir:
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


@generation_bp.route('/rename/<model_id>', methods=['POST'])
def rename_model(model_id: str):
    """Set (or with an empty name, clear) a character's display name. Files keep their names."""
    model_dir = model_dir_for(model_id)
    metadata_path = model_dir / 'metadata.json' if model_dir else None
    if not metadata_path or not metadata_path.exists():
        return jsonify({'success': False, 'error': 'Model not found'}), 404
    data = request.get_json(silent=True) or {}
    name = ' '.join(''.join(ch for ch in str(data.get('name') or '') if ch.isprintable()).split())[:60]
    metadata = load_json(metadata_path)
    if name:
        metadata['display_name'] = name
    else:
        metadata.pop('display_name', None)
    save_json(metadata, metadata_path)
    return jsonify({'success': True, 'display_name': name or None})


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
        from src.core import seethrough
        model_loaded = assembly_line.image_generator.pipeline is not None

        return jsonify({
            'success': True,
            'model_loaded': model_loaded,
            'device': assembly_line.image_generator.device,
            'model_name': assembly_line.image_generator.model_name,
            'is_dummy': False,
            'seethrough': seethrough.available(),
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
                'config_model': igen['model_name'],  # what an empty model_id means
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
        model_dir = model_dir_for(model_id)
        assets_dir = model_dir / "assets" if model_dir else None
        if not assets_dir or not assets_dir.exists():
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
        model_dir = model_dir_for(model_id)
        # Prevent path traversal.
        layer_path = model_dir / "assets" / Path(filename).name if model_dir else None
        if not layer_path or not layer_path.is_file():
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


def model_dir_for(model_id: str):
    """The output folder for a model id from a URL, or None. Never escapes OUTPUT_DIR."""
    if not model_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in model_id):
        return None
    base = config.OUTPUT_DIR.resolve()
    candidate = (base / model_id).resolve()
    return candidate if candidate.parent == base and candidate.is_dir() else None


@generation_bp.route('/artifact/<model_id>/<kind>', methods=['GET'])
def get_artifact(model_id: str, kind: str):
    """Download a rig-ready artifact: 'psd', 'ora', 'flat' or 'package' (zip)."""
    try:
        model_dir = model_dir_for(model_id)
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
