"""
API routes for model generation
"""
from flask import Blueprint, request, jsonify, send_file
from pathlib import Path
import json
import logging
from functools import lru_cache
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
    if gen_request.operation == "expressions":
        return assembly_line.add_expressions(gen_request.target_model, task_id=task_id, toggles=gen_request.toggles,
                                             visemes=gen_request.visemes, variant=gen_request.variant)
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

    toggles = _toggles(data)
    visemes = bool(toggles and VISEMES_ID in toggles)
    toggles = [t for t in toggles or [] if t != VISEMES_ID] or None
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
        toggles=toggles, visemes=visemes,
    )


# The Studio offers the vowel mouths as one more toggle chip; jobs carry them as ``visemes``.
VISEMES_ID = 'visemes'


def _toggles(data):
    """The "toggles" list (names from expressions.TOGGLES, or VISEMES_ID), or None. Raises ValueError."""
    from src.core.expressions import TOGGLES

    toggles = data.get('toggles') or None
    if toggles is not None and (not isinstance(toggles, list) or any(t not in TOGGLES and t != VISEMES_ID for t in toggles)):
        raise ValueError(f"Unknown expression toggle; choose from {', '.join(TOGGLES)}")
    return list(dict.fromkeys(toggles)) if toggles else None


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
    if request.form.get('crop'):
        # "x0,y0,x1,y1" as fractions: the character the user boxed on a sheet or group picture.
        try:
            x0, y0, x1, y1 = (float(v) for v in request.form['crop'].split(','))
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError
        except ValueError:
            return jsonify({'success': False, 'error': "'crop' must be four fractions x0,y0,x1,y1"}), 400
        w, h = image.size
        image = image.crop((round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)))
    if min(image.size) < 256:
        return jsonify({'success': False, 'error': 'The image is too small; use at least 256 px on each side'}), 400
    image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)  # See-through works at 1024-1280 anyway
    title = Path(upload.filename).stem.strip()[:60] or 'artwork'
    form = request.form
    try:
        enhance = _number(form, 'enhance', 0.0, float, 0.0, 0.6)
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    options = {'frame': form.get('frame', '1') != '0', 'plain_background': form.get('plain_background', '1') != '0',
               'enhance': enhance}
    # A prepared sketch first (seconds; ~1 min with enhance): check the framing, edit or extend,
    # then separate the layers. "separate=1" goes straight on to the full ~11 minute run.
    return _queue_artwork(image, {'prompt': f'Uploaded artwork: {title}', 'style': 'anime',
                                  'framing': 'upper_body', 'source': 'upload'},
                          import_options=options, sketch=form.get('separate') != '1')


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


def _queue_artwork(image, brief: dict, import_options=None, sketch: bool = False):
    """Queue a job that separates ``image`` as is (its image stage imports instead of generating)."""
    import uuid

    uploads = config.OUTPUT_DIR / '_uploads'
    uploads.mkdir(parents=True, exist_ok=True)
    path = uploads / f'{uuid.uuid4().hex}.png'
    image.save(path)
    job = job_queue.submit(GenerationRequest(**{**brief, 'width': image.width, 'height': image.height,
                                                'reference_image': str(path), 'sketch': sketch,
                                                'operation': 'generate', 'mask_image': None, 'pose_image': None, 'edit_prompt': None,
                                                'import_options': import_options}))
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


def _saved_brief(model_dir: Path) -> dict:
    """A saved character's request fields that GenerationRequest accepts."""
    from dataclasses import fields

    metadata_path = model_dir / 'metadata.json'
    saved = (load_json(metadata_path).get('request') or {}) if metadata_path.exists() else {}
    known = {f.name for f in fields(GenerationRequest)}
    return {k: v for k, v in saved.items() if k in known and k not in ('reference_image','mask_image','pose_image')}


def _stage(image, name: str) -> str:
    import uuid

    uploads = config.OUTPUT_DIR / '_uploads'
    uploads.mkdir(parents=True, exist_ok=True)
    path = uploads / f'{uuid.uuid4().hex}-{name}.png'
    image.save(path)
    return str(path)


@generation_bp.route('/parts/<model_id>', methods=['GET'])
def list_parts(model_id: str):
    """Parts that can be selected for a region edit of this character."""
    from src.core import regions

    model_dir = model_dir_for(model_id)
    if not model_dir or not (model_dir / 'base_image.png').exists():
        return jsonify({'success': False, 'error': 'Model not found'}), 404
    return jsonify({'success': True, 'parts': [{'id': p, 'label': regions.LABELS[p]}
                                               for p in regions.available_parts(model_dir)]})


@generation_bp.route('/edit/<model_id>', methods=['POST'])
def edit_region(model_id: str):
    """Repaint part of a character's artwork into a new sketch.

    JSON: {"parts": ["hair", ...], "mask": "<PNG data URL painted in the Studio>",
           "prompt": "short blue hair", "strength": 0.85, "seed": null}
    The selection is the union of the parts and the painted mask. An empty prompt repaints
    with the character's own brief (a "retry" of that region).
    """
    from secrets import randbits
    from PIL import Image
    from src.core import regions

    model_dir = model_dir_for(model_id)
    base = model_dir / 'base_image.png' if model_dir else None
    if not base or not base.is_file():
        return jsonify({'success': False, 'error': 'Model not found'}), 404
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'success': False, 'error': 'Send the edit as a JSON object'}), 400
    with Image.open(base) as img:
        image = img.convert('RGB')
    try:
        parts = data.get('parts') or []
        if not isinstance(parts, list) or any(p not in regions.LABELS for p in parts):
            raise ValueError('Unknown part')
        mask = regions.part_mask(model_dir, parts, image.size) | regions.painted_mask(data.get('mask'), image.size)
        prompt = str(data.get('prompt') or '').strip()
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f'Keep prompts under {MAX_PROMPT_CHARS} characters')
        strength = _number(data, 'strength', 0.75, float, 0.3, 1.0)
        seed = _number(data, 'seed', None, int, 0, 2**32 - 1)
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    if not mask.any():
        return jsonify({'success': False, 'error': 'Select a part or paint over the area to change'}), 400

    brief = _saved_brief(model_dir)
    brief['prompt'] = brief.get('prompt') or f'Artwork of {model_id[:8]}'
    job = job_queue.submit(GenerationRequest(**{
        **brief, 'width': image.width, 'height': image.height, 'operation': 'edit', 'sketch': True,
        'source': 'upload' if brief.get('source') == 'upload' else 'artwork',
        'reference_image': _stage(image, 'source'), 'mask_image': _stage(regions.to_image(mask), 'mask'),
        'edit_prompt': prompt or brief['prompt'], 'strength': strength,
        'seed': seed if seed is not None else randbits(31)}))
    return jsonify({'success': True, 'task_id': job.id, 'job': job_queue.describe(job), 'message': 'Queued'}), 202


@lru_cache(maxsize=4)
def _character_mask(path: str, _mtime: float):
    """The segmentation is the slow part of a fit guide; dragging a guide re-plans often."""
    from PIL import Image
    from src.core import outpaint

    with Image.open(path) as img:
        return outpaint.character_mask(img.convert('RGB'))


def _extension_plan(model_dir: Path, mode: str, proportions: str, keep_face: bool, guides=None):
    from PIL import Image
    from src.core import outpaint

    base = model_dir / 'base_image.png'
    with Image.open(base) as img:
        image = img.convert('RGB')
    geometry = outpaint.anatomy(model_dir)
    return outpaint.plan(image, _character_mask(str(base), base.stat().st_mtime), mode,
                         geometry=geometry, proportions=proportions, keep_face=keep_face,
                         identity=outpaint.identity_mask(model_dir, geometry) if geometry and keep_face else None,
                         guides=outpaint.check_guides(guides, image.size))


@generation_bp.route('/extend/<model_id>/plan', methods=['GET'])
def extension_plan(model_id: str):
    """Read-only fit guide: measured source placement before spending a GPU generation."""
    from src.core import outpaint

    model_dir = model_dir_for(model_id)
    if not model_dir or not (model_dir / 'base_image.png').is_file():
        return jsonify({'success': False, 'error': 'Model not found'}), 404
    proportions = request.args.get('proportions', 'balanced')
    if proportions not in outpaint.PROPORTIONS:
        return jsonify({'success': False, 'error': 'Choose balanced, compact or tall proportions'}), 400
    try:
        guides = json.loads(request.args['guides']) if request.args.get('guides') else None
        _, _, info = _extension_plan(model_dir, 'full_body', proportions, True, guides)
    except ValueError as e:  # includes malformed JSON
        return jsonify({'success': False, 'error': str(e)}), 400
    return jsonify({'success': True, 'info': info})


@generation_bp.route('/extend/<model_id>', methods=['POST'])
def extend_artwork(model_id: str):
    """Paint the rest of a character cut off by the image border into a new sketch.

    JSON: {"mode": "full_body" | "uncrop", "prompt": "optional extra description", "seed": null,
           "separate": false} (true continues through layers and Live Preview)
    """
    from secrets import randbits
    from PIL import Image
    from src.core import outpaint

    model_dir = model_dir_for(model_id)
    base = model_dir / 'base_image.png' if model_dir else None
    if not base or not base.is_file():
        return jsonify({'success': False, 'error': 'Model not found'}), 404
    data = request.get_json(silent=True) or {}
    mode = data.get('mode') or 'full_body'
    if mode not in ('full_body', 'uncrop'):
        return jsonify({'success': False, 'error': 'Choose full_body or uncrop'}), 400
    try:
        extra = str(data.get('prompt') or '').strip()
        if len(extra) > MAX_PROMPT_CHARS:
            raise ValueError(f'Keep prompts under {MAX_PROMPT_CHARS} characters')
        seed = _number(data, 'seed', None, int, 0, 2**32 - 1)
        separate = data.get('separate', False)
        if not isinstance(separate, bool):
            raise ValueError('separate must be true or false')
        proportions = data.get('proportions', 'balanced')
        if not isinstance(proportions, str) or proportions not in outpaint.PROPORTIONS:
            raise ValueError('Choose balanced, compact or tall proportions')
        keep_face = data.get('keep_face', True)
        if not isinstance(keep_face, bool):
            raise ValueError('keep_face must be true or false')
        guided = data.get('guided', True)
        if not isinstance(guided, bool):
            raise ValueError('guided must be true or false')
        canvas, mask, info = _extension_plan(model_dir, mode, proportions, keep_face,
                                             data.get('guides') if mode == 'full_body' else None)
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    info['source_model'] = model_id
    info['guided'] = guided and mode == 'full_body'
    info['outfit'] = extra
    if mode == 'uncrop' and not info['touched']:
        return jsonify({'success': False, 'error': 'This character is not cut off by the border'}), 400

    brief = _saved_brief(model_dir)
    brief['prompt'] = brief.get('prompt') or f'Artwork of {model_id[:8]}'
    if mode == 'full_body':  # the new character's brief too, for its later edits and expressions
        brief['prompt'] = outpaint.full_body_prompt(', '.join(p for p in (brief['prompt'], extra) if p))
    tags = outpaint.FULL_BODY_TAGS if mode == 'full_body' else ''
    negative = outpaint.FULL_BODY_NEGATIVE if mode == 'full_body' else outpaint.UNCROP_NEGATIVE
    if mode == 'full_body' and proportions != 'compact':
        tags += ', balanced proportions, proportional head, long legs, symmetrical shoulders'
        negative += ', chibi, oversized head, short legs, twisted torso, uneven shoulders, long neck, floating head'
    job = job_queue.submit(GenerationRequest(**{
        **brief, 'width': canvas.width, 'height': canvas.height, 'operation': 'extend', 'sketch': not separate,
        'extension': info,
        'pose_image': _stage(outpaint.pose_image(info), 'pose') if info['guided'] else None,
        'source': 'upload' if brief.get('source') == 'upload' else 'artwork',
        'framing': 'full_body' if mode == 'full_body' else brief.get('framing', 'upper_body'),
        'reference_image': _stage(canvas, 'canvas'), 'mask_image': _stage(Image.fromarray(mask), 'mask'),
        'edit_prompt': ', '.join(p for p in (brief['prompt'] if mode == 'full_body' else extra or brief['prompt'], tags) if p),
        'negative_prompt': ', '.join(p for p in (brief.get('negative_prompt'), negative) if p),
        'strength': 0.9 if mode == 'full_body' else 1.0, 'seed': seed if seed is not None else randbits(31)}))
    return jsonify({'success': True, 'task_id': job.id, 'job': job_queue.describe(job), 'info': info,
                    'message': 'Queued'}), 202


@generation_bp.route('/refine/<model_id>', methods=['POST'])
def refine_parts(model_id: str):
    """Redraw the hands or the face in more detail into a new sketch.

    JSON: {"target": "hands" | "face", "strength": 0.3–0.7 (optional), "seed": null}
    """
    from secrets import randbits
    from PIL import Image
    from src.core import refine

    model_dir = model_dir_for(model_id)
    base = model_dir / 'base_image.png' if model_dir else None
    if not base or not base.is_file():
        return jsonify({'success': False, 'error': 'Model not found'}), 404
    data = request.get_json(silent=True) or {}
    target = data.get('target')
    if target not in refine.PROMPTS:
        return jsonify({'success': False, 'error': 'Choose hands or face'}), 400
    try:
        strength = _number(data, 'strength', refine.STRENGTH[target], float, 0.3, 0.7)
        seed = _number(data, 'seed', None, int, 0, 2**32 - 1)
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    boxes = refine.boxes(model_dir, target)
    if not boxes:
        return jsonify({'success': False, 'error': f'Could not find the {target}: separate the layers first, '
                                                   'or paint over them in Edit a region'}), 400
    with Image.open(base) as img:
        image = img.convert('RGB')
    brief = _saved_brief(model_dir)
    prompt, negative = refine.PROMPTS[target]
    job = job_queue.submit(GenerationRequest(**{
        **brief, 'prompt': brief.get('prompt') or f'Artwork of {model_id[:8]}',
        'width': image.width, 'height': image.height, 'operation': 'refine', 'sketch': True,
        'source': 'upload' if brief.get('source') == 'upload' else 'artwork',
        'reference_image': _stage(image, 'source'), 'regions': boxes, 'edit_prompt': prompt,
        'negative_prompt': negative, 'strength': strength,
        'seed': seed if seed is not None else randbits(31)}))
    return jsonify({'success': True, 'task_id': job.id, 'job': job_queue.describe(job), 'boxes': boxes,
                    'message': 'Queued'}), 202


@generation_bp.route('/expressions/<model_id>', methods=['POST'])
def paint_expressions(model_id: str):
    """Queue painting closed eyes and the other mouth state onto a separated character, or with
    JSON {"toggles": ["blush", ...]} only those expression toggles."""
    model_dir = model_dir_for(model_id)
    metadata_path = model_dir / 'metadata.json' if model_dir else None
    if not metadata_path or not metadata_path.exists():
        return jsonify({'success': False, 'error': 'Model not found'}), 404
    saved = load_json(metadata_path)
    if not any(a.get('metadata', {}).get('backend') == 'see-through' for a in saved.get('assets', [])):
        return jsonify({'success': False, 'error': 'Expressions need a character separated by See-through'}), 400
    try:
        toggles = _toggles(request.get_json(silent=True) or {}) or []
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    visemes = VISEMES_ID in toggles
    toggles = [t for t in toggles if t != VISEMES_ID] or None
    prompt = (saved.get('request') or {}).get('prompt') or 'character'
    job = job_queue.submit(GenerationRequest(prompt=prompt, operation='expressions', target_model=model_dir.name,
                                             source='artwork', toggles=toggles, visemes=visemes))
    return jsonify({'success': True, 'task_id': job.id, 'job': job_queue.describe(job), 'message': 'Queued'}), 202


@generation_bp.route('/variant/<model_id>', methods=['POST'])
def paint_variant(model_id: str):
    """Queue an alternate hairstyle or outfit for a separated character, switchable in Live Preview.

    JSON: {"part": "hair" | "top" | "bottom" | "headwear", "prompt": "short black bob"}
    """
    from src.core import variants

    model_dir = model_dir_for(model_id)
    metadata_path = model_dir / 'metadata.json' if model_dir else None
    if not metadata_path or not metadata_path.exists():
        return jsonify({'success': False, 'error': 'Model not found'}), 404
    saved = load_json(metadata_path)
    if not any(a.get('metadata', {}).get('backend') == 'see-through' for a in saved.get('assets', [])):
        return jsonify({'success': False, 'error': 'Alternate looks need a character separated by See-through'}), 400
    data = request.get_json(silent=True) or {}
    part, text = data.get('part'), ' '.join(str(data.get('prompt') or '').split())
    if part not in variants.PARTS:
        return jsonify({'success': False, 'error': 'Choose hair, top, bottom or headwear'}), 400
    if not text or len(text) > 200:
        return jsonify({'success': False, 'error': 'Describe the new design in up to 200 characters'}), 400
    prompt = (saved.get('request') or {}).get('prompt') or 'character'
    job = job_queue.submit(GenerationRequest(prompt=prompt, operation='expressions', target_model=model_dir.name,
                                             source='artwork', variant={'part': part, 'prompt': text}))
    return jsonify({'success': True, 'task_id': job.id, 'job': job_queue.describe(job), 'message': 'Queued'}), 202


@generation_bp.route('/open-project', methods=['POST'])
def open_project():
    """Reopen a character exported from the Gallery (multipart field "project", the Export zip)
    as a new result. Response: {"success": true, "model_id": ...}."""
    from src.core import project

    upload = request.files.get('project')
    if not upload or not upload.filename:
        return jsonify({'success': False, 'error': 'Choose an exported character (.zip)'}), 400
    try:
        model_id = project.import_project(upload.stream, config.OUTPUT_DIR)
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    logger.info("Opened exported character %s as %s", upload.filename, model_id)
    return jsonify({'success': True, 'model_id': model_id})


@generation_bp.route('/studio/<model_id>', methods=['PUT'])
def save_studio_settings(model_id: str):
    """Keep the Live Preview's adjustments (sliders, layers, anchors) with the character, so they
    survive preview rebuilds, other browsers and project export."""
    model_dir = model_dir_for(model_id)
    if not model_dir:
        return jsonify({'success': False, 'error': 'Model not found'}), 404
    if (request.content_length or 0) > 1_000_000:
        return jsonify({'success': False, 'error': 'Settings are too large'}), 413
    settings = request.get_json(silent=True)
    if not isinstance(settings, dict) or not isinstance(settings.get('params'), dict):
        return jsonify({'success': False, 'error': 'Expected the Live Preview settings JSON'}), 400
    save_json(settings, model_dir / 'studio.json')
    return jsonify({'success': True})


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
        from src.core import expressions
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
            'toggles': [{'id': k, 'label': t.label} for k, t in expressions.TOGGLES.items()]
                       + [{'id': VISEMES_ID, 'label': 'A I U E O mouths'}],
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
    """Download a rig-ready artifact: 'psd', 'ora', 'flat', 'package' (zip) or 'pngtuber' (zip)."""
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

        if kind == 'pngtuber':
            from src.core import pngtuber
            try:
                archive = pngtuber.pack(model_dir, model_id)
            except (FileNotFoundError, ValueError) as exc:
                return jsonify({'success': False, 'error': str(exc)}), 422
            return send_file(archive, mimetype='application/zip', as_attachment=True,
                             download_name=f'{model_id}_pngtuber.zip')

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
