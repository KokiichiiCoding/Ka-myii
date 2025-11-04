"""
API routes for expression management
"""
from flask import Blueprint, request, jsonify, send_file
from pathlib import Path
import logging
from typing import List

from src.core.expression_generator import ExpressionGenerator, Expression
from src.core.image_editor import ImageEditor
from src.core.accessory_generator import AccessoryGenerator, Accessory, AccessoryType
from src.utils.helpers import save_json, load_json
import config

logger = logging.getLogger(__name__)

# Create blueprint
expression_bp = Blueprint('expression', __name__, url_prefix='/api/expression')

# Global instances
expression_generator = ExpressionGenerator()
image_editor = ImageEditor()
accessory_generator = AccessoryGenerator()


@expression_bp.route('/generate', methods=['POST'])
def generate_expressions():
    """
    Generate expressions for a model

    Request JSON:
    {
        "model_id": "uuid",
        "expressions": ["happy", "sad", "angry", ...]
    }

    Response:
    {
        "success": true,
        "expressions": {...}
    }
    """
    try:
        data = request.json
        model_id = data.get('model_id')
        expressions = data.get('expressions', None)

        if not model_id:
            return jsonify({'success': False, 'error': 'model_id is required'}), 400

        # Get model base image
        model_dir = config.OUTPUT_DIR / model_id
        base_image_path = model_dir / "base_image.png"

        if not base_image_path.exists():
            return jsonify({'success': False, 'error': 'Model not found'}), 404

        # Generate expressions
        output_dir = model_dir / "expressions"
        expression_set = expression_generator.generate_expression_set(
            base_image_path,
            output_dir,
            expressions
        )

        # Create previews
        previews = {}
        for expr_name in expression_set.list_expressions():
            assets = expression_set.get_expression(expr_name)
            if assets:
                preview_path = output_dir / expr_name / f"preview_{expr_name}.png"
                from PIL import Image
                base_img = Image.open(base_image_path)
                expression_generator.create_expression_preview(
                    assets,
                    base_img,
                    preview_path
                )
                previews[expr_name] = str(preview_path.relative_to(config.OUTPUT_DIR))

        # Save expression metadata
        expr_metadata = {
            "model_id": model_id,
            "expressions": expression_set.list_expressions(),
            "previews": previews,
            "asset_count": sum(len(expression_set.get_expression(e) or [])
                             for e in expression_set.list_expressions())
        }

        metadata_path = output_dir / "expressions.json"
        save_json(expr_metadata, metadata_path)

        return jsonify({
            'success': True,
            'expressions': expr_metadata
        })

    except Exception as e:
        logger.error(f"Expression generation failed: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@expression_bp.route('/list', methods=['GET'])
def list_available_expressions():
    """
    List all available expression types

    Response:
    {
        "success": true,
        "expressions": [...]
    }
    """
    expressions = list(ExpressionGenerator.EXPRESSION_DEFINITIONS.keys())

    return jsonify({
        'success': True,
        'expressions': expressions,
        'count': len(expressions)
    })


@expression_bp.route('/preview/<model_id>/<expression_name>', methods=['GET'])
def get_expression_preview(model_id: str, expression_name: str):
    """Get preview image for a specific expression"""
    try:
        preview_path = config.OUTPUT_DIR / model_id / "expressions" / expression_name / f"preview_{expression_name}.png"

        if not preview_path.exists():
            return jsonify({'success': False, 'error': 'Preview not found'}), 404

        return send_file(preview_path, mimetype='image/png')

    except Exception as e:
        logger.error(f"Failed to get expression preview: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@expression_bp.route('/model/<model_id>/expressions', methods=['GET'])
def get_model_expressions(model_id: str):
    """Get all expressions for a specific model"""
    try:
        metadata_path = config.OUTPUT_DIR / model_id / "expressions" / "expressions.json"

        if not metadata_path.exists():
            return jsonify({
                'success': True,
                'expressions': [],
                'message': 'No expressions generated yet'
            })

        expr_data = load_json(metadata_path)

        return jsonify({
            'success': True,
            'expressions': expr_data
        })

    except Exception as e:
        logger.error(f"Failed to get model expressions: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@expression_bp.route('/edit', methods=['POST'])
def edit_image():
    """
    Edit an image with various operations

    Request JSON:
    {
        "model_id": "uuid",
        "image_type": "base_image" | "expression" | "asset",
        "image_path": "relative/path/to/image.png",
        "operations": [
            {"type": "brightness", "factor": 1.2},
            {"type": "contrast", "factor": 1.1},
            ...
        ]
    }

    Response:
    {
        "success": true,
        "edited_image_path": "..."
    }
    """
    try:
        data = request.json
        model_id = data.get('model_id')
        image_path_str = data.get('image_path')
        operations = data.get('operations', [])

        if not model_id or not image_path_str:
            return jsonify({'success': False, 'error': 'model_id and image_path required'}), 400

        # Construct full path
        if image_path_str.startswith('/'):
            image_path_str = image_path_str[1:]

        image_path = config.OUTPUT_DIR / model_id / image_path_str

        if not image_path.exists():
            return jsonify({'success': False, 'error': 'Image not found'}), 404

        # Apply operations
        current_path = image_path
        output_dir = image_path.parent / "edited"
        output_dir.mkdir(exist_ok=True)

        for i, op in enumerate(operations):
            op_type = op.get("type")
            temp_path = output_dir / f"temp_{i}_{image_path.name}"

            if op_type == "brightness":
                current_path = image_editor.adjust_brightness(
                    current_path, op.get("factor", 1.0), temp_path
                )
            elif op_type == "contrast":
                current_path = image_editor.adjust_contrast(
                    current_path, op.get("factor", 1.0), temp_path
                )
            elif op_type == "saturation":
                current_path = image_editor.adjust_saturation(
                    current_path, op.get("factor", 1.0), temp_path
                )
            elif op_type == "sharpness":
                current_path = image_editor.adjust_sharpness(
                    current_path, op.get("factor", 1.0), temp_path
                )
            elif op_type == "filter":
                current_path = image_editor.apply_filter(
                    current_path, op.get("filter_type", "blur"), temp_path
                )
            elif op_type == "rotate":
                current_path = image_editor.rotate(
                    current_path, op.get("angle", 0), temp_path
                )
            elif op_type == "flip":
                current_path = image_editor.flip(
                    current_path, op.get("direction", "horizontal"), temp_path
                )

        # Save final result
        final_path = output_dir / f"edited_{image_path.name}"
        from PIL import Image
        final_img = Image.open(current_path)
        final_img.save(final_path)

        # Clean up temp files
        for i in range(len(operations)):
            temp_path = output_dir / f"temp_{i}_{image_path.name}"
            if temp_path.exists():
                temp_path.unlink()

        relative_path = final_path.relative_to(config.OUTPUT_DIR / model_id)

        return jsonify({
            'success': True,
            'edited_image_path': str(relative_path),
            'full_path': str(final_path)
        })

    except Exception as e:
        logger.error(f"Image editing failed: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@expression_bp.route('/accessories/generate', methods=['POST'])
def generate_accessories():
    """
    Generate accessories for a model

    Request JSON:
    {
        "model_id": "uuid",
        "accessories": [
            {
                "type": "cat_ears",
                "name": "Pink Cat Ears",
                "color": [255, 182, 193, 255],
                "size_scale": 1.0
            },
            ...
        ]
        OR
        "preset": "cute" | "cool" | "elegant" | "fantasy" | "casual"
    }

    Response:
    {
        "success": true,
        "accessories": [...]
    }
    """
    try:
        data = request.json
        model_id = data.get('model_id')

        if not model_id:
            return jsonify({'success': False, 'error': 'model_id is required'}), 400

        # Get model base image
        model_dir = config.OUTPUT_DIR / model_id
        base_image_path = model_dir / "base_image.png"

        if not base_image_path.exists():
            return jsonify({'success': False, 'error': 'Model not found'}), 404

        # Get accessories list
        accessories = []

        if 'preset' in data:
            # Use preset
            accessories = accessory_generator.get_preset_accessories(data['preset'])
        elif 'accessories' in data:
            # Custom accessories
            for acc_data in data['accessories']:
                accessories.append(Accessory(
                    accessory_type=acc_data.get('type'),
                    name=acc_data.get('name', acc_data.get('type')),
                    color=tuple(acc_data.get('color', [255, 255, 255, 255])),
                    size_scale=acc_data.get('size_scale', 1.0)
                ))
        else:
            return jsonify({'success': False, 'error': 'accessories or preset required'}), 400

        # Generate accessories
        output_dir = model_dir / "accessories"
        assets = accessory_generator.generate_accessory_set(
            base_image_path,
            accessories,
            output_dir
        )

        # Save metadata
        acc_metadata = {
            "model_id": model_id,
            "accessories": [
                {
                    "type": asset.metadata.get('accessory_type'),
                    "name": asset.metadata.get('name'),
                    "path": str(asset.file_path.relative_to(config.OUTPUT_DIR))
                }
                for asset in assets
            ]
        }

        metadata_path = output_dir / "accessories.json"
        save_json(acc_metadata, metadata_path)

        return jsonify({
            'success': True,
            'accessories': acc_metadata['accessories'],
            'count': len(assets)
        })

    except Exception as e:
        logger.error(f"Accessory generation failed: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@expression_bp.route('/accessories/list', methods=['GET'])
def list_accessory_types():
    """List all available accessory types"""
    types = [
        attr for attr in dir(AccessoryType)
        if not attr.startswith('_') and attr.isupper()
    ]

    return jsonify({
        'success': True,
        'accessory_types': [getattr(AccessoryType, t) for t in types],
        'presets': ['cute', 'cool', 'elegant', 'fantasy', 'casual']
    })


@expression_bp.route('/accessories/preview/<model_id>', methods=['GET'])
def get_accessories_preview(model_id: str):
    """Get accessories for a model"""
    try:
        metadata_path = config.OUTPUT_DIR / model_id / "accessories" / "accessories.json"

        if not metadata_path.exists():
            return jsonify({
                'success': True,
                'accessories': [],
                'message': 'No accessories generated yet'
            })

        acc_data = load_json(metadata_path)

        return jsonify({
            'success': True,
            'accessories': acc_data
        })

    except Exception as e:
        logger.error(f"Failed to get accessories: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500
