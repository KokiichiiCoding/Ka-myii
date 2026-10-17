"""
Ka-myii: Automated VTuber Model Generation System
Main Flask application
"""
import os
import sys
from pathlib import Path

# PyTorch's Windows CPU allocator (mimalloc) returns freed memory to the OS only during later
# allocations, so the idle app kept ~7.5 GB of unloaded SDXL weights while See-through needed
# the RAM. Must be set before torch loads.
os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")
from flask import Flask, render_template, jsonify, abort, url_for
from flask_cors import CORS
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.utils.helpers import setup_logging
from src.api.generation import generation_bp, init_assembly_line
from src.api.models import models_bp
from src.api.expression import expression_bp

# Try to import SocketIO for real-time progress
try:
    from flask_socketio import SocketIO
    from src.utils.progress_tracker import init_progress_manager
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False
    logging.getLogger(__name__).warning("flask-socketio not installed - real-time progress updates disabled")

# Setup logging
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=Path("logs/kamyii.log")
)

logger = logging.getLogger(__name__)


def create_app(use_dummy_generators: bool = False):
    """
    Create and configure the Flask application

    Args:
        use_dummy_generators: Use dummy generators for testing (no GPU required)

    Returns:
        Flask application (and SocketIO if available)
    """
    app = Flask(__name__)
    app.config['SECRET_KEY'] = config.SECRET_KEY
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

    # Enable CORS
    CORS(app)

    # Initialize SocketIO for real-time progress
    socketio = None
    if SOCKETIO_AVAILABLE:
        socketio = SocketIO(app, cors_allowed_origins="*")
        init_progress_manager(socketio)
        logger.info("SocketIO initialized for real-time progress updates")

        # SocketIO event handlers
        @socketio.on('connect', namespace='/progress')
        def handle_connect():
            logger.info("Client connected to progress namespace")

        @socketio.on('disconnect', namespace='/progress')
        def handle_disconnect():
            logger.info("Client disconnected from progress namespace")

    # Store socketio in app config
    app.socketio = socketio

    # Initialize assembly line
    init_assembly_line(use_dummy=use_dummy_generators)

    # Register blueprints
    app.register_blueprint(generation_bp)
    app.register_blueprint(models_bp)
    app.register_blueprint(expression_bp)

    # Routes
    @app.route('/')
    def index():
        """Main page"""
        return render_template('index.html')

    @app.route('/generator')
    def generator():
        """Model generator page"""
        return render_template('generator.html')

    @app.route('/gallery')
    def gallery():
        """Model gallery page"""
        return render_template('gallery.html')

    @app.route('/preview/<model_id>')
    def live_preview(model_id):
        from src.api.generation import preview_model_dir
        model_dir = preview_model_dir(model_id)
        if not model_dir or not (model_dir / 'assets').is_dir():
            abort(404)
        from src.utils.helpers import load_json
        metadata_path = model_dir / 'metadata.json'
        name = load_json(metadata_path).get('name', model_id[:8]) if metadata_path.exists() else model_id[:8]
        if name.startswith('model_'):
            name = 'Study ' + name[6:].upper()
        return render_template('preview.html', model_id=model_id, character_name=name,
                               preview_psd=url_for('generation.get_artifact', model_id=model_id, kind='preview_psd'))

    @app.route('/editor/<model_id>')
    def editor(model_id):
        """Expression editor page"""
        return render_template('editor.html', model_id=model_id)

    @app.route('/health')
    def health():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'version': getattr(config, 'APP_VERSION', '2.0.0'),
            'service': getattr(config, 'APP_NAME', 'Ka-myii')
        })

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal error: {error}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

    logger.info("Flask application created")
    return app


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Ka-myii VTuber Model Generator')
    parser.add_argument(
        '--host',
        default=config.HOST,
        help='Host to bind to'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=config.PORT,
        help='Port to bind to'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        default=config.DEBUG,
        help='Enable debug mode'
    )
    parser.add_argument(
        '--dummy',
        action='store_true',
        help='Use dummy generators (no GPU required, for testing)'
    )

    args = parser.parse_args()

    # Create app
    app = create_app(use_dummy_generators=args.dummy)

    logger.info("=" * 60)
    logger.info("Ka-myii: Automated VTuber Model Generation System")
    logger.info("=" * 60)
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Debug mode: {args.debug}")
    logger.info(f"Dummy mode: {args.dummy}")
    logger.info(f"Output directory: {config.OUTPUT_DIR}")
    logger.info("=" * 60)

    if args.dummy:
        logger.warning("Running in DEMO mode - no actual AI generation")
        logger.warning("   This mode is for testing the pipeline without GPU")

    # Run server with SocketIO if available
    if SOCKETIO_AVAILABLE and hasattr(app, 'socketio') and app.socketio:
        logger.info("Starting server with SocketIO support")
        app.socketio.run(
            app,
            host=args.host,
            port=args.port,
            debug=args.debug,
            allow_unsafe_werkzeug=True
        )
    else:
        logger.info("Starting server without SocketIO")
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )


if __name__ == '__main__':
    main()
