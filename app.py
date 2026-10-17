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
from flask import Flask, render_template, jsonify, abort, url_for, request
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.utils.helpers import setup_logging
from src.api.generation import generation_bp, init_assembly_line

# Setup logging
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=Path("logs/kamyii.log")
)

logger = logging.getLogger(__name__)


LOOPBACK_HOSTS = {'127.0.0.1', 'localhost', '[::1]'}


def create_app(use_dummy_generators: bool = False):
    """
    Create and configure the Flask application

    Args:
        use_dummy_generators: Use dummy generators for testing (no GPU required)

    Returns:
        Flask application
    """
    app = Flask(__name__)
    app.config['SECRET_KEY'] = config.SECRET_KEY
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
    # No CORS: the UI is same-origin, and allowing other origins would let any website the
    # user visits drive this local server (queue jobs, delete results).
    app.config['LOCAL_ONLY'] = False

    @app.before_request
    def reject_foreign_hosts():
        # DNS rebinding: a website can point its own hostname at 127.0.0.1 and become "same origin".
        if app.config['LOCAL_ONLY'] and request.host.rsplit(':', 1)[0] not in LOOPBACK_HOSTS:
            abort(403)

    # Initialize assembly line
    init_assembly_line(use_dummy=use_dummy_generators)

    # Register blueprints
    app.register_blueprint(generation_bp)

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
        from src.api.generation import model_dir_for
        model_dir = model_dir_for(model_id)
        if not model_dir or not (model_dir / 'assets').is_dir():
            abort(404)
        from src.utils.helpers import load_json
        metadata_path = model_dir / 'metadata.json'
        name = load_json(metadata_path).get('name', model_id[:8]) if metadata_path.exists() else model_id[:8]
        if name.startswith('model_'):
            name = 'Study ' + name[6:].upper()
        return render_template('preview.html', model_id=model_id, character_name=name,
                               preview_psd=url_for('generation.get_artifact', model_id=model_id, kind='preview_psd'))

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
    parser.add_argument('--open', action='store_true', help='Open the Studio in the default browser')

    args = parser.parse_args()

    # Create app
    app = create_app(use_dummy_generators=args.dummy)
    app.config['LOCAL_ONLY'] = args.host in ('127.0.0.1', 'localhost', '::1')

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

    if args.open:
        import threading
        import webbrowser
        host = '127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host
        threading.Timer(1.5, webbrowser.open, [f'http://{host}:{args.port}/generator']).start()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == '__main__':
    main()
