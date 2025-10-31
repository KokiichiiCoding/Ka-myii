"""
Ka-myii: Automated VTuber Model Generation System
Main Flask application
"""
import os
import sys
from pathlib import Path
from flask import Flask, render_template, jsonify
from flask_cors import CORS
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from src.utils.helpers import setup_logging
from src.api.generation import generation_bp, init_assembly_line
from src.api.models import models_bp

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
        Flask application
    """
    app = Flask(__name__)
    app.config['SECRET_KEY'] = config.SECRET_KEY
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

    # Enable CORS
    CORS(app)

    # Initialize assembly line
    init_assembly_line(use_dummy=use_dummy_generators)

    # Register blueprints
    app.register_blueprint(generation_bp)
    app.register_blueprint(models_bp)

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

    @app.route('/health')
    def health():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'version': '0.1.0',
            'service': 'Ka-myii'
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
        logger.warning("⚠️  Running in DUMMY mode - no actual AI generation")
        logger.warning("   This mode is for testing the pipeline without GPU")

    # Run server
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )


if __name__ == '__main__':
    main()
