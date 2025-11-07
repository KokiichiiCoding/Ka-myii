# Ka-myii 🎨✨

**Automated VTuber Model Generation System**

Ka-myii is a Python-based web application that automates the creation of high-quality Live2D style VTuber models using AI-powered generation, intelligent asset separation, and automated model assembly.

![Status](https://img.shields.io/badge/status-alpha-orange)
![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 🌟 Features

### Core Generation
- **AI-Powered Image Generation**: Uses Stable Diffusion models to generate high-quality character images
- **Intelligent Asset Separation**: Automatically separates characters into layered components (body, head, eyes, mouth, hair, etc.)
- **Live2D Model Assembly**: Combines separated assets into Live2D-compatible model formats
- **Assembly Line Pipeline**: Streamlined workflow from prompt to finished model

### Expression & Emotion System 😊😢😠
- **15+ VTuber Expressions**: Happy, sad, angry, surprised, cry, frustrated, smug, heart eyes, blush, shocked, sleepy, embarrassed, determined, pouty, wink, worried, excited, and more!
- **Live Expression Toggle**: Switch between expressions in real-time to preview your character
- **Automatic Expression Generation**: AI-generated eyes, mouth, eyebrows, blush, and tears for each emotion
- **Expression Preview**: See exactly how each expression looks before exporting

### Accessory System 👑🎀
- **20+ Accessory Types**: Cat ears, bunny ears, glasses, bows, halos, horns, crowns, headbands, chokers, and more
- **Preset Themes**: Cute, Cool, Elegant, Fantasy, and Casual accessory sets
- **Special Effects**: Sparkles, hearts, stars, and other decorative elements
- **Customizable**: Adjust colors, sizes, and positions

### Image Editing Tools 🎨
- **Brightness/Contrast/Saturation**: Fine-tune image appearance
- **Filters**: Blur, sharpen, smooth, edge enhance, emboss, and more
- **Transformations**: Rotate, flip, crop, and resize
- **Background Removal**: AI-powered background removal
- **Auto-Enhance**: Automatic quality improvement

### Advanced AI Features 🤖
- **ControlNet Expression Generation**: Enhanced expression generation using ControlNet for better control
- **Segment Anything (SAM)**: Superior automatic segmentation for cleaner, more precise layers
- **Expression Variations**: Generate multiple variations of each expression automatically
- **AI-Powered Segmentation**: Intelligent layer detection and classification

### Professional Export & Integration 📦
- **Live2D Cubism Export**: Generate complete Live2D packages with proper model3.json structure
- **Auto Physics Setup**: Automatically generate physics3.json based on detected hair/clothing parts
- **Expression Animations**: Create GIF/WebM previews of blink, talking, and expression transitions
- **VRM Support**: Framework for VRM conversion (coming soon)

### Workflow & History 📊
- **Prompt History System**: Store all generations with thumbnails, seeds, and parameters for easy re-use
- **Model Versioning**: Auto-versioning with full reproducibility tracking (config.json, seed, hash)
- **Favorites & Search**: Save and search through your prompt history
- **Statistics Dashboard**: Track usage patterns and popular prompts

### Interface & Management
- **Web-Based Interface**: Beautiful, intuitive web UI built with Flask and Bootstrap
- **Expression Editor**: Dedicated interface for managing expressions and accessories
- **Gallery Management**: View, download, and manage all your generated models
- **Batch Processing**: Generate multiple models in sequence
- **Live Preview**: Real-time preview of all changes
- **Animation Previews**: View expression transitions and blink/talk animations

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended, but not required for testing)
- 8GB+ RAM (16GB+ recommended)
- 10GB+ free disk space

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Ka-myii.git
   cd Ka-myii
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional)
   ```bash
   cp .env.example .env
   # Edit .env with your preferred settings
   ```

### Running the Application

#### Development Mode (with dummy generators - no GPU required)

Perfect for testing the interface without needing a GPU:

```bash
python app.py --dummy
```

#### Production Mode (with actual AI generation)

Requires a CUDA-capable GPU and ~10GB VRAM:

```bash
python app.py
```

#### Custom Configuration

```bash
python app.py --host 0.0.0.0 --port 8080 --debug
```

### Accessing the Web UI

Once the server is running, open your browser and navigate to:

```
http://localhost:5000
```

## 📚 Usage

### Generating Your First Model

1. **Navigate to the Generator page** by clicking "Start Creating" or visiting `/generator`

2. **Enter a character description** in the prompt field:
   ```
   anime girl with blue hair and green eyes, wearing a school uniform
   ```

3. **Choose your art style** (Anime, Live2D, VTuber, or Realistic)

4. **Adjust advanced settings** (optional):
   - Image dimensions
   - Generation steps
   - Guidance scale
   - Random seed

5. **Click "Generate Model"** and wait for the pipeline to complete

6. **Download your model** when generation is complete!

### Adding Expressions & Accessories

After generating a model, click **"Edit Expressions & Accessories"** to:

1. **Generate Expressions**:
   - Select from 15+ emotions (happy, sad, angry, cry, smug, heart eyes, etc.)
   - Click "Generate Expressions" to create all selected emotions
   - Use the live toggle to preview each expression

2. **Add Accessories**:
   - Choose a preset theme (Cute, Cool, Elegant, Fantasy, Casual)
   - Or select individual accessories (cat ears, glasses, bows, etc.)
   - Click "Add Accessories" to apply them to your model

3. **Adjust Image Settings**:
   - Fine-tune brightness, contrast, and saturation
   - Apply filters and effects
   - See changes in real-time preview

4. **Export**:
   - Download all expressions and accessories as a package
   - Export for Live2D integration

### Viewing Your Models

Visit the **Gallery** page to:
- View all generated models
- Search and filter models
- **Edit expressions and accessories** for any model
- Download models as ZIP files
- View detailed model information
- Delete unwanted models

## 🏗️ Project Structure

```
Ka-myii/
├── app.py                 # Main Flask application
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── .gitignore           # Git ignore rules
├── .env.example         # Environment template
│
├── src/                 # Source code
│   ├── api/            # API routes
│   │   ├── generation.py   # Model generation endpoints
│   │   └── models.py       # Model management endpoints
│   │
│   ├── core/           # Core functionality
│   │   ├── image_generator.py         # Stable Diffusion integration
│   │   ├── asset_separator.py         # Asset layer separation
│   │   ├── model_assembler.py         # Live2D model assembly
│   │   ├── expression_generator.py    # Expression generation ⭐
│   │   ├── accessory_generator.py     # Accessory creation ⭐
│   │   ├── image_editor.py            # Image editing tools ⭐
│   │   ├── controlnet_expression.py   # ControlNet integration ⭐
│   │   ├── sam_segmentation.py        # Segment Anything integration ⭐
│   │   ├── auto_physics.py            # Auto physics generation ⭐
│   │   ├── live2d_exporter.py         # Live2D export helper ⭐
│   │   ├── expression_animator.py     # GIF/animation creator ⭐
│   │   └── rigging.py                 # Auto-rigging (placeholder)
│   │
│   ├── models/         # Data models
│   │   └── vtuber_model.py            # VTuber model data structures
│   │
│   ├── utils/          # Utilities
│   │   ├── helpers.py                 # Helper functions
│   │   ├── prompt_history.py          # Prompt history system ⭐
│   │   └── model_versioning.py        # Model versioning ⭐
│   │
│   └── pipeline/       # Pipeline orchestration
│       └── assembly_line.py      # Main generation pipeline
│
├── templates/          # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── generator.html
│   └── gallery.html
│
├── static/             # Static assets
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   ├── main.js
│   │   ├── generator.js
│   │   └── gallery.js
│   └── img/
│
├── outputs/            # Generated models (created at runtime)
├── models/             # Pre-trained model storage
├── logs/               # Application logs
└── tests/              # Unit tests
```

## 🔧 Configuration

Configuration is managed through `config.py` and environment variables.

### Key Configuration Options

**Image Generation Settings:**
- `default_width`: Default image width (512)
- `default_height`: Default image height (512)
- `default_steps`: Number of diffusion steps (30)
- `model_name`: Stable Diffusion model to use

**Asset Separation:**
- `layers`: List of layer types to generate
- `use_ai_segmentation`: Enable AI-based segmentation

**Model Assembly:**
- `texture_size`: Size of texture atlas (2048)
- `include_physics`: Include physics in model

## 🎯 Pipeline Stages

### 1. Image Generation
Uses Stable Diffusion to generate a character image based on your prompt.

**Technologies:**
- Diffusers library
- PyTorch
- DPM++ scheduler for quality

### 2. Asset Separation
Separates the generated image into layers:
- Background
- Body
- Head
- Eyes (with variants: open, closed, happy)
- Mouth (with variants: closed, open, smile)
- Hair (front and back)
- Accessories
- Clothing

**Technologies:**
- OpenCV for image processing
- RemBG for background removal
- Custom segmentation algorithms

### 3. Model Assembly
Combines separated assets into a Live2D-compatible model structure:
- Creates model3.json configuration
- Generates physics3.json for physics simulation
- Organizes textures and assets
- Creates motion definitions

### 4. Auto-Rigging (Coming Soon)
Automatically applies rigging and deformers to enable animation.

## 📖 API Documentation

### Generate Model

```http
POST /api/generation/generate
Content-Type: application/json

{
  "prompt": "anime girl with blue hair",
  "negative_prompt": "low quality, blurry",
  "style": "live2d",
  "width": 512,
  "height": 512,
  "steps": 30,
  "guidance_scale": 7.5,
  "seed": null,
  "include_rigging": false
}
```

### List Models

```http
GET /api/generation/list
```

### Get Model Status

```http
GET /api/generation/status/{model_id}
```

### Download Model

```http
GET /api/generation/download/{model_id}
```

### Get Statistics

```http
GET /api/models/stats
```

### Generate Expressions

```http
POST /api/expression/generate
Content-Type: application/json

{
  "model_id": "uuid",
  "expressions": ["happy", "sad", "angry", "cry", "smug", "heart_eyes"]
}
```

### List Available Expressions

```http
GET /api/expression/list
```

### Get Expression Preview

```http
GET /api/expression/preview/{model_id}/{expression_name}
```

### Generate Accessories

```http
POST /api/expression/accessories/generate
Content-Type: application/json

{
  "model_id": "uuid",
  "preset": "cute"  // or "cool", "elegant", "fantasy", "casual"
}
```

### Edit Image

```http
POST /api/expression/edit
Content-Type: application/json

{
  "model_id": "uuid",
  "image_path": "base_image.png",
  "operations": [
    {"type": "brightness", "factor": 1.2},
    {"type": "contrast", "factor": 1.1},
    {"type": "saturation", "factor": 1.0}
  ]
}
```

## 🛠️ Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
```

### Linting

```bash
flake8 src/
```

## 🚧 Roadmap

- [x] Basic image generation
- [x] Asset separation
- [x] Model assembly
- [x] Web interface
- [ ] Auto-rigging implementation
- [ ] Advanced segmentation models
- [ ] Real-time progress updates (WebSocket)
- [ ] User authentication
- [ ] Cloud deployment guides
- [ ] Model fine-tuning interface
- [ ] Animation preview
- [ ] Direct Live2D export
- [ ] VRM format support

## ⚠️ Limitations

- **Auto-rigging is not yet implemented** - Manual rigging in Live2D Cubism is currently required
- **GPU recommended** - CPU generation is very slow
- **Large model size** - Stable Diffusion models are several GB
- **Segmentation quality** - Asset separation is currently basic and may need manual refinement

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion) for image generation
- [Live2D](https://www.live2d.com/) for the model format inspiration
- [Hugging Face](https://huggingface.co/) for model hosting
- The VTuber community for inspiration

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/Ka-myii/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/Ka-myii/discussions)

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Made with ❤️ for the VTuber community**
