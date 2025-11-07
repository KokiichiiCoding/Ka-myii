# Ka-myii Troubleshooting Guide

## Issue: Nothing generates when clicking "Generate Model"

This is usually because the AI model hasn't been downloaded yet. Here are the solutions:

### Solution 1: Use the built-in model downloader (EASIEST)

1. Start the application:
   ```bash
   # Windows
   run.bat

   # Linux/Mac
   python app.py
   ```

2. Open http://localhost:5000/generator in your browser

3. You'll see a **yellow warning banner** at the top that says:
   ```
   Model Status: Model not loaded yet - will load on first generation
   [Download Model Now] button
   ```

4. Click the **"Download Model Now"** button
   - This will download the Stable Diffusion model (~4-5 GB)
   - Takes 10-30 minutes depending on internet speed
   - The page will show download progress

5. Once complete, the banner will turn **green** and say:
   ```
   Model Status: Model loaded and ready (CUDA/CPU)
   ```

6. Now you can generate models!

### Solution 2: Use the standalone download script

1. Run the download script before starting the app:
   ```bash
   python download_model.py
   ```

2. Wait for the download to complete (shows progress)

3. Once done, start the application:
   ```bash
   run.bat  # Windows
   python app.py  # Linux/Mac
   ```

### Solution 3: Start in Dummy Mode (for testing without GPU/model)

If you just want to test the interface without actually generating images:

```bash
# Windows
run.bat
# Then select option 2 (Dummy Mode)

# Linux/Mac
python app.py --dummy
```

Dummy mode creates placeholder images instantly - great for testing!

## Common Errors

### Error: "CUDA out of memory"

**Cause:** Not enough GPU memory

**Solutions:**
1. Reduce image size in advanced settings (try 384x384 instead of 512x512)
2. Reduce steps (try 20 instead of 30)
3. Close other GPU-using programs
4. Run in CPU mode (slower but works):
   ```bash
   python app.py --device cpu
   ```

### Error: "Connection timeout" when downloading model

**Cause:** Slow internet or Hugging Face server issues

**Solutions:**
1. Check your internet connection
2. Try again later
3. Use a VPN if Hugging Face is blocked
4. Manually download from: https://huggingface.co/runwayml/stable-diffusion-v1-5

### Error: "diffusers library not found"

**Cause:** Missing dependencies

**Solution:**
```bash
pip install -r requirements.txt
```

Or specifically:
```bash
pip install diffusers transformers accelerate torch
```

### Error: "No module named 'torch'"

**Cause:** PyTorch not installed

**Solution:**

For **CUDA GPU** (NVIDIA):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For **CPU only**:
```bash
pip install torch torchvision torchaudio
```

The `run.bat` script on Windows does this automatically!

## Checking System Status

### Check if model is loaded:
1. Go to http://localhost:5000/generator
2. Look at the colored banner at the top:
   - 🟢 **Green** = Model loaded and ready
   - 🟡 **Yellow** = Model not loaded (click Download button)
   - 🔴 **Red** = Error (check logs)

### Check GPU:
```bash
# Check if CUDA is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Check GPU name
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

### Check logs:
Look at the console where you ran `app.py` or `run.bat` for detailed error messages.

## Performance Tips

### First generation is slow
- **Expected!** The model loads on first use
- Takes 30-60 seconds to load into memory
- Subsequent generations are faster

### Generation takes forever (10+ minutes)
- **Check if using CPU instead of GPU**
  - Look at the yellow/green banner
  - Should say "(CUDA)" not "(CPU)"
- CPU mode is 20-50x slower than GPU
- Solution: Make sure CUDA is installed (run.bat does this)

### Out of disk space
- Model cache is in: `~/.cache/huggingface/`
- Needs ~10 GB free space
- Clear old models: `rm -rf ~/.cache/huggingface/` (will re-download)

## Still Not Working?

### Debug Steps:

1. **Check browser console** (F12):
   ```javascript
   // Look for errors in the Console tab
   // Network tab should show API calls
   ```

2. **Check server logs**:
   ```
   Look at the terminal where you ran app.py
   Should see logs like:
   - "ImageGenerator initialized"
   - "Loading model: runwayml/stable-diffusion-v1-5"
   - "Model loaded successfully"
   ```

3. **Test API directly**:
   ```bash
   curl http://localhost:5000/api/generation/model-status
   ```
   Should return JSON with model status

4. **Restart the application**:
   - Close the server (Ctrl+C)
   - Run it again
   - Try generating again

## Getting Help

If none of these solutions work:

1. **Collect information**:
   - What error messages do you see?
   - Browser console errors (F12)
   - Server terminal output
   - Your system: Windows/Linux/Mac, GPU or CPU

2. **Check the logs** carefully - they usually explain the issue

3. **Try dummy mode** to verify the app itself works:
   ```bash
   python app.py --dummy
   ```
   If dummy mode works but real mode doesn't, it's a model/GPU issue

## Quick Reference

| Problem | Solution |
|---------|----------|
| Nothing generates | Click "Download Model Now" button on generator page |
| First time setup | Run `run.bat` (Windows) or `python download_model.py` |
| Just want to test | Use `--dummy` mode |
| CUDA errors | Run `run.bat` to install CUDA-enabled PyTorch |
| Slow generation | Check if using GPU (should say "CUDA" in status banner) |
| Out of memory | Reduce image size and steps in Advanced Settings |
