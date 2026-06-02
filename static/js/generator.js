// Ka-myii Studio — generation controls, live progress, layer studio, downloads.
(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);
  let socket = null;
  let currentTaskId = null;
  let lastSeed = null;
  let busy = false;

  // ---- element refs ----
  const els = {
    form: $('genForm'), prompt: $('prompt'), negative: $('negative'),
    styleChips: $('styleChips'), model: $('model'), modelHint: $('modelHint'),
    sampler: $('sampler'), resolution: $('resolution'),
    steps: $('steps'), stepsVal: $('stepsVal'), cfg: $('cfg'), cfgVal: $('cfgVal'),
    seed: $('seed'), randSeed: $('randSeed'), reuseSeed: $('reuseSeed'),
    clipSkip: $('clipSkip'), clipSkipVal: $('clipSkipVal'),
    width: $('width'), widthVal: $('widthVal'), height: $('height'), heightVal: $('heightVal'),
    lora: $('lora'), advToggle: $('advToggle'), advBody: $('advBody'), advChevron: $('advChevron'),
    generateBtn: $('generateBtn'),
    canvasEmpty: $('canvasEmpty'), resultImg: $('resultImg'),
    overlay: $('progressOverlay'), pLabel: $('progressLabel'), pFill: $('progressFill'), pEta: $('progressEta'),
    resultMeta: $('resultMeta'), metaStatus: $('metaStatus'), metaLayers: $('metaLayers'),
    metaTime: $('metaTime'), metaSeed: $('metaSeed'),
    dlPackage: $('dlPackage'), dlPsd: $('dlPsd'), dlOra: $('dlOra'), regenBtn: $('regenBtn'),
    layerPanel: $('layerPanel'), layerGrid: $('layerGrid'), layerCount: $('layerCount'),
    errorPanel: $('errorPanel'), errorMsg: $('errorMsg'),
    modelNotice: $('modelNotice'),
  };

  let selectedStyle = 'vtuber';

  // ---- init ----
  document.addEventListener('DOMContentLoaded', async () => {
    bindControls();
    connectSocket();
    await loadOptions();
    await refreshModelNotice();
    addUIEnhancements();
  });

  function bindControls() {
    els.steps.oninput = () => (els.stepsVal.textContent = els.steps.value);
    els.cfg.oninput = () => (els.cfgVal.textContent = els.cfg.value);
    els.clipSkip.oninput = () => (els.clipSkipVal.textContent = els.clipSkip.value);
    els.width.oninput = () => (els.widthVal.textContent = els.width.value);
    els.height.oninput = () => (els.heightVal.textContent = els.height.value);
    els.randSeed.onclick = () => (els.seed.value = Math.floor(Math.random() * 2147483647));
    els.reuseSeed.onclick = () => { if (lastSeed != null) els.seed.value = lastSeed; };
    els.advToggle.onclick = () => {
      els.advBody.classList.toggle('hidden');
      els.advChevron.classList.toggle('fa-chevron-down');
      els.advChevron.classList.toggle('fa-chevron-up');
    };
    els.resolution.onchange = () => {
      const opt = els.resolution.selectedOptions[0];
      if (opt && opt.dataset.w) {
        els.width.value = opt.dataset.w; els.widthVal.textContent = opt.dataset.w;
        els.height.value = opt.dataset.h; els.heightVal.textContent = opt.dataset.h;
      }
    };
    els.model.onchange = updateModelHint;
    els.form.onsubmit = onGenerate;
    els.regenBtn.onclick = resetToForm;
  }

  async function loadOptions() {
    try {
      const { data } = await axios.get('/api/generation/options');
      if (!data.success) return;

      // styles
      els.styleChips.innerHTML = '';
      data.styles.forEach((s) => {
        const chip = document.createElement('div');
        chip.className = 'chip' + (s === selectedStyle ? ' active' : '');
        chip.textContent = s;
        chip.onclick = () => {
          selectedStyle = s;
          [...els.styleChips.children].forEach((c) => c.classList.remove('active'));
          chip.classList.add('active');
        };
        els.styleChips.appendChild(chip);
      });

      // checkpoints
      els.model.innerHTML = '';
      data.checkpoints.forEach((c) => {
        const o = document.createElement('option');
        o.value = c.id; o.textContent = c.name + (c.available === false ? ' (missing)' : '');
        o.dataset.note = c.note || ''; o.dataset.source = c.source || '';
        if (c.available === false) o.disabled = false; // selectable as a hint
        els.model.appendChild(o);
      });
      if (data.defaults.model) els.model.value = data.defaults.model;
      updateModelHint();

      // samplers
      els.sampler.innerHTML = '';
      data.samplers.forEach((s) => {
        const o = document.createElement('option'); o.value = s; o.textContent = s;
        els.sampler.appendChild(o);
      });
      els.sampler.value = data.defaults.sampler;

      // resolutions
      els.resolution.innerHTML = '';
      data.resolutions.forEach((r) => {
        const o = document.createElement('option');
        o.value = r.label; o.textContent = r.label; o.dataset.w = r.width; o.dataset.h = r.height;
        els.resolution.appendChild(o);
      });

      // loras
      els.lora.innerHTML = '<option value="">None</option>';
      (data.loras || []).forEach((l) => {
        const o = document.createElement('option'); o.value = l.id; o.textContent = l.name;
        els.lora.appendChild(o);
      });

      // defaults
      els.steps.value = data.defaults.steps; els.stepsVal.textContent = data.defaults.steps;
      els.cfg.value = data.defaults.guidance_scale; els.cfgVal.textContent = data.defaults.guidance_scale;
      els.width.value = data.defaults.width; els.widthVal.textContent = data.defaults.width;
      els.height.value = data.defaults.height; els.heightVal.textContent = data.defaults.height;
      els.clipSkip.value = data.defaults.clip_skip; els.clipSkipVal.textContent = data.defaults.clip_skip;
      els.negative.placeholder = (data.defaults.negative_prompt || '').slice(0, 90) + '…';
    } catch (e) {
      console.error('Failed to load options', e);
    }
  }

  function updateModelHint() {
    const opt = els.model.selectedOptions[0];
    els.modelHint.textContent = opt ? (opt.dataset.note || '') : '';
  }

  async function refreshModelNotice() {
    try {
      const { data } = await axios.get('/api/generation/model-status');
      els.modelNotice.classList.toggle('hidden', !data.is_dummy);
    } catch (e) { /* ignore */ }
  }

  // ---- websocket progress ----
  function connectSocket() {
    if (typeof io === 'undefined') return;
    try {
      socket = io('/progress');
      socket.on('progress_update', (d) => {
        if (currentTaskId && d.task_id && d.task_id !== currentTaskId) return;
        if (typeof d.progress === 'number') els.pFill.style.width = d.progress + '%';
        if (d.message) els.pLabel.textContent = d.message;
        if (d.eta && d.eta > 1) {
          const s = Math.round(d.eta);
          els.pEta.textContent = s >= 60 ? `ETA ${Math.floor(s / 60)}m ${s % 60}s` : `ETA ${s}s`;
        } else { els.pEta.textContent = ''; }
      });
    } catch (e) { console.warn('socket init failed', e); }
  }

  // ---- generate ----
  async function onGenerate(e) {
    e.preventDefault();
    if (busy) return;
    const prompt = els.prompt.value.trim();
    if (!prompt) { els.prompt.focus(); return; }

    busy = true;
    showProgress();

    const seedRaw = parseInt(els.seed.value, 10);
    const payload = {
      prompt,
      negative_prompt: els.negative.value.trim(),
      style: selectedStyle,
      model_id: els.model.value,
      sampler: els.sampler.value,
      steps: parseInt(els.steps.value, 10),
      guidance_scale: parseFloat(els.cfg.value),
      width: parseInt(els.width.value, 10),
      height: parseInt(els.height.value, 10),
      clip_skip: parseInt(els.clipSkip.value, 10),
      seed: (isNaN(seedRaw) || seedRaw < 0) ? null : seedRaw,
      loras: els.lora.value ? [{ id: els.lora.value, weight: 1.0 }] : null,
    };
    lastSeed = payload.seed;

    try {
      const { data } = await axios.post('/api/generation/generate', payload);
      if (!data.success) throw new Error(data.error || 'Generation failed');
      currentTaskId = data.task_id;
      els.pFill.style.width = '100%';
      await showResult(data);
    } catch (err) {
      const msg = err.response?.data?.error || err.message || String(err);
      showError(msg);
    } finally {
      busy = false;
    }
  }

  function showProgress() {
    els.errorPanel.classList.add('hidden');
    els.resultMeta.classList.add('hidden');
    els.layerPanel.classList.add('hidden');
    els.canvasEmpty.classList.add('hidden');
    els.resultImg.classList.add('hidden');
    els.overlay.classList.remove('hidden');
    els.pFill.style.width = '0%';
    els.pLabel.textContent = 'Starting…';
    els.pEta.textContent = '';
    els.generateBtn.disabled = true;
  }

  async function showResult(data) {
    const model = data.model || {};
    const meta = model.metadata || {};
    const mid = data.model_id;

    els.resultImg.src = `/api/generation/preview/${mid}?t=${Date.now()}`;
    els.resultImg.onload = () => {
      els.overlay.classList.add('hidden');
      els.resultImg.classList.remove('hidden');
    };
    els.resultImg.onerror = () => { els.overlay.classList.add('hidden'); };

    els.metaStatus.textContent = model.status || 'completed';
    els.metaLayers.textContent = (meta.layers || []).length || '—';
    els.metaTime.textContent = model.generation_time ? model.generation_time.toFixed(1) + 's' : '—';
    els.metaSeed.textContent = lastSeed != null ? lastSeed : 'random';
    els.resultMeta.classList.remove('hidden');

    els.dlPackage.href = `/api/generation/artifact/${mid}/package`;
    els.dlPsd.href = `/api/generation/artifact/${mid}/psd`;
    els.dlOra.href = `/api/generation/artifact/${mid}/ora`;

    els.generateBtn.disabled = false;
    await loadLayers(mid);
  }

  async function loadLayers(mid) {
    try {
      const { data } = await axios.get(`/api/generation/layers/${mid}`);
      if (!data.success || !data.count) return;
      els.layerGrid.innerHTML = '';
      data.layers.forEach((l) => {
        const card = document.createElement('div');
        card.className = 'layer-card';
        card.innerHTML = `<div class="layer-thumb"><img loading="lazy" src="${l.url}" alt="${l.name}"></div>
                          <div class="layer-name">${l.name}</div>`;
        els.layerGrid.appendChild(card);
      });
      els.layerCount.textContent = data.count + ' parts';
      els.layerPanel.classList.remove('hidden');
    } catch (e) { console.warn('layers load failed', e); }
  }

  function showError(msg) {
    els.overlay.classList.add('hidden');
    els.canvasEmpty.classList.remove('hidden');
    els.errorMsg.textContent = msg;
    els.errorPanel.classList.remove('hidden');
    els.generateBtn.disabled = false;
  }

  function resetToForm() {
    els.resultMeta.classList.add('hidden');
    els.layerPanel.classList.add('hidden');
    els.resultImg.classList.add('hidden');
    els.canvasEmpty.classList.remove('hidden');
  }

  // ---- UI enhancements ----
  function addUIEnhancements() {
    // Add smooth scrolling for form controls
    document.querySelectorAll('input[type="range"]').forEach(slider => {
      slider.addEventListener('input', (e) => {
        const val = (e.target.value - e.target.min) / (e.target.max - e.target.min);
        e.target.style.background = `linear-gradient(to right, var(--accent) 0%, var(--accent) ${val * 100}%, var(--border) ${val * 100}%, var(--border) 100%)`;
      });
      // Initialize
      const val = (slider.value - slider.min) / (slider.max - slider.min);
      slider.style.background = `linear-gradient(to right, var(--accent) 0%, var(--accent) ${val * 100}%, var(--border) ${val * 100}%, var(--border) 100%)`;
    });

    // Add ripple effect to buttons
    document.querySelectorAll('.btn, .chip').forEach(btn => {
      btn.addEventListener('click', function(e) {
        const ripple = document.createElement('span');
        ripple.style.cssText = `
          position: absolute;
          border-radius: 50%;
          background: rgba(255, 255, 255, 0.3);
          transform: scale(0);
          animation: ripple 0.6s ease-out;
          pointer-events: none;
        `;
        const rect = this.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        ripple.style.width = ripple.style.height = size + 'px';
        ripple.style.left = e.clientX - rect.left - size / 2 + 'px';
        ripple.style.top = e.clientY - rect.top - size / 2 + 'px';

        this.style.position = 'relative';
        this.style.overflow = 'hidden';
        this.appendChild(ripple);

        setTimeout(() => ripple.remove(), 600);
      });
    });

    // Add CSS animation for ripple if not exists
    if (!document.getElementById('ripple-style')) {
      const style = document.createElement('style');
      style.id = 'ripple-style';
      style.textContent = '@keyframes ripple { to { transform: scale(4); opacity: 0; } }';
      document.head.appendChild(style);
    }

    // Smooth reveal for panels
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.style.animation = 'slideIn 0.6s ease-out forwards';
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1 });

    document.querySelectorAll('.panel, .feature').forEach(el => {
      if (!el.classList.contains('sticky')) {
        observer.observe(el);
      }
    });

    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      // Ctrl/Cmd + Enter to generate
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !busy) {
        e.preventDefault();
        els.generateBtn.click();
      }
      // Escape to reset
      if (e.key === 'Escape' && !busy) {
        resetToForm();
      }
    });

    // Show keyboard shortcuts hint on first load
    if (!localStorage.getItem('km_shortcuts_shown')) {
      setTimeout(() => {
        const hint = document.createElement('div');
        hint.style.cssText = `
          position: fixed; bottom: 20px; right: 20px;
          background: linear-gradient(135deg, rgba(177, 108, 234, 0.95), rgba(92, 200, 255, 0.95));
          color: white; padding: 12px 18px; border-radius: 10px;
          box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
          font-size: 0.85rem; z-index: 1000;
          animation: slideIn 0.4s ease-out;
          cursor: pointer;
        `;
        hint.innerHTML = '💡 <strong>Tip:</strong> Ctrl+Enter to generate, Esc to reset';
        document.body.appendChild(hint);
        hint.onclick = () => hint.remove();
        setTimeout(() => hint.remove(), 8000);
        localStorage.setItem('km_shortcuts_shown', 'true');
      }, 2000);
    }
  }
})();
