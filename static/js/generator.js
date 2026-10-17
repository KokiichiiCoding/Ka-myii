// Ka-myii Studio — generation controls, live progress, layer studio, downloads.
(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);
  let currentTaskId = null;
  let pollTimer = null;
  const pageTitle = document.title;
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
    lora: $('lora'), loraWeight: $('loraWeight'), loraWeightVal: $('loraWeightVal'), advToggle: $('advToggle'), advBody: $('advBody'), advChevron: $('advChevron'),
    generateBtn: $('generateBtn'), cancelBtn: $('cancelBtn'),
    queuePanel: $('queuePanel'), queueList: $('queueList'), queueNote: $('queueNote'),
    canvasEmpty: $('canvasEmpty'), resultImg: $('resultImg'),
    overlay: $('progressOverlay'), pLabel: $('progressLabel'), pFill: $('progressFill'), pEta: $('progressEta'),
    resultMeta: $('resultMeta'), metaStatus: $('metaStatus'), metaLayers: $('metaLayers'),
    metaTime: $('metaTime'), metaSeed: $('metaSeed'),
    dlPackage: $('dlPackage'), dlPsd: $('dlPsd'), dlOra: $('dlOra'), regenBtn: $('regenBtn'),
    layerPanel: $('layerPanel'), layerGrid: $('layerGrid'), layerCount: $('layerCount'),
    errorPanel: $('errorPanel'), errorMsg: $('errorMsg'),
    modelNotice: $('modelNotice'),
    sourceChips: $('sourceChips'), uploadFields: $('uploadFields'), promptFields: $('promptFields'),
    artwork: $('artwork'), artworkThumb: $('artworkThumb'),
  };

  let selectedStyle = 'vtuber';
  let configModel = null; // the checkpoint an empty model_id resolves to on the server
  let source = 'prompt'; // or 'upload': decompose the user's own artwork instead of generating

  // ---- init ----
  document.addEventListener('DOMContentLoaded', async () => {
    bindControls();
    await loadOptions();
    await refreshModelNotice();
    addUIEnhancements();
    refreshQueue();
    setInterval(refreshQueue, 3000);
    if (await resumeActiveJob()) return;
    const savedId = new URLSearchParams(location.search).get('model');
    if (savedId) {
      try {
        const {data} = await axios.get(`/api/generation/status/${encodeURIComponent(savedId)}`);
        if (data.success) {
          const saved = data.model.request;
          if (saved) {
            els.prompt.value = saved.reference_image ? '' : (saved.prompt || '');
            els.negative.value = saved.negative_prompt || '';
            $('framing').value = saved.framing || 'full_body';
            if (saved.seed != null) { lastSeed = saved.seed; els.seed.value = saved.seed; }
            for (const [key, value] of [['steps',saved.steps],['cfg',saved.guidance_scale],['width',saved.width],['height',saved.height],['clipSkip',saved.clip_skip]]) {
              if (value != null) { els[key].value=value; els[key+'Val'].textContent=value; }
            }
            const usedModel = saved.model_id || configModel;
            if ([...els.model.options].some(o=>o.value===usedModel)) els.model.value=usedModel;
            if ([...els.sampler.options].some(o=>o.value===saved.sampler)) els.sampler.value=saved.sampler;
            const lora = saved.loras?.[0];
            if (lora && [...els.lora.options].some(o=>o.value===(lora.id || lora.path))) {
              els.lora.value = lora.id || lora.path;
              els.loraWeight.value = lora.weight ?? 1; els.loraWeightVal.textContent = els.loraWeight.value;
            }
            const resolution=[...els.resolution.options].find(o=>Number(o.dataset.w)===saved.width&&Number(o.dataset.h)===saved.height);
            if (resolution) els.resolution.value=resolution.value;
            if (saved.style) selectedStyle=saved.style;
            for (const chip of els.styleChips.children) {
              const active=chip.textContent===selectedStyle;
              chip.classList.toggle('active',active); chip.setAttribute('aria-pressed',String(active));
            }
            updateModelHint();
          }
          els.canvasEmpty.classList.add('hidden');
          await showResult({model_id:savedId, model:data.model});
        }
      } catch (err) { showError('This character could not be opened. Return to the collection and try again.'); }
    }
  });

  function bindControls() {
    els.steps.oninput = () => (els.stepsVal.textContent = els.steps.value);
    els.cfg.oninput = () => (els.cfgVal.textContent = els.cfg.value);
    els.clipSkip.oninput = () => (els.clipSkipVal.textContent = els.clipSkip.value);
    els.loraWeight.oninput = () => (els.loraWeightVal.textContent = els.loraWeight.value);
    els.width.oninput = () => (els.widthVal.textContent = els.width.value);
    els.height.oninput = () => (els.heightVal.textContent = els.height.value);
    els.randSeed.onclick = () => (els.seed.value = Math.floor(Math.random() * 2147483647));
    els.reuseSeed.onclick = () => { if (lastSeed != null) els.seed.value = lastSeed; };
    els.advToggle.onclick = () => {
      els.advBody.classList.toggle('hidden');
      els.advToggle.setAttribute('aria-expanded', String(!els.advBody.classList.contains('hidden')));
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
    for (const chip of els.sourceChips.children) {
      chip.onclick = () => {
        source = chip.dataset.source;
        for (const c of els.sourceChips.children) {
          c.classList.toggle('active', c === chip); c.setAttribute('aria-pressed', String(c === chip));
        }
        els.uploadFields.classList.toggle('hidden', source !== 'upload');
        els.promptFields.classList.toggle('hidden', source === 'upload');
        setGenerateLabel(busy);
      };
    }
    els.artwork.onchange = () => {
      const file = els.artwork.files[0];
      if (els.artworkThumb.src) URL.revokeObjectURL(els.artworkThumb.src);
      els.artworkThumb.classList.toggle('hidden', !file);
      if (file) els.artworkThumb.src = URL.createObjectURL(file);
    };
    els.form.onsubmit = onGenerate;
    els.regenBtn.onclick = resetToForm;
    els.cancelBtn.onclick = async () => {
      if (!currentTaskId) return;
      els.cancelBtn.disabled = true;
      els.pLabel.textContent = 'Cancelling…';
      try { await axios.post(`/api/generation/jobs/${currentTaskId}/cancel`); }
      catch (e) { els.cancelBtn.disabled = false; }
    };
  }

  async function loadOptions() {
    try {
      const { data } = await axios.get('/api/generation/options');
      if (!data.success) return;

      // styles
      els.styleChips.innerHTML = '';
      data.styles.forEach((s) => {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'chip' + (s === selectedStyle ? ' active' : '');
        chip.textContent = s;
        chip.setAttribute('aria-pressed', String(s === selectedStyle));
        chip.onclick = () => {
          selectedStyle = s;
          [...els.styleChips.children].forEach((c) => { c.classList.remove('active'); c.setAttribute('aria-pressed', 'false'); });
          chip.classList.add('active');
          chip.setAttribute('aria-pressed', 'true');
        };
        els.styleChips.appendChild(chip);
      });

      // checkpoints
      els.model.innerHTML = '';
      data.checkpoints.forEach((c) => {
        const o = document.createElement('option');
        o.value = c.id; o.textContent = c.name;
        o.dataset.note = c.note || ''; o.dataset.source = c.source || '';
        els.model.appendChild(o);
      });
      if (data.defaults.model) els.model.value = data.defaults.model;
      configModel = data.defaults.config_model;
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
      $('seethroughNotice').classList.toggle('hidden', data.is_dummy || data.seethrough !== false);
      $('gpuNotice').classList.toggle('hidden', data.is_dummy || data.device === 'cuda');
    } catch (e) { /* ignore */ }
  }

  // ---- background job tracking ----
  // Generation runs as a server-side job; polling (not the socket) is the single source of truth,
  // so a reloaded or reopened tab re-attaches to the running job.
  async function resumeActiveJob() {
    try {
      const { data } = await axios.get('/api/generation/jobs');
      const jobs = data.jobs || [];
      const active = jobs.find((j) => j.state === 'running') || jobs.find((j) => j.state === 'queued');
      if (!active) return false;
      if (active.prompt && !els.prompt.value) els.prompt.value = active.prompt;
      track(active);
      return true;
    } catch (e) { return false; }
  }

  function track(job) {
    currentTaskId = job.id;
    busy = true;
    showProgress();
    renderJob(job);
    clearTimeout(pollTimer);
    const tick = async () => {
      if (currentTaskId !== job.id) return;
      try {
        const { data } = await axios.get(`/api/generation/jobs/${job.id}`);
        const j = data.job;
        renderJob(j);
        if (j.state === 'completed') {
          endJob();
          const status = await axios.get(`/api/generation/status/${encodeURIComponent(j.model_id)}`);
          await showResult({ model_id: j.model_id, model: status.data.model });
          return;
        }
        if (j.state === 'failed') { endJob(); showError(j.error || 'Generation failed'); return; }
        if (j.state === 'cancelled') { endJob(); els.overlay.classList.add('hidden'); resetToForm(); return; }
      } catch (err) {
        if (err.response?.status === 404) {
          endJob();
          showError('The server no longer knows this job — it was probably restarted. Finished characters are in the Gallery.');
          return;
        }
        // Transient network hiccup: keep polling.
      }
      pollTimer = setTimeout(tick, 1000);
    };
    pollTimer = setTimeout(tick, 1000);
  }

  function renderJob(j) {
    els.pFill.style.width = (j.progress || 0) + '%';
    // A real run takes ~15 min; show progress on the tab while the user is elsewhere.
    document.title = (j.state === 'queued' ? 'Queued' : Math.floor(j.progress || 0) + '%') + ' · ' + pageTitle;
    if (j.state === 'queued') {
      els.pLabel.textContent = j.position ? `Waiting in queue — ${j.position} ahead` : 'Waiting for the GPU…';
    } else if (j.message) {
      const step = { image: 'Step 1 of 3 · ', separate: 'Step 2 of 3 · ', assemble: 'Step 3 of 3 · ' }[j.stage] || '';
      els.pLabel.textContent = step + j.message;
    }
    if (j.eta && j.eta > 1) {
      const s = Math.round(j.eta);
      els.pEta.textContent = (s >= 60 ? `about ${Math.floor(s / 60)}m ${s % 60}s` : `about ${s}s`) + ' left in this step';
    } else { els.pEta.textContent = ''; }
    els.cancelBtn.disabled = !!j.cancel_requested;
  }

  function endJob() {
    clearTimeout(pollTimer);
    document.title = pageTitle;
    busy = false;
    setGenerateLabel(false);
    els.cancelBtn.classList.add('hidden');
    els.cancelBtn.disabled = false;
  }

  // Every job except the one on the canvas, so several characters can be lined up overnight.
  async function refreshQueue() {
    let jobs;
    try { jobs = (await axios.get('/api/generation/jobs')).data.jobs || []; } catch (e) { return; }
    const others = jobs.filter((j) => (j.state === 'queued' || j.state === 'running') && j.id !== currentTaskId)
      .sort((a, b) => (a.state === 'running' ? -1 : 0) - (b.state === 'running' ? -1 : 0) || (a.position ?? 0) - (b.position ?? 0));
    els.queueList.innerHTML = '';
    for (const j of others) {
      const li = document.createElement('li');
      li.className = 'queue-item';
      const label = document.createElement('span');
      label.className = 'q-prompt'; label.textContent = j.prompt; label.title = j.prompt;
      const state = document.createElement('span');
      state.className = 'q-state';
      state.textContent = j.state === 'running' ? `${Math.floor(j.progress)}%` : `#${(j.position ?? 0) + 1}`;
      const watch = document.createElement('button');
      watch.type = 'button'; watch.title = 'Watch this job'; watch.setAttribute('aria-label', 'Watch ' + j.prompt);
      watch.innerHTML = '<i class="fas fa-eye"></i>';
      watch.onclick = () => { els.prompt.value = j.prompt; track(j); refreshQueue(); };
      const cancel = document.createElement('button');
      cancel.type = 'button'; cancel.title = 'Cancel'; cancel.setAttribute('aria-label', 'Cancel ' + j.prompt);
      cancel.innerHTML = '<i class="fas fa-xmark"></i>';
      cancel.disabled = j.cancel_requested;
      cancel.onclick = async () => { cancel.disabled = true; await axios.post(`/api/generation/jobs/${j.id}/cancel`).catch(() => {}); refreshQueue(); };
      li.append(label, state, watch, cancel);
      els.queueList.appendChild(li);
    }
    els.queuePanel.classList.toggle('hidden', !others.length && !els.queueNote.textContent);
  }

  function setGenerateLabel(queueing) {
    els.generateBtn.innerHTML = queueing
      ? '<i class="fas fa-plus"></i> Add to queue'
      : source === 'upload'
        ? '<i class="fas fa-layer-group"></i> Separate layers'
        : '<i class="fas fa-wand-magic-sparkles"></i> Create character';
  }

  // ---- generate ----
  let submitting = false;
  async function onGenerate(e) {
    e.preventDefault();
    if (submitting) return;
    let send;
    if (source === 'upload') {
      const file = els.artwork.files[0];
      if (!file) { els.artwork.focus(); return; }
      const form = new FormData();
      form.append('image', file);
      send = () => axios.post('/api/generation/upload', form);
    } else {
      const prompt = els.prompt.value.trim();
      if (!prompt) { els.prompt.focus(); return; }
      send = briefSender(prompt);
    }

    // While a job is on the canvas, new briefs join the queue instead of replacing it.
    const queueing = busy;
    if (!queueing) { busy = true; showProgress(); }
    els.queueNote.textContent = '';

    submitting = true;
    els.generateBtn.disabled = true;
    try {
      const { data } = await send();
      if (!data.success) throw new Error(data.error || 'Generation failed');
      if (queueing) refreshQueue(); else track(data.job);
    } catch (err) {
      const msg = err.response?.data?.error || err.message || String(err);
      if (queueing) { els.queueNote.textContent = msg; refreshQueue(); }
      else { endJob(); showError(msg); }
    } finally {
      submitting = false;
      els.generateBtn.disabled = false;
    }
  }

  function briefSender(prompt) {
    const seedRaw = parseInt(els.seed.value, 10);
    const payload = {
      prompt,
      negative_prompt: els.negative.value.trim(),
      style: selectedStyle,
      framing: $('framing').value,
      model_id: els.model.value,
      sampler: els.sampler.value,
      steps: parseInt(els.steps.value, 10),
      guidance_scale: parseFloat(els.cfg.value),
      width: parseInt(els.width.value, 10),
      height: parseInt(els.height.value, 10),
      clip_skip: parseInt(els.clipSkip.value, 10),
      seed: (isNaN(seedRaw) || seedRaw < 0) ? null : seedRaw,
      loras: els.lora.value ? [{ id: els.lora.value, weight: parseFloat(els.loraWeight.value) }] : null,
    };
    if (!busy) lastSeed = payload.seed;
    return () => axios.post('/api/generation/generate', payload);
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
    setGenerateLabel(true);
    els.cancelBtn.classList.remove('hidden');
  }

  async function showResult(data) {
    const model = data.model || {};
    const meta = model.metadata || {};
    const mid = data.model_id;
    if (model.request?.seed != null) lastSeed = model.request.seed;

    els.resultImg.src = `/api/generation/preview/${mid}?t=${Date.now()}`;
    els.resultImg.onload = () => {
      els.overlay.classList.add('hidden');
      els.resultImg.classList.remove('hidden');
    };
    els.resultImg.onerror = () => { els.overlay.classList.add('hidden'); };

    els.metaStatus.textContent = model.status || 'completed';
    els.metaLayers.textContent = (meta.layers || []).length || '—';
    els.metaTime.textContent = model.generation_time ? model.generation_time.toFixed(1) + 's' : '—';
    els.metaSeed.textContent = model.request?.reference_image ? '—' : (model.request?.seed ?? lastSeed ?? 'random');
    els.resultMeta.classList.remove('hidden');

    $('livePreview').href = `/preview/${encodeURIComponent(mid)}`;
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
        card.innerHTML = `<div class="layer-thumb"><img loading="lazy" src="${l.url}?crop=1" alt="${l.name}"></div>
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

    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      // Ctrl/Cmd + Enter to generate
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        els.generateBtn.click();
      }
      // Escape to reset
      if (e.key === 'Escape' && !busy) {
        resetToForm();
      }
    });

  }
})();
