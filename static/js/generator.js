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
    sketchBtn: $('sketchBtn'), separateBtn: $('separateBtn'),
    editPanel: $('editPanel'), partChips: $('partChips'), partCanvas: $('partCanvas'), maskCanvas: $('maskCanvas'),
    paintToggle: $('paintToggle'), eraseToggle: $('eraseToggle'), brushSize: $('brushSize'),
    clearSelection: $('clearSelection'), editPrompt: $('editPrompt'), editStrength: $('editStrength'),
    editStrengthVal: $('editStrengthVal'), editBtn: $('editBtn'), editRetryBtn: $('editRetryBtn'), editNote: $('editNote'),
    extendFullBtn: $('extendFullBtn'), uncropBtn: $('uncropBtn'),
  };
  let shownModelId = null; // the result on the canvas

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
            els.prompt.value = saved.source === 'upload' ? '' : (saved.prompt || '');
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
        els.sketchBtn.classList.toggle('hidden', source === 'upload');
        setGenerateLabel(busy);
      };
    }
    els.artwork.onchange = () => {
      const file = els.artwork.files[0];
      if (els.artworkThumb.src) URL.revokeObjectURL(els.artworkThumb.src);
      $('cropStage').classList.toggle('hidden', !file);
      $('cropHint').classList.toggle('hidden', !file);
      setCrop(null);
      if (file) els.artworkThumb.src = URL.createObjectURL(file);
    };
    els.form.onsubmit = onGenerate;
    els.regenBtn.onclick = resetToForm;
    bindEditor();
    bindCrop();
    els.separateBtn.onclick = async () => {
      if (!shownModelId) return;
      els.separateBtn.disabled = true;
      try {
        const { data } = await axios.post(`/api/generation/retry/${encodeURIComponent(shownModelId)}`);
        if (busy) refreshQueue(); else track(data.job);
      } catch (err) { showError(err.response?.data?.error || 'Could not queue the separation'); }
      finally { els.separateBtn.disabled = false; }
    };
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
      const step = { image: 'Artwork · ', separate: 'Layers · ', expressions: 'Expressions · ', assemble: 'Packaging · ' }[j.stage] || '';
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
        ? '<i class="fas fa-file-import"></i> Import artwork'
        : '<i class="fas fa-wand-magic-sparkles"></i> Create character';
  }

  // ---- import crop: a box around one character of a sheet or group picture ----
  let crop = null; // [x0, y0, x1, y1] as fractions of the artwork
  function setCrop(c) {
    crop = c;
    $('cropBox').hidden = $('cropClear').hidden = !c;
    if (c) Object.assign($('cropBox').style, { left: `${c[0] * 100}%`, top: `${c[1] * 100}%`,
      width: `${(c[2] - c[0]) * 100}%`, height: `${(c[3] - c[1]) * 100}%` });
  }
  function bindCrop() {
    const stage = $('cropStage');
    const clamp = (v) => Math.min(1, Math.max(0, v));
    const at = (e) => {
      const r = els.artworkThumb.getBoundingClientRect();
      return [clamp((e.clientX - r.left) / r.width), clamp((e.clientY - r.top) / r.height)];
    };
    let start = null;
    stage.onpointerdown = (e) => { start = at(e); stage.setPointerCapture(e.pointerId); e.preventDefault(); };
    stage.onpointermove = (e) => {
      if (!start) return;
      const p = at(e);
      setCrop([Math.min(start[0], p[0]), Math.min(start[1], p[1]), Math.max(start[0], p[0]), Math.max(start[1], p[1])]);
    };
    stage.onpointerup = stage.onpointercancel = () => {
      start = null;
      if (crop && (crop[2] - crop[0] < 0.05 || crop[3] - crop[1] < 0.05)) setCrop(null); // a click, not a box
    };
    $('cropClear').onclick = () => setCrop(null);
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
      form.append('frame', $('importFrame').checked ? '1' : '0');
      form.append('plain_background', $('importWhite').checked ? '1' : '0');
      form.append('enhance', $('importEnhance').value);
      form.append('separate', $('importSeparate').checked ? '1' : '0');
      if (crop) form.append('crop', crop.map((v) => v.toFixed(4)).join(','));
      send = () => axios.post('/api/generation/upload', form);
    } else {
      const prompt = els.prompt.value.trim();
      if (!prompt) { els.prompt.focus(); return; }
      send = briefSender(prompt, e.submitter === els.sketchBtn);
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

  function briefSender(prompt, sketch) {
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
      sketch,
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
    els.metaSeed.textContent = model.request?.source === 'upload' ? '—' : (model.request?.seed ?? lastSeed ?? 'random');
    // A sketch has artwork only: offer to separate it instead of the layer downloads.
    shownModelId = mid;
    prepareEditor(mid);
    const sketch = model.status === 'sketch';
    els.separateBtn.classList.toggle('hidden', !sketch);
    for (const el of [$('livePreview'), els.dlPackage, els.dlPsd, els.dlOra]) el.classList.toggle('hidden', sketch);
    els.resultMeta.classList.remove('hidden');

    $('livePreview').href = `/preview/${encodeURIComponent(mid)}`;
    els.dlPackage.href = `/api/generation/artifact/${mid}/package`;
    els.dlPsd.href = `/api/generation/artifact/${mid}/psd`;
    els.dlOra.href = `/api/generation/artifact/${mid}/ora`;
    // Needs the painted closed eyes / second mouth shape (Gallery: Add blink & talk).
    $('dlPngtuber').classList.toggle('hidden', sketch || !model.metadata?.expressions);
    $('dlPngtuber').href = `/api/generation/artifact/${mid}/pngtuber`;

    els.generateBtn.disabled = false;
    if (sketch) els.layerPanel.classList.add('hidden'); else await loadLayers(mid);
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

  // ---- region edits: select parts and/or paint, describe, repaint into a new sketch ----
  // Mirrors src/core/regions.py PARTS (the server builds the real mask; this only highlights).
  const PART_LAYERS = {
    hair: ['front hair', 'back hair', 'side hair'],
    face: ['face', 'nose', 'mouth', 'eyebrow-l', 'eyebrow-r', 'eyewhite-l', 'eyewhite-r', 'irides-l', 'irides-r', 'eyelash-l', 'eyelash-r'],
    eyes: ['eyewhite-l', 'eyewhite-r', 'irides-l', 'irides-r', 'eyelash-l', 'eyelash-r'],
    mouth: ['mouth'],
    top: ['topwear', 'neckwear', 'handwear-l', 'handwear-r'],
    bottom: ['bottomwear', 'legwear', 'footwear'],
    accessories: ['headwear', 'earwear', 'eyewear', 'tail', 'wings', 'objects'],
  };
  const selectedParts = new Set();
  const layerImages = {};
  let editLayers = [];
  let painted = false, paintMode = false, eraseMode = false, lastEdit = null;

  function bindEditor() {
    els.editStrength.oninput = () => (els.editStrengthVal.textContent = els.editStrength.value);
    els.paintToggle.onclick = () => setPaint(!paintMode, false);
    els.eraseToggle.onclick = () => setPaint(true, !eraseMode);
    els.clearSelection.onclick = () => {
      selectedParts.clear();
      for (const c of els.partChips.children) { c.classList.remove('active'); c.setAttribute('aria-pressed', 'false'); }
      clearMask(); drawParts();
    };
    els.editBtn.onclick = () => submitEdit(false);
    els.extendFullBtn.onclick = () => submitExtend('full_body');
    els.uncropBtn.onclick = () => submitExtend('uncrop');
    els.editRetryBtn.onclick = () => submitEdit(true);
    const mask = els.maskCanvas;
    let last = null;
    const point = (e) => {
      const r = mask.getBoundingClientRect();
      return [(e.clientX - r.left) * mask.width / r.width, (e.clientY - r.top) * mask.height / r.height, mask.width / r.width];
    };
    const stroke = (a, b) => {
      const ctx = mask.getContext('2d');
      ctx.globalCompositeOperation = eraseMode ? 'destination-out' : 'source-over';
      ctx.strokeStyle = 'rgb(205,142,115)'; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      ctx.lineWidth = Number(els.brushSize.value) * b[2];
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0] + 0.01, b[1]); ctx.stroke();
      if (!eraseMode) painted = true;
    };
    mask.addEventListener('pointerdown', (e) => { if (!paintMode) return; last = point(e); stroke(last, last); mask.setPointerCapture(e.pointerId); });
    mask.addEventListener('pointermove', (e) => { if (!last) return; const p = point(e); stroke(last, p); last = p; });
    for (const ev of ['pointerup', 'pointercancel']) mask.addEventListener(ev, () => { last = null; });
  }

  function setPaint(on, erase) {
    paintMode = on; eraseMode = on && erase;
    els.maskCanvas.classList.toggle('painting', on);
    els.paintToggle.classList.toggle('active', on && !eraseMode); els.paintToggle.setAttribute('aria-pressed', String(on && !eraseMode));
    els.eraseToggle.classList.toggle('active', eraseMode); els.eraseToggle.setAttribute('aria-pressed', String(eraseMode));
  }

  function clearMask() {
    const m = els.maskCanvas;
    m.getContext('2d').clearRect(0, 0, m.width, m.height);
    painted = false;
  }

  async function prepareEditor(mid) {
    selectedParts.clear(); setPaint(false, false);
    const size = () => {
      for (const c of [els.maskCanvas, els.partCanvas]) {
        c.width = els.resultImg.naturalWidth || 1; c.height = els.resultImg.naturalHeight || 1;
        c.classList.remove('hidden');
      }
      clearMask(); drawParts();
    };
    if (els.resultImg.complete && els.resultImg.naturalWidth) size(); else els.resultImg.addEventListener('load', size, { once: true });
    els.partChips.innerHTML = '';
    try {
      const [{ data: parts }, layers] = await Promise.all([
        axios.get(`/api/generation/parts/${encodeURIComponent(mid)}`),
        axios.get(`/api/generation/layers/${encodeURIComponent(mid)}`).then((r) => r.data.layers || []).catch(() => []),
      ]);
      editLayers = layers;
      for (const part of parts.parts || []) {
        const chip = document.createElement('button');
        chip.type = 'button'; chip.className = 'chip'; chip.textContent = part.label;
        chip.setAttribute('aria-pressed', 'false');
        chip.title = part.id === 'background' ? 'Everything around the character' : '';
        chip.onclick = () => {
          const on = !selectedParts.has(part.id);
          if (on) selectedParts.add(part.id); else selectedParts.delete(part.id);
          chip.classList.toggle('active', on); chip.setAttribute('aria-pressed', String(on));
          drawParts();
        };
        els.partChips.appendChild(chip);
      }
    } catch (e) { /* editing stays possible with the brush */ }
  }

  async function drawParts() {
    const c = els.partCanvas, ctx = c.getContext('2d');
    const chosen = (l) => [...selectedParts].some((p) => (PART_LAYERS[p] || []).includes(l.name));
    const first = editLayers.findIndex(chosen);
    // Back to front: selected layers paint, every layer above them erases what it hides.
    const used = first < 0 ? [] : editLayers.slice(first);
    const images = await Promise.all(used.map((l) => layerImages[l.url] || (layerImages[l.url] = new Promise((ok) => {
      const im = new Image(); im.onload = () => ok(im); im.onerror = () => ok(null); im.src = l.url;
    }))));
    ctx.globalCompositeOperation = 'source-over';
    ctx.clearRect(0, 0, c.width, c.height);
    used.forEach((l, i) => {
      if (!images[i]) return;
      ctx.globalCompositeOperation = chosen(l) ? 'source-over' : 'destination-out';
      ctx.drawImage(images[i], 0, 0, c.width, c.height);
    });
    ctx.globalCompositeOperation = 'source-in';
    ctx.fillStyle = 'rgb(80,200,190)'; ctx.fillRect(0, 0, c.width, c.height);
  }

  async function submitExtend(mode) {
    if (!shownModelId) return;
    els.editNote.textContent = '';
    for (const b of [els.extendFullBtn, els.uncropBtn]) b.disabled = true;
    try {
      const { data } = await axios.post(`/api/generation/extend/${encodeURIComponent(shownModelId)}`,
        { mode, prompt: els.editPrompt.value.trim() });
      if (busy) { els.editNote.textContent = 'Added to the queue.'; refreshQueue(); } else track(data.job);
    } catch (err) { els.editNote.textContent = err.response?.data?.error || 'Could not queue the extension'; }
    finally { for (const b of [els.extendFullBtn, els.uncropBtn]) b.disabled = false; }
  }

  async function submitEdit(again) {
    if (!shownModelId && !(again && lastEdit)) return;
    const edit = again && lastEdit ? lastEdit : {
      source: shownModelId, parts: [...selectedParts], mask: painted ? els.maskCanvas.toDataURL('image/png') : null,
      prompt: els.editPrompt.value.trim(), strength: parseFloat(els.editStrength.value),
    };
    if (!edit.parts.length && !edit.mask) { els.editNote.textContent = 'Pick a part or paint over the area you want to change.'; return; }
    els.editNote.textContent = '';
    lastEdit = edit;
    els.editBtn.disabled = els.editRetryBtn.disabled = true;
    try {
      const { data } = await axios.post(`/api/generation/edit/${encodeURIComponent(edit.source)}`,
        { parts: edit.parts, mask: edit.mask, prompt: edit.prompt, strength: edit.strength });
      els.editRetryBtn.classList.remove('hidden');
      if (busy) { els.editNote.textContent = 'Added to the queue.'; refreshQueue(); } else track(data.job);
    } catch (err) { els.editNote.textContent = err.response?.data?.error || 'Could not queue the edit'; }
    finally { els.editBtn.disabled = els.editRetryBtn.disabled = false; }
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
