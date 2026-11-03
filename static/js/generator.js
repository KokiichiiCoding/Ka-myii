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
    fullBodyProportions: $('fullBodyProportions'), fullBodyKeepFace: $('fullBodyKeepFace'), fullBodyGuided: $('fullBodyGuided'), fullBodyPanel: $('fullBodyPanel'), fullBodyPrompt: $('fullBodyPrompt'), fullBodyOutput: $('fullBodyOutput'), fullBodyNote: $('fullBodyNote'),
    extendFullBtn: $('extendFullBtn'), uncropBtn: $('uncropBtn'), fullBodyFront: $('fullBodyFront'),
    fullBodyReference: $('fullBodyReference'), fullBodyReferenceVal: $('fullBodyReferenceVal'),
  };
  let shownModelId = null; // the result on the canvas

  let selectedStyle = 'vtuber';
  let configModel = null; // the checkpoint an empty model_id resolves to on the server
  let source = 'prompt'; // or 'upload': decompose the user's own artwork instead of generating
  let can3d = false; // the optional 3D engine is installed

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
    // Gallery "Make a variation": the same brief and settings with a new seed, no result shown.
    const briefId = new URLSearchParams(location.search).get('brief');
    if (briefId && !savedId) {
      try {
        const {data} = await axios.get(`/api/generation/status/${encodeURIComponent(briefId)}`);
        if (data.success && data.model.request && data.model.request.source !== 'upload') {
          fillForm(data.model.request);
          els.seed.value = -1; saveDraft(); els.prompt.focus();
          return;
        }
      } catch (err) { /* fall back to the saved draft */ }
    }
    if (!savedId && !els.prompt.value) restoreDraft();   // also brings back steps, sampler, toggles...
    if (savedId) {
      try {
        const {data} = await axios.get(`/api/generation/status/${encodeURIComponent(savedId)}`);
        if (data.success) {
          const saved = data.model.request;
          if (saved) fillForm(saved);
          els.canvasEmpty.classList.add('hidden');
          await showResult({model_id:savedId, model:data.model});
          // Live Preview's "Extend to full body" link opens the extension controls.
          if (location.hash === '#front') { els.fullBodyFront.checked = true; els.fullBodyFront.dispatchEvent(new Event('change')); }
          if (['#extend','#fullbody','#front'].includes(location.hash)) { els.fullBodyPanel.open = true; els.extendFullBtn.scrollIntoView({block: 'center'}); els.extendFullBtn.focus(); }
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
      try { localStorage.setItem('kamyii.studio.advanced', els.advBody.classList.contains('hidden') ? '' : '1'); } catch { /* private mode */ }
    };
    try { if (localStorage.getItem('kamyii.studio.advanced')) els.advToggle.onclick(); } catch { /* private mode */ }
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
        $('batchRow').classList.toggle('hidden', source === 'upload');
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
    // Drop or paste an image anywhere on the page to import it (designs copied from a chat or site).
    const useArtwork = (file) => {
      if (!file || !/^image\/(png|jpeg|webp)$/.test(file.type)) return false;
      const list = new DataTransfer(); list.items.add(file);
      els.sourceChips.querySelector('[data-source="upload"]').click();
      els.artwork.files = list.files; els.artwork.onchange();
      els.uploadFields.scrollIntoView({behavior: 'smooth', block: 'nearest'});
      return true;
    };
    const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes('Files');
    document.addEventListener('dragover', (e) => { if (hasFiles(e)) { e.preventDefault(); els.form.classList.add('drop-target'); } });
    document.addEventListener('dragleave', (e) => { if (!e.relatedTarget) els.form.classList.remove('drop-target'); });
    document.addEventListener('drop', (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault(); els.form.classList.remove('drop-target');
      if (!useArtwork(e.dataTransfer.files[0])) alert('Drop a PNG, JPG or WebP image.');
    });
    document.addEventListener('paste', (e) => {
      if (/^(INPUT|TEXTAREA)$/.test(e.target.tagName)) return;  // normal text paste
      const item = [...(e.clipboardData?.items || [])].find((i) => i.kind === 'file');
      if (item && useArtwork(item.getAsFile())) e.preventDefault();
    });
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
    $('revealBtn').onclick = async () => {
      if (!shownModelId) return;
      try { await axios.post(`/api/generation/reveal/${encodeURIComponent(shownModelId)}`); }
      catch (err) { els.editNote.textContent = `Could not open the folder: ${err.response?.data?.error || err.message}`; }
    };
    $('addExpressions').onclick = async () => {
      if (!shownModelId) return;
      $('addExpressions').disabled = true;
      try {
        const { data } = await axios.post(`/api/generation/expressions/${encodeURIComponent(shownModelId)}`);
        if (!data.success) throw new Error(data.error);
        if (busy) refreshQueue(); else track(data.job);
      } catch (err) { showError(err.response?.data?.error || 'Could not queue blink & talk'); }
      finally { $('addExpressions').disabled = false; }
    };
    $('make3d').onclick = async () => {
      if (!shownModelId) return;
      $('make3d').disabled = true;
      try {
        const { data } = await axios.post(`/api/generation/avatar3d/${encodeURIComponent(shownModelId)}`, {});
        if (!data.success) throw new Error(data.error);
        if (busy) refreshQueue(); else track(data.job);
      } catch (err) { showError(err.response?.data?.error || 'Could not queue the 3D avatar'); }
      finally { $('make3d').disabled = false; }
    };
    els.cancelBtn.onclick = async () => {
      if (!currentTaskId) return;
      els.cancelBtn.disabled = true;
      els.pLabel.textContent = 'Cancelling…';
      try { await axios.post(`/api/generation/jobs/${currentTaskId}/cancel`); }
      catch (e) { els.cancelBtn.disabled = false; }
    };
  }

  $('avatarKind').addEventListener('change', () => {
    const three = $('avatarKind').value === '3d';
    if (three) $('framing').value = 'full_body';
    $('framing').disabled = three;
  });

  async function loadOptions() {
    try {
      const { data } = await axios.get('/api/generation/options');
      if (!data.success) return;

      can3d = !!data.avatar3d?.available;
      $('avatarKind').options[1].disabled = !can3d;
      if (!can3d) $('avatarHint').textContent = `3D avatars need the optional 3D engine (about 10 GB): run ${data.avatar3d?.install || 'python scripts/install_3d.py'}.`;
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
      $('toggleChips').replaceChildren(...(data.toggles || []).map((t) => {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'chip';
        chip.textContent = t.label;
        chip.dataset.toggle = t.id;
        chip.setAttribute('aria-pressed', 'false');
        chip.onclick = () => chip.setAttribute('aria-pressed', String(chip.classList.toggle('active')));
        return chip;
      }));

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
      $('fullBodyNotice').textContent=data.is_dummy?'Demo mode uses placeholder artwork. Start Ka-myii in real mode to draw a detailed full body.':'Artwork takes a few minutes; preparing layers and expressions takes longer. You can watch progress or cancel in the queue.';
      $('seethroughNotice').classList.toggle('hidden', data.is_dummy || data.seethrough !== false);
      $('gpuNotice').classList.toggle('hidden', data.is_dummy || data.device === 'cuda');
    } catch (e) { /* ignore */ }
  }

  // ---- background job tracking ----
  // Generation runs as a server-side job; polling (not the socket) is the single source of truth,
  // so a reloaded or reopened tab re-attaches to the running job.
  // Each painted toggle adds about 15 s after the layers.
  function toggleCount() {
    const n = document.querySelectorAll('#toggleChips .active').length;
    $('togglesLabel').textContent = n ? `Expression toggles · ${n} selected, about ${n * 15} s` : 'Expression toggles';
  }

  // Put a saved character's brief and settings back into the form.
  function fillForm(saved) {
    if (saved.extension?.mode === 'front') {
      els.fullBodyReference.value = saved.extension.reference_strength ?? 0.65;
      els.fullBodyReference.oninput();
      els.fullBodyPrompt.value = saved.extension.outfit || '';
      els.fullBodyProportions.value = saved.extension.proportions || 'balanced';
    }
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
    for (const chip of document.querySelectorAll('#toggleChips [data-toggle]')) {
      const on = (saved.toggles || []).includes(chip.dataset.toggle);
      chip.classList.toggle('active', on); chip.setAttribute('aria-pressed', String(on));
    }
    if (saved.avatar3d && !$('avatarKind').options[1].disabled) { $('avatarKind').value = '3d'; $('avatarKind').dispatchEvent(new Event('change')); }
    toggleCount();
    updateModelHint();
  }

  // The unsent brief survives a reload or a trip to the Gallery (this browser only).
  const DRAFT_KEY = 'kamyii.studio.draft';
  // The whole brief (never the seed), shaped like a saved request so fillForm can restore it.
  function restoreDraft() {
    try {
      const draft = JSON.parse(localStorage.getItem(DRAFT_KEY) || 'null');
      if (!draft) return;
      if (draft.negative !== undefined && draft.negative_prompt === undefined) draft.negative_prompt = draft.negative;   // older drafts
      fillForm({...draft, seed: null});
    } catch { /* storage unavailable: start empty */ }
  }
  function saveDraft() {
    const draft = {
      prompt: els.prompt.value, negative_prompt: els.negative.value, style: selectedStyle, framing: $('framing').value,
      model_id: els.model.value, sampler: els.sampler.value, steps: Number(els.steps.value), guidance_scale: Number(els.cfg.value),
      width: Number(els.width.value), height: Number(els.height.value), clip_skip: Number(els.clipSkip.value),
      loras: els.lora.value ? [{id: els.lora.value, weight: Number(els.loraWeight.value)}] : null,
      toggles: [...document.querySelectorAll('#toggleChips .active')].map((c) => c.dataset.toggle),
      avatar3d: $('avatarKind').value === '3d',
    };
    try { localStorage.setItem(DRAFT_KEY, JSON.stringify(draft)); } catch { /* private mode */ }
  }

  // A random, VTuber-friendly starting brief in the tag style the checkpoints expect.
  const IDEAS = {
    who: ['1girl', '1girl', '1girl', '1boy'],
    hair: ['silver hair', 'pink hair', 'black hair', 'blonde hair', 'light blue hair', 'red hair', 'white hair', 'purple hair', 'brown hair', 'green hair', 'two-tone hair', 'gradient hair'],
    style: ['long hair, twintails', 'short hair, bob cut', 'long hair, ponytail', 'medium hair, side braid', 'very long hair, hime cut', 'short hair, messy hair', 'long hair, twin braids', 'wavy hair, side ponytail'],
    eyes: ['blue eyes', 'red eyes', 'golden eyes', 'green eyes', 'purple eyes', 'heterochromia', 'pink eyes', 'aqua eyes'],
    outfit: ['sailor uniform', 'oversized hoodie', 'maid outfit', 'witch hat, black robe', 'idol costume, frills', 'school blazer, ribbon', 'kimono, obi', 'gothic lolita dress', 'track jacket', 'sweater, turtleneck', 'cyberpunk jacket, neon trim', 'shrine maiden outfit', 'knight armor, cape', 'chef uniform', 'pajamas, sleep mask on head'],
    extra: ['cat ears', 'fox ears', 'star hair ornament', 'headphones', 'hair ribbon', 'glasses', 'halo', 'small horns', 'flower hair ornament', 'choker', 'hairclip', 'beret', '', ''],
    mood: ['gentle smile', 'confident smile', 'shy, blush', 'smug', 'cheerful, open mouth', 'sleepy', 'serious', 'playful wink'],
  };
  const pick = (list) => list[Math.floor(Math.random() * list.length)];
  function surpriseBrief() {
    return [pick(IDEAS.who), pick(IDEAS.hair), pick(IDEAS.style), pick(IDEAS.eyes), pick(IDEAS.outfit), pick(IDEAS.extra), pick(IDEAS.mood)].filter(Boolean).join(', ');
  }

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
          endJob(); flagTab('✓ Ready');
          const status = await axios.get(`/api/generation/status/${encodeURIComponent(j.model_id)}`);
          await showResult({ model_id: j.model_id, model: status.data.model });
          return;
        }
        if (j.state === 'failed') { endJob(); flagTab('✕ Failed'); showError(j.error || 'Generation failed'); return; }
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
    // The artwork is ready long before its layers: show it, so a miss can be cancelled early.
    const art = $('progressArt'), src = j.artwork ? `/api/generation/preview/${encodeURIComponent(j.artwork)}?size=thumb` : '';
    if (src && art.dataset.src !== src) { art.dataset.src = src; art.src = src; art.onload = () => art.classList.remove('hidden'); }
    if (!src) { art.classList.add('hidden'); art.dataset.src = ''; }
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

  // A finished job in a background tab keeps saying so until the user looks.
  function flagTab(text) {
    if (!document.hidden) return;
    document.title = `${text} · ${pageTitle}`;
    document.addEventListener('visibilitychange', () => { document.title = pageTitle; }, {once: true});
  }

  function endJob() {
    clearTimeout(pollTimer);
    $('progressArt').classList.add('hidden'); $('progressArt').dataset.src = '';
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
    // A batch queued by mistake: cancel every waiting job at once (the running one keeps going).
    const waiting = jobs.filter((j) => j.state === 'queued' && j.id !== currentTaskId);
    $('cancelQueued').classList.toggle('hidden', waiting.length < 2);
    $('cancelQueued').onclick = async () => {
      $('cancelQueued').disabled = true;
      for (const j of waiting) await axios.post(`/api/generation/jobs/${j.id}/cancel`).catch(() => {});
      $('cancelQueued').disabled = false;
      refreshQueue();
    };
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
    let send, extra = [];
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
      if (!prompt) {   // say what is missing instead of silently doing nothing
        els.prompt.placeholder = 'Describe your character first, or press ✦ Surprise me (↺ Recent reuses an earlier brief).';
        els.prompt.classList.remove('flash'); void els.prompt.offsetWidth; els.prompt.classList.add('flash');
        els.prompt.focus(); return;
      }
      const sketch = e.submitter === els.sketchBtn, count = Number($('batchCount').value) || 1;
      send = briefSender(prompt, sketch);
      // "How many": the rest of the batch joins the queue behind the first, one seed each.
      if (count > 1) extra = Array.from({length: count - 1}, (_, i) => briefSender(prompt, sketch, i + 1));
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
      for (const next of extra) {   // a failure here must not end the first job's progress view
        try { await next(); } catch (err) { els.queueNote.textContent = err.response?.data?.error || err.message; break; }
      }
      if (extra.length) refreshQueue();
    } catch (err) {
      const msg = err.response?.data?.error || err.message || String(err);
      if (queueing) { els.queueNote.textContent = msg; refreshQueue(); }
      else { endJob(); showError(msg); }
    } finally {
      submitting = false;
      els.generateBtn.disabled = false;
    }
  }

  function briefSender(prompt, sketch, offset = 0) {
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
      seed: (isNaN(seedRaw) || seedRaw < 0) ? null : seedRaw + offset,
      loras: els.lora.value ? [{ id: els.lora.value, weight: parseFloat(els.loraWeight.value) }] : null,
      sketch,
      toggles: [...document.querySelectorAll('#toggleChips .active')].map((c) => c.dataset.toggle),
      avatar3d: $('avatarKind').value === '3d' && !sketch ? { quality: 'standard' } : false,
    };
    if (!busy && !offset) lastSeed = payload.seed;
    return () => axios.post('/api/generation/generate', payload);
  }

  // Something to read during a ten-minute wait: what to try once the character is ready.
  const TIPS = [
    'In Live Preview, press N to nod, H to shake the head, T for a puzzled tilt, Y for a happy bounce.',
    'Live Preview keys 1–7 switch expressions; Shift+1–9 switch painted toggles like blush or tears.',
    'Turn on the camera in Live Preview and the character follows your head, blinks and talks. Nothing leaves your machine.',
    "Add blink & talk paints closed eyes and an open mouth in this character's own style (about 2 minutes).",
    'Make 3D avatar turns a full-body, front-facing character into a rigged VRM for VSeeFace or VRChat-style apps.',
    'The 3D stage has a green screen background and a Bust framing preset for OBS.',
    'Gallery → More options → Make a variation reuses this brief with a new seed.',
    'Export saves the layered PSD and the Cubism scaffold; Open exported character brings a project back.',
    'Drop or paste a picture anywhere in the Studio to start from your own artwork.',
  ];
  let tipTimer = null;
  function rotateTips() {
    clearInterval(tipTimer);
    let i = Math.floor(Math.random() * TIPS.length);
    const show = () => { $('progressTip').textContent = TIPS[i++ % TIPS.length]; };
    show(); tipTimer = setInterval(() => { if (els.overlay.classList.contains('hidden')) clearInterval(tipTimer); else show(); }, 9000);
  }

  function showProgress() {
    rotateTips();
    // The Create button sits below a long form; bring the progress into view.
    els.overlay.parentElement.scrollIntoView({behavior: 'smooth', block: 'nearest'});
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

    const name = model.display_name || (model.name?.startsWith('model_') ? `Study ${model.name.slice(6).toUpperCase()}` : (model.name || ''));
    $('resultName').value = $('resultName').dataset.saved = name;
    els.metaStatus.textContent = model.status || 'completed';
    els.metaLayers.textContent = (meta.layers || []).length || '—';
    els.metaTime.textContent = model.generation_time ? model.generation_time.toFixed(1) + 's' : '—';
    els.metaSeed.textContent = model.request?.source === 'upload' ? '—' : (model.request?.seed ?? lastSeed ?? 'random');
    // A sketch has artwork only: offer to separate it instead of the layer downloads.
    shownModelId = mid;
    extensionGuideRequest++; fitGuides = null; fitView = null;
    if ($('fullBodyFit').open) drawExtensionGuide();
    // Reload/bookmark the result now on the canvas, including a newly extended character.
    const resultUrl = new URL(location.href);
    resultUrl.searchParams.set('model', mid);
    history.replaceState(null, '', resultUrl.pathname + resultUrl.search + resultUrl.hash);
    prepareEditor(mid);
    renderChecks(model, mid);
    const referenceId = model.request?.extension?.source_model;
    const hasReference = model.request?.extension?.mode === 'front' && typeof referenceId === 'string' && /^[\w-]+$/.test(referenceId);
    $('redrawComparison').classList.toggle('hidden', !hasReference);
    if (hasReference) {
      $('redrawSourceImage').src = `/api/generation/preview/${encodeURIComponent(referenceId)}`;
      $('redrawResultImage').src = els.resultImg.src;
      $('redrawSourceLink').href = `/generator?model=${encodeURIComponent(referenceId)}`;
    } else {
      for (const id of ['redrawSourceImage', 'redrawResultImage']) $(id).removeAttribute('src');
      $('redrawSourceLink').removeAttribute('href');
    }
    const sketch = model.status === 'sketch';
    els.separateBtn.classList.toggle('hidden', !sketch);
    for (const el of [$('livePreview'), els.dlPackage, els.dlPsd, els.dlOra]) el.classList.toggle('hidden', sketch);
    els.resultMeta.classList.remove('hidden');

    $('livePreview').href = `/preview/${encodeURIComponent(mid)}`;
    $('open3d').classList.toggle('hidden', !meta.avatar3d);
    $('open3d').href = `/3d?model=${encodeURIComponent(mid)}`;
    $('avatar3dNote').classList.toggle('hidden', !(meta.avatar3d_error && !meta.avatar3d));
    $('avatar3dNote').textContent = `The 3D avatar could not be built: ${meta.avatar3d_error || ''}. The 2D character is complete; Make 3D avatar tries again.`;
    $('make3d').classList.toggle('hidden', !can3d || !!meta.avatar3d || !['completed', 'sketch'].includes(model.status));
    els.dlPackage.href = `/api/generation/artifact/${mid}/package`;
    els.dlPsd.href = `/api/generation/artifact/${mid}/psd`;
    els.dlOra.href = `/api/generation/artifact/${mid}/ora`;
    const separated = model.status === 'completed' && (model.assets || []).some((a) => a.metadata?.backend === 'see-through');
    $('addExpressions').classList.toggle('hidden', !separated || !!model.metadata?.expressions);
    // Needs the painted closed eyes / second mouth shape (Add blink & talk).
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
    els.fullBodyReference.oninput = () => (els.fullBodyReferenceVal.textContent = `${Math.round(Number(els.fullBodyReference.value) * 100)}%`);
    els.paintToggle.onclick = () => setPaint(!paintMode, false);
    els.eraseToggle.onclick = () => setPaint(true, !eraseMode);
    els.clearSelection.onclick = () => {
      selectedParts.clear();
      for (const c of els.partChips.children) { c.classList.remove('active'); c.setAttribute('aria-pressed', 'false'); }
      clearMask(); drawParts();
    };
    els.editBtn.onclick = () => submitEdit(false);
    els.extendFullBtn.onclick = () => submitExtend(els.fullBodyFront.checked ? 'front' : 'full_body');
    // A front redraw keeps nothing of the old picture: face preservation and the fit guide don't apply.
    els.fullBodyFront.addEventListener('change', () => {
      const front = els.fullBodyFront.checked;
      els.fullBodyGuided.disabled = els.fullBodyKeepFace.disabled = front;
      $('fullBodyFit').hidden = front;
      $('frontReferenceControls').hidden = !front;
      els.extendFullBtn.textContent = front ? 'Redraw facing forward' : 'Generate full body';
    });
    $('fullBodyFit').addEventListener('toggle', () => { if ($('fullBodyFit').open) drawExtensionGuide(); });
    els.fullBodyProportions.addEventListener('change', () => { fitGuides = null; if ($('fullBodyFit').open) drawExtensionGuide(); });
    $('fitReset').onclick = () => { fitGuides = null; drawExtensionGuide(); };
    // Drag a shoulder, hip or ankle dot; its mirror moves with it, so pairs stay level.
    const guide = $('fullBodyGuide'); let dragging = null;
    const at = (e) => { const r = guide.getBoundingClientRect(); return [(e.clientX - r.left) * guide.width / r.width, (e.clientY - r.top) * guide.height / r.height]; };
    const handleAt = (e) => {
      if (!fitView?.g.pose_points) return null;
      const s = guide.height / fitView.g.canvas[1], [x, y] = at(e);
      for (const [name, pair] of Object.entries(GUIDE_POINTS))
        for (const i of pair) { const [px, py] = fitView.g.pose_points[i]; if (Math.hypot(px * s - x, py * s - y) < 14) return name; }
      return null;
    };
    guide.addEventListener('pointerdown', (e) => {
      dragging = handleAt(e);
      if (dragging) { fitGuides = currentGuides(); guide.setPointerCapture(e.pointerId); e.preventDefault(); }
    });
    guide.addEventListener('pointermove', (e) => {
      if (!dragging) { guide.style.cursor = handleAt(e) ? 'grab' : ''; return; }
      const g = fitView.g, s = guide.height / g.canvas[1], [x, y] = at(e).map((v) => v / s), axis = g.pose_points[1][0];
      const order = GUIDE_NAMES.indexOf(dragging), above = fitGuides[GUIDE_NAMES[order - 1]], below = fitGuides[GUIDE_NAMES[order + 1]];
      const half = Math.min(Math.max(Math.abs(x - axis), 8), g.canvas[0] / 2 - 2);
      const yy = Math.min(Math.max(y, above ? above[1] + 12 : 4), below ? below[1] - 12 : g.canvas[1] * .97);
      fitGuides[dragging] = [half, yy];
      const [a, b] = GUIDE_POINTS[dragging];
      g.pose_points[a] = [axis - half, yy]; g.pose_points[b] = [axis + half, yy];
      guide.style.cursor = 'grabbing'; paintFitGuide();
    });
    for (const ev of ['pointerup', 'pointercancel']) guide.addEventListener(ev, () => { if (dragging) { dragging = null; drawExtensionGuide(); } });
    $('showFullBody').onclick = () => { els.fullBodyPanel.open = true; els.fullBodyPanel.scrollIntoView({block:'center',behavior:'smooth'}); els.fullBodyPrompt.focus(); };
    els.uncropBtn.onclick = () => submitExtend('uncrop');
    for (const [id, target] of [['refineHandsBtn', 'hands'], ['refineFaceBtn', 'face']]) $(id).onclick = async () => {
      if (!shownModelId) return;
      els.editNote.textContent = ''; $(id).disabled = true;
      try {
        const { data } = await axios.post(`/api/generation/refine/${encodeURIComponent(shownModelId)}`, { target });
        if (busy) { els.editNote.textContent = 'Added to the queue.'; refreshQueue(); } else track(data.job);
      } catch (err) { els.editNote.textContent = err.response?.data?.error || 'Could not queue the refinement'; }
      finally { $(id).disabled = false; }
    };
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
    // The edit panel sits below a tall canvas; bring the picture back into view to paint on it.
    if (on) els.maskCanvas.scrollIntoView({block: 'nearest'});
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

  let extensionGuideRequest = 0, fitGuides = null, fitView = null;
  // OpenPose joints the owner can drag (outpaint.pose_points): [left, right] of each pair.
  const GUIDE_POINTS = { shoulders: [2, 5], hips: [8, 11], ankles: [10, 13] }, GUIDE_NAMES = Object.keys(GUIDE_POINTS);
  function currentGuides() {
    const p = fitView.g.pose_points, out = {};
    for (const [name, [a, b]] of Object.entries(GUIDE_POINTS)) out[name] = [(p[b][0] - p[a][0]) / 2, p[a][1]];
    return out;
  }
  async function drawExtensionGuide() {
    if (!shownModelId) return;
    const revision = ++extensionGuideRequest, mid = shownModelId;
    const note = $('fullBodyFitNote'); if (!fitView) note.textContent = 'Measuring the head and standing axis…';
    try {
      const params = { proportions: els.fullBodyProportions.value };
      if (fitGuides) params.guides = JSON.stringify(fitGuides);
      const { data } = await axios.get(`/api/generation/extend/${encodeURIComponent(mid)}/plan`, { params });
      if (revision !== extensionGuideRequest || mid !== shownModelId) return;
      let source = fitView?.mid === mid ? fitView.source : null;
      if (!source) {
        source = new Image();
        source.src = `/api/generation/preview/${encodeURIComponent(mid)}`;
        await new Promise((resolve, reject) => { source.onload = resolve; source.onerror = reject; });
        if (revision !== extensionGuideRequest || mid !== shownModelId) return;
      }
      fitView = { mid, g: data.info, source };
      paintFitGuide();
      $('fitReset').hidden = !fitGuides;
      note.textContent = (data.info.measured
        ? 'Measured from this character. The face is retained; the outfit blends into a newly drawn body.'
        : 'Approximate fit: separate the current artwork first for a precise head measurement.')
        + (data.info.pose_points ? ' Drag the shoulder, hip and ankle dots to set the stance; pairs stay level.' : '') + ' Guide only—generated anatomy can vary.';
    } catch (err) { if (revision===extensionGuideRequest) note.textContent = err.response?.data?.error || 'Could not load the fit guide. You can still generate a full body.'; }
  }

  function paintFitGuide() {
    const { g, source } = fitView, c = $('fullBodyGuide'), ctx = c.getContext('2d');
    {
      const s = c.height / g.canvas[1], [x,y,w,h] = g.placed;
      ctx.clearRect(0,0,c.width,c.height); ctx.fillStyle='#f2eee6';ctx.fillRect(0,0,c.width,c.height);
      ctx.drawImage(source,x*s,y*s,w*s,h*s);
      // The same shoulder/hip/ankle targets used by generation, shown before spending GPU time.
      if(g.pose_points){
        const links=[[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13]];
        ctx.strokeStyle='#bc9d87';ctx.lineWidth=2;
        for(const [a,b] of links){ctx.beginPath();ctx.moveTo(...g.pose_points[a].map(v=>v*s));ctx.lineTo(...g.pose_points[b].map(v=>v*s));ctx.stroke();}
      }
      ctx.strokeStyle='#9b735d';ctx.lineWidth=1;ctx.setLineDash([4,4]);ctx.beginPath();
      ctx.moveTo(g.standing_axis*s,0);ctx.lineTo(g.standing_axis*s,g.foot_y*s);
      ctx.moveTo(20,g.foot_y*s);ctx.lineTo(c.width-20,g.foot_y*s);ctx.stroke();ctx.setLineDash([]);
      if (g.result_geometry) {
        const a=g.result_geometry,hh=a.head_height*s,crown=a.crown_y*s;
        ctx.strokeStyle='#9b735d';ctx.font='11px sans-serif';ctx.fillStyle='#624b3e';
        for(let i=0;i<=Math.floor(g.target_heads);i++) {const yy=crown+hh*i;
          ctx.beginPath();ctx.moveTo(12,yy);ctx.lineTo(26,yy);ctx.stroke();ctx.fillText(String(i),30,yy+4);}
      }
      if (g.pose_points) for (const pair of Object.values(GUIDE_POINTS)) for (const i of pair) {
        const [px, py] = g.pose_points[i];
        ctx.beginPath(); ctx.arc(px * s, py * s, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#d0674f'; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
      }
    }
  }

  // Automatic checks of a drawn full body (src/core/checks.py), each with a one-click fix.
  const CHECK_FIXES = { reseed: 'Try another seed', uncrop: 'Make room', repaint: 'Repaint this area', refine_hands: 'Refine hands' };
  function renderChecks(model, mid) {
    const box = $('resultChecks'), found = model.metadata?.checks, ext = model.request?.extension || {};
    box.innerHTML = '';
    box.classList.toggle('hidden', !Array.isArray(found));
    if (!Array.isArray(found)) return;
    const title = document.createElement('div');
    title.textContent = found.length ? 'Automatic checks found:' : 'Automatic checks: no second face, cropped shoes or uneven shoulders found.';
    box.appendChild(title);
    for (const f of found) {
      const row = document.createElement('div'); row.className = 'check-row';
      const text = document.createElement('span'); text.textContent = f.message; row.appendChild(text);
      if (CHECK_FIXES[f.fix] && (f.fix !== 'reseed' || ext.source_model)) {
        const b = document.createElement('button');
        b.type = 'button'; b.className = 'btn'; b.textContent = CHECK_FIXES[f.fix];
        b.onclick = () => fixCheck(f, mid, ext, b);
        row.appendChild(b);
      }
      box.appendChild(row);
    }
    const note = document.createElement('div'); note.className = 'edit-note'; note.id = 'checkNote'; box.appendChild(note);
  }

  async function fixCheck(f, mid, ext, button) {
    const id = encodeURIComponent(mid);
    button.disabled = true; $('checkNote').textContent = '';
    try {
      let res;
      if (f.fix === 'reseed') res = await axios.post(`/api/generation/extend/${encodeURIComponent(ext.source_model)}`, {
        mode: ext.mode === 'front' ? 'front' : 'full_body', prompt: ext.outfit || '', proportions: ext.proportions || 'balanced',
        keep_face: ext.keep_face !== false, guided: !!ext.guided, guides: ext.guides || null,
        reference_strength: ext.reference_strength ?? 0.65, separate: false });
      else if (f.fix === 'uncrop') res = await axios.post(`/api/generation/extend/${id}`, { mode: 'uncrop' });
      else if (f.fix === 'repaint') res = await axios.post(`/api/generation/edit/${id}`, { parts: [], mask: boxMask(f.box), prompt: f.prompt || '', strength: 0.65 });
      else res = await axios.post(`/api/generation/refine/${id}`, { target: 'hands' });
      if (busy) { $('checkNote').textContent = 'Added to the queue.'; refreshQueue(); } else track(res.data.job);
    } catch (err) { $('checkNote').textContent = err.response?.data?.error || 'Could not queue the fix'; }
    finally { button.disabled = false; }
  }

  function boxMask([x0, y0, x1, y1]) {
    const c = document.createElement('canvas'), ctx = c.getContext('2d');
    c.width = els.resultImg.naturalWidth; c.height = els.resultImg.naturalHeight;
    ctx.fillStyle = '#fff'; ctx.beginPath();
    ctx.ellipse((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2, 0, 0, Math.PI * 2); ctx.fill();
    return c.toDataURL('image/png');
  }

  async function submitExtend(mode) {
    if (!shownModelId) return;
    const body = mode !== 'uncrop';
    const note = body?els.fullBodyNote:els.editNote;
    note.textContent = '';
    for (const b of [els.extendFullBtn, els.uncropBtn]) b.disabled = true;
    try {
      const { data } = await axios.post(`/api/generation/extend/${encodeURIComponent(shownModelId)}`,
        { mode, prompt: (body?els.fullBodyPrompt:els.editPrompt).value.trim(),
          proportions: els.fullBodyProportions.value, keep_face: els.fullBodyKeepFace.checked, guided: els.fullBodyGuided.checked,
          guides: mode === 'full_body' ? fitGuides : null,
          reference_strength: mode === 'front' ? Number(els.fullBodyReference.value) : undefined,
          separate: body && els.fullBodyOutput.value==='preview' });
      if (busy) { note.textContent = 'Added to the queue.'; refreshQueue(); } else track(data.job);
    } catch (err) { note.textContent = err.response?.data?.error || 'Could not queue the extension'; }
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
    // The raw CUDA message means nothing to most people.
    if (/out of memory|CUDA_ERROR_OUT_OF_MEMORY|CUBLAS_STATUS_ALLOC_FAILED/i.test(msg)) {
      msg = 'The graphics card ran out of memory. Close games, video apps or other AI tools using the GPU, '
        + 'then try again. If the artwork was already drawn, it is in the Gallery with a Retry layers button.';
    }
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

    // Rename the character on the canvas in place (same endpoint as the Gallery).
    $('resultName').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') e.target.blur();
      if (e.key === 'Escape') { e.target.value = e.target.dataset.saved; e.target.blur(); }
    });
    $('resultName').addEventListener('change', async (e) => {
      const input = e.target, wanted = input.value.trim();
      if (!shownModelId || !wanted || wanted === input.dataset.saved) { input.value = input.dataset.saved; return; }
      try {
        const { data } = await axios.post(`/api/generation/rename/${encodeURIComponent(shownModelId)}`, { name: wanted });
        if (!data.success) throw new Error(data.error);
        input.value = input.dataset.saved = data.display_name;
        input.classList.remove('saved'); void input.offsetWidth; input.classList.add('saved');
      } catch (err) { input.value = input.dataset.saved; }
    });
    els.prompt.addEventListener('input', saveDraft);
    els.negative.addEventListener('input', saveDraft);
    // Rough totals from measured runs on the 8 GB laptop: a sketch ~1 min, layers ~11 min, 3D +4 min.
    const batchHint = () => {
      const n = Number($('batchCount').value) || 1, full = $('avatarKind').value === '3d' ? 15 : 11;
      $('batchHint').textContent = n === 1 ? 'Pick the best, then separate it.' : `About ${n} min as sketches, or ${n * full} min as finished characters.`;
    };
    $('batchCount').addEventListener('change', batchHint);
    $('avatarKind').addEventListener('change', batchHint);
    // Settings and chips too, so the next visit starts from the same checkpoint, steps and look.
    els.form.addEventListener('change', (e) => { if (e.target.id !== 'seed' && e.target.id !== 'artwork') saveDraft(); });
    els.form.addEventListener('click', (e) => {
      if (e.target.closest('#toggleChips .chip')) toggleCount();
      if (e.target.closest('#styleChips .chip, #toggleChips .chip')) saveDraft();
    });
    // The whole brief is remembered, so offer a way back to an empty one with default settings.
    $('clearBrief').addEventListener('click', () => {
      if (els.prompt.value.trim() && !confirm('Clear the brief and go back to the default settings?')) return;
      try { localStorage.removeItem(DRAFT_KEY); } catch { /* private mode */ }
      location.href = '/generator';
    });
    // Recent briefs: the last distinct prompts from the collection, newest first.
    const closeRecent = () => { $('recentList').classList.add('hidden'); $('recentBtn').setAttribute('aria-expanded', 'false'); };
    $('recentBtn').addEventListener('click', async () => {
      if (!$('recentList').classList.contains('hidden')) { closeRecent(); return; }
      let models = [];
      try { models = (await axios.get('/api/generation/list')).data.models || []; } catch (e) { /* offline: empty list */ }
      const seen = new Set(), recent = [];
      for (const m of models) {
        const p = m.request?.source === 'upload' ? '' : (m.request?.prompt || '').trim();
        if (p && !seen.has(p) && recent.length < 8) { seen.add(p); recent.push(m.request); }
      }
      $('recentList').replaceChildren(...(recent.length ? recent.map((r) => {
        const item = document.createElement('li'); item.setAttribute('role', 'option'); item.tabIndex = 0; item.textContent = item.title = r.prompt;
        const pick = () => { els.prompt.value = r.prompt; els.negative.value = r.negative_prompt || ''; saveDraft(); closeRecent(); els.prompt.focus(); };
        item.addEventListener('click', pick);
        item.addEventListener('keydown', (e) => { if (e.key === 'Enter') pick(); if (e.key === 'Escape') closeRecent(); });
        return item;
      }) : [Object.assign(document.createElement('li'), {textContent: 'No briefs yet: your characters\' briefs appear here.', className: 'muted'})]));
      $('recentList').classList.remove('hidden'); $('recentBtn').setAttribute('aria-expanded', 'true');
      $('recentList').querySelector('li')?.focus();
    });
    document.addEventListener('click', (e) => { if (!e.target.closest('#recentList, #recentBtn')) closeRecent(); });
    $('surpriseBtn').addEventListener('click', () => {
      els.prompt.value = surpriseBrief(); saveDraft();
      els.prompt.classList.remove('flash'); void els.prompt.offsetWidth; els.prompt.classList.add('flash');
      els.prompt.focus();
    });

    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      // Ctrl/Cmd + Enter to generate
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        els.generateBtn.click();
      }
      // Escape to reset (not while typing, nor when it is closing a dialog such as the shortcuts list)
      if (e.key === 'Escape' && !busy && !document.querySelector('dialog[open]') && !/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) {
        resetToForm();
      }
    });

  }
})();
