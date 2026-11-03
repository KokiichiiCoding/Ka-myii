/* The collection shares the same saved models and preview adapter as Studio. */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  let models = [];
  let shownModels = [], restoredScroll = false;
  const separating = new Set();   // sketches queued for separation on this page (no double queueing)   // what the grid shows now, in order (quick look flips through it)
  // Per-browser view preferences; the page works the same without storage.
  const saved = (() => { try { return JSON.parse(localStorage.getItem('kamyii.gallery') || '{}'); } catch { return {}; } })();
  let filter = saved.filter || 'all';
  const FILTERS = {
    all: () => true,
    favorite: m => !!m.favorite,
    ready: m => m.status === 'completed',
    sketch: m => m.status === 'sketch',
    '3d': m => !!m.metadata?.avatar3d,
    failed: m => m.status === 'failed' || !!(m.metadata?.avatar3d_error && !m.metadata?.avatar3d),
  };
  const SORTS = {
    newest: (a, b) => String(b.created_at).localeCompare(String(a.created_at)),
    oldest: (a, b) => String(a.created_at).localeCompare(String(b.created_at)),
    name: (a, b) => displayName(a).localeCompare(displayName(b), undefined, {numeric: true, sensitivity: 'base'}),
  };
  function remember() { try { localStorage.setItem('kamyii.gallery', JSON.stringify({filter, sort: $('sortSelect').value})); } catch { /* private mode */ } }
  // "3 days ago" reads faster than a date; the exact time stays in the tooltip.
  const when = (iso) => {
    const date = new Date(iso), days = Math.floor((Date.now() - date) / 864e5);
    const text = days < 1 ? 'today' : days === 1 ? 'yesterday' : days < 14 ? `${days} days ago` : date.toLocaleDateString();
    return Object.assign(document.createElement('time'), {textContent: text, dateTime: iso, title: date.toLocaleString()});
  };
  const displayName = (m) => m.display_name || (m.name?.startsWith('model_') ? `Study ${m.name.slice(6).toUpperCase()}` : (m.name || `Character ${m.id.slice(0,8)}`));
  function render() {
    const query = $('searchInput').value.trim().toLowerCase();
    const shown = models.filter(m => FILTERS[filter](m) && `${displayName(m)} ${m.name} ${m.id} ${m.status} ${m.request?.prompt || ''} ${m.notes || ''}`.toLowerCase().includes(query))
      .sort(SORTS[$('sortSelect').value] || SORTS.newest);
    for (const chip of $('filterChips').children) {
      const on = chip.dataset.filter === filter, count = models.filter(FILTERS[chip.dataset.filter]).length;
      chip.classList.toggle('active', on); chip.setAttribute('aria-pressed', String(on));
      chip.dataset.count = count; chip.hidden = !count && chip.dataset.filter !== 'all' && !on;
    }
    shownModels = shown;
    // Starred sketches can all be separated at once (pick the keepers from a batch, then go).
    const starredSketches = filter === 'favorite' ? shown.filter(m => m.status === 'sketch' && !separating.has(m.id)) : [];
    if (filter !== 'favorite') $('bulkNote').textContent = '';
    $('bulkSeparate').hidden = !starredSketches.length;
    $('bulkRow').hidden = !starredSketches.length && !$('bulkNote').textContent;
    $('bulkSeparate').textContent = `Separate ${starredSketches.length} starred sketch${starredSketches.length === 1 ? '' : 'es'}`;
    $('bulkSeparate').onclick = async () => {
      $('bulkSeparate').disabled = true;
      let queued = 0;
      for (const m of starredSketches) {
        try { if ((await axios.post(`/api/generation/retry/${encodeURIComponent(m.id)}`)).data.success) { queued++; separating.add(m.id); } } catch (e) { /* counted below */ }
      }
      $('bulkSeparate').disabled = false;
      $('bulkNote').textContent = `Queued ${queued} of ${starredSketches.length}. Each becomes a new character when its layers are ready.`;
      render();
    };
    const grid = $('galleryGrid');
    grid.replaceChildren();
    $('collectionCount').textContent = `${shown.length} of ${models.length} characters`;
    $('emptyGallery').hidden = models.length > 0;
    grid.hidden = !models.length;
    for (const model of shown) {
      const id = encodeURIComponent(model.id);
      const card = document.createElement('article');
      card.className = 'model-card';
      const art = document.createElement('a');
      art.className = 'model-art'; art.href = `/generator?model=${id}`;
      const image = document.createElement('img');
      image.src = `/api/generation/preview/${id}?size=thumb`;
      image.alt = model.name || 'Character artwork'; image.loading = 'lazy';
      if (model.request?.prompt) art.title = model.request.prompt;
      art.append(image);
      if (model.metadata?.avatar3d) { const tag = document.createElement('span'); tag.className = 'art-tag'; tag.textContent = '3D'; tag.title = 'Has a rigged 3D avatar'; art.append(tag); }
      if (model.metadata?.expressions) { const tag = document.createElement('span'); tag.className = 'art-tag art-tag-2'; tag.textContent = 'Blink & talk'; art.append(tag); }
      const body = document.createElement('div'); body.className = 'model-card-body';
      const status = document.createElement('span'); status.className = 'eyebrow';
      status.textContent = model.status === 'completed' ? 'CHARACTER / READY' : `CHARACTER / ${(model.status || 'unknown').toUpperCase()}`;
      const name = document.createElement('h2'); name.textContent = displayName(model);
      // A star kept with the character (metadata.json), so favourites survive export and reopen.
      const star = document.createElement('button'); star.type = 'button'; star.className = 'star-btn';
      const paintStar = () => { star.textContent = model.favorite ? '★' : '☆'; star.setAttribute('aria-pressed', String(!!model.favorite));
        star.setAttribute('aria-label', model.favorite ? 'Remove from favorites' : 'Add to favorites'); star.title = star.getAttribute('aria-label'); };
      paintStar();
      star.addEventListener('click', async () => {
        const wanted = !model.favorite;
        try {
          const {data} = await axios.post(`/api/generation/favorite/${id}`, {favorite: wanted});
          if (!data.success) throw new Error(data.error);
          model.favorite = wanted; paintStar();
          if (filter === 'favorite' && !wanted) render(); else for (const chip of $('filterChips').children) if (chip.dataset.filter === 'favorite') { chip.dataset.count = models.filter(FILTERS.favorite).length; chip.hidden = chip.dataset.count === '0'; }
        } catch (error) { $('galleryError').textContent = 'Could not update favorites.'; $('galleryError').hidden = false; }
      });

      const details = document.createElement('p'); details.className = 'card-meta';
      details.textContent = `${model.assets?.filter(a=>a.layer_type!=='character').length || 0} layers · `;
      if (model.status === 'failed') details.textContent = `Failed: ${model.error_message || 'unknown error'} · `;
      if (model.status === 'sketch') details.textContent = 'Sketch · ';
      details.append(when(model.created_at));
      const note = document.createElement('p'); note.className = 'card-note'; note.textContent = model.notes || ''; note.hidden = !model.notes;
      const actions = document.createElement('div'); actions.className = 'actions';
      const links = [[`/generator?model=${id}`, 'Artwork & layers', 'btn'], [`/api/generation/download/${id}`, 'Export', 'btn']];
      if (model.status === 'completed' && model.assets?.length) links.unshift([`/preview/${id}`, 'Live Preview ↗', 'btn btn-primary']);
      for (const [href,label,cls] of links) {
        const link = document.createElement('a'); link.href=href;link.textContent=label;link.className=cls;actions.append(link);
      }
      if (model.status === 'failed' || model.status === 'sketch') {
        // Separate the saved artwork as a new character (a sketch, or after running out of memory).
        const retry = document.createElement('button'); retry.type = 'button'; retry.className = 'btn btn-primary';
        retry.textContent = model.status === 'sketch' ? 'Separate layers' : 'Retry layers';
        retry.addEventListener('click', async () => {
          retry.disabled = true;
          try {
            const {data} = await axios.post(`/api/generation/retry/${id}`);
            if (!data.success) throw new Error(data.error);
            location.href = '/generator';  // the Studio re-attaches to the queued job
          } catch (error) {
            $('galleryError').textContent = 'Could not queue a retry. Is the server still running?';
            $('galleryError').hidden = false; retry.disabled = false;
          }
        });
        actions.prepend(retry);
      }
      // Closed eyes and the other mouth shape painted from the character's own art (~2 min, GPU).
      const seethrough = model.status === 'completed' && model.assets?.some(a => a.metadata?.backend === 'see-through');
      const queueExpressions = async (button) => {
        button.disabled = true;
        try {
          const {data} = await axios.post(`/api/generation/expressions/${id}`);
          if (!data.success) throw new Error(data.error);
          location.href = `/generator`;  // the Studio follows the queued job
        } catch (error) {
          $('galleryError').textContent = `Could not queue expressions: ${error.response?.data?.error || error.message}`;
          $('galleryError').hidden = false; button.disabled = false;
        }
      };
      // A rigged 3D VRM avatar built from the artwork (Pixal3D + Ka-myii's rig, ~4 min on the GPU).
      const avatar = model.metadata?.avatar3d;
      const queue3d = async (button, reuseMesh = false) => {
        button.disabled = true;
        try {
          const {data} = await axios.post(`/api/generation/avatar3d/${id}`, reuseMesh ? {reuse_mesh: true} : {});
          if (!data.success) throw new Error(data.error);
          location.href = `/generator?model=${encodeURIComponent(id)}`;  // fast jobs still reopen their result
        } catch (error) {
          $('galleryError').textContent = `Could not queue the 3D avatar: ${error.response?.data?.error || error.message}`;
          $('galleryError').hidden = false; button.disabled = false;
        }
      };
      if (avatar) {
        const open = document.createElement('a'); open.href = `/3d?model=${id}`; open.className = 'btn';
        open.textContent = '3D avatar ↗'; actions.append(open);
        if (avatar.rig_outdated) {
          const update = document.createElement('button'); update.type = 'button'; update.className = 'btn btn-update';
          update.textContent = 'Update 3D rig ✦';
          update.title = 'New since this avatar was made: Joy, Angry, Sorrow and Fun expressions for VRM apps, and closed eyes that fully cover the open ones. Uses the saved mesh: about 30 s, no GPU.';
          update.addEventListener('click', () => queue3d(update, true));
          actions.append(update);
        }
      } else if (['completed', 'sketch'].includes(model.status)) {
        const make = document.createElement('button'); make.type = 'button'; make.className = 'btn';
        make.textContent = 'Make 3D avatar';
        make.title = 'Build a rigged VRM from this artwork (~4 min on the GPU). Full-body, front-facing art works best; separated characters also blink and talk.';
        make.addEventListener('click', () => queue3d(make));
        actions.append(make);
      }
      if (model.metadata?.avatar3d_error && !avatar) details.append(` · 3D failed: ${model.metadata.avatar3d_error}`);
      if (seethrough && !model.metadata?.expressions) {
        const add = document.createElement('button'); add.type = 'button'; add.className = 'btn';
        add.textContent = 'Add blink & talk'; add.title = 'Paint closed eyes and an open mouth from this character (~2 min)';
        add.addEventListener('click', () => queueExpressions(add));
        actions.append(add);
      }
      const more = document.createElement('details'); more.className = 'model-more';
      const summary = document.createElement('summary'); summary.textContent = 'More options';
      const remove = document.createElement('button'); remove.type='button'; remove.textContent='Delete character';
      remove.addEventListener('click', async () => {
        remove.disabled=true;
        try {
          const {data}=await axios.delete(`/api/generation/delete/${id}`);
          if (!data.success) throw new Error(data.error);
          models=models.filter(m=>m.id!==model.id); render();
          undoToast(model, name.textContent);
        } catch (error) {
          $('galleryError').textContent='Could not delete the character. Try refreshing the collection.';
          $('galleryError').hidden=false; remove.disabled=false;
        }
      });
      // A note kept with the character (what it is for, which outfit...); searchable.
      const notes = document.createElement('button'); notes.type = 'button'; notes.className = 'more-plain'; notes.textContent = 'Notes';
      notes.addEventListener('click', () => {
        const box = document.createElement('textarea'); box.className = 'note-input'; box.maxLength = 500; box.rows = 3;
        box.value = model.notes || ''; box.placeholder = 'What is this character for? (Ctrl+Enter saves, Esc cancels)';
        box.setAttribute('aria-label', 'Notes'); note.hidden = true; note.after(box); box.focus();
        let done = false;
        const finish = async (save) => {
          if (done) return; done = true;
          if (save && box.value.trim() !== (model.notes || '')) {
            try {
              const {data} = await axios.post(`/api/generation/notes/${id}`, {notes: box.value});
              if (data.success) model.notes = data.notes || undefined;
            } catch (error) { $('galleryError').textContent = 'Could not save the notes.'; $('galleryError').hidden = false; }
          }
          render();
        };
        box.addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) finish(true); if (e.key === 'Escape') finish(false); });
        box.addEventListener('blur', () => finish(true));
      });
      const rename = document.createElement('button'); rename.type = 'button'; rename.className = 'more-plain'; rename.textContent = 'Rename';
      rename.addEventListener('click', () => {
        const input = document.createElement('input'); input.type = 'text'; input.maxLength = 60;
        input.value = displayName(model); input.className = 'rename-input'; input.setAttribute('aria-label', 'Character name');
        name.replaceWith(input); input.focus(); input.select();
        let done = false;
        const finish = async (save) => {
          if (done) return; done = true;
          if (save && input.value.trim() !== displayName(model)) {
            try {
              const {data} = await axios.post(`/api/generation/rename/${id}`, {name: input.value});
              if (data.success) model.display_name = data.display_name;
            } catch (error) { $('galleryError').textContent = 'Could not rename the character.'; $('galleryError').hidden = false; }
          }
          render();
        };
        input.addEventListener('keydown', (e) => { if (e.key === 'Enter') finish(true); if (e.key === 'Escape') finish(false); });
        input.addEventListener('blur', () => finish(true));
      });
      // A dynamic pose (side-on, mid-action) redrawn as a standing, front-facing VTuber in the Studio.
      const front = document.createElement('a'); front.href = `/generator?model=${id}#front`; front.textContent = 'Redraw facing forward';
      more.append(summary,rename,notes);
      if (model.request?.prompt && model.request.source !== 'upload') {
        // The same brief and settings with a new seed: a sibling of a character you like.
        const variation = document.createElement('a'); variation.href = `/generator?brief=${id}`; variation.textContent = 'Make a variation';
        const copy = document.createElement('button'); copy.type = 'button'; copy.className = 'more-plain'; copy.textContent = 'Copy prompt';
        copy.addEventListener('click', async () => {
          try { await navigator.clipboard.writeText(model.request.prompt); copy.textContent = 'Copied ✓'; }
          catch (error) { copy.textContent = 'Copy failed'; }
          setTimeout(() => { copy.textContent = 'Copy prompt'; }, 1500);
        });
        more.append(variation, copy);
      }
      more.append(front);
      const reveal = document.createElement('button'); reveal.type = 'button'; reveal.textContent = 'Open folder';
      reveal.className = 'more-plain'; reveal.title = "Show this character's files (PSD, layers, VRM) in your file manager";
      reveal.addEventListener('click', async () => {
        try { await axios.post(`/api/generation/reveal/${id}`); }
        catch (error) { $('galleryError').textContent = `Could not open the folder: ${error.response?.data?.error || error.message}`; $('galleryError').hidden = false; }
      });
      more.append(reveal);
      if (seethrough && model.metadata?.expressions) {
        const repaint = document.createElement('button'); repaint.type = 'button'; repaint.textContent = 'Repaint blink & talk';
        repaint.addEventListener('click', () => queueExpressions(repaint));
        more.append(repaint);
      }
      if (avatar) {
        const vrm = document.createElement('a'); vrm.href = `/api/generation/artifact/${id}/vrm`; vrm.textContent = 'Download VRM';
        const rebuild = document.createElement('button'); rebuild.type = 'button'; rebuild.textContent = 'Rebuild 3D avatar';
        rebuild.addEventListener('click', () => queue3d(rebuild));
        more.append(vrm, rebuild);
        if (avatar.can_refresh_rig) {
          const refresh = document.createElement('button'); refresh.type = 'button'; refresh.textContent = 'Refresh rig & expressions';
          refresh.title = 'Update the rig and face expressions using the saved 3D mesh, without regenerating it.';
          refresh.addEventListener('click', () => queue3d(refresh, true));
          more.append(refresh);
        }
      }
      more.append(remove);
      const look = document.createElement('button'); look.type = 'button'; look.className = 'look-btn';
      look.textContent = '⤢'; look.title = look.ariaLabel = 'Quick look (← → to flip, Esc to close)';
      look.addEventListener('click', () => quickLook(shownModels.indexOf(model)));
      body.append(status,name,details,note,actions,more);card.append(art,star,look,body);grid.append(card);
    }
    if (models.length && !shown.length) {
      const message=document.createElement('p');message.className='collection-none';
      message.textContent = query ? 'No characters match your search.' : 'No characters in this group yet.';grid.append(message);
    }
  }
  // Full-size artwork over the page; arrow keys flip through the filtered, sorted list.
  let lookIndex = -1;
  function quickLook(index) {
    if (!shownModels.length) return;
    lookIndex = (index + shownModels.length) % shownModels.length;
    const m = shownModels[lookIndex], id = encodeURIComponent(m.id);
    $('quickLookImage').src = `/api/generation/preview/${id}`;
    $('quickLookImage').alt = displayName(m);
    $('quickLookName').textContent = `${displayName(m)} · ${lookIndex + 1} / ${shownModels.length}`;
    if (m.notes) { const small = document.createElement('small'); small.textContent = m.notes; $('quickLookName').append(small); }
    $('quickLookOpen').href = `/generator?model=${id}`;
    $('quickLookPreview').href = `/preview/${id}`;
    $('quickLookPreview').hidden = !(m.status === 'completed' && m.assets?.length);
    $('quickStar').textContent = m.favorite ? '★ Favorite' : '☆ Favorite';
    $('quickStar').setAttribute('aria-pressed', String(!!m.favorite));
    $('quickSeparate').hidden = !['sketch', 'failed'].includes(m.status);
    $('quickSeparate').disabled = separating.has(m.id);
    $('quickSeparate').textContent = separating.has(m.id) ? 'Queued ✓' : 'Separate layers';
    if (!$('quickLook').open) $('quickLook').showModal();
  }
  // Triage a batch without leaving the viewer: star the keepers, queue their separation.
  $('quickStar').addEventListener('click', async () => {
    const m = shownModels[lookIndex], wanted = !m.favorite;
    try {
      const {data} = await axios.post(`/api/generation/favorite/${encodeURIComponent(m.id)}`, {favorite: wanted});
      if (!data.success) throw new Error(data.error);
      m.favorite = wanted; quickLook(lookIndex);
    } catch (error) { $('quickLookName').textContent = 'Could not update favorites.'; }
  });
  $('quickSeparate').addEventListener('click', async () => {
    const m = shownModels[lookIndex];
    $('quickSeparate').disabled = true;
    try {
      const {data} = await axios.post(`/api/generation/retry/${encodeURIComponent(m.id)}`);
      if (!data.success) throw new Error(data.error);
      separating.add(m.id);
      $('quickSeparate').textContent = 'Queued ✓';
    } catch (error) { $('quickSeparate').textContent = 'Could not queue'; $('quickSeparate').disabled = false; }
  });
  $('quickLook').addEventListener('close', () => render());   // stars changed in the viewer
  $('quickPrev').addEventListener('click', () => quickLook(lookIndex - 1));
  $('quickNext').addEventListener('click', () => quickLook(lookIndex + 1));
  $('quickClose').addEventListener('click', () => $('quickLook').close());
  $('quickLook').addEventListener('click', (e) => { if (e.target === $('quickLook')) $('quickLook').close(); });   // backdrop
  // Swipe on touch screens: a mostly horizontal drag of 50 px or more flips.
  let swipeX = null, swipeY = 0;
  $('quickLookImage').addEventListener('pointerdown', (e) => { swipeX = e.clientX; swipeY = e.clientY; });
  $('quickLookImage').addEventListener('pointerup', (e) => {
    if (swipeX === null) return;
    const dx = e.clientX - swipeX, dy = e.clientY - swipeY; swipeX = null;
    if (Math.abs(dx) >= 50 && Math.abs(dx) > 2 * Math.abs(dy)) quickLook(lookIndex + (dx < 0 ? 1 : -1));
  });
  $('quickLook').addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') { e.preventDefault(); quickLook(lookIndex - 1); }
    if (e.key === 'ArrowRight') { e.preventDefault(); quickLook(lookIndex + 1); }
  });
  // Jobs still being made, above the collection; artwork appears as soon as it is drawn.
  async function pollJobs() {
    let jobs = [];
    try { jobs = (await axios.get('/api/generation/jobs')).data.jobs || []; } catch (e) { /* offline */ }
    const active = jobs.filter(j => j.state === 'running' || j.state === 'queued').reverse();   // oldest first
    const strip = $('jobStrip');
    strip.hidden = !active.length;
    strip.replaceChildren(...active.map(j => {
      const card = document.createElement('a'); card.className = 'job-card'; card.href = '/generator';
      const art = document.createElement('div'); art.className = 'job-art';
      if (j.artwork) { const img = document.createElement('img'); img.src = `/api/generation/preview/${encodeURIComponent(j.artwork)}?size=thumb`; img.alt = ''; art.append(img); }
      else art.textContent = j.state === 'queued' ? '…' : '✦';
      const text = document.createElement('div');
      const title = document.createElement('strong'); title.textContent = j.prompt || 'Imported artwork';
      const state = document.createElement('small');
      state.textContent = j.state === 'queued' ? `Queued${j.position ? ` · ${j.position} ahead` : ''}` : `${Math.round(j.progress || 0)}% · ${j.message || 'Working'}`;
      const bar = document.createElement('span'); bar.className = 'job-bar'; bar.style.setProperty('--p', `${j.state === 'queued' ? 0 : j.progress || 0}%`);
      text.append(title, state, bar); card.append(art, text);
      return card;
    }));
    setTimeout(pollJobs, active.length ? 3000 : 10000);
  }
  pollJobs();
  // Deleting moves a character to outputs/_trash; offer an undo for a few seconds.
  let toastTimer = null;
  function undoToast(model, label) {
    const toast = $('undoToast');
    $('undoText').textContent = `Deleted ${label}.`;
    toast.hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => { toast.hidden = true; }, 8000);
    $('undoBtn').onclick = async () => {
      toast.hidden = true;
      try {
        const {data} = await axios.post(`/api/generation/restore/${encodeURIComponent(model.id)}`);
        if (!data.success) throw new Error(data.error);
        await load();
      } catch (error) {
        $('galleryError').textContent = `Could not restore ${label}: ${error.response?.data?.error || error.message}. It stays in outputs/_trash for 7 days.`;
        $('galleryError').hidden = false;
      }
    };
  }
  async function load() {
    $('loadingGallery').hidden = false;$('galleryError').hidden = true;
    try {
      const {data} = await axios.get('/api/generation/list');
      if (!data.success) throw new Error(data.error || 'Could not open the collection.');
      models = data.models || [];render();
      // Coming back (browser Back) to a long collection: return to where the user was.
      const back = !restoredScroll && performance.getEntriesByType('navigation')[0]?.type === 'back_forward';
      restoredScroll = true;
      try { const y = Number(sessionStorage.getItem('kamyii.gallery.scroll')); if (back && y) scrollTo(0, y); } catch { /* no storage */ }
    } catch (error) {
      $('galleryError').textContent = 'Could not open your collection. Check that the local server is running, then refresh.';
      $('galleryError').hidden = false;
    } finally { $('loadingGallery').hidden = true; }
  }
  $('searchInput').addEventListener('input',render);
  $('sortSelect').value = SORTS[saved.sort] ? saved.sort : 'newest';
  $('sortSelect').addEventListener('change', () => { remember(); render(); });
  for (const chip of $('filterChips').children) chip.addEventListener('click', () => { filter = chip.dataset.filter; remember(); render(); });
  if (!FILTERS[filter]) filter = 'all';
  document.addEventListener('keydown', (e) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName);
    if (e.key === '/' && !typing) { e.preventDefault(); $('searchInput').focus(); }
    else if (e.key === 'Escape' && e.target === $('searchInput')) { $('searchInput').value = ''; render(); $('searchInput').blur(); }
  });
  $('refreshBtn').addEventListener('click',load);
  window.addEventListener('kamyii:job-done', load);
  window.addEventListener('pagehide', () => { try { sessionStorage.setItem('kamyii.gallery.scroll', String(scrollY)); } catch { /* no storage */ } });  // the nav saw a job finish
  // Reopen a character saved with Export (on this or another machine).
  $('openProjectBtn').addEventListener('click', () => $('projectFile').click());
  $('projectFile').addEventListener('change', async () => {
    const file = $('projectFile').files[0];
    if (!file) return;
    const form = new FormData(); form.append('project', file);
    $('openProjectBtn').disabled = true; $('galleryError').hidden = true;
    $('collectionCount').textContent = `Opening ${file.name}…`;
    try {
      const {data} = await axios.post('/api/generation/open-project', form);
      if (!data.success) throw new Error(data.error);
      await load();
    } catch (error) {
      const reason = error.response?.status === 413 ? 'the file is too large' : (error.response?.data?.error || error.message);
      $('galleryError').textContent = `Could not open ${file.name}: ${reason}`;
      $('galleryError').hidden = false; render();
    } finally { $('openProjectBtn').disabled = false; $('projectFile').value = ''; }
  });
  // Drop an exported character zip anywhere on the page to reopen it (same path as the button).
  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes('Files');
  document.addEventListener('dragover', (e) => { if (hasFiles(e)) { e.preventDefault(); document.body.classList.add('dropping'); } });
  document.addEventListener('dragleave', (e) => { if (!e.relatedTarget) document.body.classList.remove('dropping'); });
  document.addEventListener('drop', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault(); document.body.classList.remove('dropping');
    const file = e.dataTransfer.files[0];
    if (!/\.zip$/i.test(file?.name || '')) { $('galleryError').textContent = 'Drop a character .zip saved with Export.'; $('galleryError').hidden = false; return; }
    const list = new DataTransfer(); list.items.add(file);
    $('projectFile').files = list.files; $('projectFile').dispatchEvent(new Event('change'));
  });
  load();
})();
