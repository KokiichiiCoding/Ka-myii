/* The collection shares the same saved models and preview adapter as Studio. */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  let models = [];
  const displayName = (m) => m.display_name || (m.name?.startsWith('model_') ? `Study ${m.name.slice(6).toUpperCase()}` : (m.name || `Character ${m.id.slice(0,8)}`));
  function render() {
    const query = $('searchInput').value.trim().toLowerCase();
    const shown = models.filter(m => `${displayName(m)} ${m.name} ${m.id} ${m.status} ${m.request?.prompt || ''}`.toLowerCase().includes(query));
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
      image.src = `/api/generation/preview/${id}`;
      image.alt = model.name || 'Character artwork'; image.loading = 'lazy';
      art.append(image);
      const body = document.createElement('div'); body.className = 'model-card-body';
      const status = document.createElement('span'); status.className = 'eyebrow';
      status.textContent = model.status === 'completed' ? 'CHARACTER / READY' : `CHARACTER / ${(model.status || 'unknown').toUpperCase()}`;
      const name = document.createElement('h2'); name.textContent = displayName(model);
      const details = document.createElement('p'); details.className = 'card-meta';
      details.textContent = `${model.assets?.filter(a=>a.layer_type!=='character').length || 0} layers · ${new Date(model.created_at).toLocaleDateString()}`;
      if (model.status === 'failed') details.textContent = `Failed: ${model.error_message || 'unknown error'} · ${new Date(model.created_at).toLocaleDateString()}`;
      if (model.status === 'sketch') details.textContent = `Sketch · ${new Date(model.created_at).toLocaleDateString()}`;
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
        if (!confirm(`Delete ${name.textContent} and all its generated files? This cannot be undone.`)) return;
        remove.disabled=true;
        try {
          const {data}=await axios.delete(`/api/generation/delete/${id}`);
          if (!data.success) throw new Error(data.error);
          models=models.filter(m=>m.id!==model.id); render();
        } catch (error) {
          $('galleryError').textContent='Could not delete the character. Try refreshing the collection.';
          $('galleryError').hidden=false; remove.disabled=false;
        }
      });
      const rename = document.createElement('button'); rename.type = 'button'; rename.textContent = 'Rename';
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
      more.append(summary,rename);
      if (seethrough && model.metadata?.expressions) {
        const repaint = document.createElement('button'); repaint.type = 'button'; repaint.textContent = 'Repaint blink & talk';
        repaint.addEventListener('click', () => queueExpressions(repaint));
        more.append(repaint);
      }
      more.append(remove);
      body.append(status,name,details,actions,more);card.append(art,body);grid.append(card);
    }
    if (models.length && !shown.length) {
      const message=document.createElement('p');message.textContent='No characters match your search.';grid.append(message);
    }
  }
  async function load() {
    $('loadingGallery').hidden = false;$('galleryError').hidden = true;
    try {
      const {data} = await axios.get('/api/generation/list');
      if (!data.success) throw new Error(data.error || 'Could not open the collection.');
      models = data.models || [];render();
    } catch (error) {
      $('galleryError').textContent = 'Could not open your collection. Check that the local server is running, then refresh.';
      $('galleryError').hidden = false;
    } finally { $('loadingGallery').hidden = true; }
  }
  $('searchInput').addEventListener('input',render);
  $('refreshBtn').addEventListener('click',load);
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
  load();
})();
