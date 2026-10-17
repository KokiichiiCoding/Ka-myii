/* The collection shares the same saved models and preview adapter as Studio. */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  let models = [];
  function render() {
    const query = $('searchInput').value.trim().toLowerCase();
    const shown = models.filter(m => `${m.name} ${m.id} ${m.status} ${m.request?.prompt || ''}`.toLowerCase().includes(query));
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
      status.textContent = model.status === 'completed' ? 'CHARACTER / READY' : `CHARACTER / ${model.status || 'UNKNOWN'}`;
      const name = document.createElement('h2'); name.textContent = model.name?.startsWith('model_') ? `Study ${model.name.slice(6).toUpperCase()}` : (model.name || `Character ${model.id.slice(0,8)}`);
      const details = document.createElement('p'); details.className = 'card-meta';
      details.textContent = `${model.assets?.filter(a=>a.layer_type!=='character').length || 0} layers · ${new Date(model.created_at).toLocaleDateString()}`;
      const actions = document.createElement('div'); actions.className = 'actions';
      const links = [[`/generator?model=${id}`, 'Artwork & layers', 'btn'], [`/api/generation/download/${id}`, 'Export', 'btn']];
      if (model.status === 'completed' && model.assets?.length) links.unshift([`/preview/${id}`, 'Live Preview ↗', 'btn btn-primary']);
      for (const [href,label,cls] of links) {
        const link = document.createElement('a'); link.href=href;link.textContent=label;link.className=cls;actions.append(link);
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
      more.append(summary,remove);
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
  load();
})();
