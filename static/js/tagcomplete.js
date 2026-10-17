// Danbooru tag autocomplete for the brief fields. Illustrious / NoobAI / Pony checkpoints are
// trained on these tags, so suggesting real ones (most-used first) improves prompts.
// Tag data: a1111-sd-webui-tagcomplete (MIT), see static/vendor/tags.
(function () {
  'use strict';

  const SOURCE = '/static/vendor/tags/danbooru.csv';
  const MAX = 8;
  const CATEGORY = { 0: 'general', 1: 'artist', 3: 'series', 4: 'character', 5: 'meta' };
  let tags = null; // [{name, key, cat, count, aliases}], sorted by count (file order)
  let loading = null;

  // name,category,count[,alias | ,"alias,alias"] — names never contain commas in this list.
  function parse(text) {
    const out = [];
    for (const line of text.split('\n')) {
      if (!line) continue;
      const [name, cat, count, ...rest] = line.split(',');
      const aliases = rest.join(',').replace(/"/g, '').split(',').filter(Boolean);
      out.push({ name: name.replace(/_/g, ' '), key: name.toLowerCase(), cat: +cat, count: +count, aliases });
    }
    return out;
  }

  function load() {
    loading = loading || fetch(SOURCE).then((r) => r.text()).then((text) => { tags = parse(text); })
      .catch(() => { tags = []; });
    return loading;
  }

  // The list is stored most-used first, so the first MAX matches are the most popular ones.
  function search(list, query) {
    const q = query.toLowerCase().replace(/ /g, '_');
    const found = [];
    for (const t of list) {
      // prefix, word start ("miku" → "hatsune miku") or an alias ("twin tails" → "twintails")
      if (t.key.startsWith(q) || t.key.includes('_' + q) || t.aliases.some((x) => x.startsWith(q))) {
        if (found.push(t) >= MAX) break;
      }
    }
    return found;
  }

  function fmt(n) { return n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : n >= 1e3 ? Math.round(n / 1e3) + 'k' : String(n); }

  function attach(field) {
    const pop = document.createElement('ul');
    pop.className = 'tc-pop hidden';
    pop.id = field.id + 'Suggestions';
    pop.setAttribute('role', 'listbox');
    field.parentElement.style.position = 'relative';
    field.parentElement.appendChild(pop);
    field.setAttribute('aria-autocomplete', 'list');
    field.setAttribute('aria-controls', pop.id);
    let items = [], active = 0;

    // The tag being typed: from the last comma before the caret up to the caret.
    function token() {
      const before = field.value.slice(0, field.selectionStart);
      const start = before.lastIndexOf(',') + 1;
      return { start, text: before.slice(start).trimStart() };
    }

    function close() { pop.classList.add('hidden'); items = []; field.removeAttribute('aria-activedescendant'); }

    function render() {
      pop.innerHTML = '';
      items.forEach((t, i) => {
        const li = document.createElement('li');
        li.id = `${pop.id}-${i}`;
        li.setAttribute('role', 'option');
        li.setAttribute('aria-selected', String(i === active));
        li.className = 'tc-item tc-' + (CATEGORY[t.cat] || 'general') + (i === active ? ' active' : '');
        const name = document.createElement('span'); name.textContent = t.name;
        const meta = document.createElement('span'); meta.className = 'tc-count';
        meta.textContent = (t.cat ? CATEGORY[t.cat] + ' · ' : '') + fmt(t.count);
        li.append(name, meta);
        li.onmousedown = (e) => { e.preventDefault(); accept(i); };
        pop.appendChild(li);
      });
      pop.classList.toggle('hidden', !items.length);
      if (items.length) field.setAttribute('aria-activedescendant', `${pop.id}-${active}`);
    }

    function accept(i) {
      const { start } = token();
      const after = field.value.slice(field.selectionStart).replace(/^[^,]*,?\s*/, '');
      const head = field.value.slice(0, start) + (start ? ' ' : '') + items[i].name + ', ';
      field.value = head + after;
      field.selectionStart = field.selectionEnd = head.length;
      close();
      field.dispatchEvent(new Event('input'));
    }

    field.addEventListener('focus', load, { once: true });
    field.addEventListener('input', async (e) => {
      if (!e.isTrusted) return; // our own accept() re-dispatches input
      const { text } = token();
      if (text.length < 2) { close(); return; }
      await load();
      if (token().text !== text) return; // user kept typing while the list loaded
      items = search(tags, text);
      active = 0;
      render();
    });
    field.addEventListener('keydown', (e) => {
      if (!items.length) return;
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        active = (active + (e.key === 'ArrowDown' ? 1 : items.length - 1)) % items.length;
        render();
      } else if ((e.key === 'Enter' || e.key === 'Tab') && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        e.stopPropagation();
        accept(active);
      } else if (e.key === 'Escape') {
        e.stopPropagation();
        close();
      }
    });
    field.addEventListener('blur', close);
  }

  globalThis.KamyiiTagcomplete = { parse, search }; // for tests/tagcomplete.test.cjs

  document.addEventListener('DOMContentLoaded', () => {
    for (const id of ['prompt', 'negative']) {
      const field = document.getElementById(id);
      if (field) attach(field);
    }
  });
})();
