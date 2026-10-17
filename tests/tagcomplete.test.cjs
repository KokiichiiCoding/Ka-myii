// Tag autocomplete parsing/ranking against the vendored Danbooru list: node tests/tagcomplete.test.cjs
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const root = path.join(__dirname, '..');
const sandbox = { document: { addEventListener() {} } };
sandbox.globalThis = sandbox;
vm.runInNewContext(fs.readFileSync(path.join(root, 'static/js/tagcomplete.js'), 'utf8'), sandbox);
const { parse, search } = sandbox.KamyiiTagcomplete;

const tags = parse(fs.readFileSync(path.join(root, 'static/vendor/tags/danbooru.csv'), 'utf8'));
assert(tags.length > 50000, 'tag list loaded');
assert(tags.every((t) => Number.isFinite(t.count) && Number.isFinite(t.cat)), 'every row parsed');
const names = (q) => search(tags, q).map((t) => t.name);

assert.strictEqual(names('twin')[0], 'twintails', 'prefix, most-used first');
assert(names('miku').includes('hatsune miku'), 'word-start match');
assert.strictEqual(names('gray h')[0], 'grey hair', 'unquoted single alias');
assert.strictEqual(names('twin tails')[0], 'twintails', 'quoted alias list');
assert(names('cat ea').includes('cat ears'), 'spaces match underscores');
assert(names('zzzzqx').length === 0, 'no false matches');
assert(search(tags, 'a').length <= 8, 'capped');
console.log('Tag autocomplete: parsing, aliases, word-start matches and popularity order passed.');
