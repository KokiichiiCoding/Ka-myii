"""Quality-of-life UI: nav job indicator, Gallery filters/sort, Studio draft and Surprise me."""
import json
import math
import time

import pytest

from test_journeys import base_url, page, expect, JOB


def _wait_jobs(page, base_url, timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        jobs = page.request.get(f"{base_url}/api/generation/jobs").json()["jobs"]
        if jobs and all(j["state"] not in ("queued", "running") for j in jobs):
            return jobs
        time.sleep(.5)
    raise AssertionError("demo jobs did not finish")


def test_nav_shows_a_running_job_then_links_to_the_result(base_url, page):
    job = {"id": "j1", "prompt": "x", "model_id": "abc", "progress": 42.4, "message": "Separating", "state": "running"}
    # The Gallery polls the same endpoint, so the job stays running until the test finishes it.
    page.route("**/api/generation/jobs", lambda route: route.fulfill(
        content_type="application/json", body=json.dumps({"success": True, "jobs": [job]})))
    page.goto(f"{base_url}/gallery")
    expect(page.locator("#navModelText")).to_have_text("Creating · 42%")
    expect(page.locator("#navModelStatus")).to_have_attribute("href", "/generator")
    expect(page).to_have_title("42% · Ka-myii — Collection")
    job["state"] = "completed"
    expect(page.locator("#navModelText")).to_have_text("Character ready ↗", timeout=10_000)
    expect(page.locator("#navModelStatus")).to_have_attribute("href", "/generator?model=abc")
    expect(page).to_have_title("Ka-myii — Collection")
    assert not page.errors


def test_gallery_filters_sorts_and_remembers(base_url, page):
    for prompt, sketch in (("1girl, red hair", False), ("1girl, blue hair", True)):
        r = page.request.post(f"{base_url}/api/generation/generate", data={"prompt": prompt, "sketch": sketch})
        assert r.status == 202
        _wait_jobs(page, base_url)
    models = page.request.get(f"{base_url}/api/generation/list").json()["models"]
    for model, name in zip(sorted(models, key=lambda m: m["created_at"]), ("Zed", "Amy")):
        page.request.post(f"{base_url}/api/generation/rename/{model['id']}", data={"name": name})

    page.goto(f"{base_url}/gallery")
    names = page.locator(".model-card h2")
    expect(names).to_have_count(2)
    expect(names.first).to_have_text("Amy")  # newest first
    page.select_option("#sortSelect", "oldest")
    expect(names.first).to_have_text("Zed")
    expect(page.locator('[data-filter="sketch"]')).to_have_attribute("data-count", "1")
    expect(page.locator('[data-filter="3d"]')).to_be_hidden()  # no 3D avatars, chip hidden
    page.click('[data-filter="sketch"]')
    expect(names).to_have_count(1)
    expect(names.first).to_have_text("Amy")

    page.reload()  # filter and sort are remembered in this browser
    expect(page.locator('[data-filter="sketch"]')).to_have_attribute("aria-pressed", "true")
    expect(page.locator("#sortSelect")).to_have_value("oldest")
    page.keyboard.press("/")
    expect(page.locator("#searchInput")).to_be_focused()
    assert not page.errors


def test_studio_surprise_me_and_draft_survives_reload(base_url, page):
    page.goto(f"{base_url}/generator")
    page.click("#surpriseBtn")
    brief = page.input_value("#prompt")
    assert brief.startswith(("1girl", "1boy")) and brief.count(",") >= 4
    page.fill("#negative", "lowres")
    page.reload()
    expect(page.locator("#prompt")).to_have_value(brief)
    expect(page.locator("#negative")).to_have_value("lowres")
    assert not page.errors


def test_make_a_variation_reuses_the_brief_with_a_new_seed(base_url, page):
    r = page.request.post(f"{base_url}/api/generation/generate",
                          data={"prompt": "1girl, green hair, kimono", "seed": 1234, "sketch": True, "steps": 22})
    assert r.status == 202
    _wait_jobs(page, base_url)
    page.goto(f"{base_url}/gallery")
    page.locator(".model-more summary").first.click()
    page.get_by_role("link", name="Make a variation").click()
    expect(page.locator("#prompt")).to_have_value("1girl, green hair, kimono")
    expect(page.locator("#seed")).to_have_value("-1")
    expect(page.locator("#steps")).to_have_value("22")
    expect(page.locator("#resultMeta")).to_be_hidden()  # a fresh brief, not the old result
    assert not page.errors


def test_studio_imports_a_dropped_image(base_url, page):
    page.goto(f"{base_url}/generator")
    page.evaluate("""async () => {
      const c = document.createElement('canvas'); c.width = c.height = 8;
      const blob = await new Promise(r => c.toBlob(r, 'image/png'));
      const dt = new DataTransfer(); dt.items.add(new File([blob], 'art.png', {type: 'image/png'}));
      document.body.dispatchEvent(new DragEvent('drop', {dataTransfer: dt, bubbles: true, cancelable: true}));
    }""")
    expect(page.locator("#uploadFields")).to_be_visible()
    expect(page.locator('[data-source="upload"]')).to_have_attribute("aria-pressed", "true")
    assert page.evaluate("document.getElementById('artwork').files[0].name") == "art.png"
    expect(page.locator("#cropStage")).to_be_visible()
    assert not page.errors


def test_rename_the_character_from_the_studio_result(base_url, page):
    r = page.request.post(f"{base_url}/api/generation/generate", data={"prompt": "1girl, sketch", "sketch": True})
    assert r.status == 202
    _wait_jobs(page, base_url)
    model_id = page.request.get(f"{base_url}/api/generation/list").json()["models"][0]["id"]
    page.goto(f"{base_url}/generator?model={model_id}")
    name = page.locator("#resultName")
    expect(name).to_be_visible(timeout=JOB)
    assert name.input_value().startswith("Study ")
    name.fill("Lumi the Witch")
    name.press("Enter")
    names = lambda: {m["id"]: m.get("display_name") for m in page.request.get(f"{base_url}/api/generation/list").json()["models"]}
    for _ in range(40):
        if names()[model_id] == "Lumi the Witch":
            break
        time.sleep(.1)
    assert names()[model_id] == "Lumi the Witch"
    name.fill("Nope")
    name.press("Escape")
    expect(name).to_have_value("Lumi the Witch")
    assert not page.errors


def test_live_preview_gestures_move_the_head(base_url, page):
    r = page.request.post(f"{base_url}/api/generation/generate", data={"prompt": "1girl, silver hair"})
    assert r.status == 202
    _wait_jobs(page, base_url)
    model_id = page.request.get(f"{base_url}/api/generation/list").json()["models"][0]["id"]
    page.add_init_script("""Object.defineProperty(window,'RigRenderer',{configurable:true,set(api){
      Object.defineProperty(window,'RigRenderer',{value:{...api,create(gl){const r=api.create(gl),draw=r.draw;
      r.draw=frame=>{window.__topwear=frame.items.filter(({L})=>L.bn==='topwear').map(({L})=>Array.from(L.cur));draw(frame);};return r;}}});
    }});""")
    page.goto(f"{base_url}/preview/{model_id}")
    expect(page.locator("#btnPause")).to_be_enabled(timeout=JOB)
    for toggle in ("#tgIdle", "#tgMouse", "#tgRand"):   # hold the head still apart from the gesture
        if page.locator(toggle).get_attribute("aria-pressed") == "true":
            page.click(toggle)
    page.locator("body").press("h")   # shake head
    turns = []
    for _ in range(12):
        turns.append(page.evaluate("Anime25D.state().pose.angleX"))
        page.wait_for_timeout(80)
    assert max(turns) > .2 and min(turns) < -.2, turns
    page.wait_for_timeout(1500)
    page.locator("body").press("y")   # yay: the chest lags the hop and bounces back
    chest = []
    for _ in range(14):
        chest.append(page.evaluate("Anime25D.state().pose.chestY"))
        page.wait_for_timeout(70)
    assert max(chest) > 1 and min(chest) < -1, chest
    page.wait_for_timeout(1500)
    page.locator('body').press('y')
    page.wait_for_function('Anime25D.state().pose.chestY > 1', timeout=5000)
    page.locator('#btnPause').click()
    page.locator('#pBust').evaluate("e=>{e.value='1';e.dispatchEvent(new Event('input'));}")
    page.wait_for_timeout(150)
    moving = page.evaluate('window.__topwear.flat()')
    page.locator('#pBust').evaluate("e=>{e.value='0';e.dispatchEvent(new Event('input'));}")
    page.wait_for_timeout(150)
    still = page.evaluate('window.__topwear.flat()')
    assert len(moving) == len(still) > 0
    assert max(abs(a-b) for a,b in zip(moving,still)) > .1, 'The bounce control changes rendered clothing'
    page.locator('#pBust').evaluate("e=>{e.value='4';e.dispatchEvent(new Event('input'));}")
    page.wait_for_timeout(150)
    strong = page.evaluate('window.__topwear.flat()')
    assert all(math.isfinite(v) for v in strong), 'Extreme bounce keeps mesh positions finite'
    expect(page.locator("#tgLively")).to_have_attribute("aria-pressed", "true")
    assert not page.errors


def test_live_preview_moods_face_marks_and_animal_ears(base_url, page):
    r = page.request.post(f"{base_url}/api/generation/generate", data={"prompt": "1girl, silver hair"})
    assert r.status == 202
    _wait_jobs(page, base_url)
    model_id = page.request.get(f"{base_url}/api/generation/list").json()["models"][0]["id"]
    page.add_init_script("""Object.defineProperty(window,'RigRenderer',{configurable:true,set(api){
      Object.defineProperty(window,'RigRenderer',{value:{...api,create(gl){const r=api.create(gl),draw=r.draw;
      r.draw=frame=>{window.__drawn=Object.fromEntries(frame.items.map(({L,alpha})=>[L.name,alpha]));draw(frame);};return r;}}});
    }});""")
    page.goto(f"{base_url}/preview/{model_id}")
    expect(page.locator("#btnPause")).to_be_enabled(timeout=JOB)
    drawn = lambda name: page.evaluate(f"(window.__drawn||{{}})[{json.dumps(name)}] || 0")
    page.locator("body").press("9")   # Angry: frowning brows and its anger mark
    page.wait_for_function("Anime25D.state().activePreset === 'angry' && Anime25D.state().marks.anger === 1")
    expect(page.locator("#pBrowAngSym")).to_have_value("0.85")
    page.wait_for_function("(window.__drawn||{})['mark anger'] > .9", timeout=5000)
    page.locator("body").press("9")   # cleared: the mark and the brows go back
    page.wait_for_function("Anime25D.state().activePreset === null && Anime25D.state().marks.anger === 0")
    expect(page.locator("#pBrowAngSym")).to_have_value("0")
    page.locator("body").press("r")   # a blush of one's own...
    page.locator('[data-preset="gloomy"]').click()
    assert page.evaluate("Anime25D.state().marks") == {"blush": 1, "gloom": 1, "anger": 0, "sweat": 0}
    page.locator('[data-preset="gloomy"]').click()   # ...stays when the mood that added gloom ends
    assert page.evaluate("Anime25D.state().marks") == {"blush": 1, "gloom": 0, "anger": 0, "sweat": 0}
    expect(page.locator('[data-mark="markBlush"]')).to_have_attribute("aria-pressed", "true")
    page.wait_for_function("(window.__drawn||{})['mark blush L'] > .9 && !((window.__drawn||{})['mark gloom'] > .05)", timeout=5000)
    page.locator("#nekoStyle").select_option("4")   # bunny ears and a devil tail
    page.locator("#tailStyle").select_option("4")
    page.wait_for_function("Anime25D.state().companions.ear === 'bunny' && Anime25D.state().companions.tailKind === 'devil'")
    page.wait_for_function("(window.__drawn||{})['companion ear L'] === 1 && (window.__drawn||{})['companion tail'] === 1", timeout=5000)
    assert drawn("companion ear R") == 1
    assert not page.errors


def test_star_a_favorite_and_show_only_favorites(base_url, page):
    for prompt in ("1girl, red hair", "1girl, blue hair"):
        assert page.request.post(f"{base_url}/api/generation/generate", data={"prompt": prompt, "sketch": True}).status == 202
        _wait_jobs(page, base_url)
    page.goto(f"{base_url}/gallery")
    expect(page.locator(".model-card")).to_have_count(2)
    expect(page.locator('[data-filter="favorite"]')).to_be_hidden()   # nothing starred yet
    star = page.locator(".model-card").first.locator(".star-btn")
    star.click()
    expect(star).to_have_attribute("aria-pressed", "true")
    expect(page.locator('[data-filter="favorite"]')).to_be_visible()
    page.click('[data-filter="favorite"]')
    expect(page.locator(".model-card")).to_have_count(1)
    page.reload()   # kept with the character, not just in this browser
    expect(page.locator(".model-card .star-btn[aria-pressed=true]")).to_have_count(1)
    assert not page.errors


def test_recent_briefs_refill_the_prompt(base_url, page):
    for prompt in ("1girl, red hair, maid outfit", "1girl, blue hair, hoodie"):
        assert page.request.post(f"{base_url}/api/generation/generate",
                                 data={"prompt": prompt, "negative_prompt": "lowres", "sketch": True}).status == 202
        _wait_jobs(page, base_url)
    page.goto(f"{base_url}/generator")
    page.click("#recentBtn")
    items = page.locator("#recentList li")
    expect(items).to_have_count(2)
    expect(items.first).to_have_text("1girl, blue hair, hoodie")   # newest first
    items.nth(1).click()
    expect(page.locator("#prompt")).to_have_value("1girl, red hair, maid outfit")
    expect(page.locator("#negative")).to_have_value("lowres")
    expect(page.locator("#recentList")).to_be_hidden()
    assert not page.errors


def test_how_many_queues_a_batch_with_counting_seeds(base_url, page):
    page.goto(f"{base_url}/generator")
    page.fill("#prompt", "1girl, silver hair")
    page.fill("#seed", "100")
    page.select_option("#batchCount", "4")
    page.click("#sketchBtn")
    jobs = []
    for _ in range(240):
        jobs = page.request.get(f"{base_url}/api/generation/jobs").json()["jobs"]
        if len(jobs) == 4 and all(j["state"] == "completed" for j in jobs):
            break
        time.sleep(.5)
    seeds = sorted(m["request"]["seed"] for m in page.request.get(f"{base_url}/api/generation/list").json()["models"])
    assert seeds == [100, 101, 102, 103], seeds
    expect(page.locator("#batchRow")).to_be_visible()
    page.click('[data-source="upload"]')
    expect(page.locator("#batchRow")).to_be_hidden()
    assert not page.errors


def test_quick_look_flips_through_the_collection(base_url, page):
    for prompt in ("1girl, red hair", "1girl, blue hair"):
        assert page.request.post(f"{base_url}/api/generation/generate", data={"prompt": prompt, "sketch": True}).status == 202
        _wait_jobs(page, base_url)
    page.goto(f"{base_url}/gallery")
    page.locator(".model-card").first.hover()
    page.locator(".model-card .look-btn").first.click()
    expect(page.locator("#quickLook")).to_be_visible()
    expect(page.locator("#quickLookName")).to_contain_text("1 / 2")
    page.keyboard.press("ArrowRight")
    expect(page.locator("#quickLookName")).to_contain_text("2 / 2")
    page.keyboard.press("ArrowRight")   # wraps around
    expect(page.locator("#quickLookName")).to_contain_text("1 / 2")
    page.click("#quickStar")   # triage without leaving the viewer
    expect(page.locator("#quickStar")).to_have_attribute("aria-pressed", "true")
    page.click("#quickSeparate")
    expect(page.locator("#quickSeparate")).to_have_text("Queued ✓")
    assert any(j["state"] in ("queued", "running", "completed") and j["prompt"]
               for j in page.request.get(f"{base_url}/api/generation/jobs").json()["jobs"][:1])
    page.keyboard.press("Escape")
    expect(page.locator("#quickLook")).to_be_hidden()
    expect(page.locator(".model-card .star-btn[aria-pressed=true]")).to_have_count(1)
    assert not page.errors


def test_studio_shows_the_artwork_while_layers_are_separated(base_url, page):
    assert page.request.post(f"{base_url}/api/generation/generate", data={"prompt": "1girl", "sketch": True}).status == 202
    _wait_jobs(page, base_url)
    art_id = page.request.get(f"{base_url}/api/generation/list").json()["models"][0]["id"]
    job = {"id": "j9", "state": "running", "prompt": "1girl", "model_id": None, "progress": 40.0,
           "stage": "separate", "message": "Separating", "eta": None, "artwork": art_id}
    page.route("**/api/generation/jobs", lambda r: r.fulfill(json={"success": True, "jobs": [job]}))
    page.route("**/api/generation/jobs/j9", lambda r: r.fulfill(json={"success": True, "job": job}))
    page.goto(f"{base_url}/generator")   # re-attaches to the running job
    expect(page.locator("#progressArt")).to_be_visible()
    assert art_id in page.get_attribute("#progressArt", "src")
    assert not page.errors


def test_question_mark_lists_shortcuts_and_esc_keeps_the_result(base_url, page):
    assert page.request.post(f"{base_url}/api/generation/generate", data={"prompt": "1girl", "sketch": True}).status == 202
    _wait_jobs(page, base_url)
    model_id = page.request.get(f"{base_url}/api/generation/list").json()["models"][0]["id"]
    page.goto(f"{base_url}/gallery")
    page.keyboard.press("?")
    expect(page.locator("#kmHelp")).to_be_visible()
    expect(page.locator("#kmHelp")).to_contain_text("Search the collection")
    page.keyboard.press("Escape")
    expect(page.locator("#kmHelp")).to_be_hidden()
    page.goto(f"{base_url}/generator?model={model_id}")
    expect(page.locator("#resultMeta")).to_be_visible(timeout=JOB)
    page.keyboard.press("?")
    expect(page.locator("#kmHelp")).to_contain_text("Create the character")
    page.keyboard.press("Escape")   # closes the list without leaving the result
    expect(page.locator("#kmHelp")).to_be_hidden()
    expect(page.locator("#resultMeta")).to_be_visible()
    page.locator("#resultName").fill("Temp")
    page.locator("#resultName").press("Escape")   # undoes the name, keeps the result
    expect(page.locator("#resultMeta")).to_be_visible()
    assert not page.errors


def test_gallery_shows_jobs_in_progress(base_url, page):
    assert page.request.post(f"{base_url}/api/generation/generate", data={"prompt": "1girl", "sketch": True}).status == 202
    _wait_jobs(page, base_url)
    art_id = page.request.get(f"{base_url}/api/generation/list").json()["models"][0]["id"]
    jobs = [{"id": "q2", "state": "queued", "prompt": "1girl, blue hair", "position": 0, "progress": 0},
            {"id": "r1", "state": "running", "prompt": "1girl, red hair", "progress": 42.4,
             "message": "Separating", "artwork": art_id}]
    page.route("**/api/generation/jobs", lambda r: r.fulfill(json={"success": True, "jobs": jobs}))
    page.goto(f"{base_url}/gallery")
    cards = page.locator("#jobStrip .job-card")
    expect(cards).to_have_count(2)
    expect(cards.first).to_contain_text("42% · Separating")   # oldest (running) first
    expect(cards.first.locator("img")).to_have_attribute("src", f"/api/generation/preview/{art_id}?size=thumb")
    expect(cards.nth(1)).to_contain_text("Queued")
    assert not page.errors


def test_out_of_memory_failure_reads_as_advice(base_url, page):
    job = {"id": "oom", "state": "running", "prompt": "1girl", "progress": 20.0, "stage": "separate", "message": "x"}
    page.route("**/api/generation/jobs", lambda r: r.fulfill(json={"success": True, "jobs": [job]}))
    page.route("**/api/generation/jobs/oom", lambda r: r.fulfill(json={"success": True, "job": job}))
    page.goto(f"{base_url}/generator")
    expect(page.locator("#progressOverlay")).to_be_visible()   # attached to the running job
    job.update(state="failed", error="CUDA out of memory. Tried to allocate 2.00 GiB")
    expect(page.locator("#errorMsg")).to_contain_text("ran out of memory", timeout=10_000)
    expect(page.locator("#errorMsg")).to_contain_text("Retry layers")
    assert not page.errors


def test_back_to_the_gallery_keeps_the_scroll_position(base_url, page):
    for i in range(6):
        assert page.request.post(f"{base_url}/api/generation/generate", data={"prompt": f"1girl, {i}", "sketch": True}).status == 202
        _wait_jobs(page, base_url)
    page.set_viewport_size({"width": 900, "height": 600})
    page.goto(f"{base_url}/gallery")
    expect(page.locator(".model-card")).to_have_count(6)
    page.evaluate("scrollTo(0, 900)")
    page.locator(".model-card").nth(4).locator(".model-art").click()   # into the Studio
    page.wait_for_url("**/generator?model=*")
    page.go_back()
    expect(page.locator(".model-card")).to_have_count(6)
    page.wait_for_function("scrollY > 700")
    assert not page.errors


def test_delete_can_be_undone_from_the_toast(base_url, page):
    import config
    assert page.request.post(f"{base_url}/api/generation/generate", data={"prompt": "1girl", "sketch": True}).status == 202
    _wait_jobs(page, base_url)
    page.goto(f"{base_url}/gallery")
    expect(page.locator(".model-card")).to_have_count(1)
    page.locator(".model-card summary").click()
    page.locator(".model-card").get_by_text("Delete character").click()
    expect(page.locator(".model-card")).to_have_count(0)
    expect(page.locator("#undoToast")).to_be_visible()
    assert any((config.OUTPUT_DIR / "_trash").iterdir())
    page.click("#undoBtn")
    expect(page.locator(".model-card")).to_have_count(1)
    assert not any((config.OUTPUT_DIR / "_trash").iterdir())
    assert not page.errors


def test_studio_remembers_the_whole_brief_but_not_the_seed(base_url, page):
    page.goto(f"{base_url}/generator")
    page.fill("#prompt", "1girl, fox ears")
    page.locator("#steps").fill("22")
    page.locator("#steps").dispatch_event("change")
    page.select_option("#framing", "full_body")
    page.locator("#toggleChips .chip").first.click()
    page.fill("#seed", "777")
    page.reload()
    expect(page.locator("#prompt")).to_have_value("1girl, fox ears")
    expect(page.locator("#steps")).to_have_value("22")
    expect(page.locator("#framing")).to_have_value("full_body")
    expect(page.locator("#toggleChips .chip").first).to_have_attribute("aria-pressed", "true")
    expect(page.locator("#seed")).not_to_have_value("777")
    assert not page.errors


def test_separate_all_starred_sketches_at_once(base_url, page):
    for prompt in ("1girl, a", "1girl, b", "1girl, c"):
        assert page.request.post(f"{base_url}/api/generation/generate", data={"prompt": prompt, "sketch": True}).status == 202
        _wait_jobs(page, base_url)
    ids = [m["id"] for m in page.request.get(f"{base_url}/api/generation/list").json()["models"]]
    for model_id in ids[:2]:
        page.request.post(f"{base_url}/api/generation/favorite/{model_id}", data={"favorite": True})
    page.goto(f"{base_url}/gallery")
    expect(page.locator("#bulkRow")).to_be_hidden()
    page.click('[data-filter="favorite"]')
    expect(page.locator("#bulkSeparate")).to_have_text("Separate 2 starred sketches")
    page.click("#bulkSeparate")
    expect(page.locator("#bulkNote")).to_contain_text("Queued 2 of 2")
    expect(page.locator("#bulkSeparate")).to_be_hidden()   # no second, duplicate batch
    _wait_jobs(page, base_url, timeout=300)   # two demo separations, ~50 s each
    statuses = sorted(m["status"] for m in page.request.get(f"{base_url}/api/generation/list").json()["models"])
    assert statuses == ["completed", "completed", "sketch", "sketch", "sketch"]
    assert not page.errors


def test_status_panel_shows_space_and_empties_the_trash(base_url, page):
    assert page.request.post(f"{base_url}/api/generation/generate", data={"prompt": "1girl", "sketch": True}).status == 202
    _wait_jobs(page, base_url)
    model_id = page.request.get(f"{base_url}/api/generation/list").json()["models"][0]["id"]
    page.request.delete(f"{base_url}/api/generation/delete/{model_id}")
    page.goto(f"{base_url}/gallery")
    page.click("#navModelStatus")
    expect(page.locator("#kmStatus")).to_be_visible()
    expect(page.locator("#kmStatusTable")).to_contain_text("Demo")
    expect(page.locator("#kmStatusTable")).to_contain_text("Trash")
    page.on("dialog", lambda d: d.accept())   # the "permanently delete" confirmation
    page.click("#kmEmptyTrash")
    expect(page.locator("#kmStatusNote")).to_contain_text("Removed 1")
    expect(page.locator("#kmEmptyTrash")).to_be_hidden()
    assert not page.errors


def test_make_a_3d_avatar_from_the_studio_result(base_url, page):
    page.goto(f"{base_url}/generator")
    page.fill("#prompt", "1girl, standing")
    page.click("#generateBtn")
    expect(page.locator("#livePreview")).to_be_visible(timeout=JOB)
    expect(page.locator("#make3d")).to_be_visible()
    expect(page.locator("#open3d")).to_be_hidden()
    page.click("#make3d")
    expect(page.locator("#open3d")).to_be_visible(timeout=JOB)   # the demo mannequin VRM
    expect(page.locator("#make3d")).to_be_hidden()
    assert not page.errors


def test_notes_show_on_the_card_and_are_searchable(base_url, page):
    for prompt in ("1girl, red hair", "1girl, blue hair"):
        assert page.request.post(f"{base_url}/api/generation/generate", data={"prompt": prompt, "sketch": True}).status == 202
        _wait_jobs(page, base_url)
    page.goto(f"{base_url}/gallery")
    card = page.locator(".model-card").first
    card.locator("summary").click()
    card.get_by_role("button", name="Notes").click()
    card.locator("textarea.note-input").fill("stream intro outfit")
    card.locator("textarea.note-input").press("Control+Enter")
    expect(page.locator(".model-card .card-note:not([hidden])")).to_have_text("stream intro outfit")
    page.fill("#searchInput", "intro")
    expect(page.locator(".model-card")).to_have_count(1)
    page.reload()
    page.fill("#searchInput", "")
    expect(page.locator(".model-card .card-note:not([hidden])")).to_have_count(1)
    assert not page.errors


def test_cancel_all_queued_keeps_the_running_job(base_url, page):
    jobs = [{"id": "run", "state": "running", "prompt": "a", "progress": 10.0, "stage": "image", "message": "x"},
            {"id": "w1", "state": "queued", "prompt": "b", "position": 0},
            {"id": "w2", "state": "queued", "prompt": "c", "position": 1}]
    cancelled = []
    page.route("**/api/generation/jobs", lambda r: r.fulfill(json={"success": True, "jobs": jobs}))
    page.route("**/api/generation/jobs/run", lambda r: r.fulfill(json={"success": True, "job": jobs[0]}))
    def cancel(route):
        job_id = route.request.url.split("/jobs/")[1].split("/")[0]
        cancelled.append(job_id)
        for j in jobs:
            if j["id"] == job_id:
                j["state"] = "cancelled"
        route.fulfill(json={"success": True})
    page.route("**/api/generation/jobs/*/cancel", cancel)
    page.goto(f"{base_url}/generator")
    expect(page.locator("#cancelQueued")).to_be_visible(timeout=10_000)
    page.click("#cancelQueued")
    expect(page.locator("#cancelQueued")).to_be_hidden(timeout=10_000)
    assert sorted(cancelled) == ["w1", "w2"]
    assert not page.errors


def test_clear_brief_goes_back_to_defaults(base_url, page):
    page.goto(f"{base_url}/generator")
    default_steps = page.input_value("#steps")
    page.fill("#prompt", "1girl, cat ears")
    page.locator("#steps").fill("12")
    page.locator("#steps").dispatch_event("change")
    page.on("dialog", lambda d: d.accept())
    page.click("#clearBrief")
    page.wait_for_load_state()
    expect(page.locator("#prompt")).to_have_value("")
    expect(page.locator("#steps")).to_have_value(default_steps)
    page.reload()
    expect(page.locator("#prompt")).to_have_value("")
    assert not page.errors


def test_open_folder_from_the_studio_result(base_url, page):
    assert page.request.post(f"{base_url}/api/generation/generate", data={"prompt": "1girl", "sketch": True}).status == 202
    _wait_jobs(page, base_url)
    model_id = page.request.get(f"{base_url}/api/generation/list").json()["models"][0]["id"]
    calls = []
    # Intercepted: the real endpoint would open a file manager window on this machine.
    page.route("**/api/generation/reveal/*", lambda r: (calls.append(r.request.url), r.fulfill(json={"success": True})))
    page.goto(f"{base_url}/generator?model={model_id}")
    expect(page.locator("#revealBtn")).to_be_visible(timeout=JOB)
    page.click("#revealBtn")
    page.wait_for_timeout(300)
    assert calls and calls[0].endswith(f"/reveal/{model_id}")
    assert not page.errors


def test_empty_brief_says_what_is_missing(base_url, page):
    page.goto(f"{base_url}/generator")
    page.click("#generateBtn")
    expect(page.locator("#prompt")).to_be_focused()
    assert "Surprise me" in page.get_attribute("#prompt", "placeholder")
    assert page.request.get(f"{base_url}/api/generation/jobs").json()["jobs"] == []
    assert not page.errors


def test_3d_page_points_to_the_gallery_when_there_are_no_avatars(base_url, page):
    page.goto(f"{base_url}/3d")
    expect(page.locator("#d3-make")).to_be_hidden()   # nothing to make one from yet
    assert page.request.post(f"{base_url}/api/generation/generate", data={"prompt": "1girl", "sketch": True}).status == 202
    _wait_jobs(page, base_url)
    page.reload()
    expect(page.locator("#d3-make")).to_be_visible()
    expect(page.locator("#d3-characters")).to_contain_text("No 3D avatars yet")
    assert not page.errors
