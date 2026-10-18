"""
End-to-end journeys through the real UI in a browser, against an in-process server in demo mode
(placeholder generators: no GPU, seconds per character).

Needs Playwright (`pip install playwright`) and a browser: `python -m playwright install chromium`,
or an installed Edge/Chrome via KAMYII_E2E_CHANNEL=msedge|chrome. Skipped otherwise.
"""
import io
import json
import os
import threading
import zipfile

import pytest
from PIL import Image, ImageDraw

import config

sync_api = pytest.importorskip("playwright.sync_api")
expect = sync_api.expect
JOB = 90_000  # ms: a demo job is a few seconds, the first one also builds the preview PSD


@pytest.fixture
def base_url():
    from werkzeug.serving import make_server
    import app as app_module

    server = make_server("127.0.0.1", 0, app_module.create_app(use_dummy_generators=True), threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture
def page():
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel=os.environ.get("KAMYII_E2E_CHANNEL") or None,
                                        args=["--enable-unsafe-swiftshader"])  # WebGL without a GPU
        except sync_api.Error as e:
            pytest.skip(f"no browser for Playwright ({str(e).splitlines()[0]})")
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()
        page.errors = []
        page.on("pageerror", lambda e: page.errors.append(str(e)))
        page.on("dialog", lambda d: d.accept())  # Gallery's delete confirmation
        yield page
        browser.close()


def _artwork(path):
    """A 'character' on a busy green background, plus a second small figure (a sheet)."""
    img = Image.new("RGB", (900, 700), (40, 120, 60))
    d = ImageDraw.Draw(img)
    d.ellipse((150, 80, 450, 680), fill=(230, 190, 170))
    d.ellipse((240, 160, 360, 300), fill=(250, 220, 200))
    d.ellipse((650, 400, 760, 690), fill=(230, 190, 170))
    img.save(path)
    return path


def test_create_a_character_preview_it_and_keep_adjustments(base_url, page):
    page.goto(f"{base_url}/generator")
    page.fill("#prompt", "1girl, silver hair, blue eyes, sailor uniform")
    page.click("#generateBtn")
    expect(page.locator("#livePreview")).to_be_visible(timeout=JOB)
    expect(page.locator("#layerPanel")).to_be_visible()
    psd = page.request.get(base_url + page.get_attribute("#dlPsd", "href"))
    assert psd.status == 200 and psd.body()[:4] == b"8BPS"

    page.click("#livePreview")
    expect(page.locator("#btnPause")).to_be_enabled(timeout=JOB)  # the rig loaded and runs
    page.locator("#btnSaveSettings").dispatch_event("click")
    expect(page.locator("#appStatus")).to_contain_text("saved with the character")
    model_id = page.url.rstrip("/").split("/")[-1]
    assert json.loads((config.OUTPUT_DIR / model_id / "studio.json").read_text())["params"]
    page.reload()
    expect(page.locator("#btnPause")).to_be_enabled(timeout=JOB)
    expect(page.locator("#utilStatus")).to_contain_text("Saved adjustments restored")
    assert not page.errors


def test_import_artwork_edit_a_region_then_separate(base_url, page, tmp_path):
    page.goto(f"{base_url}/generator")
    page.click("[data-source=upload]")
    page.set_input_files("#artwork", str(_artwork(tmp_path / "my oc.png")))
    # Box the big figure only, as on a reference sheet.
    box = page.locator("#artworkThumb").bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.1, box["y"] + 2)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] * 0.6, box["y"] + box["height"] - 2, steps=5)
    page.mouse.up()
    expect(page.locator("#cropBox")).to_be_visible()
    page.click("#generateBtn")
    expect(page.locator("#separateBtn")).to_be_visible(timeout=JOB)  # a prepared sketch first
    sketch_src = page.get_attribute("#resultImg", "src")

    page.click("#editPanel summary")
    page.click("#paintToggle")
    mask = page.locator("#maskCanvas").bounding_box()
    page.mouse.move(mask["x"] + mask["width"] * 0.4, mask["y"] + mask["height"] * 0.3)
    page.mouse.down()
    page.mouse.move(mask["x"] + mask["width"] * 0.6, mask["y"] + mask["height"] * 0.35, steps=5)
    page.mouse.up()
    page.fill("#editPrompt", "red witch hat")
    page.click("#editBtn")
    expect(page.locator("#resultImg")).not_to_have_attribute("src", sketch_src, timeout=JOB)
    expect(page.locator("#separateBtn")).to_be_visible(timeout=JOB)  # the edit is a new sketch

    page.click("#separateBtn")
    expect(page.locator("#livePreview")).to_be_visible(timeout=JOB)
    models = [p for p in config.OUTPUT_DIR.iterdir() if (p / "metadata.json").exists()]
    sources = sorted(json.loads((p / "metadata.json").read_text())["status"] for p in models)
    assert sources == ["completed", "sketch", "sketch"]  # import, edit, separated character
    assert not page.errors


def test_export_and_reopen_a_character_from_the_gallery(base_url, page, tmp_path):
    page.goto(f"{base_url}/generator")
    page.fill("#prompt", "1boy, black hair, hoodie")
    page.click("#generateBtn")
    expect(page.locator("#livePreview")).to_be_visible(timeout=JOB)

    page.goto(f"{base_url}/gallery")
    expect(page.locator("#collectionCount")).to_have_text("1 of 1 characters")
    # The Export link's file (fetched directly: Edge's download UI crashes Playwright 1.47's driver).
    href = page.locator(".model-card >> text=Export").get_attribute("href")
    exported = tmp_path / "export.zip"
    exported.write_bytes(page.request.get(base_url + href).body())
    assert "metadata.json" in zipfile.ZipFile(exported).namelist()

    page.set_input_files("#projectFile", str(exported))
    expect(page.locator("#collectionCount")).to_have_text("2 of 2 characters", timeout=30_000)
    expect(page.locator(".model-card h2", has_text="(imported)")).to_have_count(0)  # unnamed: shown by id
    reopened = page.locator(".model-card").nth(1)
    reopened.locator("text=Live Preview").click()
    expect(page.locator("#btnPause")).to_be_enabled(timeout=JOB)

    page.goto(f"{base_url}/gallery")
    page.locator(".model-card").first.locator("summary").click()
    page.locator(".model-card").first.locator("text=Delete character").click()
    expect(page.locator("#collectionCount")).to_have_text("1 of 1 characters")
    assert not page.errors
