"""
End-to-end journeys through the real UI in a browser, against an in-process server in demo mode
(placeholder generators: no GPU, seconds per character).

Needs Playwright (`pip install playwright`) and a browser: `python -m playwright install chromium`,
or an installed Edge/Chrome via KAMYII_E2E_CHANNEL=msedge|chrome. Skipped otherwise.
"""
import io
import json
import math
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


def test_preview_hair_shading_and_closed_eye_adjustments_survive_reload(base_url, page):
    import config
    # A small layered character with a painted bang shadow; no image generation or GPU.
    root = config.OUTPUT_DIR / "motion-study"
    assets = root / "assets"
    assets.mkdir(parents=True)
    def part(name, box, colour):
        image = Image.new("RGBA", (320, 400))
        ImageDraw.Draw(image).rectangle(box, fill=colour)
        image.save(assets / name)
    part("01_face.png", (70, 25, 250, 230), (245, 205, 185, 255))
    part("02_eyewhite-l.png", (95, 125, 125, 145), "white")
    part("03_eyewhite-r.png", (190, 125, 220, 145), "white")
    part("04_mouth.png", (150, 180, 170, 188), (90, 30, 40, 255))
    part("05_front hair.png", (110, 20, 180, 110), (90, 75, 115, 255))
    page.goto(f"{base_url}/preview/motion-study")
    expect(page.locator("#btnPause")).to_be_enabled(timeout=JOB)
    for control in ("tgIdle", "tgBlink"):
        if page.locator("#" + control).get_attribute("aria-pressed") == "true":
            page.locator("#" + control).click()
    def slider(name, value):
        page.locator("#" + name).evaluate("(el,value)=>{el.value=value;el.dispatchEvent(new Event('input',{bubbles:true}));el.dispatchEvent(new Event('change',{bubbles:true}));}", value)
    slider("pBangC", 0.7)
    page.wait_for_function("window.Anime25D.state().shading.pixels>0.2")
    assert page.evaluate("window.Anime25D.state().shading.vertices") > 0
    slider("pShadowFollow", 0)
    page.wait_for_function("window.Anime25D.state().shading.pixels<0.001")
    slider("pEyeL", 0)
    page.locator("#btnAnchorMode").click()
    handle = page.locator('[data-key="eyeLClose"]')
    handle.focus()
    page.keyboard.press("ArrowDown")
    assert page.evaluate("window.Anime25D.state().anchors.eyeLClose.dy") == 1
    page.keyboard.press("Escape")
    slider("pShadowFollow", 0.35)
    page.locator("#btnSaveSettings").dispatch_event("click")
    expect(page.locator("#appStatus")).to_contain_text("saved with the character")
    page.reload()
    expect(page.locator("#btnPause")).to_be_enabled(timeout=JOB)
    expect(page.locator("#utilStatus")).to_contain_text("Saved adjustments restored")
    assert page.evaluate("window.Anime25D.state().params.shadowFollow") == 0.35
    assert page.evaluate("window.Anime25D.state().anchors.eyeLClose.dy") == 1
    # Old saved default depths are upgraded; a deliberate depth edit is retained.
    saved_path = root / "studio.json"
    legacy = json.loads(saved_path.read_text())
    legacy.pop("rigRevision")
    for layer in legacy["layers"]:
        if "eye_close" in layer["id"]:
            layer["depth"] = 1.12
        elif "mouth" in layer["id"]:
            layer["depth"] = 1.3
    saved_path.write_text(json.dumps(legacy))
    page.reload()
    expect(page.locator("#btnPause")).to_be_enabled(timeout=JOB)
    page.locator("#btnSaveSettings").dispatch_event("click")
    expect(page.locator("#appStatus")).to_contain_text("saved with the character")
    migrated = json.loads(saved_path.read_text())
    assert all(layer["depth"] == 1 for layer in migrated["layers"] if "eye_close" in layer["id"])
    assert all(layer["depth"] == 1.3 for layer in migrated["layers"] if "mouth" in layer["id"])
    assert not page.errors


def test_shadow_shader_moves_painted_colour_without_moving_cutout(base_url, page):
    page.goto(base_url + "/health")
    page.add_script_tag(url=base_url + "/static/vendor/anime25d/lib/renderer.js")
    actual = page.evaluate("""() => {
      const canvas=document.createElement('canvas');canvas.width=canvas.height=64;
      const gl=canvas.getContext('webgl',{alpha:true,premultipliedAlpha:true});
      const r=RigRenderer.create(gl),part={},pixels=new Uint8ClampedArray(64*64*4);
      for(let y=8;y<56;y++)for(let x=8;x<56;x++){
        const i=(y*64+x)*4,dark=x>=24&&x<28;
        pixels.set(dark?[30,25,20,255]:[240,200,180,255],i);
      }
      r.upload(part,{positions:new Float32Array([0,0,64,0,0,64,64,64]),uvs:new Float32Array([0,0,1,0,0,1,1,1]),indices:new Uint16Array([0,1,2,1,3,2]),image:new ImageData(pixels,64,64)});
      const frame={width:64,height:64,masks:[],items:[{L:part,alpha:1}]};
      function capture(){r.draw(frame);const data=new Uint8Array(64*64*4);gl.readPixels(0,0,64,64,gl.RGBA,gl.UNSIGNED_BYTE,data);return data;}
      const before=capture();r.flow(part,new Float32Array([.125,0,.125,0,.125,0,.125,0]));const after=capture();
      let alphaSame=true,bx=0,ax=0,bn=0,an=0;
      for(let i=0;i<before.length;i+=4){if(before[i+3]!==after[i+3])alphaSame=false;const x=(i/4)%64;
        if(before[i+3]>200&&before[i]<70){bx+=x;bn++;}if(after[i+3]>200&&after[i]<70){ax+=x;an++;}}
      const error=gl.getError();r.dispose(part);
      return {alphaSame,before:bx/bn,after:ax/an,error};
    }""")
    assert actual["alphaSame"], "Moving the hair shadow must preserve the face silhouette"
    assert abs(actual["after"] - actual["before"] - 8) < 0.1
    assert actual["error"] == 0
    assert not page.errors


def test_manual_blink_editor_moves_rendered_art_and_keeps_old_models(base_url, page):
    root = config.OUTPUT_DIR / "old-blink-model"
    assets = root / "assets"
    assets.mkdir(parents=True)
    def part(name, box, colour):
        image = Image.new("RGBA", (320, 400))
        ImageDraw.Draw(image).rectangle(box, fill=colour)
        image.save(assets / name)
    part("01_face.png", (70, 25, 250, 230), (245, 205, 185, 255))
    part("02_eyewhite-l.png", (95, 125, 125, 145), "white")
    part("03_eyewhite-r.png", (190, 125, 220, 145), "white")
    part("04_mouth.png", (150, 180, 170, 188), (90, 30, 40, 255))
    expression_dir = root / "expressions"
    expression_dir.mkdir()
    painted = Image.new("RGBA", (320, 400))
    d = ImageDraw.Draw(painted)
    d.rectangle((90, 115, 130, 150), fill=(245, 205, 185, 255))
    d.line((95, 140, 125, 140), fill="black", width=3)
    d.rectangle((185, 105, 225, 145), fill=(245, 205, 185, 255))
    d.line((190, 130, 220, 130), fill="black", width=3)
    painted.save(expression_dir / "eye_close.png")
    (expression_dir / "expressions.json").write_text(json.dumps({"eye_close": "expressions/eye_close.png"}))
    # Observe real renderer vertices rather than just checking slider/state values.
    page.add_init_script("""Object.defineProperty(window,'RigRenderer',{configurable:true,set(api){
      Object.defineProperty(window,'RigRenderer',{value:{...api,create(gl){const r=api.create(gl),positions=r.positions;
        r.positions=(L,p)=>{if(L.fade==='eyeClose'){window.__closed=window.__closed||{};window.__closed[L.side]=Array.from(p);}positions(L,p);};return r;}}});
    }});""")
    page.goto(f"{base_url}/preview/old-blink-model")
    expect(page.locator("#btnBlinkEdit")).to_be_enabled(timeout=JOB)
    page.locator("#btnBlinkEdit").click()
    page.wait_for_function("window.Anime25D.state().blinkEditor && window.Anime25D.state().pose.eyeL===0")
    before = page.evaluate("window.__closed.L")
    page.locator("#blinkLX").fill("7")
    page.locator("#blinkLY").fill("-9")
    page.locator("#blinkLY").press("Tab")
    page.wait_for_function("window.Anime25D.state().anchors.eyeLClose?.dy===-9")
    page.wait_for_timeout(80)
    after = page.evaluate("window.__closed.L")
    assert all(abs(after[i] - before[i] - (7 if i % 2 == 0 else -9)) < 0.001 for i in range(len(before)))
    assert "eyeRClose" not in page.evaluate("window.Anime25D.state().anchors")
    # Nudge through the actual on-stage control, then undo without losing either axis.
    page.locator('[data-key="eyeLClose"]').focus()
    page.keyboard.press("Shift+ArrowRight")
    assert page.evaluate("window.Anime25D.state().anchors.eyeLClose.dx") == 17
    page.locator("#btnUndo").click()
    assert page.evaluate("window.Anime25D.state().anchors.eyeLClose.dx") == 7
    page.locator('[data-blink-view="open"]').click()
    page.wait_for_function("window.Anime25D.state().pose.eyeL===1 && window.Anime25D.state().pose.eyeR===1")
    page.locator('[data-blink-view="R"]').click()
    page.wait_for_function("window.Anime25D.state().pose.eyeL===1 && window.Anime25D.state().pose.eyeR===0")
    page.locator("#pEyeTiltR").evaluate("el=>{el.value=6;el.dispatchEvent(new Event('input'));el.dispatchEvent(new Event('change'));}")
    page.locator("#btnBlinkSave").click()
    expect(page.locator("#appStatus")).to_contain_text("saved with the character")
    kept = json.loads((root / "studio.json").read_text())
    assert kept["params"]["eyeOpenL"] == kept["params"]["eyeOpenR"] == 1, "Do not save the temporary alignment pose"
    page.reload()
    expect(page.locator("#btnBlinkEdit")).to_be_enabled(timeout=JOB)
    assert not page.evaluate("window.Anime25D.state().blinkEditor")
    expect(page.locator("#blinkLX")).to_have_value("7")
    expect(page.locator("#blinkLY")).to_have_value("-9")
    expect(page.locator("#pEyeTiltR")).to_have_value("6")
    page.locator("#btnBlinkReset").click()
    expect(page.locator("#blinkLX")).to_have_value("0")
    expect(page.locator("#pEyeTiltR")).to_have_value("0")
    page.locator("#btnBlinkMatch").click()
    fitted = page.evaluate("window.Anime25D.state().anchors")
    assert fitted["eyeLClose"]["dy"] < fitted["eyeRClose"]["dy"], "Correct the two independently painted heights against the open eyes"
    page.locator("#blinkStyle").select_option("1")
    assert page.evaluate("window.Anime25D.state().synth.eye"), "Replace damaged painting with the clean fallback"
    page.locator("#btnUndo").click()
    assert not page.evaluate("window.Anime25D.state().synth.eye"), "Undo restores the painted blink"
    page.locator("#btnRedo").click()
    assert page.evaluate("window.Anime25D.state().synth.eye")
    page.locator("#btnBlinkSave").click()
    expect(page.locator("#appStatus")).to_contain_text("saved with the character")
    page.reload()
    expect(page.locator("#btnBlinkEdit")).to_be_enabled(timeout=JOB)
    expect(page.locator("#blinkStyle")).to_have_value("1")
    assert page.evaluate("window.Anime25D.state().synth.eye"), "Keep the chosen artwork across reopening"
    page.locator("#btnBlinkEdit").click()
    page.locator("#btnBlinkDone").click()
    assert not page.evaluate("window.Anime25D.state().blinkEditor")
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
    assert not page.errors


def test_fluid_speech_bouncy_motion_and_fitted_nekomimi_survive_reopening(base_url, page):
    root = config.OUTPUT_DIR / "companion-study"
    assets = root / "assets"
    assets.mkdir(parents=True)
    for name, box, colour in [
        ("01_back hair.png", (65, 60, 255, 350), (135, 115, 155, 255)),
        ("02_topwear.png", (55, 245, 265, 399), (75, 90, 125, 255)),
        ("03_face.png", (85, 90, 235, 240), (245, 205, 185, 255)),
        ("04_eyewhite-l.png", (105, 145, 130, 163), "white"),
        ("05_eyewhite-r.png", (190, 145, 215, 163), "white"),
        ("06_mouth.png", (148, 200, 172, 216), (90, 30, 40, 255)),
        ("07_front hair.png", (80, 60, 240, 130), (145, 125, 165, 255)),
    ]:
        image = Image.new("RGBA", (320, 400))
        ImageDraw.Draw(image).rectangle(box, fill=colour)
        image.save(assets / name)
    emotes=root/"expressions";emotes.mkdir()
    for key,colour in [("hearts",(245,100,150,255)),("stars",(255,220,90,255)),("squeeze",(60,35,45,255))]:
        image=Image.new("RGBA",(320,400));draw=ImageDraw.Draw(image)
        for x in (100,185):draw.rectangle((x,140,x+35,170),fill=colour)
        image.save(emotes/f"toggle_{key}.png")
    (emotes/"expressions.json").write_text(json.dumps({"toggles":{key:f"expressions/toggle_{key}.png" for key in ("hearts","stars","squeeze")}}))
    page.add_init_script("""Object.defineProperty(window,'RigRenderer',{configurable:true,set(api){
      Object.defineProperty(window,'RigRenderer',{value:{...api,create(gl){const r=api.create(gl),draw=r.draw;
        const upload=r.upload;r.upload=(L,m)=>{if(L.expression){window.__mouthTextures=window.__mouthTextures||{};const d=m.image.data;let white=0;for(let i=0;i<d.length;i+=4)if(d[i]>220&&d[i+1]>210&&d[i+2]>195&&d[i+3]>200)white++;window.__mouthTextures[L.expression]={white};}upload(L,m);};
        r.draw=frame=>{window.__motionFrame=frame.items.map(({L,alpha})=>({name:L.name,alpha,points:Array.from(L.cur)}));draw(frame);window.__glError=gl.getError();};return r;}}});
    }});""")
    page.add_init_script("""const anchorClick=HTMLAnchorElement.prototype.click;
      HTMLAnchorElement.prototype.click=function(){if(this.download.endsWith('.png')){
        fetch(this.href).then(r=>r.blob()).then(blob=>{const reader=new FileReader();reader.onload=()=>window.__exportedPng=reader.result;reader.readAsDataURL(blob);});
      }else anchorClick.call(this);};""")
    page.goto(f"{base_url}/preview/companion-study")
    expect(page.locator("#btnPause")).to_be_enabled(timeout=JOB)
    assert not page.evaluate("window.Anime25D.state().companions.ears")
    def slider(name, value):
        page.locator("#" + name).evaluate("(el,value)=>{el.value=value;el.dispatchEvent(new Event('input'));el.dispatchEvent(new Event('change'));}", value)
    for control in ("tgIdle", "tgBlink"):
        if page.locator("#" + control).get_attribute("aria-pressed") == "true":
            page.locator("#" + control).click()
    page.locator("#toggleBtns button").filter(has_text="Heart eyes").click()
    assert page.locator("#toggleBtns button").count()==3, "Each split-eye design has one control"
    slider("pEyeL",0)
    page.wait_for_function("window.Anime25D.state().pose.eyeL<.001")
    page.wait_for_timeout(120)
    assert not page.evaluate("window.__motionFrame.some(L=>L.name==='toggle heart eyes_l')")
    assert page.evaluate("window.__motionFrame.some(L=>L.name==='toggle heart eyes_r')")
    assert page.evaluate("window.__motionFrame.some(L=>L.name==='eye_close_l')"), "The closed lid replaces the heart overlay"
    page.locator("#toggleBtns button").filter(has_text="Star eyes").click()
    page.wait_for_function("!window.__motionFrame.some(L=>L.name.startsWith('toggle heart'))")
    page.locator("#toggleBtns button").filter(has_text="> < eyes").click()
    page.wait_for_timeout(100)
    assert not page.evaluate("window.__motionFrame.some(L=>L.name.startsWith('eye_close')||L.name.startsWith('eyewhite'))"), "Squeezed-eye art replaces both original eye states"
    page.locator("#toggleBtns button").filter(has_text="> < eyes").click()
    slider("pEyeL",1)
    page.locator("#btnNaturalSway").click()
    expect(page.locator("#pWeightShift")).to_have_value("0.9")
    page.wait_for_function("Math.abs(window.Anime25D.state().pose.torso.shift)>.03")
    page.locator("#tgIdle").click()
    page.locator("#tgPhys").click()
    slider("pMouthOpen", .1)
    page.wait_for_function("window.Anime25D.state().pose.mouth>.099")
    small = page.evaluate("window.__motionFrame.find(L=>L.name==='mouth_open').alpha")
    assert .02 < small < .7, "Small speech openings blend into compressed art instead of popping"
    slider("pMouthOpen", 1)
    page.wait_for_function("window.Anime25D.state().pose.mouth>.999")
    page.wait_for_timeout(100)
    opened = page.evaluate("window.__motionFrame.find(L=>L.name==='mouth_open')")
    def span(points, axis):
        return max(points[axis::2]) - min(points[axis::2])
    slider("pMouthRound", 1)
    page.wait_for_timeout(600)
    rounded = page.evaluate("window.__motionFrame.find(L=>L.name==='mouth_round')")
    assert rounded["alpha"] > .95 and not page.evaluate("window.__motionFrame.some(L=>L.name==='mouth_open')")
    page.locator('[data-preset="grin"]').click()
    page.wait_for_function("window.__motionFrame.some(L=>L.name==='mouth_grin' && L.alpha>.95)")
    assert page.evaluate("window.__mouthTextures.grin.white") > 150
    page.locator('[data-preset="jawdrop"]').click()
    page.wait_for_function("window.__motionFrame.some(L=>L.name==='mouth_round' && L.alpha>.95)")
    assert page.evaluate("window.__mouthTextures.round.white") == 0
    page.locator("#btnSaveSettings").dispatch_event("click")
    expect(page.locator("#appStatus")).to_contain_text("saved with the character")
    page.reload()
    expect(page.locator("#btnPause")).to_be_enabled(timeout=JOB)
    assert page.evaluate("window.Anime25D.state().activePreset") == "jawdrop"
    page.locator('[data-preset="jawdrop"]').click()
    page.wait_for_function("window.Anime25D.state().params.jawDrop===0 && window.Anime25D.state().params.mouthRound===0")
    page.locator('[data-preset="smile"]').click()
    page.locator("#tgTalk").click()
    page.wait_for_function("window.Anime25D.state().pose.mouth>.2")
    assert page.evaluate("window.Anime25D.state().activePreset") == "smile", "Talking must work while retaining the chosen expression"
    page.locator("#tgTalk").click()
    page.locator('[data-preset="smile"]').click()
    page.locator("#btnSpringNod").click()
    page.locator("#nekoStyle").select_option("1")
    page.locator("#tailStyle").select_option("1")
    page.wait_for_function("Math.abs(window.Anime25D.state().pose.nod)>.04")
    samples = []
    for _ in range(16):
        samples.append(page.evaluate("""()=>{
          const mean=L=>L.points.filter((_,i)=>i%2).reduce((a,b)=>a+b,0)/(L.points.length/2);
          return {head:mean(window.__motionFrame.find(L=>L.name==='face')),
            torso:mean(window.__motionFrame.find(L=>L.name==='topwear'))};
        }"""))
        page.wait_for_timeout(55)
    head_span = max(s["head"] for s in samples) - min(s["head"] for s in samples)
    torso_span = max(s["torso"] for s in samples) - min(s["torso"] for s in samples)
    assert head_span > 15 and torso_span < head_span * .5, "The rendered head must lead the quieter torso"
    page.locator("#btnSaveSettings").dispatch_event("click")
    expect(page.locator("#appStatus")).to_contain_text("saved with the character")
    page.reload()
    expect(page.locator("#btnPause")).to_be_enabled(timeout=JOB)
    expect(page.locator("#pNodDepth")).to_have_value("0.85")
    expect(page.locator("#pNodSpeed")).to_have_value("2.6")
    # Presets remain mutually understandable: Bouncy returns to the earlier idle motion.
    page.locator('[data-motion="1.4"]').click()
    expect(page.locator("#pNodDepth")).to_have_value("0")
    for control in ("tgIdle", "tgBlink"):
        if page.locator("#" + control).get_attribute("aria-pressed") == "true":
            page.locator("#" + control).click()
    page.locator("#nekoStyle").select_option("1")
    page.locator("#tailStyle").select_option("1")
    page.locator("#furPalette").select_option("3")
    page.locator("#tailSide").select_option("-1")
    slider("pEarSize", .85)
    slider("pEarLift", -.12)
    slider("pTailSize", .7)
    page.locator('[data-motion="1.4"]').click()
    page.locator("#tgIdle").click()
    page.wait_for_function("window.__motionFrame.filter(L=>L.name.startsWith('companion')).length===3")
    first = page.evaluate("window.__motionFrame.filter(L=>L.name.startsWith('companion'))")
    page.wait_for_timeout(450)
    second = page.evaluate("window.__motionFrame.filter(L=>L.name.startsWith('companion'))")
    assert all(a["points"] != b["points"] for a, b in zip(first, second)), "Both ears and the tail animate in the actual renderer"
    assert page.evaluate("window.__glError") == 0
    for turn in (1, -1, 1, -1, 0):
        slider("pAngleX", turn)
        page.wait_for_timeout(150)
        assert page.evaluate("window.__motionFrame.every(L=>L.points.every(Number.isFinite))"), "Rapid turns must keep all deformed meshes finite"
    page.locator("#tgPhys").click()
    page.locator("#tgIdle").click()
    page.wait_for_timeout(1500)
    still = page.evaluate("window.__motionFrame.find(L=>L.name==='companion tail').points")
    page.wait_for_timeout(150)
    assert still == page.evaluate("window.__motionFrame.find(L=>L.name==='companion tail').points")
    # A full head tilt rolls the rendered head about 16° on the neck; the torso stays nearly upright.
    def diagonal(name):
        p = page.evaluate(f"window.__motionFrame.find(L=>L.name==='{name}').points")
        return math.atan2(p[-1] - p[1], p[-2] - p[0])
    rest = diagonal("face"), diagonal("topwear")
    shirt = page.evaluate("window.__motionFrame.find(L=>L.name==='topwear').points")
    collar = min(range(0, len(shirt), 2), key=lambda i: (shirt[i] - 160) ** 2 + (shirt[i + 1] - 245) ** 2)
    slider("pAngleZ", 1)
    page.wait_for_function("window.Anime25D.state().pose.tilt>.999")
    page.wait_for_timeout(100)
    assert .26 < abs(diagonal("face") - rest[0]) < .36, "Full tilt is a visible head roll"
    assert .012 < abs(diagonal("topwear") - rest[1]) < .065, "The torso visibly follows while the head retains its own larger roll"
    tilted = page.evaluate("window.__motionFrame.find(L=>L.name==='topwear').points")
    assert (tilted[collar] - shirt[collar]) - (tilted[0] - shirt[0]) > 4, "The collar bends with the neck, beyond the torso's lean"
    slider("pAngleZ", 0)
    slider("pShrug", 1)
    page.wait_for_timeout(600)
    shrugged = page.evaluate("window.__motionFrame.find(L=>L.name==='topwear').points")
    assert shirt[1] - shrugged[1] > 5 and abs(shrugged[collar + 1] - shirt[collar + 1]) < 1, "Shrug lifts the shoulders, not the collar"
    assert abs(shrugged[-1] - shirt[-1]) < .5, "A cropped chest stays on the canvas bottom"
    slider("pShrug", 0)
    page.locator("#btnHeadSway").click()
    expect(page.locator("#pNodDepth")).to_have_value("0")
    page.wait_for_function("window.Anime25D.state().pose.tilt<-.5")
    page.wait_for_function("window.Anime25D.state().pose.tilt>.5")
    page.locator('[data-motion="1.4"]').click()
    expect(page.locator("#pSwayDepth")).to_have_value("0")
    page.locator("#tgPhys").click()
    page.locator("#tgIdle").click()
    page.locator("#btnSaveSettings").dispatch_event("click")
    expect(page.locator("#appStatus")).to_contain_text("saved with the character")
    kept = json.loads((root / "studio.json").read_text())["params"]
    assert kept["neko"] == kept["tail"] == 1 and kept["furPalette"] == 3
    assert kept["earLift"] == -.12 and kept["motionBounce"] == 1.4
    page.reload()
    expect(page.locator("#btnPause")).to_be_enabled(timeout=JOB)
    expect(page.locator("#nekoStyle")).to_have_value("1")
    expect(page.locator("#tailSide")).to_have_value("-1")
    expect(page.locator("#pTailSize")).to_have_value("0.7")
    page.wait_for_function("window.__motionFrame.filter(L=>L.name.startsWith('companion')).length===3")
    canvas = page.evaluate("window.Anime25D.state().canvas")
    assert canvas["width"] > 320 or canvas["height"] > 400, "Add transparent space for companions without changing artist coordinates"
    page.locator("#btnPng").click()
    page.wait_for_function("window.__exportedPng")
    import base64
    png = Image.open(io.BytesIO(base64.b64decode(page.evaluate("window.__exportedPng.split(',')[1]"))))
    assert png.size == (canvas["width"], canvas["height"]) and png.mode == "RGBA"
    assert png.getextrema()[3] == (0, 255), "Transparent PNG export includes rendered art"
    page.locator("#tailStyle").select_option("0")
    page.wait_for_function("!window.__motionFrame.some(L=>L.name==='companion tail')")
    page.locator("#btnUndo").click()
    page.wait_for_function("window.__motionFrame.some(L=>L.name==='companion tail')")
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
    assert not page.errors


def test_preview_full_body_action_supports_finished_models_and_artwork_review(base_url, page):
    page.goto(f"{base_url}/generator")
    page.fill("#prompt", "1girl, upper body, silver hair, blue hoodie")
    page.click("#generateBtn")
    expect(page.locator("#livePreview")).to_be_visible(timeout=JOB)
    original = page.get_attribute("#livePreview", "href").rsplit("/", 1)[-1]
    original_art = (config.OUTPUT_DIR / original / "base_image.png").read_bytes()
    page.locator("#livePreview").click()
    expect(page.locator("#btnPause")).to_be_enabled(timeout=JOB)
    page.locator("#btnFullBody").click()
    expect(page.locator("#fullBodyPanel")).to_have_attribute("open", "")
    expect(page.locator("#extendFullBtn")).to_be_visible()
    expect(page.locator("#fullBodyNotice")).to_contain_text("Demo mode")
    page.locator("#fullBodyFit > summary").click()
    expect(page.locator("#fullBodyFitNote")).to_contain_text("Measured from this character",timeout=JOB)
    page.locator("#fullBodyProportions").select_option("tall")
    page.wait_for_function("document.querySelector('#fullBodyGuide').getContext('2d').getImageData(0,0,416,608).data.some((v,i)=>i%4===3&&v>0)")
    expect(page.locator("#fullBodyFitNote")).to_contain_text("Drag the shoulder, hip and ankle dots", timeout=JOB)
    # Drag the left hip dot outwards and down: both hips move, and the stance goes with the request.
    plan = page.request.get(f"{base_url}/api/generation/extend/{original}/plan?proportions=tall").json()["info"]
    guide = page.locator("#fullBodyGuide").bounding_box()
    k = 608 / plan["canvas"][1] * guide["width"] / 416
    hip = plan["pose_points"][8]
    page.mouse.move(guide["x"] + hip[0] * k, guide["y"] + hip[1] * k)
    page.mouse.down()
    page.mouse.move(guide["x"] + hip[0] * k - 12, guide["y"] + hip[1] * k + 6, steps=4)
    page.mouse.up()
    expect(page.locator("#fitReset")).to_be_visible(timeout=JOB)
    page.fill("#fullBodyPrompt", "black boots")
    page.locator("#extendFullBtn").click()
    expect(page.locator("#livePreview")).not_to_have_attribute("href", f"/preview/{original}", timeout=JOB)
    expect(page.locator("#livePreview")).to_be_visible(timeout=JOB)
    new_id = page.get_attribute("#livePreview", "href").rsplit("/", 1)[-1]
    assert new_id != original
    assert f"model={new_id}" in page.url, "Reload must reopen the full-body result, not its parent"
    model = json.loads((config.OUTPUT_DIR / new_id / "metadata.json").read_text())
    assert model["status"] == "completed" and model["request"]["framing"] == "full_body"
    assert not model["request"]["sketch"] and "black boots" in model["request"]["prompt"]
    assert model["request"]["extension"]["proportions"]=="tall" and model["request"]["pose_image"] is None
    hips = model["request"]["extension"]["guides"]["hips"]
    assert hips[0] > (plan["pose_points"][11][0] - plan["pose_points"][8][0]) / 2 and hips[1] > hip[1], "The dragged stance was used"
    expect(page.locator("#resultChecks")).to_contain_text("Automatic checks")
    assert (config.OUTPUT_DIR / original / "base_image.png").read_bytes() == original_art
    page.locator("#showFullBody").click()
    page.locator("#fullBodyOutput").select_option("artwork")
    page.locator("#extendFullBtn").click()
    expect(page.locator("#separateBtn")).to_be_visible(timeout=JOB)
    result_url = page.url
    page.reload()
    expect(page.locator("#separateBtn")).to_be_visible(timeout=JOB)
    assert page.url == result_url
    assert not page.errors
