"""Local 3D stage: self-contained file safety and rendering, no GPU generation."""
import json
import os
from pathlib import Path
import struct
import threading
import time
import urllib.request
import pytest

sync_api = pytest.importorskip("playwright.sync_api")

@pytest.fixture
def stage():
    from werkzeug.serving import make_server
    import app
    server=make_server("127.0.0.1",0,app.create_app(use_dummy_generators=True),threaded=True)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    yield f"http://127.0.0.1:{server.server_port}/3d"
    # Jobs still running must finish before outputs/ is pointed back at the real folder.
    deadline=time.time()+60
    while time.time()<deadline:
        jobs=json.load(urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/generation/jobs"))["jobs"]
        if not any(j["state"] in ("queued","running") for j in jobs):break
        time.sleep(.25)
    server.shutdown()

@pytest.fixture
def page():
    with sync_api.sync_playwright() as p:
        try:
            browser=p.chromium.launch(channel=os.environ.get("KAMYII_E2E_CHANNEL") or None,
                ignore_default_args=["--headless=old"], args=["--headless=new", "--enable-unsafe-swiftshader"])
        except sync_api.Error as e:
            pytest.skip(f"No browser: {e}")
        page=browser.new_page(viewport={"width":1400,"height":1000});page.errors=[]
        page.on("pageerror",lambda e:page.errors.append(str(e)))
        page.console_messages=[]
        page.on("console",lambda m:page.console_messages.append((m.type,m.text)) if m.type in ("warning","error") else None)
        yield page
        browser.close()

def glb(external=False):
    data=struct.pack("<9f",-.5,0,0,.5,0,0,0,1,0)
    manifest={"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],
        "meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],
        "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[-.5,0,0],"max":[.5,1,0]}],
        "bufferViews":[{"buffer":0,"byteLength":len(data)}],"buffers":[{"byteLength":len(data)}]}
    if external:manifest["buffers"][0]["uri"]="http://127.0.0.1:1/should-not-be-requested.bin"
    body=json.dumps(manifest,separators=(",",":")).encode();body+=b" "*(-len(body)%4)
    return struct.pack("<III",0x46546c67,2,20+len(body)+8+len(data))+struct.pack("<II",len(body),0x4e4f534a)+body+struct.pack("<II",len(data),0x004e4942)+data

def upload(page,data,name="model.glb"):
    page.locator("#d3-file").set_input_files({"name":name,"mimeType":"model/gltf-binary","buffer":data})

def test_3d_glb_local_only(stage,page):
    page.goto(stage);page.wait_for_function("window.Kamyii3D")
    seen=[];page.on("request",lambda r:seen.append(r.url))
    upload(page,glb(True))
    sync_api.expect(page.locator("#d3-status")).to_contain_text("external resources")
    assert not any("should-not-be-requested" in url for url in seen)
    upload(page,b"not a real model")
    sync_api.expect(page.locator("#d3-status")).to_contain_text("valid VRM")
    upload(page,glb())
    sync_api.expect(page.locator("#d3-status")).to_contain_text("GLB ready",timeout=20000)
    assert page.evaluate("Kamyii3D.state.loaded && !Kamyii3D.state.vrm")
    assert page.locator("#d3-camera").is_disabled() and page.locator("#d3-fit").is_enabled()
    upload(page,b"bad replacement")
    sync_api.expect(page.locator("#d3-status")).to_contain_text("valid VRM")
    assert page.evaluate("Kamyii3D.state.loaded"), "A failed replacement keeps the existing model"
    page.set_viewport_size({"width":390,"height":844})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth+1")
    assert not page.errors
    print("GLB console messages:",page.console_messages)

def test_real_vrm(stage,page):
    path=os.environ.get("KAMYII_TEST_VRM")
    if not path:pytest.skip("Set KAMYII_TEST_VRM to a licensed local VRM sample")
    page.add_init_script("""(() => {
      window.__cameraRequests = 0;
      Object.defineProperty(navigator.mediaDevices, 'getUserMedia', {value: async () => {
        window.__cameraRequests++; throw new DOMException('Denied for test', 'NotAllowedError');
      }});
    })();""")
    page.goto(stage);page.wait_for_function("window.Kamyii3D")
    page.locator("#d3-file").set_input_files(str(Path(path).resolve()))
    page.wait_for_function("Kamyii3D.state.vrm",timeout=60000)
    assert page.locator("#d3-camera").is_enabled()
    page.locator("#d3-idle").click()
    page.locator("#d3-turn").fill("0.7")
    page.wait_for_function("Kamyii3D.state.headY > .25",timeout=5000)
    page.locator("#d3-mouth").fill("0.7")
    assert page.locator("#d3-mouth-value").inner_text()=="70%"
    chips=page.locator("#d3-expressions button")   # the avatar's own emotions, e.g. VRM 0 Joy -> happy
    assert chips.count()>0 and page.locator("#d3-expressions-wrap").is_visible()
    chips.first.click();assert chips.first.get_attribute("aria-pressed")=="true"
    chips.first.click();assert chips.first.get_attribute("aria-pressed")=="false"
    page.screenshot(path="outputs/vrm-review/vrm-stage-final.png")
    page.locator("#d3-reset").click();page.wait_for_function("Math.abs(Kamyii3D.state.headY) < .02",timeout=5000)
    for attempt in (1,2):
        page.locator("#d3-camera").click()
        sync_api.expect(page.locator("#d3-camera-status")).to_contain_text("permission was denied",timeout=20000)
        sync_api.expect(page.locator("#d3-camera")).to_have_text("Turn camera on")
        assert not page.evaluate("Kamyii3D.state.cameraOn")
        assert page.evaluate("window.__cameraRequests") == attempt
    assert not page.errors
    print("VRM console messages:",page.console_messages)

def test_gallery_makes_a_3d_avatar_and_the_stage_opens_it(stage,page):
    """Demo mode: Gallery "Make 3D avatar" queues the job (a mannequin VRM without a GPU), then the
    card's 3D link opens it on the stage, posed and listed among the character's avatars."""
    import time
    import config
    from PIL import Image
    d=config.OUTPUT_DIR/"char-3d";d.mkdir(parents=True)
    Image.new("RGB",(64,96),(220,150,160)).save(d/"base_image.png")
    (d/"metadata.json").write_text(json.dumps({"id":"char-3d","name":"model_3d0001","status":"sketch","created_at":"2026-10-28T00:00:00",
        "request":{"prompt":"1girl"},"metadata":{},"assets":[]}))
    base=stage[:-3];page.on("dialog",lambda dialog:dialog.accept())
    page.goto(base+"/gallery")
    page.get_by_role("button",name="Make 3D avatar").click()
    page.wait_for_url("**/generator?model=char-3d")
    for _ in range(150):
        if json.loads((d/"metadata.json").read_text()).get("metadata",{}).get("avatar3d"):break
        time.sleep(.2)
    else:pytest.fail("the demo 3D job did not finish")
    page.goto(base+"/gallery")
    page.get_by_role("link",name="3D avatar ↗").click()
    page.wait_for_url("**/3d?model=char-3d")
    page.wait_for_function("window.Kamyii3D && Kamyii3D.state.vrm",timeout=60000)
    sync_api.expect(page.locator("#d3-model-name")).to_have_text("Study 3D0001")
    assert page.locator("#d3-characters").input_value()=="char-3d" and page.locator("#d3-download").is_visible()
    page.locator("#d3-turn").fill("0.7")
    page.wait_for_function("Kamyii3D.state.headY > .25",timeout=5000)
    assert not page.errors

def test_studio_creates_a_2d_and_3d_character(stage,page):
    """Studio "Avatar: 2D + 3D": full body is forced, and the result offers the 3D avatar (demo mannequin)."""
    base=stage[:-3];page.on("dialog",lambda dialog:dialog.accept())
    page.goto(base+"/generator")
    page.wait_for_function("!document.querySelector('#avatarKind').options[1].disabled")
    page.select_option("#avatarKind","3d")
    assert page.locator("#framing").input_value()=="full_body" and page.locator("#framing").is_disabled()
    page.fill("#prompt","1girl, pink hair, twintails")
    page.click("#generateBtn")
    sync_api.expect(page.locator("#open3d")).to_be_visible(timeout=120000)
    sync_api.expect(page.locator("#livePreview")).to_be_visible()
    page.locator("#open3d").click()
    page.wait_for_url("**/3d?model=*")
    page.wait_for_function("window.Kamyii3D && Kamyii3D.state.vrm",timeout=60000)
    assert not page.errors
