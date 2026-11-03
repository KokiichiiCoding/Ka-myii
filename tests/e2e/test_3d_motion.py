"""Exercise the actual VRM bones in the local 3D stage."""
import os
from pathlib import Path

import pytest
from test_3d_stage import stage, page


def test_head_turn_has_smaller_torso_follow_and_returns_to_neutral(stage, page):
    path = os.environ.get('KAMYII_TEST_GENERATED_VRM') or os.environ.get('KAMYII_TEST_VRM')
    if not path or not Path(path).is_file():
        pytest.skip('Set KAMYII_TEST_GENERATED_VRM or KAMYII_TEST_VRM to a local VRM')
    page.goto(stage)
    page.locator('#d3-file').set_input_files(str(Path(path).resolve()))
    page.wait_for_function('window.Kamyii3D && Kamyii3D.state.vrm', timeout=60000)
    page.locator('#d3-idle').click()
    page.locator('#d3-turn').fill('1')
    page.wait_for_function('Kamyii3D.state.headY > .43 && Kamyii3D.state.chestY + Kamyii3D.state.spineY > .075', timeout=6000)
    state = page.evaluate('Kamyii3D.state')
    assert state['chestY'] + state['spineY'] < state['headY'] * .25
    page.locator('#d3-reset').click()
    page.wait_for_function('Math.abs(Kamyii3D.state.headY) < .02 && Math.abs(Kamyii3D.state.chestY + Kamyii3D.state.spineY) < .02', timeout=6000)
    assert not page.errors


def test_tracking_drives_face_and_falls_back_to_idle(stage, page):
    path = os.environ.get('KAMYII_TEST_GENERATED_VRM')
    if not path or not Path(path).is_file():
        pytest.skip('Set KAMYII_TEST_GENERATED_VRM to a generated VRM')
    page.route('**/static/js/3d_tracking.js', lambda route: route.fulfill(
        content_type='application/javascript', body='''
        window.Kamyii3DTracking={create(o){
          window.__pose=p=>o.onPose(p);
          return {start:async()=>{o.onState('on','Camera on');return true},
            stop:()=>{o.onPose(null);o.onState('off','Camera off')},
            calibrate:async()=>true,dispose:()=>{}};
        }};'''))
    page.goto(stage)
    page.locator('#d3-file').set_input_files(str(Path(path).resolve()))
    page.wait_for_function('window.Kamyii3D && Kamyii3D.state.vrm', timeout=60000)
    page.locator('#d3-camera').click()
    page.evaluate('''__pose({ax:.7,ay:.2,az:.1,eL:1,eR:0,mo:.8})''')
    page.wait_for_function('''Kamyii3D.state.headY > .3 &&
        Kamyii3D.state.expressions.blinkLeft > .9 &&
        Kamyii3D.state.expressions.blinkRight < .1 &&
        Kamyii3D.state.expressions.aa > .7''', timeout=6000)
    assert not page.evaluate('Kamyii3D.state.idleActive')
    page.evaluate('__pose(null)')
    assert page.evaluate('Kamyii3D.state.cameraOn && Kamyii3D.state.idleActive')
    page.wait_for_function('Math.abs(Kamyii3D.state.headY) < .07', timeout=6000)
    assert not page.errors


def test_gestures_raise_the_arm_hop_and_return_to_rest(stage, page):
    path = os.environ.get('KAMYII_TEST_GENERATED_VRM') or os.environ.get('KAMYII_TEST_VRM')
    if not path or not Path(path).is_file():
        pytest.skip('Set KAMYII_TEST_GENERATED_VRM or KAMYII_TEST_VRM to a local VRM')
    page.goto(stage)
    page.locator('#d3-file').set_input_files(str(Path(path).resolve()))
    page.wait_for_function('window.Kamyii3D && Kamyii3D.state.vrm', timeout=60000)
    page.locator('#d3-idle').click()
    page.get_by_role('button', name='Wave').click()
    page.wait_for_function('Kamyii3D.state.gesture === "wave" && Kamyii3D.state.rightArmLift > .5', timeout=4000)
    page.wait_for_function('Kamyii3D.state.gesture === null && Math.abs(Kamyii3D.state.rightArmLift) < 1e-3', timeout=5000)
    page.locator('body').press('4')  # Hop
    page.wait_for_function('Kamyii3D.state.hipsLift > .005', timeout=3000)  # busts bob less
    page.wait_for_function('Kamyii3D.state.gesture === null && Math.abs(Kamyii3D.state.hipsLift) < 1e-6', timeout=4000)
    assert not page.errors


def test_microphone_alone_drives_the_mouth(stage, page):
    path = os.environ.get('KAMYII_TEST_GENERATED_VRM') or os.environ.get('KAMYII_TEST_VRM')
    if not path or not Path(path).is_file():
        pytest.skip('Set KAMYII_TEST_GENERATED_VRM or KAMYII_TEST_VRM to a local VRM')
    # A synthetic voice: an oscillator stream instead of a real microphone.
    page.add_init_script('''navigator.mediaDevices.getUserMedia = async () => {
        const ac = new AudioContext(), o = ac.createOscillator(), d = ac.createMediaStreamDestination();
        o.type = 'sawtooth'; o.frequency.value = 180; o.connect(d); o.start(); return d.stream; };''')
    page.goto(stage)
    page.locator('#d3-file').set_input_files(str(Path(path).resolve()))
    page.wait_for_function('window.Kamyii3D && Kamyii3D.state.vrm', timeout=60000)
    page.get_by_role('button', name='Use microphone').click()
    page.wait_for_function('''Kamyii3D.state.micOn && ['aa','ih','ou','ee','oh']
        .reduce((s, k) => s + (Kamyii3D.state.expressions[k] || 0), 0) > .5''', timeout=8000)
    page.get_by_role('button', name='Microphone off').click()
    page.wait_for_function('''!Kamyii3D.state.micOn && ['aa','ih','ou','ee','oh']
        .reduce((s, k) => s + (Kamyii3D.state.expressions[k] || 0), 0) < .05''', timeout=4000)
    assert not page.errors


def test_record_a_clip_on_the_green_screen(stage, page):
    path = os.environ.get('KAMYII_TEST_GENERATED_VRM') or os.environ.get('KAMYII_TEST_VRM')
    if not path or not Path(path).is_file():
        pytest.skip('Set KAMYII_TEST_GENERATED_VRM or KAMYII_TEST_VRM to a local VRM')
    page.goto(stage)
    page.locator('#d3-file').set_input_files(str(Path(path).resolve()))
    page.wait_for_function('window.Kamyii3D && Kamyii3D.state.vrm', timeout=60000)
    page.get_by_role('button', name='Green screen').click()
    page.get_by_role('button', name='Record clip').click()
    page.wait_for_timeout(1500)
    with page.expect_download() as download:
        page.get_by_role('button', name='Stop recording').click()
    assert download.value.suggested_filename == 'kamyii-3d.webm'
    assert page.get_by_role('button', name='Record clip').is_visible()
    assert not page.errors
    if Path(download.value.path()).stat().st_size < 10_000:
        # Headless Chrome with SwiftShader captures no frames from a WebGL canvas on the page (a
        # detached one works); a real browser recorded ~0.3 MB/s of the stage.
        can_capture = page.evaluate('''async () => {
            const c = document.body.appendChild(document.createElement('canvas'));   // on the page, like the stage
            const gl = c.getContext('webgl', {preserveDrawingBuffer: true});
            const tick = () => { gl.clearColor(Math.random(), 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT); requestAnimationFrame(tick); }; tick();
            const r = new MediaRecorder(c.captureStream(30)), sizes = []; r.ondataavailable = e => sizes.push(e.data.size);
            r.start(200); await new Promise(res => setTimeout(res, 1000)); await new Promise(res => { r.onstop = res; r.stop(); });
            return sizes.some(n => n > 0); }''')
        if can_capture:
            raise AssertionError('the stage recorded an empty clip although this browser can capture WebGL')
        pytest.skip('this browser captures no frames from WebGL canvases; clip size not checked')


def _stick_vrm(path):
    """A stick figure with a chest, rigged like a real avatar (pink texture), written to ``path``."""
    import io
    import numpy as np
    from PIL import Image
    from src.core import rig3d
    from tests.test_avatar3d import _with_chest_front
    pos, faces, which, j, radii = _with_chest_front()
    w, _ = rig3d.skin_weights(pos, faces, j, radii)
    springs = {"bones": [], "roots": [], "colliders": [], "joints": {}}
    w = rig3d.bust_springs(pos, None, w, j, springs)
    png = io.BytesIO()
    Image.new("RGB", (8, 8), "pink").save(png, "PNG")
    rig3d.write_vrm(path, pos, rig3d.vertex_normals(pos, faces), np.zeros((len(pos), 2)), faces, w,
                    {**j, **springs.pop("joints")}, [], png.getvalue(), "image/png", path.stem,
                    Image.new("RGB", (8, 8)), springs)
    return path


def test_chest_springs_bounce_on_a_hop_and_follow_the_slider(stage, page, tmp_path):
    vrm = _stick_vrm(tmp_path / "chest.vrm")
    page.goto(stage)
    page.locator('#d3-file').set_input_files(str(vrm))
    page.wait_for_function('window.Kamyii3D && Kamyii3D.state.vrm', timeout=60000)
    assert page.locator('#d3-bounce').is_visible()
    rigged = page.evaluate('Kamyii3D.state.chestSprings')
    assert len(rigged) == 2 and rigged[0]['stiffness'] == pytest.approx(1.2)   # 50%: as rigged
    page.locator('#d3-idle').click()
    page.locator('body').press('4')  # Hop
    page.wait_for_function('Kamyii3D.state.chestSprings.some(s => s.swing > .02)', timeout=3000)
    page.locator('#d3-bounce').fill('0')
    page.wait_for_function('Kamyii3D.state.chestSprings.every(s => s.stiffness === 4 && s.drag === 1)', timeout=2000)
    page.wait_for_function('Kamyii3D.state.chestSprings.every(s => s.swing < .00001)', timeout=3000)
    page.wait_for_function('Kamyii3D.state.gesture === null', timeout=5000)
    page.get_by_role('button', name='Hop').click()
    page.wait_for_function('Kamyii3D.state.hipsLift > .005', timeout=3000)
    assert all(s['swing'] < .00001 for s in page.evaluate('Kamyii3D.state.chestSprings')), 'Zero keeps chest still during a hop'
    assert page.evaluate("JSON.parse(localStorage.getItem('kamyii.3d')).bounce") == 0
    page.locator('#d3-bounce').fill('0.5')
    page.wait_for_function('Kamyii3D.state.gesture === null', timeout=5000)
    page.get_by_role('button', name='Hop').click()
    page.wait_for_function('Kamyii3D.state.chestSprings.some(s => s.swing > .02)', timeout=4000)
    assert not page.errors


def test_animal_ears_and_tail_fit_the_avatar_and_move(stage, page, tmp_path):
    vrm = _stick_vrm(tmp_path / "kemono.vrm")
    page.goto(stage)
    page.locator('#d3-file').set_input_files(str(vrm))
    page.wait_for_function('window.Kamyii3D && Kamyii3D.state.vrm', timeout=60000)
    assert page.locator('#d3-tail').is_enabled(), 'A full-body avatar can have a tail'
    page.locator('#d3-ears').select_option('fox')
    page.locator('#d3-tail').select_option('fox')
    page.wait_for_function('Kamyii3D.state.companions.hair', timeout=5000)   # read off the rendered hair
    state = page.evaluate('Kamyii3D.state.companions')
    assert state['ear'] == 'fox' and state['tail'] == 'fox' and len(state['ears']) == 2
    left, right = state['ears']
    assert left[0] < right[0] and min(left[1], right[1]) > .35, 'One ear each side, on top of the head'
    assert state['hair'][0] > state['hair'][2], 'Match hair takes the pink texture'
    sways = []
    for _ in range(10):   # Idle sways the tail
        sways.append(page.evaluate('Kamyii3D.state.companions.tailSway'))
        page.wait_for_timeout(150)
    assert max(sways) - min(sways) > .01, sways
    page.wait_for_function('Kamyii3D.state.companions.earAngles.every(a => a > .1)', timeout=3000)   # leaning out
    page.locator('#d3-ears').select_option('off')
    page.wait_for_function('Kamyii3D.state.companions.ears.length === 0', timeout=3000)
    saved = page.evaluate("JSON.parse(localStorage.getItem('kamyii.3d')).companions")
    assert list(saved.values())[0]['ear'] == 'off' and list(saved.values())[0]['tail'] == 'fox'
    page.reload()   # remembered per character
    page.locator('#d3-file').set_input_files(str(vrm))
    page.wait_for_function('window.Kamyii3D && Kamyii3D.state.vrm', timeout=60000)
    page.wait_for_function("Kamyii3D.state.companions.tail === 'fox'", timeout=5000)
    assert page.locator('#d3-tail').input_value() == 'fox'
    assert not page.errors
