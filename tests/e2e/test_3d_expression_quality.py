"""Interactions on a real generated avatar; the fixture is optional and remains private."""
import os
from pathlib import Path
import pytest
expect = pytest.importorskip("playwright.sync_api").expect
from test_3d_stage import stage, page


def generated_avatar():
    path = os.environ.get("KAMYII_TEST_GENERATED_VRM")
    if not path or not Path(path).is_file():
        pytest.skip("Set KAMYII_TEST_GENERATED_VRM to a generated avatar with painted expressions")
    return Path(path)


def test_painted_expressions_and_vowels_do_not_overlap(stage, page):
    avatar = generated_avatar()
    page.goto(stage)
    page.wait_for_function("window.Kamyii3D")
    page.locator("#d3-file").set_input_files(str(avatar))
    page.wait_for_function("Kamyii3D.state.vrm", timeout=60000)
    choices=page.locator("#d3-mouth-shapes button")
    assert choices.all_text_contents()==["A", "I", "U", "E", "O"]
    page.locator("#d3-mouth").fill("0.8")
    page.get_by_role("button",name="I",exact=True).click()
    page.wait_for_function("Kamyii3D.state.expressions.ih > .75 && Kamyii3D.state.expressions.aa === 0")
    tongue=page.get_by_role("button",name="Tongue out",exact=True)
    cat=page.get_by_role("button",name="Cat mouth",exact=True)
    tongue.click()
    page.wait_for_function("Kamyii3D.state.expressions['Tongue out'] === 1 && Kamyii3D.state.expressions.ih === 0")
    cat.click()
    expect(tongue).to_have_attribute("aria-pressed","false")
    page.wait_for_function("Kamyii3D.state.expressions['Tongue out'] === 0 && Kamyii3D.state.expressions['Cat mouth'] === 1")
    cat.click()
    page.wait_for_function("Kamyii3D.state.expressions.ih > .75 && Kamyii3D.state.expressions['Cat mouth'] === 0")
    heart=page.get_by_role("button",name="Heart eyes",exact=True)
    star=page.get_by_role("button",name="Star eyes",exact=True)
    blush=page.get_by_role("button",name="Blush",exact=True)
    heart.click(); blush.click(); star.click()
    expect(heart).to_have_attribute("aria-pressed","false")
    expect(star).to_have_attribute("aria-pressed","true")
    expect(blush).to_have_attribute("aria-pressed","true")
    page.wait_for_function("Kamyii3D.state.expressions['Heart eyes'] === 0 && Kamyii3D.state.expressions.Blush === 1")
    # The painted eye decal fades when a real auto-blink closes the underlying eyes.
    page.wait_for_function("Kamyii3D.state.expressions['Star eyes'] < .7",timeout=12000)
    page.wait_for_function("Kamyii3D.state.expressions['Star eyes'] > .9",timeout=3000)
    page.screenshot(path="outputs/3d-expressions-review/witch-expressions.png",full_page=True)
    page.set_viewport_size({"width":390,"height":844})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth+1")
    assert not page.errors


def test_failed_character_and_file_replacements_preserve_download(stage,page):
    avatar=generated_avatar().read_bytes()
    page.route("**/api/generation/list",lambda route:route.fulfill(json={"models":[
        {"id":"good","name":"Good","metadata":{"avatar3d":True}},
        {"id":"broken","name":"Broken","metadata":{"avatar3d":True}}]}))
    page.route("**/api/generation/artifact/good/vrm",lambda route:route.fulfill(status=200,body=avatar,content_type="model/gltf-binary"))
    page.route("**/api/generation/artifact/broken/vrm",lambda route:route.fulfill(status=404,body="missing"))
    page.goto(stage);page.wait_for_function("window.Kamyii3D && document.querySelector('#d3-characters').options.length === 3")
    page.select_option("#d3-characters","good")
    page.wait_for_function("Kamyii3D.state.vrm",timeout=60000)
    expect(page.locator("#d3-download")).to_have_attribute("href","/api/generation/artifact/good/vrm")
    page.select_option("#d3-characters","broken")
    expect(page.locator("#d3-status")).to_contain_text("no 3D avatar yet")
    expect(page.locator("#d3-download")).to_have_attribute("href","/api/generation/artifact/good/vrm")
    assert page.locator("#d3-characters").input_value()=="good" and page.evaluate("Kamyii3D.state.vrm")
    page.locator("#d3-file").set_input_files({"name":"invalid.vrm","mimeType":"model/gltf-binary","buffer":b"bad"})
    expect(page.locator("#d3-status")).to_contain_text("valid VRM")
    expect(page.locator("#d3-download")).to_have_attribute("href","/api/generation/artifact/good/vrm")
    assert page.locator("#d3-characters").input_value()=="good" and page.evaluate("Kamyii3D.state.vrm")
    assert not page.errors
