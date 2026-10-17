"""Preview format, cache, route boundaries and framing, all without a GPU."""
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from psd_tools import PSDImage
from src.core.live_preview import ensure_preview_psd, preview_layers


def write_layer(folder,name,color,box,size=(100,120)):
    folder.mkdir(exist_ok=True,parents=True)
    image=Image.new('RGBA',size)
    image.paste(color,box)
    image.save(folder/name)


def test_preview_psd_merges_pairs_and_preserves_canvas_and_order(tmp_path):
    assets=tmp_path/'assets'
    write_layer(assets,'01_back hair.png',(20,30,40,255),(10,10,90,100))
    write_layer(assets,'02_face.png',(220,180,160,255),(25,25,75,75))
    write_layer(assets,'03_eyewhite-l.png',(255,255,255,255),(30,35,40,45))
    write_layer(assets,'04_eyewhite-r.png',(250,250,250,255),(60,35,70,45))
    write_layer(assets,'05_mouth.png',(100,20,20,255),(45,60,55,65))
    write_layer(assets,'06_front hair.png',(10,20,30,255),(15,10,85,30))
    write_layer(assets,'character.png',(0,0,0,255),(0,0,100,120))
    source=(assets/'03_eyewhite-l.png').read_bytes()
    out=ensure_preview_psd(tmp_path)
    psd=PSDImage.open(out)
    assert psd.size==(100,120), 'Do not crop away transparent canvas margins'
    assert [l.name for l in psd]==['back hair','face','eyewhite','mouth_open','front hair']
    eyes=next(l for l in psd if l.name=='eyewhite')
    assert eyes.offset==(30,35)
    rgba=eyes.topil().convert('RGBA')
    assert rgba.getpixel((5,5))==(255,255,255,255)
    assert rgba.getpixel((35,5))==(250,250,250,255)
    assert rgba.getpixel((20,5))[3]==0
    assert (assets/'03_eyewhite-l.png').read_bytes()==source
    # Flattening the PSD must preserve the exact source painter's order.
    expected=Image.new('RGBA',psd.size)
    for _,img in preview_layers(assets):expected.alpha_composite(img)
    actual = np.asarray(psd.composite().convert("RGBA"))
    expected = np.asarray(expected)
    assert np.array_equal(actual[..., 3], expected[..., 3])
    visible = expected[..., 3] > 0
    assert np.array_equal(actual[visible], expected[visible])


def test_preview_cache_reuses_then_invalidates_on_new_layer(tmp_path):
    write_layer(tmp_path/'assets','01_face.png',(200,150,100,255),(5,5,70,80))
    first=ensure_preview_psd(tmp_path);stamp=first.stat().st_mtime_ns
    assert ensure_preview_psd(tmp_path)==first
    assert first.stat().st_mtime_ns==stamp
    write_layer(tmp_path/'assets','02_mouth.png',(100,20,20,255),(40,60,60,65))
    assert ensure_preview_psd(tmp_path)!=first


def test_demo_names_and_empty_or_misaligned_layers(tmp_path):
    assets=tmp_path/'assets'
    write_layer(assets,'01_head.png',(255,200,150,255),(5,5,80,80))
    write_layer(assets,'02_eye_L.png',(255,255,255,255),(20,30,30,40))
    write_layer(assets,'03_eye_R.png',(255,255,255,255),(60,30,70,40))
    assert [n for n,_ in preview_layers(assets)]==['face','eyewhite']
    write_layer(assets,'04_bad.png',(0,0,0,255),(0,0,10,10),size=(20,20))
    with pytest.raises(ValueError,match='aligned'):ensure_preview_psd(tmp_path)
    with pytest.raises(FileNotFoundError):ensure_preview_psd(tmp_path/'absent')


def test_preview_routes_reject_paths_and_handle_missing_layers(tmp_path,monkeypatch):
    import app
    import config
    monkeypatch.setattr(config,'OUTPUT_DIR',tmp_path)
    client=app.create_app(use_dummy_generators=True).test_client()
    model=tmp_path/'unit-model';model.mkdir()
    assert client.get('/preview/absent').status_code==404
    assert client.get('/api/generation/artifact/unit-model/preview_psd').status_code==422
    assert client.get('/api/generation/artifact/..%5Coutside/preview_psd').status_code==404
    write_layer(model/'assets','01_face.png',(255,200,150,255),(10,10,80,90))
    page=client.get('/preview/unit-model')
    assert page.status_code==200 and b'/api/generation/artifact/unit-model/preview_psd' in page.data
    response=client.get('/api/generation/artifact/unit-model/preview_psd')
    assert response.status_code==200 and response.data[:4]==b'8BPS'
    assert client.post('/api/generation/generate',json={'prompt':'test','framing':'invalid'}).status_code==400


def test_upper_body_prompt_and_request_roundtrip():
    from src.core.image_generator import ImageGenerator
    from src.models.vtuber_model import GenerationRequest,VTuberModel
    request=GenerationRequest(prompt='mint hair',style='vtuber',framing='upper_body',seed=42)
    gen=ImageGenerator.__new__(ImageGenerator)  # prompt builder needs no torch
    prompt,_=gen._build_prompts(request)
    assert 'upper body' in prompt and 'full body' not in prompt and 'reference sheet' not in prompt
    model=VTuberModel(request=request)
    loaded=VTuberModel.from_dict(model.to_dict())
    assert loaded.request.seed==42 and loaded.request.framing=='upper_body'
    prompt,_=gen._build_prompts(GenerationRequest(prompt='mint hair'))
    assert 'full body' in prompt


def test_cubism_package_contains_project_not_source_workfiles(tmp_path, monkeypatch):
    import app
    import config
    import io
    import zipfile
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    client = app.create_app(use_dummy_generators=True).test_client()
    model = tmp_path / "package-model"
    project = model / "model" / "study"
    project.mkdir(parents=True)
    (project / "kamyii_manifest.json").write_text("{}")
    (project / "RIGGING_GUIDE.md").write_text("Artist guide")
    (model / "base_image.png").write_bytes(b"source image")
    response = client.get("/api/generation/artifact/package-model/package")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.data)) as package:
        assert "RIGGING_GUIDE.md" in package.namelist()
        assert "base_image.png" not in package.namelist()
    response.close()


def test_artist_psd_keeps_canvas_margins_and_source_offsets(tmp_path):
    from src.core.psd_exporter import export_psd, PYTOSHOP_AVAILABLE
    from src.models.vtuber_model import Asset
    if not PYTOSHOP_AVAILABLE:
        pytest.skip("pytoshop unavailable")
    write_layer(tmp_path, "part.png", (100, 150, 200, 255), (20, 30, 60, 80))
    output = export_psd([Asset("face", tmp_path / "part.png")], tmp_path / "artist.psd")
    assert output is not None
    psd = PSDImage.open(output)
    assert psd.size == (100, 120)
    assert psd[0].offset == (20, 30)


def test_silhouette_hole_cannot_reject_a_real_mouth(tmp_path, monkeypatch):
    from src.core import seethrough
    from src.core.asset_separator import AssetSeparator
    source = Image.new("RGBA", (100, 120), "white")
    source.paste((180, 130, 100, 255), (15, 15, 85, 110))
    source.paste((80, 20, 10, 255), (40, 65, 60, 75))
    source_path = tmp_path / "source.png"
    source.save(source_path)
    cutout = source.copy()
    alpha = Image.new("L", source.size)
    alpha.paste(255, (15, 15, 85, 110))
    alpha.paste(0, (33, 58, 67, 82))  # erroneous internal cutout surrounds the mouth
    cutout.putalpha(alpha)
    monkeypatch.setattr(AssetSeparator, "_foreground_cutout", lambda *args: cutout.copy())
    psd = PSDImage.new("RGBA", source.size)
    psd.create_pixel_layer(Image.new("RGBA", (20, 10), (80, 20, 10, 255)), name="mouth", left=40, top=65)
    psd.create_pixel_layer(Image.new("RGBA", (5, 5), (0, 0, 255, 255)), name="headwear", left=0, top=0)
    psd_path = tmp_path / "source.psd"
    psd.save(psd_path)
    monkeypatch.setattr(seethrough, "available", lambda: True)
    monkeypatch.setattr(seethrough, "decompose", lambda *args, **kwargs: psd_path)
    assets = AssetSeparator().separate(source_path, tmp_path / "assets")
    assert "mouth" in {a.layer_type for a in assets}
    assert "headwear" not in {a.layer_type for a in assets}


def test_visible_source_colours_are_retained_but_hidden_inpainting_is_preserved(tmp_path, monkeypatch):
    from src.core import seethrough
    from src.core.asset_separator import AssetSeparator
    source = Image.new("RGBA", (100, 120), "white")
    source.paste((200, 160, 120, 255), (10, 10, 90, 110))
    source.paste((220, 170, 130, 255), (30, 30, 70, 70))
    source_path = tmp_path / "source.png"
    source.save(source_path)
    cutout = source.copy()
    alpha = Image.new("L", source.size)
    alpha.paste(255, (10, 10, 90, 110))
    cutout.putalpha(alpha)
    monkeypatch.setattr(AssetSeparator, "_foreground_cutout", lambda *args: cutout.copy())
    psd = PSDImage.new("RGBA", source.size)
    psd.create_pixel_layer(Image.new("RGBA", (80, 100), (180, 140, 100, 255)), name="neck", left=10, top=10)
    psd.create_pixel_layer(Image.new("RGBA", (40, 40), (200, 150, 110, 255)), name="face", left=30, top=30)
    psd_path = tmp_path / "parts.psd"
    psd.save(psd_path)
    monkeypatch.setattr(seethrough, "available", lambda: True)
    monkeypatch.setattr(seethrough, "decompose", lambda *args, **kwargs: psd_path)
    assets = AssetSeparator().separate(source_path, tmp_path / "assets")
    neck = Image.open(next(a.file_path for a in assets if a.layer_type == "neck"))
    face = Image.open(next(a.file_path for a in assets if a.layer_type == "face"))
    assert neck.getpixel((20, 20)) == (200, 160, 120, 255)
    assert neck.getpixel((40, 40)) == (180, 140, 100, 255)
    assert face.getpixel((40, 40)) == (220, 170, 130, 255)
