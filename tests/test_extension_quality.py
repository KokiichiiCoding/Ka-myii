"""Extension layout and diffusion-mask contracts, without torch or model downloads."""
import sys
from types import SimpleNamespace
import numpy as np
from PIL import Image, ImageDraw


def test_pose_guide_tracks_retained_head_and_fixed_standing_axis():
    from src.core.outpaint import pose_points, pose_image
    info={"canvas":[832,1216],"target_heads":6.75,"foot_y":1143,
          "result_geometry":{"face":[360,80,468,210],"head_height":162,"axis_x":416,"crown_y":48}}
    p=pose_points(info)
    assert len(p)==18 and all(0<x<832 and 0<y<1216 for x,y in p)
    assert p[0][1]<p[1][1]<p[2][1]<p[8][1]<p[9][1]<p[10][1]
    assert abs(p[2][0]+p[5][0]-832)<.01 and p[10][1]==p[13][1]
    assert 150<p[0][1]<190 and p[1][1]<250, "The neck begins under the retained face"
    img=np.asarray(pose_image(info))
    assert img.shape==(1216,832,3) and img.max()==255 and (img.any(-1)).sum()>10000


def test_head_preservation_is_solid_for_see_through_soft_alpha(tmp_path):
    from src.core.outpaint import identity_mask
    assets=tmp_path/"assets";assets.mkdir()
    face=Image.new("RGBA",(200,300));ImageDraw.Draw(face).ellipse((60,30,140,140),fill=(240,210,180,250))
    face.save(assets/"01_face.png")
    hair=Image.new("RGBA",face.size);ImageDraw.Draw(hair).rectangle((55,15,145,60),fill=(150,125,160,250))
    hair.save(assets/"02_front hair.png")
    before=(assets/"01_face.png").read_bytes()
    keep=identity_mask(tmp_path,{"face":[60,30,141,141],"head_height":126})
    assert keep[80,100]==255 and keep[35,100]==255, "Soft source alpha must not make the whole head repaintable"
    assert keep[190,100]==0 and keep[0,100]==0, "Outfit and cropped border remain free"
    assert (assets/"01_face.png").read_bytes()==before


def test_differential_inpainting_protects_layout_on_the_first_step(monkeypatch):
    from src.core.image_generator import ImageGenerator
    class Tensor:
        def __init__(self,a):self.a=np.array(a)
        def __getitem__(self,key):return Tensor(self.a[key])
        def to(self,other):return self
        def expand_as(self,other):return Tensor(np.broadcast_to(self.a,other.a.shape))
        def copy_(self,other):self.a[...] = other.a;return self
        def __gt__(self,n):return Tensor(self.a>n)
    class Pipe:
        tokenizer_2=None
        num_timesteps=10
        def __call__(self,**call):
            initial=np.asarray(call["mask_image"])
            assert initial[initial.shape[0]//3,initial.shape[1]//2]==0
            assert initial[-5,initial.shape[1]//2]==255
            mask=Tensor(np.ones((1,1,call["height"]//8,call["width"]//8)))
            call["callback_on_step_end"](self,0,None,{"mask":mask})
            assert mask.a[0,0,mask.a.shape[2]//3,mask.a.shape[3]//2]==0
            call["callback_on_step_end"](self,8,None,{"mask":mask})
            assert mask.a[0,0,mask.a.shape[2]//3,mask.a.shape[3]//2]==1
            return SimpleNamespace(images=[call["image"]])
    pipe=Pipe()
    monkeypatch.setitem(sys.modules,"diffusers",SimpleNamespace(AutoPipelineForInpainting=SimpleNamespace(from_pipe=lambda _:pipe)))
    gen=object.__new__(ImageGenerator);gen.pipeline=pipe;gen.device="cpu"
    gen._torch=SimpleNamespace(from_numpy=Tensor)
    image=Image.new("RGB",(100,120),"white");mask=np.full((120,100),128,np.uint8);mask[80:]=255
    result=gen.inpaint(image,Image.fromarray(mask),"standing",steps=10,graded=True,detail=128)
    assert result.size==image.size


def _figure(tilt=0, hair_over_left=False, size=(400, 600), feet_cut=False):
    """A flat test figure: brown hair, skin face, blue torso (shoulders tilted by ``tilt`` px)."""
    img = Image.new("RGB", size, "white"); d = ImageDraw.Draw(img)
    d.ellipse((150, 40, 250, 150), fill=(90, 60, 40))            # hair
    d.ellipse((165, 70, 235, 150), fill=(240, 210, 190))         # face
    d.rectangle((190, 150, 210, 175), fill=(240, 210, 190))      # neck
    d.polygon([(200 - 70, 175 + tilt), (200 + 70, 175 - tilt), (200 + 60, 380), (200 - 60, 380)], fill=(60, 80, 160))
    d.rectangle((170, 380, 230, size[1] - 1 if feet_cut else 560), fill=(60, 60, 60))
    if hair_over_left:
        d.rectangle((120, 120, 160, 330), fill=(90, 60, 40))
    fg = np.asarray(img).sum(-1) < 750
    return img, fg


def test_full_body_review_finds_cropped_feet_tilted_shoulders_and_duplicates():
    from src.core import checks
    g = {"target_heads": 5.5, "result_geometry": {"face": [165, 70, 235, 150], "head_height": 110, "axis_x": 200, "crown_y": 40}}
    img, fg = _figure()
    assert checks.review(img, fg, g, {"scores": {"solo": .9}}) == []
    assert abs(checks.heads_tall(fg, g["result_geometry"]) - (560 - 40) / (1.33 * 70)) < .05
    assert "too big" in checks.review(img, fg, {**g, "target_heads": 6.75})[0]["message"], "5.6 heads aiming for 6.75"
    assert "too small" in checks.review(img, fg, {**g, "target_heads": 4.5})[0]["message"]
    img, fg = _figure(hair_over_left=True)
    assert checks.review(img, fg, g) == [], "A long lock of hair over one shoulder is not a tilt"
    img, fg = _figure(tilt=25)
    found = {f["id"]: f for f in checks.review(img, fg, g)}
    assert set(found) == {"shoulders"} and found["shoulders"]["tilt"] > 0 and found["shoulders"]["fix"] == "repaint"
    x0, y0, x1, y1 = found["shoulders"]["box"]
    assert x0 < 130 and x1 > 270 and y0 == 150, "The repaint box covers both shoulders"
    img, fg = _figure(feet_cut=True)
    found = checks.review(img, fg, g, {"scores": {"2girls": .6, "bad hands": .5}})
    assert [f["id"] for f in found] == ["duplicate", "feet", "hands"]
    assert [f["fix"] for f in found] == ["reseed", "uncrop", "refine_hands"]
    assert checks.review(img, fg, None, None)[0]["id"] == "feet", "Pixel checks work without the tagger or geometry"
    # A second head on the chest: the whole picture still reads "solo", one body tile has eyes.
    img, fg = _figure()
    tiles = checks.face_tiles(img, g)
    assert len(tiles) >= 2 and tiles[0].size == (198, 198)
    body, face = {"scores": {"skirt": .9, "head out of frame": .8}}, {"scores": {"blue eyes": .8, "looking at viewer": .7}}
    assert checks.review(img, fg, g, {"scores": {"solo": .95}}, [body, body]) == []
    assert [f["id"] for f in checks.review(img, fg, g, {"scores": {"solo": .95}}, [body, face])] == ["duplicate"]
    assert checks.face_tiles(img, None) == []


def test_extended_body_keeps_matching_source_head_layers(tmp_path):
    import json
    from src.core import heads
    from src.models.vtuber_model import GenerationRequest
    from src.pipeline.assembly_line import AssemblyLine

    src, new = tmp_path / "src", tmp_path / "new"
    (src / "assets").mkdir(parents=True); (new / "assets").mkdir(parents=True); (src / "expressions").mkdir()
    base = Image.new("RGB", (100, 140), "white"); d = ImageDraw.Draw(base)
    d.ellipse((30, 20, 70, 70), fill=(240, 210, 190)); d.rectangle((25, 10, 75, 139), outline=(90, 60, 40), width=6)
    base.save(src / "base_image.png")
    def layer(name, box, colour, folder="assets"):
        im = Image.new("RGBA", (100, 140)); ImageDraw.Draw(im).ellipse(box, fill=colour); im.save(src / folder / name)
    layer("01_face.png", (30, 20, 70, 70), (240, 210, 190, 255))
    for i, part in enumerate(("eyewhite-l", "irides-l", "eyelash-l", "mouth")):
        layer(f"0{i + 2}_{part}.png", (35, 28 + 6 * i, 65, 42 + 6 * i), (240, 210, 190, 255))
    hair = Image.new("RGBA", (100, 140)); ImageDraw.Draw(hair).rectangle((25, 10, 75, 139), outline=(90, 60, 40, 255), width=6)
    hair.save(src / "assets" / "06_front hair.png")
    layer("eye_close.png", (40, 35, 50, 40), (10, 10, 10, 255), "expressions")
    (src / "expressions" / "expressions.json").write_text(json.dumps({"eye_close": "expressions/eye_close.png", "mouth_drawn": "closed"}))
    info = {"placed": [30, 5, 40, 56], "source_model": "src"}
    canvas = Image.new("RGB", (100, 140), "white"); canvas.paste(base.resize((40, 56), Image.Resampling.LANCZOS), (30, 5))
    ImageDraw.Draw(canvas).rectangle((30, 40, 70, 61), fill=(50, 120, 200))   # the long locks were redrawn
    canvas.save(new / "base_image.png")
    for name in ("01_face", "02_eyewhite-l", "03_irides-l", "04_eyelash-l", "05_mouth", "06_front hair", "07_topwear"):
        Image.new("RGBA", (100, 140), (1, 2, 3, 255)).save(new / "assets" / f"{name}.png")
    before = {p.name: p.read_bytes() for p in (src / "assets").iterdir()}

    kept = heads.transfer(src, new, info)
    assert set(kept) == {"eyewhite-l", "irides-l", "eyelash-l", "mouth"}, "Features only; redrawn hair keeps the new separation"
    iris = np.asarray(Image.open(new / "assets" / "03_irides-l.png"))
    assert iris[0, 0, 3] == 0 and iris[19, 50, 3] == 255, "The source iris, scaled into the placement"
    assert (np.asarray(Image.open(new / "assets" / "01_face.png"))[0, 0] == (1, 2, 3, 255)).all(), "The new face stays"
    assert (np.asarray(Image.open(new / "assets" / "06_front hair.png"))[0, 0] == (1, 2, 3, 255)).all()
    assert (np.asarray(Image.open(new / "assets" / "07_topwear.png"))[0, 0] == (1, 2, 3, 255)).all()
    assert json.loads((new / "expressions" / "expressions.json").read_text())["eye_close"] == "expressions/eye_close.png"
    assert np.asarray(Image.open(new / "expressions" / "eye_close.png")).shape == (140, 100, 4)
    assert before == {p.name: p.read_bytes() for p in (src / "assets").iterdir()}, "The source character is untouched"

    line = AssemblyLine(output_base_dir=tmp_path, use_dummy_generators=True)
    for crafted in ("../src", "C:/x", ""):
        assert line._keep_head(GenerationRequest(prompt="x", extension={**info, "source_model": crafted}), new) == []
