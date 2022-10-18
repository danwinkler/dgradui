import pathlib
import shutil
import subprocess
import sys
import tempfile

import trimesh
from solid import *
from solid.utils import *


def extrude_paths_into_base(paths, border=10, height=10, path_depth=5):
    minx, miny, maxx, maxy = None, None, None, None

    for path in paths:
        for p in path:
            if minx == None or p[0] < minx:
                minx = p[0]
            if miny == None or p[1] < miny:
                miny = p[1]
            if maxx == None or p[0] > maxx:
                maxx = p[0]
            if maxy == None or p[1] > maxy:
                maxy = p[1]

    base = translate([minx - border, miny - border, 0])(
        cube([maxx - minx + border * 2, maxy - miny + border * 2, height])
    )

    for path in paths:
        extruded = linear_extrude(height=path_depth + 1)(polygon(path, convexity=10))
        base = base - up(height - path_depth)(extruded)

    return base


def write_solid_to_file(obj, path):
    scad_render_to_file(obj, path)


def solid_to_glb(obj, openscad_path=None):
    if openscad_path is None:
        if sys.platform == "darwin":
            openscad_path = "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD"

    # get tempfile
    scad_file = tempfile.NamedTemporaryFile(suffix=".scad", delete=False)
    stl_file = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    glb_file = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)

    # Write obj to tempfile
    scad_render_to_file(obj, scad_file.name)

    # Run openscad
    subprocess.run([openscad_path, "-o", stl_file.name, scad_file.name])

    assert pathlib.Path(stl_file.name).exists()

    # Copy to test.stl
    shutil.copy(stl_file.name, "test.stl")

    # Convert stl to obj
    mesh = trimesh.load(stl_file.name)
    scene = trimesh.Scene(mesh)
    scene.export(glb_file.name)

    return glb_file.name
