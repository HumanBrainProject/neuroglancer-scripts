#!/usr/bin/env python
import json

import nibabel
import numpy as np
import pyvtk

input_filename="colin.surf.gii"

mesh = nibabel.load(input_filename)
mesh.print_summary()
points_list = mesh.get_arrays_from_intent("NIFTI_INTENT_POINTSET")
assert len(points_list) == 1
points = points_list[0].data
triangles_list = mesh.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")
assert len(triangles_list) == 1
triangles = triangles_list[0].data
# Gifti uses millimetres, Neuroglancer expects nanometres
points *= 1e6
# Workaround: dtype must be np.int_ (pyvtk does not recognize int32 as
# integers)
triangles = triangles.astype(np.int_)
vtk_mesh = pyvtk.PolyData(points, polygons=triangles)

rgba_labels = mesh.get_arrays_from_intent("NIFTI_INTENT_RGBA_VECTOR")[0].data
assert np.all(rgba_labels[:, 3] == 1)

rgb_dtype = [("r", "u1"), ("g", "u1"), ("b", "u1")]
rgb_array = np.empty((rgba_labels.shape[0],), dtype=rgb_dtype)
rgb_array[:]["r"] = rgba_labels[:, 0] * 255
rgb_array[:]["g"] = rgba_labels[:, 1] * 255
rgb_array[:]["b"] = rgba_labels[:, 2] * 255

unique_values, labels  = np.unique(rgb_array, return_inverse=True)

with open("labels.json") as f:
    labels_metadata = json.load(f)

def generate_rgb_to_index_dict():
    for m in labels_metadata["atlas"]["data"]["label"]:
        s = m.split(":")
        if not s[0].strip():
            continue
        index = int(s[1])
        r, g, b = int(s[8]), int(s[9]), int(s[10])
        #rgb = np.array((r, g, b), dtype=rgb_dtype)  # scalar array
        yield (r, g, b), index

rgb_to_index = {rgb: index for rgb, index in generate_rgb_to_index_dict()}
rgb_to_index = {rgb: index for rgb, index in generate_rgb_to_index_dict()}

# Some RGB values of the texture do not appear in the JSON label map…
label_map = np.empty_like(unique_values, dtype=np.uint8)
for unique_label, rgb in enumerate(unique_values):
    label_map[unique_label] = rgb_to_index.get(tuple(rgb), 0)

final_vertex_labels = label_map[labels]

sc = pyvtk.Scalars(final_vertex_labels, name="label", lookup_table="none")
pd = pyvtk.PointData(sc)
vtk_data = pyvtk.VtkData(
    vtk_mesh,
    "Converted using "
    "https://github.com/HumanBrainProject/neuroglancer-scripts",
    pd)

vtk_data.tofile("jubrain-mpm-surf.vtk")

# PROBLEM: for the Scalars pyvtk uses space as an ASCII separator, while
# Neuroglancer expects newline. Therefore the VTK file must be post-processed
# to correct this.

print("GLSL colormap snippet")
print("=====================")
for rgb, index in generate_rgb_to_index_dict():
    print("  if(label == {:#.0f})\n    return vec3({:#.0f}, {:#.0f}, {:#.0f});"
          .format(index, *rgb))
