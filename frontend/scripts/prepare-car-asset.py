"""Prepare the licensed Khronos CarConcept GLB for AeroVoxel.

Run with --source pointing to the original official GLB. Geometry and original
copyright metadata are preserved. Branding/color/emissive textures are removed;
normal, occlusion, and roughness maps retain their original UV transforms.
See public/models/ATTRIBUTION.md for author, source, license and modifications.
"""

import argparse
import json
from pathlib import Path
import struct


def prepare(source: Path, target: Path) -> None:
    data = source.read_bytes()
    magic, version, length = struct.unpack_from("<III", data)
    if (magic, version, length) != (0x46546C67, 2, len(data)):
        raise ValueError("Expected a complete glTF 2.0 binary asset")
    json_length, json_type = struct.unpack_from("<II", data, 12)
    if json_type != 0x4E4F534A:
        raise ValueError("Missing GLB JSON chunk")
    document = json.loads(data[20 : 20 + json_length])
    binary_offset = 20 + json_length
    binary_length, binary_type = struct.unpack_from("<II", data, binary_offset)
    if binary_type != 0x004E4942:
        raise ValueError("Missing embedded binary chunk")
    binary = data[binary_offset + 8 : binary_offset + 8 + binary_length]
    retained_textures = set()
    for material in document["materials"]:
        pbr = material.get("pbrMetallicRoughness", {})
        pbr.pop("baseColorTexture", None)
        material.pop("emissiveTexture", None)
        if material.get("name") not in {"Headlight", "Brakelight", "Signallight"}:
            material.pop("emissiveFactor", None)
        extensions = material.get("extensions", {})
        extensions.pop("KHR_materials_transmission", None)
        extensions.pop("KHR_materials_iridescence", None)
        if "emissiveFactor" not in material:
            extensions.pop("KHR_materials_emissive_strength", None)
        for info in [material.get("normalTexture"), material.get("occlusionTexture"), pbr.get("metallicRoughnessTexture")]:
            if info:
                retained_textures.add(info["index"])

    # Keep accessor buffer views verbatim, repack only the retained image bytes.
    image_views = {image["bufferView"] for image in document["images"]}
    retained_images = sorted({document["textures"][i]["source"] for i in retained_textures})
    required_image_views = {document["images"][i]["bufferView"] for i in retained_images}
    new_binary = bytearray()
    new_views = []
    view_map = {}
    for i, view in enumerate(document["bufferViews"]):
        if i in image_views and i not in required_image_views:
            continue
        while len(new_binary) % 4:
            new_binary.append(0)
        view_map[i] = len(new_views)
        updated = dict(view, byteOffset=len(new_binary))
        new_views.append(updated)
        start = view.get("byteOffset", 0)
        new_binary.extend(binary[start : start + view["byteLength"]])
    for accessor in document["accessors"]:
        if "bufferView" in accessor:
            accessor["bufferView"] = view_map[accessor["bufferView"]]
        if "sparse" in accessor:
            for key in ("indices", "values"):
                entry = accessor["sparse"][key]
                entry["bufferView"] = view_map[entry["bufferView"]]
    images = []
    image_map = {}
    for i in retained_images:
        image_map[i] = len(images)
        image = dict(document["images"][i])
        image["bufferView"] = view_map[image["bufferView"]]
        images.append(image)
    textures = []
    texture_map = {}
    for i in sorted(retained_textures):
        texture_map[i] = len(textures)
        texture = dict(document["textures"][i])
        texture["source"] = image_map[texture["source"]]
        textures.append(texture)
    for material in document["materials"]:
        for info in [material.get("normalTexture"), material.get("occlusionTexture"), material.get("pbrMetallicRoughness", {}).get("metallicRoughnessTexture")]:
            if info:
                info["index"] = texture_map[info["index"]]
    document["bufferViews"] = new_views
    document["images"] = images
    document["textures"] = textures
    document["buffers"][0]["byteLength"] = len(new_binary)
    removed = {"KHR_materials_transmission", "KHR_materials_iridescence"}
    for key in ("extensionsUsed", "extensionsRequired"):
        if key in document:
            document[key] = [name for name in document[key] if name not in removed]
    encoded = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded += b" " * (-len(encoded) % 4)
    new_binary.extend(b"\x00" * (-len(new_binary) % 4))
    total = 12 + 8 + len(encoded) + 8 + len(new_binary)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(struct.pack("<III", magic, version, total) + struct.pack("<II", len(encoded), json_type) + encoded + struct.pack("<II", len(new_binary), binary_type) + new_binary)
    print(f"Prepared {target}: {total:,} bytes, {len(images)} detail maps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "public/models/concept-car.glb")
    args = parser.parse_args()
    prepare(args.source, args.output)
