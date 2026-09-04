"""Optional neural image-to-3D reconstruction integration.

The core AeroVoxel install does not import the heavy reconstruction stack.  The
module exposes capability detection and a Stable Fast 3D runner when the
optional dependencies are installed.
"""

from app.recon.capability import get_recon_capability
from app.recon.export_to_aerovoxel import mesh_to_center_slice_mask
from app.recon.integration import register_mask_for_simulation

__all__ = ["get_recon_capability", "mesh_to_center_slice_mask", "register_mask_for_simulation"]
