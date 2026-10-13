from __future__ import annotations
import io
import json
from contextlib import nullcontext
from types import SimpleNamespace, ModuleType
import sys
import numpy as np
import pytest
import trimesh
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app
from app.recon.export_to_aerovoxel import fit_mask_to_domain, mesh_to_center_slice_mask
from app.recon.integration import register_mask_array
from app.recon.engine import select_reconstruction_engine, AutoReconstructionEngine, DepthAnythingEngine, StableFast3DEngine
import app.recon.engine as engines
import app.routes.simulate as simulation


def capability(**overrides):
    return {"ready_for_inference": True, "depth_anything_available": True,
            "cuda_available": True, "cuda_device_count": 1, "cuda_memory_mb": [4096],
            "recommended_device": "cuda", **overrides}


def test_mesh_xy_orientation_aspect_and_airflow_margins(tmp_path):
    source = tmp_path / 'wide.glb'
    trimesh.creation.box(extents=(3, 1, 0.5)).export(source)
    destination = tmp_path / 'mask.npy'
    mesh_to_center_slice_mask(str(source), str(destination))
    mask = np.load(destination)
    ys, xs = np.where(mask)
    assert (xs.max()-xs.min()+1)/(ys.max()-ys.min()+1) == pytest.approx(3, abs=.18)
    assert xs.min() >= 34 and xs.max() <= 93
    assert ys.min() >= 16 and ys.max() <= 47
    assert not mask[0].any() and not mask[-1].any() and not mask[:, 0].any() and not mask[:, -1].any()


def test_mask_normalization_keeps_asymmetric_shape_orientation():
    shape = np.zeros((30, 90), dtype=bool)
    shape[:10, :] = True
    shape[:, :10] = True
    fitted = fit_mask_to_domain(shape)
    ys, xs = np.where(fitted)
    assert (xs.max()-xs.min()+1)/(ys.max()-ys.min()+1) == pytest.approx(3, abs=.1)
    assert fitted[ys.min(), xs.max()] and fitted[ys.max(), xs.min()]
    assert not fitted[ys.max(), xs.max()]


@pytest.mark.parametrize('memory,device,expected', [(4096,'cuda',DepthAnythingEngine),(8192,'cpu',DepthAnythingEngine),(8192,'cuda',StableFast3DEngine)])
def test_auto_respects_device_and_vram(monkeypatch,memory,device,expected):
    monkeypatch.setattr(engines,'get_recon_capability',lambda:capability(cuda_memory_mb=[memory]))
    selected=select_reconstruction_engine('auto',device=device)
    first=selected.candidates[0] if isinstance(selected,AutoReconstructionEngine) else selected
    assert isinstance(first,expected)


def test_explicit_cuda_does_not_silently_run_cpu(monkeypatch):
    monkeypatch.setattr(engines,'get_recon_capability',lambda:capability(cuda_available=False))
    with pytest.raises(RuntimeError,match='CUDA was requested'):
        select_reconstruction_engine(device='cuda')
    with pytest.raises(RuntimeError,match='only supports CPU'):
        select_reconstruction_engine('opencv',device='cuda')


def test_auto_preserves_failure_before_fallback(tmp_path):
    class Failing:
        name='failed-ml'
        def reconstruct(self,image,output_dir,device=None):
            from pathlib import Path
            Path(output_dir).mkdir(parents=True)
            Path(output_dir,'partial.txt').write_text('failed model artifact')
            raise RuntimeError('out of memory')
    class Working:
        name='cpu-ml'
        def reconstruct(self,*args,**kwargs):
            return {'engine':self.name,'device':'cpu'}
    chain=AutoReconstructionEngine([Failing(),Working()])
    result=chain.reconstruct('input.jpg',str(tmp_path))
    assert [r['status'] for r in result['attempts']] == ['FAILED','SUCCEEDED']
    assert (tmp_path/'attempt_failed-ml'/'partial.txt').exists()
    assert json.loads((tmp_path/'attempts.json').read_text())[0]['error'].endswith('out of memory')


@pytest.mark.parametrize('bad', ['nan','nonbinary','boundary','empty'])
def test_import_rejects_invalid_solver_masks(tmp_path,monkeypatch,bad):
    monkeypatch.setattr(simulation,'UPLOADS_DIR',str(tmp_path))
    mask=np.zeros((64,128),dtype=np.float32)
    mask[24:40,50:78]=1
    if bad=='nan':mask[30,60]=np.nan
    elif bad=='nonbinary':mask[30,60]=.5
    elif bad=='boundary':mask[0,60]=1
    else:mask[:]=0
    stream=io.BytesIO();np.save(stream,mask)
    response=TestClient(app).post('/api/recon/import',files={'file':('mask.npy',stream.getvalue(),'application/octet-stream')})
    assert response.status_code==400
    assert not list(tmp_path.glob('mask_*.npy'))


def test_import_previews_and_runs_real_lbm(tmp_path,monkeypatch):
    monkeypatch.setattr(simulation,'UPLOADS_DIR',str(tmp_path))
    mask=fit_mask_to_domain(np.ones((10,40),dtype=bool))
    stream=io.BytesIO();np.save(stream,mask)
    client=TestClient(app)
    response=client.post('/api/recon/import',files={'file':('mask.npy',stream.getvalue(),'application/octet-stream')})
    assert response.status_code==200,response.text
    job=response.json()
    assert (tmp_path/f"preview_{job['job_id']}.jpg").exists()
    result=client.post('/api/simulate/simple',json={'job_id':job['job_id'],'wind_speed':15,'wind_angle_deg':0})
    assert result.status_code==200,result.text
    assert np.isfinite(result.json()['metrics']['cd_force_proxy'])
    from app.ml.features import char_length_from_mask
    assert result.json()['metrics']['solver_conditions']['char_length_lu']==char_length_from_mask(mask)


def test_import_bounds_header_before_allocation(tmp_path,monkeypatch):
    monkeypatch.setattr(simulation,'UPLOADS_DIR',str(tmp_path))
    stream=io.BytesIO()
    np.lib.format.write_array_header_1_0(stream,{'descr':'|b1','fortran_order':False,'shape':(10**9,10**9)})
    response=TestClient(app).post('/api/recon/import',files={'file':('mask.npy',stream.getvalue())})
    assert response.status_code==400


def test_sf3d_adapter_returns_public_evaluator_validation_contract(tmp_path,monkeypatch):
    # Exercises the adapter and real mesh export without downloading SF3D weights.
    image=tmp_path/'alpha.png';Image.new('RGBA',(32,32),(200,200,200,128)).save(image)
    torch=ModuleType('torch');torch.cuda=SimpleNamespace(is_available=lambda:False)
    torch.inference_mode=nullcontext
    class Model:
        @staticmethod
        def from_pretrained(*args,**kwargs):return Model()
        def to(self,*args):return self
        def eval(self):return self
        def run_image(self,*args,**kwargs):return trimesh.creation.box(extents=(2,1,.5)),{}
    system=ModuleType('sf3d.system');system.SF3D=Model
    utils=ModuleType('sf3d.utils');utils.resize_foreground=lambda image,ratio:image
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setitem(sys.modules,'sf3d',ModuleType('sf3d'))
    monkeypatch.setitem(sys.modules,'sf3d.system',system)
    monkeypatch.setitem(sys.modules,'sf3d.utils',utils)
    from app.recon.sf3d_runner import run_sf3d
    result=run_sf3d(str(image),str(tmp_path/'output'),device='cpu')
    assert result['engine']=='stable-fast-3d'
    assert result['serialized_mesh']['valid']
    assert result['mask']['shape']==[64,128]

def test_original_photo_upload_still_runs_cpu_simulation(tmp_path,monkeypatch):
    import cv2
    import app.routes.upload as upload
    monkeypatch.setattr(upload,'UPLOADS_DIR',str(tmp_path))
    monkeypatch.setattr(simulation,'UPLOADS_DIR',str(tmp_path))
    image=np.full((300,600,3),240,dtype=np.uint8)
    cv2.circle(image,(300,150),60,(20,20,20),-1)
    ok,jpeg=cv2.imencode('.jpg',image)
    assert ok
    client=TestClient(app)
    uploaded=client.post('/api/upload',files={'file':('synthetic-circle.jpg',jpeg.tobytes(),'image/jpeg')})
    assert uploaded.status_code==200,uploaded.text
    result=client.post('/api/simulate/simple',json={'job_id':uploaded.json()['job_id'],'wind_speed':15,'wind_angle_deg':0})
    assert result.status_code==200,result.text
    assert np.isfinite(result.json()['metrics']['cd_force_proxy'])
