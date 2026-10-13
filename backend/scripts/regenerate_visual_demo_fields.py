"""Regenerate car/drone CPU caches from the exact displayed side silhouettes.

Stages output, preserves old arrays in a fresh evidence directory, and records
hashes and educational force/field proxies. Does not claim full 3D CFD.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
from app.services.cache_generator import make_car_mask, make_drone_mask, NX, NY, STEPS, TAU, U_INLET
from app.solvers.lbm_2d import LbmSolver2D, educational_force_metrics
from app.ml.features import char_length_from_mask


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default=f'evaluation_outputs/visual_fields_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}')
    args=parser.parse_args();out=(ROOT/args.output_dir).resolve()
    if out.exists():raise ValueError('Choose a fresh output directory to preserve old evidence')
    out.mkdir(parents=True)
    flow=ROOT/'backend/app/assets/flow';(out/'before').mkdir();(out/'after').mkdir()
    geometry_bytes=(ROOT/'shared/demoGeometry.json').read_bytes();geometry=json.loads(geometry_bytes)
    result={'generated_utc':datetime.now(timezone.utc).isoformat(),'geometry_sha256':hashlib.sha256(geometry_bytes).hexdigest(),'car_asset_sha256':geometry['car_asset_sha256'],'source':'Exact display-model side projection, CPU D2Q9, educational proxies','cases':{}}
    for name,builder in [('sports_car_v1',make_car_mask),('drone_v1',make_drone_mask)]:
        for kind in ['mask','velocity','pressure']:
            source=flow/f'{name}_{kind}.npy'
            if source.exists():shutil.copyfile(source,out/'before'/source.name)
        mask=builder();solver=LbmSolver2D(nx=NX,ny=NY,tau=TAU,u_inlet=U_INLET)
        started=time.perf_counter();velocity,pressure=solver.solve(mask,steps=STEPS)
        if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(pressure)):raise ValueError(f'{name} produced non-finite fields')
        force=educational_force_metrics(pressure,mask,U_INLET,char_length_from_mask(mask),velocity=velocity,tau=TAU,momentum_force_lu=solver.last_force_lu,momentum_method=solver.last_force_method)
        wake_pixels=int(np.sum((velocity[0]<.02)&~mask));wake=float(np.clip(wake_pixels/(NX*(NY//2)),.05,.99))
        drag=float(np.clip(.15+wake_pixels/(NX*NY)*3.5,.05,1.8));lift=float(np.clip((np.sum(pressure[NY//2:])-np.sum(pressure[:NY//2]))*1.5,-.8,1.5))
        hashes={}
        for kind,array in [('mask',mask),('velocity',velocity.astype(np.float32)),('pressure',pressure.astype(np.float32))]:
            target=out/'after'/f'{name}_{kind}.npy';np.save(target,array,allow_pickle=False);hashes[kind]=hashlib.sha256(target.read_bytes()).hexdigest()
        result['cases'][name]={'nx':NX,'ny':NY,'steps':STEPS,'tau':TAU,'u_inlet':U_INLET,'wall_time_s':round(time.perf_counter()-started,3),'solid_cells':int(np.sum(mask)),'array_sha256':hashes,'metrics':{'drag_coefficient_estimate':drag,'lift_coefficient_estimate':lift,'wake_score':wake,'cd_force_proxy':float(force['cd_force_proxy']),'confidence_label':'Educational proxies computed from this cached CPU field; not certified CFD'}}
        print(name,result['cases'][name]['metrics'])
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    # Publish only after both staged simulations validated. Atomic file replacement.
    for source in (out/'after').glob('*.npy'):
        temp=flow/(source.name+'.tmp');shutil.copyfile(source,temp);temp.replace(flow/source.name)
    (flow/'visual_demo_generation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Published validated CPU caches; before/after evidence preserved at',out)


if __name__=='__main__':main()
