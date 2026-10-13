"""Verify typed reconstruction inputs through real CPU model inference and LBM.

Does not reinterpret synthetic/public inputs as phone captures. Every invocation
uses a fresh output directory unless --output-dir is explicitly supplied.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from app.main import app
from app.recon.evaluation import evaluate_manifest
from app.recon.integration import register_mask_for_simulation
from app.routes import simulate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--engine', choices=['depth_anything', 'opencv', 'auto'], default='depth_anything')
    args = parser.parse_args()
    out = Path(args.output_dir or f"../evaluation_outputs/cpu_e2e_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    if out.exists():
        raise ValueError('Output directory already exists; choose a fresh path to preserve prior attempts')
    try:
        import torch
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    except ImportError:
        pass
    summary = evaluate_manifest(args.manifest, str(out), device='cpu', engine=args.engine)
    client = TestClient(app)
    for case in summary['results']:
        case['lbm_status'] = 'NOT_RUN'
        if case['status'] != 'SUCCEEDED':
            continue
        try:
            recon = case['model_result']
            job = register_mask_for_simulation(recon['mask']['mask_path'], simulate.UPLOADS_DIR)
            response = client.post('/api/simulate/simple', json={'job_id': job['job_id'], 'wind_speed': 15, 'wind_angle_deg': 0})
            if response.status_code != 200:
                raise RuntimeError(response.text)
            payload = response.json()
            arrays = {}
            simulation_dir = Path(case['output_dir']) / 'simulation'
            simulation_dir.mkdir(parents=True, exist_ok=True)
            for name in ['mask','velocity','pressure']:
                array = np.load(Path(simulate.UPLOADS_DIR) / f'{name}_{job["job_id"]}.npy', allow_pickle=False)
                if not np.all(np.isfinite(array)):
                    raise ValueError(f'{name} is non-finite')
                target = simulation_dir / f'{name}.npy'
                np.save(target, array)
                arrays[name] = {'shape': list(array.shape), 'path': str(target.resolve()), 'finite': True}
            case.update(lbm_status='SUCCEEDED', simulation=payload, simulation_arrays=arrays)
        except Exception as exc:
            case.update(lbm_status='FAILED', lbm_error=f'{type(exc).__name__}: {exc}')
    summary['n_lbm_succeeded'] = sum(case['lbm_status'] == 'SUCCEEDED' for case in summary['results'])
    summary['n_lbm_failed'] = sum(case['lbm_status'] == 'FAILED' for case in summary['results'])
    (out / 'cpu_e2e.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({key: summary[key] for key in ['n_cases','n_succeeded','n_failed','n_lbm_succeeded','n_lbm_failed']}, indent=2))
    return 0 if summary['n_lbm_succeeded'] == summary['n_cases'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
