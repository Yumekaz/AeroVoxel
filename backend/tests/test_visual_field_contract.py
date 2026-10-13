from pathlib import Path
import hashlib
import json
import numpy as np
import pytest
from app.services.cache_generator import make_car_mask, make_drone_mask
from app.routes.demo_cases import _available_demo_cases

ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('case_id,builder', [('sports_car_v1',make_car_mask),('drone_v1',make_drone_mask)])
def test_published_cpu_field_uses_display_model_projection(case_id,builder):
    geometry=json.loads((ROOT/'shared/demoGeometry.json').read_text())
    generation=json.loads((ROOT/'backend/app/assets/flow/visual_demo_generation.json').read_text())
    mask=builder()
    assert mask.shape==(64,128) and mask.dtype==np.bool_
    assert mask.any() and not mask[0].any() and not mask[-1].any()
    expected=np.array(geometry['cases'][case_id]['mask']).reshape(64,128).astype(bool)
    assert np.array_equal(mask,expected)
    for kind in ['mask','velocity','pressure']:
        path=ROOT/f'backend/app/assets/flow/{case_id}_{kind}.npy'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==generation['cases'][case_id]['array_sha256'][kind]
        array=np.load(path,allow_pickle=False)
        assert np.isfinite(array).all()
        if kind=='mask':assert np.array_equal(array,expected)
    assert hashlib.sha256((ROOT/'frontend/public/models/concept-car.glb').read_bytes()).hexdigest()==generation['car_asset_sha256']


def test_demo_api_estimates_match_regenerated_cpu_evidence():
    evidence=json.loads((ROOT/'backend/app/assets/flow/visual_demo_generation.json').read_text())
    catalog={case['case_id']:case for case in _available_demo_cases()}
    for name,record in evidence['cases'].items():
        assert catalog[name]['metrics_source']=='Cached CPU field educational proxies'
        for key in ['drag_coefficient_estimate','lift_coefficient_estimate','wake_score']:
            assert catalog[name][key]==record['metrics'][key]
