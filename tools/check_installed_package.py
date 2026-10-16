"""Validate an installed gato-mpc wheel from OUTSIDE the source checkout.

CPU by default (what the publish workflow runs on a GPU-less runner): the
package imports from site-packages, carries its native sources, resolves the
pinned grid-rbd release, exposes the vendored robots, and generates code for a
tiny URDF into --work. ``--gpu`` additionally registers that robot, lets the
first solver compile its module into a cache under --work, solves, and — when
a CUDA torch is installed — checks the device path against the host path.

    python -I tools/check_installed_package.py --work /tmp/gato-smoke [--gpu]
"""
import argparse
import json
from pathlib import Path
import sys

URDF = '''<robot name="package_probe">
  <link name="base"/>
  <link name="upper"><inertial><origin xyz="0.2 0 0"/><mass value="1"/>
    <inertia ixx="0.05" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><origin xyz="0.2 0 0"/><geometry><cylinder radius="0.04" length="0.4"/></geometry></collision></link>
  <link name="lower"><inertial><origin xyz="0.2 0 0"/><mass value="0.5"/>
    <inertia ixx="0.02" iyy="0.05" izz="0.05" ixy="0" ixz="0" iyz="0"/></inertial>
    <collision><origin xyz="0.2 0 0"/><geometry><cylinder radius="0.03" length="0.4"/></geometry></collision></link>
  <link name="tool"/>
  <joint name="shoulder" type="revolute"><parent link="base"/><child link="upper"/>
    <axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="20" velocity="10"/></joint>
  <joint name="elbow" type="revolute"><parent link="upper"/><child link="lower"/><origin xyz="0.4 0 0"/>
    <axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="10" velocity="10"/></joint>
  <joint name="EE" type="fixed"><parent link="lower"/><child link="tool"/><origin xyz="0.4 0 0"/></joint>
</robot>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)

    import gato
    from gato import _cache, _grid

    root = Path(gato.__file__).resolve().parent
    assert 'site-packages' in root.parts or 'dist-packages' in root.parts, f'not an installed package: {root}'
    native = _cache.native_root()
    assert native == root / '_native', f'wheel must carry its native sources, got {native}'
    for rel in ('CMakeLists.txt', 'python/bindings.cu', 'tools/native_identity.py',
                'gato/bsqp/bsqp.cuh', 'gato/dynamics/plant.cuh', 'gato/dynamics/indy7/grid.cuh'):
        assert (native / rel).is_file(), rel
    provenance = _grid.require()
    assert provenance['peers']['GLASS']['commit'] == _grid.GLASS_PIN
    assert (_grid.collision_dir() / 'grid_collision_geometry.cuh').is_file()
    # every vendored robot (the registry): URDF bundled, headers resolvable
    from gato.builder import load_registry
    vendored = sorted(load_registry())
    assert {'indy7', 'iiwa14', 'go2', 'fr3', 'gen3', 'ur5e'} <= set(vendored), vendored
    for name in vendored:
        assert gato.robot_urdf(name).is_file(), name
        assert _cache.resolve(name) is not None, name

    cache = args.work / 'cache'
    urdf = args.work / 'probe.urdf'
    urdf.write_text(URDF)
    meta = gato.register_robot('package_probe', urdf, N=[8], cache_dir_override=cache)
    assert meta['nq'] == meta['nv'] == 2 and meta['ee_frame'] == 'EE', meta
    entry = _cache.robot_entry('package_probe', cache)
    assert (cache / 'robots' / entry['codegen_key'] / 'package_probe' / 'grid.cuh').is_file()

    report = {'result': 'PASS', 'version': gato.__version__, 'root': str(root),
              'grid_rbd': _grid.installed_version(), 'glass': provenance['peers']['GLASS']['commit'][:7]}
    if args.gpu:
        import os
        import numpy as np
        os.environ['GATO_CACHE_DIR'] = str(cache)
        s = gato.BSQP(urdf, batch_size=2, N=8, dt=0.01, plant_type='package_probe', max_sqp_iters=3)
        x = np.zeros((2, s.nx), np.float32)
        goals = np.zeros((2, 8 * 6), np.float32)
        goals[:, 0::6] = 0.8
        host = s.solve(x, goals)
        assert np.isfinite(host.xu).all()
        assert _cache.find_module('package_probe', 8, cache=cache) is not None
        report['module'] = str(_cache.find_module('package_probe', 8, cache=cache).name)
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and torch.cuda.is_available():
            s2 = gato.BSQP(urdf, batch_size=2, N=8, dt=0.01, plant_type='package_probe', max_sqp_iters=3)
            dev = s2.solve(torch.from_numpy(x).cuda(), torch.from_numpy(goals).cuda())
            np.testing.assert_array_equal(dev.xu.cpu().numpy(), host.xu)
            report['device_path'] = 'bit-identical (torch CUDA)'
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    sys.exit(main())
