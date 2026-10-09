"""Reproduce DEM observation/checkpoint defects using immutable incoming git source."""
import argparse,hashlib,importlib.util,json,subprocess,sys,tempfile
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path.cwd())
    p.add_argument('--output',type=Path,default=Path('/tmp/tensordem3d-pre-fix.json'))
    args=p.parse_args();commit='6dd9e7824c4f90a8eebad1c454cc946ef04d5fdd'
    source=subprocess.check_output(['git','show',commit+':src/tensordem/dem3d.py'],cwd=args.repo)
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='dem3d-readonly-old-source-') as folder:
        path=Path(folder)/'dem3d_old.py';path.write_bytes(source)
        spec=importlib.util.spec_from_file_location('dem3d_pre_fix_isolated',path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        cfg=module.DEM3DConfig(nx=3,ny=3,nz=2,drag=0.,fix_x_edges=False)
        s=module.IceDEM3D(cfg);s.positions=s.initial_positions*1.03;before=s.state_dict();s.diagnostics()
        t=module.IceDEM3D(cfg);snap=t.state_dict();snap['alive']=torch.full(snap['alive'].shape,float('nan'),dtype=torch.float64);t.load_state_dict(snap)
        report=dict(source_head=commit,source_sha256=hashlib.sha256(source).hexdigest(),
                    reproducer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    diagnostics_new_broken=int((before['alive']&~s.alive).sum()),
                    diagnostics_changed_failure_history=not torch.equal(before['failure_mode'],s.failure_mode),
                    diagnostics_time_s=s.time,nan_alive_dtype_accepted=bool(t.alive.all()),
                    scope='isolated exact pre-fix git source; observation and checkpoint validation only; not physical energy qualification')
        args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
