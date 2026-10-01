"""Inspect a copied molecular cache and thermo log without loading pair histories.

Usage: python -m tools.diagnose_cached_motion RUN_DIR --output REPORT_DIR
This produces motion/thermo diagnostics, not an equilibration or reaction test.
COM masses below use standard C/H/N/O masses for the present CPD/MAL mixture.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from MartiniFuncs.io.thermo import discover_thermo_file, read_lammps_thermo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cache = args.run_dir/'cache'
    frames = pd.read_csv(cache/'frames.csv')
    molecules = json.loads((cache/'molecules.json').read_text())
    species = np.load(cache/'species.npy')
    masses = np.array([{'C5H6': 66.103, 'C11H9NO2': 187.198}[s] for s in species])
    com = np.load(cache/'com_angstrom.npy', mmap_mode='r')
    wrapped = np.load(cache/'wrapped_com_angstrom.npy', mmap_mode='r')
    box = np.load(cache/'box_lengths_angstrom.npy', mmap_mode='r')
    times = frames.time_ps.to_numpy()-frames.time_ps.iloc[0]
    assert len(com) == len(frames)
    raw = np.empty((len(com), len(species)))
    drift = np.empty((len(com), 3))
    corrected = np.empty_like(raw)
    max_increment = max_tor_error = 0.
    for start in range(0, len(com), 10000):
        end = min(start+10000, len(com))
        positions = np.asarray(com[start:end])
        assert np.isfinite(positions).all()
        d = positions-com[0]
        drift[start:end] = np.einsum('tma,m->ta', d, masses)/masses.sum()
        raw[start:end] = np.sum(d*d, axis=-1)
        corrected[start:end] = np.sum((d-drift[start:end, None, :])**2, axis=-1)
        lo = max(0, start-1)
        increments = np.diff(com[lo:end], axis=0)
        dw = np.diff(wrapped[lo:end], axis=0)
        lengths = box[lo+1:end, None, :]
        expected = dw-lengths*np.rint(dw/lengths)
        max_tor_error = max(max_tor_error, float(np.max(np.abs(increments-expected))))
        max_increment = max(max_increment, float(np.linalg.norm(increments, axis=-1).max()))
    thermo_path = discover_thermo_file(args.run_dir)
    thermo = read_lammps_thermo(thermo_path, units='metal')
    assert thermo.thermo_block.nunique() == 1
    thermo['elapsed_ps'] = thermo.thermo_time_ps-frames.time_ps.iloc[0]
    # P_kin = 2 KE/(3 V), eV/Angstrom^3 -> bar. Printed KE has finite precision.
    thermo['kinetic_pressure_bar'] = 2*thermo.KinEng/(3*thermo.Volume)*1602176.6208
    thermo['pressure_minus_kinetic_bar'] = thermo.Press-thermo.kinetic_pressure_bar
    summary = dict(frames=len(com), duration_ps=float(times[-1]),
        frame_spacing_ps=float(np.median(np.diff(times))),
        max_box_change_angstrom=float(np.max(np.abs(box-box[0]))),
        max_tor_increment_consistency_error_angstrom=max_tor_error,
        max_saved_molecular_com_step_angstrom=max_increment,
        max_system_com_displacement_angstrom=float(np.linalg.norm(drift, axis=1).max()),
        pressure_minus_kinetic_max_abs_bar=float(thermo.pressure_minus_kinetic_bar.abs().max()),
        temperature_mean_K=float(thermo.Temp.mean()), temperature_range_K=[float(thermo.Temp.min()),float(thermo.Temp.max())])
    windows = [(0,850),(850,880),(900,950),(1000,1200),(1400,1500.005)]
    rows = []
    for lo,hi in windows:
        keep = (times>=lo)&(times<hi)
        table = thermo[(thermo.elapsed_ps>=lo)&(thermo.elapsed_ps<hi)]
        for s in sorted(set(species)):
            mask = species == s
            rows.append(dict(start_ps=lo,end_ps=hi,species=s,
                mean_reference_msd_angstrom2=float(raw[keep][:,mask].mean()),
                drift_corrected_msd_angstrom2=float(corrected[keep][:,mask].mean()),
                mean_temperature_K=float(table.Temp.mean()),
                mean_potential_energy_eV=float(table.PotEng.mean())))
    # Rank motion at the largest MAL peak in the 880--950 ps event.
    keep = np.flatnonzero((times>=880)&(times<=950))
    peak = keep[np.argmax(raw[keep][:,species=='C11H9NO2'].mean(axis=1))]
    summary['event_peak_elapsed_ps'] = float(times[peak])
    ranking = []
    for s in sorted(set(species)):
        indices = np.flatnonzero(species==s)
        order = indices[np.argsort(raw[peak,indices])[::-1]]
        summary[s+'_peak_top5_fraction'] = float(raw[peak,order[:5]].sum()/raw[peak,indices].sum())
        for i in order:
            ranking.append(dict(species=s,molecule_index=int(i),time_ps=float(times[peak]),
                squared_displacement_angstrom2=float(raw[peak,i]),
                lammps_atom_ids=[int(a)+1 for a in molecules[i]['atom_ids']]))
    args.output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output/'window_summary.csv',index=False)
    (args.output/'event_molecules.json').write_text(json.dumps(ranking,indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes = plt.subplots(4,1,figsize=(11,11),sharex=True,layout='constrained')
    display_indices = np.arange(0,len(times),100)  # 0.5 ps display; metrics use all frames
    for s in sorted(set(species)):
        mask = species==s
        line, = axes[0].plot(times[display_indices],raw[display_indices][:,mask].mean(axis=1),label=s)
        axes[0].plot(times[display_indices],corrected[display_indices][:,mask].mean(axis=1),
                     color=line.get_color(),ls='--',label=s+' (system drift removed)')
    axes[0].set_ylabel(r'Reference MSD ($\mathrm{\AA}^2$)')
    axes[0].legend(fontsize=8,ncol=2)
    axes[1].plot(thermo.elapsed_ps,thermo.Temp,lw=.6)
    axes[1].axhline(400,color='black',ls='--',lw=.8)
    axes[1].set_ylabel('Temperature (K)')
    axes[2].plot(thermo.elapsed_ps,thermo.PotEng-thermo.PotEng.mean(),lw=.6,alpha=.5,label='Printed samples')
    axes[2].plot(thermo.elapsed_ps,thermo.PotEng.rolling(100,center=True).mean()-thermo.PotEng.mean(),
                 color='black',label='50 ps moving mean')
    axes[2].set_ylabel('PE minus run mean (eV)')
    axes[2].legend(fontsize=8)
    axes[3].plot(times[display_indices],np.linalg.norm(drift[display_indices],axis=1))
    axes[3].set_ylabel(r'System COM displacement ($\mathrm{\AA}$)')
    axes[3].set_xlabel('Time since NVT start (ps)')
    for ax in axes:
        ax.axvspan(880,950,color='gray',alpha=.12)
        ax.grid(alpha=.2)
    fig.savefig(args.output/'motion_thermo.png',dpi=160)
    plt.close(fig)
    print(json.dumps(summary,indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
