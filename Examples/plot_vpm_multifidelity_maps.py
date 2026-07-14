"""Generate VPM F1A acoustic maps for every RPM and fidelity.

The map layout intentionally matches ``VPM/VPM-F1A.ipynb``: a 30 x 30
observer half-plane over 5 m, mirrored by ``Plotter`` and plotted as OSPL.
The final five rotations are used because that is the largest common native
window across the low, mid, and high multifidelity histories.

Overall levels are integrated without the solver's optional 20 Hz zero
padding, which otherwise dilutes short high-RPM histories. OSPL is therefore
the RMS of the solver's mean-removed pressure and OASPL uses its native FFT.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401 - registers the SciencePlots styles
import torch

from FastBEMT import (
    Environment,
    F1A,
    Plotter,
    Propeller,
    Simulation,
    load_propeller_geometry,
)
from FastBEMT.Aeroacoustics import uniform_observer_grid
from FastBEMT.Aeroacoustics.Utils import (
    spl_spectrum_to_overall_level,
    time_domain_to_spl_spectrum,
)
plt.style.use(["science", "no-latex"])
warnings.filterwarnings(
    "ignore",
    message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "Data" / "VPM"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Figures" / "VPM-F1A" / "multifidelity"
PROPELLER_GEOMETRY_FILE = REPO_ROOT / "Data" / "10x7E.pkl"

EXPECTED_RPMS = (3000, 4000, 5000, 6000, 7000)
FIDELITIES = ("steady", "low", "mid", "high")
CASE_PATTERN = re.compile(
    r"^loadingsAPC10x7E_(?P<fidelity>steady|low|mid|high)_"
    r"(?P<rpm>\d+)RPM\.pt$"
)
CACHE_VERSION = 2
STEADY_SAMPLES_PER_ROTATION = 360
SUMMARY_FIELDS = (
    "status",
    "rpm",
    "fidelity",
    "loadings_file",
    "effective_loadings_file",
    "derived_periodic_steady",
    "steady_blade_frame_relative_rms",
    "steady_derivative_relative_rms",
    "steady_samples_per_rotation",
    "retained_source_samples",
    "retained_sections",
    "retained_source_rotations",
    "observer_rotations",
    "observer_samples",
    "sample_spacing_s",
    "nyquist_hz",
    "frequency_resolution_hz",
    "map_min_db",
    "map_max_db",
    "solver_padding_delta_mean_db",
    "elapsed_seconds",
    "map_file",
    "data_file",
    "error",
)


@dataclass(frozen=True)
class Case:
    rpm: int
    fidelity: str
    path: Path

    @property
    def title(self) -> str:
        fidelity = (
            "Steady (periodic)"
            if self.fidelity == "steady"
            else self.fidelity.title()
        )
        return f"{fidelity} - {self.rpm} RPM"

    @property
    def output_stem(self) -> str:
        return f"APC10x7E_{self.fidelity}_{self.rpm}RPM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rpm", dest="rpms", type=int, action="append")
    parser.add_argument(
        "--fidelity",
        dest="fidelities",
        choices=FIDELITIES,
        action="append",
    )
    parser.add_argument("--metric", choices=("ospl", "oaspl"), default="ospl")
    parser.add_argument("--last-rotations", type=float, default=5.0)
    parser.add_argument("--domain-size", type=float, default=5.0)
    parser.add_argument("--grid-size", type=int, default=30)
    parser.add_argument("--observer-batch-size", type=int, default=25)
    parser.add_argument(
        "--steady-rotations",
        type=int,
        default=5,
        help=(
            "Rigid-periodic rotations synthesized from each steady loading; "
            "defaults to the common five-rotation source window."
        ),
    )
    parser.add_argument(
        "--steady-invariance-tolerance",
        type=float,
        default=0.01,
        help="Maximum relative RMS variation allowed after steady loads are moved to blade frames.",
    )
    parser.add_argument(
        "--steady-derivative-tolerance",
        type=float,
        default=0.01,
        help="Maximum RMS error allowed for the synthesized rigid-rotation derivative.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse an existing NPZ map instead of recomputing that case.",
    )
    return parser.parse_args()


def discover_cases(data_dir: Path) -> list[Case]:
    cases = []
    for path in data_dir.glob("loadingsAPC10x7E_*_*RPM.pt"):
        match = CASE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        cases.append(
            Case(
                rpm=int(match.group("rpm")),
                fidelity=match.group("fidelity"),
                path=path.resolve(),
            )
        )

    keys = [(case.rpm, case.fidelity) for case in cases]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate RPM/fidelity loading files were found.")

    expected = {(rpm, fidelity) for rpm in EXPECTED_RPMS for fidelity in FIDELITIES}
    found = set(keys)
    unexpected = sorted(found - expected)
    if unexpected:
        raise ValueError(
            f"Unexpected RPM/fidelity loading files in '{data_dir}': {unexpected}"
        )

    fidelity_order = {name: index for index, name in enumerate(FIDELITIES)}
    return sorted(cases, key=lambda case: (case.rpm, fidelity_order[case.fidelity]))


def select_cases(cases: list[Case], args: argparse.Namespace) -> list[Case]:
    selected_rpms = set(args.rpms or EXPECTED_RPMS)
    selected_fidelities = set(args.fidelities or FIDELITIES)
    invalid_rpms = sorted(selected_rpms - set(EXPECTED_RPMS))
    if invalid_rpms:
        raise ValueError(f"Unsupported RPM filters: {invalid_rpms}")
    expected = {
        (rpm, fidelity)
        for rpm in selected_rpms
        for fidelity in selected_fidelities
    }
    found = {(case.rpm, case.fidelity) for case in cases}
    missing = sorted(expected - found)
    if missing:
        raise FileNotFoundError(
            "Missing selected RPM/fidelity loading files: " + str(missing)
        )
    selected = [
        case
        for case in cases
        if case.rpm in selected_rpms and case.fidelity in selected_fidelities
    ]
    return selected


def map_paths(case: Case, output_dir: Path, metric: str) -> tuple[Path, Path]:
    base = output_dir / f"{case.output_stem}_{metric}_map"
    return base.with_suffix(".png"), base.with_suffix(".npz")


def contour_for(levels: np.ndarray) -> list[float] | None:
    minimum = float(np.nanmin(levels))
    maximum = float(np.nanmax(levels))
    return [60.0] if minimum < 60.0 < maximum else None


def rotation_matrices(
    source_times: torch.Tensor,
    *,
    rpm: int,
    blade_count: int,
    global_to_blade: bool,
) -> torch.Tensor:
    """Return the x-axis rotor transforms used by ``Kinematics``."""
    omega = 2.0 * math.pi * float(rpm) / 60.0
    phase_offsets = (
        2.0
        * math.pi
        / blade_count
        * torch.arange(blade_count, dtype=source_times.dtype)
    )
    angles = omega * source_times[:, None] + phase_offsets[None, :]
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    zeros = torch.zeros_like(cosine)
    ones = torch.ones_like(cosine)
    signed_sine = sine if global_to_blade else -sine
    return torch.stack(
        (
            torch.stack((ones, zeros, zeros), dim=-1),
            torch.stack((zeros, cosine, signed_sine), dim=-1),
            torch.stack((zeros, -signed_sine, cosine), dim=-1),
        ),
        dim=-2,
    )


def prepare_periodic_steady_loading(
    case: Case,
    *,
    args: argparse.Namespace,
) -> tuple[Path, float, float, int]:
    """Turn short steady snapshots into a verified rigid-periodic history."""
    required_rotations = max(2, int(math.ceil(float(args.last_rotations))))
    if args.steady_rotations < required_rotations:
        raise ValueError(
            "steady_rotations must be at least the requested last_rotations "
            f"window ({required_rotations})."
        )
    tolerance = float(args.steady_invariance_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("steady_invariance_tolerance must be finite and positive.")
    derivative_tolerance = float(args.steady_derivative_tolerance)
    if not np.isfinite(derivative_tolerance) or derivative_tolerance <= 0.0:
        raise ValueError("steady_derivative_tolerance must be finite and positive.")

    payload = torch.load(case.path, map_location="cpu")
    loaded_table = payload["loadings"]
    table = loaded_table.detach().to(device="cpu", dtype=torch.float32)
    sections = payload["sections"]
    rotation_period = 60.0 / float(case.rpm)
    source_times = (
        rotation_period
        / STEADY_SAMPLES_PER_ROTATION
        * torch.arange(table.shape[0], dtype=table.dtype)
    )
    global_forces = table
    source_time_basis = "steady snapshots at 0, 1, and 2 degrees"
    blade_count = int(table.shape[1])
    global_to_blade = rotation_matrices(
        source_times,
        rpm=case.rpm,
        blade_count=blade_count,
        global_to_blade=True,
    )
    blade_forces = torch.einsum(
        "tbij,tbsj->tbsi",
        global_to_blade,
        global_forces,
    )
    steady_blade_forces = torch.mean(blade_forces, dim=0)
    residual_rms = torch.sqrt(
        torch.mean((blade_forces - steady_blade_forces[None, ...]) ** 2)
    )
    force_rms = torch.sqrt(torch.mean(steady_blade_forces**2))
    scale = max(float(force_rms), torch.finfo(table.dtype).eps)
    relative_rms = float(residual_rms) / scale
    if relative_rms > tolerance:
        raise ValueError(
            f"{case.title} is not steady in the blade frame: relative RMS "
            f"variation {relative_rms:.6f} exceeds {tolerance:.6f}."
        )

    samples_per_rotation = STEADY_SAMPLES_PER_ROTATION
    reconstructed_dt = rotation_period / samples_per_rotation

    interval_count = args.steady_rotations * samples_per_rotation
    periodic_times = source_times[0] + reconstructed_dt * torch.arange(
        interval_count + 1,
        dtype=table.dtype,
    )
    blade_to_global = rotation_matrices(
        periodic_times,
        rpm=case.rpm,
        blade_count=blade_count,
        global_to_blade=False,
    )
    periodic_forces = torch.einsum(
        "tbij,bsj->tbsi",
        blade_to_global,
        steady_blade_forces,
    )
    omega_vector = periodic_forces.new_tensor(
        [2.0 * math.pi * float(case.rpm) / 60.0, 0.0, 0.0]
    ).expand_as(periodic_forces)
    expected_derivative = torch.linalg.cross(
        omega_vector,
        periodic_forces,
        dim=-1,
    )
    numerical_derivative = torch.gradient(
        periodic_forces,
        spacing=(periodic_times,),
        dim=(0,),
        edge_order=2,
    )[0]
    derivative_error_rms = torch.sqrt(
        torch.mean((numerical_derivative - expected_derivative) ** 2)
    )
    derivative_scale = torch.sqrt(torch.mean(expected_derivative**2))
    derivative_relative_rms = float(derivative_error_rms) / max(
        float(derivative_scale),
        torch.finfo(table.dtype).eps,
    )
    if derivative_relative_rms > derivative_tolerance:
        raise ValueError(
            f"{case.title} synthesized derivative RMS error "
            f"{derivative_relative_rms:.6f} exceeds {derivative_tolerance:.6f}."
        )
    periodic_time_channel = periodic_times[:, None, None, None].expand(
        -1,
        table.shape[1],
        table.shape[2],
        1,
    )
    periodic_table = torch.cat((periodic_time_channel, periodic_forces), dim=-1)

    derived_dir = args.data_dir / "derived_periodic_steady"
    derived_dir.mkdir(parents=True, exist_ok=True)
    derived_path = derived_dir / f"{case.path.stem}_periodic.pt"
    torch.save(
        {
            "loadings": periodic_table.contiguous(),
            "sections": dict(sections),
            "periodic_steady_derivation": {
                "source_file": case.path.name,
                "source_shape": tuple(loaded_table.shape),
                "source_samples": int(table.shape[0]),
                "source_times_s": source_times.tolist(),
                "source_time_basis": source_time_basis,
                "source_blade_frame_relative_rms": relative_rms,
                "derivative_relative_rms": derivative_relative_rms,
                "samples_per_rotation": samples_per_rotation,
                "rotations": int(args.steady_rotations),
                "method": "blade-frame mean followed by analytic rigid rotation",
            },
        },
        derived_path.with_suffix(".pt.tmp"),
    )
    derived_path.with_suffix(".pt.tmp").replace(derived_path)
    return derived_path, relative_rms, derivative_relative_rms, samples_per_rotation


def render_case_map(
    plotter: Plotter,
    case: Case,
    levels: np.ndarray,
    *,
    args: argparse.Namespace,
    path: Path,
) -> None:
    plotter.plot_acoustic_maps(
        {case.title: levels},
        grid_size=args.grid_size,
        domain_size=args.domain_size,
        metric=args.metric,
        columns=1,
        figsize=(4.5, 3.0),
        cmap="magma",
        contour_levels=contour_for(levels),
        mirror=True,
        save_path=str(path),
    )
    plt.close("all")


def unpadded_overall_levels(f1a: F1A) -> tuple[torch.Tensor, torch.Tensor]:
    """Return OSPL/OASPL without short-record zero-padding dilution."""
    pressure_rms = torch.sqrt(torch.mean(f1a.p_tot.square(), dim=1))
    p_ref = float(f1a.environment.p_ref)
    ospl = 20.0 * torch.log10(
        pressure_rms.clamp_min(torch.finfo(pressure_rms.dtype).tiny) / p_ref
    )
    native_frequencies, native_spl = time_domain_to_spl_spectrum(
        f1a.p_tot,
        f1a.sample_spacing,
        p_ref,
        time_dim=1,
        frequency_bin_width=None,
    )
    oaspl = spl_spectrum_to_overall_level(
        native_spl,
        native_frequencies,
        weighted=True,
        frequency_dim=1,
    )
    return ospl, oaspl


def run_case(
    case: Case,
    geometry: dict,
    environment: Environment,
    observers: np.ndarray,
    *,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object]]:
    started = time.perf_counter()
    effective_path = case.path
    derived_periodic_steady = False
    steady_relative_rms = float("nan")
    steady_derivative_relative_rms = float("nan")
    steady_samples_per_rotation = 0
    if case.fidelity == "steady":
        (
            effective_path,
            steady_relative_rms,
            steady_derivative_relative_rms,
            steady_samples_per_rotation,
        ) = prepare_periodic_steady_loading(case, args=args)
        derived_periodic_steady = True

    simulation = Simulation(
        revolutions=1,
        timesteps_per_revolution=1,
        device=args.device,
    )
    propeller = Propeller(geometry, environment, simulation)
    f1a = F1A(
        propeller=propeller,
        loadings=effective_path,
        rpm=float(case.rpm),
        last_rotations=args.last_rotations,
    )
    source_times = f1a.kinematics.source_times.detach().cpu().numpy()
    source_rotations = (
        float(source_times[-1] - source_times[0]) * float(case.rpm) / 60.0
    )

    f1a.run(
        observers=observers,
        observer_duration=simulation.observer_duration,
        n_observer_times=simulation.n_timesteps,
        observer_batch_size=args.observer_batch_size,
    )
    corrected_ospl, corrected_oaspl = unpadded_overall_levels(f1a)
    padding_delta = corrected_ospl - f1a.ospl
    sample_spacing = float(f1a.sample_spacing)
    nyquist = 0.5 / sample_spacing
    frequency_resolution = 1.0 / (f1a.n_observer_times * sample_spacing)
    ospl = corrected_ospl.detach().cpu().numpy()
    oaspl = corrected_oaspl.detach().cpu().numpy()
    levels = oaspl if args.metric == "oaspl" else ospl
    png_path, npz_path = map_paths(case, args.output_dir, args.metric)
    np.savez_compressed(
        npz_path,
        levels=levels,
        ospl=ospl,
        oaspl=oaspl,
        cache_version=np.int64(CACHE_VERSION),
        rpm=np.int64(case.rpm),
        fidelity=np.asarray(case.fidelity),
        metric=np.asarray(args.metric),
        source_loading_size_bytes=np.int64(case.path.stat().st_size),
        source_loading_mtime_ns=np.int64(case.path.stat().st_mtime_ns),
        propeller_geometry_size_bytes=np.int64(
            PROPELLER_GEOMETRY_FILE.stat().st_size
        ),
        propeller_geometry_mtime_ns=np.int64(
            PROPELLER_GEOMETRY_FILE.stat().st_mtime_ns
        ),
        last_rotations=np.float64(args.last_rotations),
        derived_periodic_steady=np.bool_(derived_periodic_steady),
        steady_blade_frame_relative_rms=np.float64(steady_relative_rms),
        steady_derivative_relative_rms=np.float64(
            steady_derivative_relative_rms
        ),
        steady_samples_per_rotation=np.int64(steady_samples_per_rotation),
        steady_rotations=np.int64(args.steady_rotations),
        steady_invariance_tolerance=np.float64(
            args.steady_invariance_tolerance
        ),
        steady_derivative_tolerance=np.float64(
            args.steady_derivative_tolerance
        ),
        grid_size=np.int64(args.grid_size),
        domain_size=np.float64(args.domain_size),
        retained_source_samples=np.int64(f1a.n_source_times),
        retained_sections=np.int64(f1a.n_sections),
        retained_source_rotations=np.float64(source_rotations),
        observer_rotations=np.int64(f1a.observer_rotations),
        observer_samples=np.int64(f1a.n_observer_times),
        sample_spacing_s=np.float64(sample_spacing),
        nyquist_hz=np.float64(nyquist),
        frequency_resolution_hz=np.float64(frequency_resolution),
        solver_ospl=f1a.ospl.detach().cpu().numpy(),
        solver_oaspl=f1a.oaspl.detach().cpu().numpy(),
        solver_padding_delta_mean_db=np.float64(
            float(torch.mean(padding_delta).detach().cpu())
        ),
    )
    render_case_map(
        Plotter(propeller),
        case,
        levels,
        args=args,
        path=png_path,
    )

    record = {
        "status": "computed",
        "rpm": case.rpm,
        "fidelity": case.fidelity,
        "loadings_file": str(case.path),
        "effective_loadings_file": str(effective_path),
        "derived_periodic_steady": derived_periodic_steady,
        "steady_blade_frame_relative_rms": (
            f"{steady_relative_rms:.6f}" if derived_periodic_steady else ""
        ),
        "steady_derivative_relative_rms": (
            f"{steady_derivative_relative_rms:.6f}"
            if derived_periodic_steady
            else ""
        ),
        "steady_samples_per_rotation": (
            steady_samples_per_rotation if derived_periodic_steady else ""
        ),
        "retained_source_samples": f1a.n_source_times,
        "retained_sections": f1a.n_sections,
        "retained_source_rotations": f"{source_rotations:.6f}",
        "observer_rotations": f1a.observer_rotations,
        "observer_samples": f1a.n_observer_times,
        "sample_spacing_s": f"{sample_spacing:.12g}",
        "nyquist_hz": f"{nyquist:.6f}",
        "frequency_resolution_hz": f"{frequency_resolution:.6f}",
        "map_min_db": f"{float(np.nanmin(levels)):.6f}",
        "map_max_db": f"{float(np.nanmax(levels)):.6f}",
        "solver_padding_delta_mean_db": (
            f"{float(torch.mean(padding_delta).detach().cpu()):.6f}"
        ),
        "elapsed_seconds": f"{time.perf_counter() - started:.3f}",
        "map_file": str(png_path),
        "data_file": str(npz_path),
        "error": "",
    }
    return levels, record


def load_cached_case(
    case: Case,
    *,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object]] | None:
    png_path, npz_path = map_paths(case, args.output_dir, args.metric)
    if (
        not args.skip_existing
        or not npz_path.is_file()
        or not png_path.is_file()
    ):
        return None

    with np.load(npz_path) as saved:
        required_fields = {
            "cache_version",
            "grid_size",
            "domain_size",
            "last_rotations",
            "source_loading_size_bytes",
            "source_loading_mtime_ns",
            "propeller_geometry_size_bytes",
            "propeller_geometry_mtime_ns",
            "levels",
            "derived_periodic_steady",
            "steady_blade_frame_relative_rms",
            "steady_derivative_relative_rms",
            "steady_samples_per_rotation",
            "steady_rotations",
            "steady_invariance_tolerance",
            "steady_derivative_tolerance",
            "retained_source_samples",
            "retained_sections",
            "retained_source_rotations",
            "observer_rotations",
            "observer_samples",
            "sample_spacing_s",
            "nyquist_hz",
            "frequency_resolution_hz",
            "solver_padding_delta_mean_db",
        }
        if not required_fields.issubset(saved.files):
            return None
        if int(saved["cache_version"]) != CACHE_VERSION:
            return None
        if int(saved["grid_size"]) != args.grid_size:
            return None
        if not np.isclose(float(saved["domain_size"]), args.domain_size):
            return None
        if not np.isclose(float(saved["last_rotations"]), args.last_rotations):
            return None
        if int(saved["source_loading_size_bytes"]) != case.path.stat().st_size:
            return None
        if int(saved["source_loading_mtime_ns"]) != case.path.stat().st_mtime_ns:
            return None
        if (
            int(saved["propeller_geometry_size_bytes"])
            != PROPELLER_GEOMETRY_FILE.stat().st_size
        ):
            return None
        if (
            int(saved["propeller_geometry_mtime_ns"])
            != PROPELLER_GEOMETRY_FILE.stat().st_mtime_ns
        ):
            return None
        if case.fidelity == "steady" and int(saved["steady_rotations"]) != int(
            args.steady_rotations
        ):
            return None
        if case.fidelity == "steady" and not np.isclose(
            float(saved["steady_invariance_tolerance"]),
            args.steady_invariance_tolerance,
        ):
            return None
        if case.fidelity == "steady" and not np.isclose(
            float(saved["steady_derivative_tolerance"]),
            args.steady_derivative_tolerance,
        ):
            return None
        levels = np.asarray(saved["levels"], dtype=float)
        derived_periodic_steady = bool(saved["derived_periodic_steady"])
        steady_relative_rms = float(saved["steady_blade_frame_relative_rms"])
        steady_derivative_relative_rms = float(
            saved["steady_derivative_relative_rms"]
        )
        steady_samples_per_rotation = int(saved["steady_samples_per_rotation"])
        solver_padding_delta = float(saved["solver_padding_delta_mean_db"])
        record = {
            "status": "cached",
            "rpm": case.rpm,
            "fidelity": case.fidelity,
            "loadings_file": str(case.path),
            "effective_loadings_file": (
                str(
                    args.data_dir
                    / "derived_periodic_steady"
                    / f"{case.path.stem}_periodic.pt"
                )
                if derived_periodic_steady
                else str(case.path)
            ),
            "derived_periodic_steady": derived_periodic_steady,
            "steady_blade_frame_relative_rms": (
                f"{steady_relative_rms:.6f}" if derived_periodic_steady else ""
            ),
            "steady_derivative_relative_rms": (
                f"{steady_derivative_relative_rms:.6f}"
                if derived_periodic_steady
                else ""
            ),
            "steady_samples_per_rotation": (
                steady_samples_per_rotation if derived_periodic_steady else ""
            ),
            "retained_source_samples": int(saved["retained_source_samples"]),
            "retained_sections": int(saved["retained_sections"]),
            "retained_source_rotations": f"{float(saved['retained_source_rotations']):.6f}",
            "observer_rotations": int(saved["observer_rotations"]),
            "observer_samples": int(saved["observer_samples"]),
            "sample_spacing_s": f"{float(saved['sample_spacing_s']):.12g}",
            "nyquist_hz": f"{float(saved['nyquist_hz']):.6f}",
            "frequency_resolution_hz": (
                f"{float(saved['frequency_resolution_hz']):.6f}"
            ),
            "map_min_db": f"{float(np.nanmin(levels)):.6f}",
            "map_max_db": f"{float(np.nanmax(levels)):.6f}",
            "solver_padding_delta_mean_db": f"{solver_padding_delta:.6f}",
            "elapsed_seconds": "0.000",
            "map_file": str(png_path),
            "data_file": str(npz_path),
            "error": "",
        }
    return levels, record


def write_summary(records: list[dict[str, object]], path: Path) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def comparison_window_errors(
    records: list[dict[str, object]],
) -> tuple[list[str], set[int]]:
    errors = []
    invalid_rpms = set()
    for rpm in EXPECTED_RPMS:
        rpm_records = [
            record
            for record in records
            if record.get("status") in {"computed", "cached"}
            and int(record["rpm"]) == rpm
        ]
        observer_rotations = {
            int(record["observer_rotations"]) for record in rpm_records
        }
        if len(observer_rotations) > 1:
            invalid_rpms.add(rpm)
            errors.append(
                f"{rpm} RPM fidelities use different observer windows: "
                f"{sorted(observer_rotations)} rotations."
            )
    return errors, invalid_rpms


def render_rpm_comparisons(
    levels_by_case: dict[tuple[int, str], np.ndarray],
    geometry: dict,
    environment: Environment,
    *,
    args: argparse.Namespace,
    excluded_rpms: set[int],
) -> None:
    simulation = Simulation(revolutions=1, timesteps_per_revolution=1, device="cpu")
    plotter = Plotter(Propeller(geometry, environment, simulation))
    for rpm in EXPECTED_RPMS:
        if rpm in excluded_rpms:
            continue
        maps = {
            (
                "Steady (periodic)"
                if fidelity == "steady"
                else fidelity.title()
            ): levels_by_case[(rpm, fidelity)]
            for fidelity in FIDELITIES
            if (rpm, fidelity) in levels_by_case
        }
        if not maps:
            continue
        output_path = args.output_dir / f"APC10x7E_{rpm}RPM_fidelities_{args.metric}.png"
        plotter.plot_acoustic_maps(
            maps,
            grid_size=args.grid_size,
            domain_size=args.domain_size,
            metric=args.metric,
            columns=len(maps),
            figsize=(4.2 * len(maps), 3.2),
            cmap="magma",
            contour_levels=(
                [60.0]
                if all(contour_for(levels) for levels in maps.values())
                else None
            ),
            mirror=True,
            shared_color_scale=True,
            save_path=output_path,
        )
        plt.close("all")


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = select_cases(discover_cases(args.data_dir), args)
    geometry = load_propeller_geometry(PROPELLER_GEOMETRY_FILE)
    environment = Environment()
    observers = uniform_observer_grid(
        size=args.domain_size,
        nx=args.grid_size,
        ny=args.grid_size,
    )

    print(f"Device: {args.device}")
    print(f"Selected cases: {len(cases)}")
    records: list[dict[str, object]] = []
    levels_by_case: dict[tuple[int, str], np.ndarray] = {}
    failures: list[str] = []
    summary_path = args.output_dir / "map_summary.csv"

    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.title}", flush=True)
        try:
            cached = load_cached_case(case, args=args)
            levels, record = (
                cached
                if cached is not None
                else run_case(case, geometry, environment, observers, args=args)
            )
            levels_by_case[(case.rpm, case.fidelity)] = levels
            print(
                f"  {record['status']}: {record['map_min_db']} to "
                f"{record['map_max_db']} dB in {record['elapsed_seconds']} s",
                flush=True,
            )
        except Exception as error:  # continue so one bad fidelity does not hide others
            error_text = f"{type(error).__name__}: {error}"
            failures.append(f"{case.title}: {error_text}")
            record = {
                "status": "failed",
                "rpm": case.rpm,
                "fidelity": case.fidelity,
                "loadings_file": str(case.path),
                "error": error_text,
            }
            print(f"  failed: {record['error']}", flush=True)
        records.append(record)
        write_summary(records, summary_path)
        release_cuda()

    window_errors, invalid_rpms = comparison_window_errors(records)
    failures.extend(window_errors)
    for error in window_errors:
        print(f"Comparison skipped: {error}", flush=True)

    render_rpm_comparisons(
        levels_by_case,
        geometry,
        environment,
        args=args,
        excluded_rpms=invalid_rpms,
    )
    print(f"Summary: {summary_path}")
    if failures:
        raise RuntimeError(f"{len(failures)} map case(s) failed; see {summary_path}.")


if __name__ == "__main__":
    main()
