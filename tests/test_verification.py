"""Unit tests for the reproducible DEM verification campaign utilities."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from tensordem.verification import (
    CampaignSpec,
    CampaignValidationError,
    compare_histories,
    history_digest,
    interpolate_series,
    make_resolution_specs,
    parameter_grid,
    read_report,
    report_metadata,
    run_case,
    run_resolution_campaign,
    summarize_history,
    trapezoidal_integral,
    validate_history,
    validate_report,
    write_report,
    write_summary_csv,
)


def history(times, forces, **channels):
    rows = []
    for index, (time, force) in enumerate(zip(times, forces)):
        row = {"time": time, "reaction_x": 0.0, "reaction_y": force}
        for key, values in channels.items():
            row[key] = values[index]
        rows.append(row)
    return rows


class CampaignSpecTests(unittest.TestCase):
    def test_defaults_are_valid_and_map_to_dem(self):
        spec = CampaignSpec()
        config = spec.to_dem_config()
        geometry = spec.realized_geometry()
        self.assertEqual(config.nx, spec.nx)
        self.assertGreaterEqual(config.ny, 2)
        self.assertAlmostEqual(geometry["actual_width_m"], spec.target_width_m)
        self.assertGreater(geometry["disk_mass_kg"], 0)
        self.assertEqual(geometry["particle_count"], config.nx * config.ny)

    def test_target_width_is_fixed_across_resolution(self):
        coarse = CampaignSpec(nx=5)
        fine = CampaignSpec(nx=13)
        self.assertAlmostEqual(
            coarse.realized_geometry()["actual_width_m"],
            fine.realized_geometry()["actual_width_m"],
        )
        self.assertNotEqual(
            coarse.realized_geometry()["particle_radius_m"],
            fine.realized_geometry()["particle_radius_m"],
        )

    def test_invalid_integer_fields_rejected(self):
        for kwargs in ({"nx": True}, {"nx": 2}, {"nx": 3.5},
                       {"save_every": False}, {"save_every": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(CampaignValidationError):
                CampaignSpec(**kwargs)

    def test_invalid_positive_parameters_rejected(self):
        for name in ("target_width_m", "target_height_m", "thickness_m",
                     "duration_s", "speed_m_s", "density_kg_m3",
                     "bond_stiffness_N_m", "contact_stiffness_N_m",
                     "breaking_strain", "shear_breaking_strain", "tool_radius_m"):
            with self.subTest(name=name), self.assertRaises(CampaignValidationError):
                CampaignSpec(**{name: 0.0})
            with self.subTest(name=name), self.assertRaises(CampaignValidationError):
                CampaignSpec(**{name: math.nan})

    def test_nonnegative_parameters_and_device_are_checked(self):
        for name in ("contact_damping_N_s_m", "drag_N_s_m", "tool_gap_m"):
            with self.subTest(name=name), self.assertRaises(CampaignValidationError):
                CampaignSpec(**{name: -1.0})
        for kwargs in ({"device": "tpu"}, {"fix_edges": 1}, {"dt_s": 0.0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(CampaignValidationError):
                CampaignSpec(**kwargs)

    def test_resolution_builder_rejects_empty_and_duplicate_levels(self):
        with self.assertRaises(CampaignValidationError):
            make_resolution_specs([])
        with self.assertRaises(CampaignValidationError):
            make_resolution_specs([5, 5])


class HistoryValidationTests(unittest.TestCase):
    def test_history_is_normalized_to_float_values(self):
        result = validate_history(history([0, 1], [0, 2]))
        self.assertEqual(result[1]["time"], 1.0)
        self.assertEqual(result[1]["reaction_y"], 2.0)

    def test_requires_at_least_two_samples(self):
        with self.assertRaises(CampaignValidationError):
            validate_history([{"time": 0, "reaction_x": 0, "reaction_y": 0}])

    def test_rejects_non_mapping_rows_and_missing_channels(self):
        for bad in ([1, 2], [{"time": 0}, {"time": 1}]):
            with self.subTest(bad=bad), self.assertRaises(CampaignValidationError):
                validate_history(bad)

    def test_rejects_nonmonotonic_time(self):
        for times in ([0, 0], [0, -1], [0, 2, 1]):
            forces = [0] * len(times)
            with self.subTest(times=times), self.assertRaises(CampaignValidationError):
                validate_history(history(times, forces))

    def test_rejects_nonfinite_channels(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(CampaignValidationError):
                validate_history(history([0, 1], [0, value]))

    def test_rejects_boolean_as_numeric(self):
        with self.assertRaises(CampaignValidationError):
            validate_history(history([0, 1], [0, True]))

    def test_history_digest_is_order_independent_but_value_sensitive(self):
        first = history([0, 1], [0, 2])
        second = [{"reaction_y": 0, "time": 0, "reaction_x": 0},
                  {"reaction_y": 2, "time": 1, "reaction_x": 0}]
        self.assertEqual(history_digest(first), history_digest(second))
        self.assertNotEqual(history_digest(first), history_digest(history([0, 1], [0, 3])))


class NumericMetricTests(unittest.TestCase):
    def test_linear_interpolation(self):
        values = interpolate_series([0, 1, 2], [0, 2, 4], [0, 0.25, 1.5, 2])
        self.assertEqual(values, [0, 0.5, 3.0, 4])

    def test_interpolation_rejects_extrapolation_and_invalid_times(self):
        with self.assertRaises(CampaignValidationError):
            interpolate_series([0, 1], [0, 1], [-0.1])
        with self.assertRaises(CampaignValidationError):
            interpolate_series([0, 0], [0, 1], [0])
        with self.assertRaises(CampaignValidationError):
            interpolate_series([0], [0], [0])

    def test_trapezoidal_integral_for_constant_and_linear_signals(self):
        self.assertAlmostEqual(trapezoidal_integral([0, 2], [3, 3]), 6)
        self.assertAlmostEqual(trapezoidal_integral([0, 1, 2], [0, 1, 2]), 2)
        self.assertAlmostEqual(trapezoidal_integral([0, 1], [1, -1]), 0)

    def test_integral_rejects_invalid_input(self):
        for times, values in (([0], [1]), ([0, 1], [1]), ([1, 0], [0, 0])):
            with self.subTest(times=times), self.assertRaises(CampaignValidationError):
                trapezoidal_integral(times, values)

    def test_summary_reports_peaks_impulse_and_energy(self):
        rows = history([0, 1, 2], [-2, 4, 0],
                       broken_bonds=[0, 1, 2],
                       kinetic_energy=[0.1, 0.2, 0.3],
                       mechanical_energy=[1, 2, 3],
                       bond_energy=[0.8, 1.5, 2.0],
                       pair_contact_energy=[0.1, 0.2, 0.5],
                       tool_contact_energy=[0.1, 0.3, 0.5])
        result = summarize_history(rows)
        self.assertEqual(result["peak_positive_force_N"], 4)
        self.assertEqual(result["peak_negative_force_N"], -2)
        self.assertEqual(result["peak_absolute_force_N"], 4)
        self.assertAlmostEqual(result["signed_impulse_Ns"], 3)
        self.assertEqual(result["final_broken_bonds"], 2)
        self.assertEqual(result["final_mechanical_energy_J"], 3)
        self.assertEqual(result["maximum_broken_bonds"], 2)

    def test_identical_force_histories_have_zero_error_and_unit_correlation(self):
        rows = history([0, 1, 2], [0, 3, 0])
        result = compare_histories(rows, rows)
        self.assertEqual(result["force_difference_rms_N"], 0)
        self.assertEqual(result["force_relative_rms_difference"], 0)
        self.assertAlmostEqual(result["force_correlation"], 1)
        self.assertEqual(result["force_peak_absolute_relative_difference"], 0)

    def test_different_sampling_of_same_linear_signal_matches(self):
        coarse = history([0, 1, 2], [0, 2, 0])
        fine = history([0, 0.5, 1, 1.5, 2], [0, 1, 2, 1, 0])
        result = compare_histories(coarse, fine)
        self.assertAlmostEqual(result["force_difference_rms_N"], 0)
        self.assertEqual(result["sample_count"], 5)

    def test_comparison_rejects_nonoverlap_and_invalid_channel(self):
        with self.assertRaises(CampaignValidationError):
            compare_histories(history([0, 1], [0, 1]), history([2, 3], [0, 1]))
        with self.assertRaises(CampaignValidationError):
            compare_histories(history([0, 1], [0, 1]), history([0, 1], [0, 1]), force_channel="missing")

    def test_zero_reference_signal_does_not_divide_by_zero(self):
        result = compare_histories(history([0, 1], [0, 0]), history([0, 1], [0, 0]))
        self.assertEqual(result["force_relative_rms_difference"], 0)
        self.assertEqual(result["force_peak_absolute_relative_difference"], 0)


class ParameterGridTests(unittest.TestCase):
    def test_cartesian_grid_is_deterministic(self):
        base = CampaignSpec(nx=3, duration_s=1e-7)
        cases = parameter_grid(base, {"nx": [3, 4], "speed_m_s": [0.1, 0.2]})
        self.assertEqual(len(cases), 4)
        self.assertEqual(
            [(case.nx, case.speed_m_s) for case in cases],
            [(3, 0.1), (3, 0.2), (4, 0.1), (4, 0.2)],
        )

    def test_parameter_grid_rejects_unknown_or_empty_fields(self):
        base = CampaignSpec()
        for parameters in ({}, {"unknown": [1]}, {"nx": []}, {"nx": "35"}):
            with self.subTest(parameters=parameters), self.assertRaises(CampaignValidationError):
                parameter_grid(base, parameters)


class EndToEndCampaignTests(unittest.TestCase):
    def tiny_spec(self, nx=3, **kwargs):
        defaults = dict(nx=nx, target_width_m=0.2, target_height_m=0.1,
                        duration_s=1e-7, speed_m_s=0.2, save_every=1)
        defaults.update(kwargs)
        return CampaignSpec(**defaults)

    def test_case_runs_and_has_reproducible_numeric_history(self):
        spec = self.tiny_spec()
        first = run_case(spec)
        second = run_case(spec)
        self.assertEqual(first["history_sha256"], second["history_sha256"])
        self.assertEqual(first["solver"]["step_count"], 1)
        self.assertGreaterEqual(len(first["history"]), 2)
        self.assertEqual(first["summary"]["sample_count"], len(first["history"]))
        self.assertFalse(first["history"][-1].get("nonfinite", False))
        self.assertEqual(first["solver"]["model_scope"].startswith("2-D"), True)

    def test_case_step_limit_fails_before_long_run(self):
        with self.assertRaises(CampaignValidationError):
            run_case(self.tiny_spec(duration_s=1.0), max_steps=1)

    def test_campaign_has_one_comparison_per_case(self):
        report = run_resolution_campaign([self.tiny_spec(3), self.tiny_spec(4)])
        self.assertEqual(report["schema"], "tensordem.verification-campaign/1")
        self.assertEqual(report["case_count"], 2)
        self.assertEqual(len(report["comparisons"]), 2)
        self.assertEqual(report["reference_index"], 1)
        self.assertTrue(report["comparisons"][1]["is_reference"])
        self.assertFalse(report["methodology"]["physical_validation"])

    def test_explicit_reference_index_is_respected(self):
        report = run_resolution_campaign(
            [self.tiny_spec(3), self.tiny_spec(4)], reference_index=0
        )
        self.assertEqual(report["reference_index"], 0)
        self.assertTrue(report["comparisons"][0]["is_reference"])

    def test_invalid_campaign_arguments_are_rejected(self):
        with self.assertRaises(CampaignValidationError):
            run_resolution_campaign([])
        with self.assertRaises(CampaignValidationError):
            run_resolution_campaign([self.tiny_spec()], reference_index=3)
        with self.assertRaises(CampaignValidationError):
            run_resolution_campaign([object()])

    def test_json_and_csv_round_trip(self):
        report = run_resolution_campaign([self.tiny_spec(3), self.tiny_spec(4)])
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "nested" / "report.json"
            csv_path = Path(directory) / "summary.csv"
            self.assertEqual(write_report(report, json_path), json_path)
            loaded = read_report(json_path)
            self.assertEqual(loaded["case_count"], 2)
            self.assertEqual(report_metadata(loaded)["case_count"], 2)
            self.assertTrue(write_summary_csv(loaded, csv_path).exists())
            self.assertGreater(csv_path.stat().st_size, 100)
            parsed = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(parsed["schema"], report["schema"])

    def test_report_validation_detects_tampered_history(self):
        report = run_resolution_campaign([self.tiny_spec()])
        report["cases"][0]["history"][0]["reaction_y"] += 1
        with self.assertRaises(CampaignValidationError):
            validate_report(report)

    def test_report_validation_detects_bad_references(self):
        report = run_resolution_campaign([self.tiny_spec()])
        report["comparisons"][0]["reference_index"] = 7
        with self.assertRaises(CampaignValidationError):
            validate_report(report)

    def test_unknown_report_schema_is_rejected(self):
        with self.assertRaises(CampaignValidationError):
            validate_report({"schema": "future/99"})

    def test_write_report_rejects_nonfinite_values(self):
        report = run_resolution_campaign([self.tiny_spec()])
        report["methodology"]["bad_value"] = math.nan
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(CampaignValidationError):
                write_report(report, Path(directory) / "bad.json")


if __name__ == "__main__":
    unittest.main()
