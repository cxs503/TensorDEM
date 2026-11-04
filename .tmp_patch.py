"""One-shot patcher: friction/rolling channels in the DEM3D energy ledger."""
import io, sys

def patch(path, subs):
    src = io.open(path, encoding="utf-8").read()
    for old, new in subs:
        assert src.count(old) == 1, f"pattern not unique/found in {path}: {old[:60]!r}"
        src = src.replace(old, new)
    io.open(path, "w", encoding="utf-8").write(src)
    print("patched", path)

DEM3D = "src/tensordem/dem3d.py"
ENERGY = "src/tensordem/dem3d_energy.py"

patch(DEM3D, [
    # 1) publish per-step contact power scalars (step owner only)
    (
        "        # Rolling resistance torque accumulator per particle (N, 3).\n"
        "        self._rolling_torque = torch.zeros((n, 3), dtype=self.dtype, device=self.device)\n",
        "        # Rolling resistance torque accumulator per particle (N, 3).\n"
        "        self._rolling_torque = torch.zeros((n, 3), dtype=self.dtype, device=self.device)\n"
        "        # Power delivered to the particles by the tangential friction force and\n"
        "        # by the rolling-resistance couple, published by the step-owner force call\n"
        "        # (update_history=True) for the energy-audit ledger.  Probes leave them\n"
        "        # untouched so monitoring never clobbers a running accumulation.\n"
        "        self._last_friction_power = 0.0\n"
        "        self._last_rolling_power = 0.0\n",
    ),
    (
        "        force.index_add_(0, j, -active_bond_force)\n"
        "\n"
        "        contact_pairs = self._contact_candidate_pairs()\n"
        "        if contact_pairs.numel():\n"
        "            ci, cj = contact_pairs.T\n"
        "            cdelta = self.positions[cj] - self.positions[ci]\n"
        "            cdistance = torch.linalg.vector_norm(cdelta, dim=1)\n"
        "            overlap = (2 * cfg.radius - cdistance).clamp_min(0)\n"
        "            near = overlap > 0\n",
        "        force.index_add_(0, j, -active_bond_force)\n"
        "\n"
        "        contact_pairs = self._contact_candidate_pairs()\n"
        "        friction_power = torch.zeros((), dtype=self.dtype, device=self.device)\n"
        "        rolling_power = torch.zeros((), dtype=self.dtype, device=self.device)\n"
        "        if contact_pairs.numel():\n"
        "            ci, cj = contact_pairs.T\n"
        "            cdelta = self.positions[cj] - self.positions[ci]\n"
        "            cdistance = torch.linalg.vector_norm(cdelta, dim=1)\n"
        "            overlap = (2 * cfg.radius - cdistance).clamp_min(0)\n"
        "            near = overlap > 0\n",
    ),
    (
        "                    F_t = F_t_raw * scale[:, None]\n"
        "\n"
        "                    # Apply tangential force\n"
        "                    force.index_add_(0, ci, F_t)\n"
        "                    force.index_add_(0, cj, -F_t)\n",
        "                    F_t = F_t_raw * scale[:, None]\n"
        "\n"
        "                    # Power delivered to the particles by the friction force\n"
        "                    # (force on i is F_t, on j is -F_t): P = F_t . (v_i - v_j).\n"
        "                    # Negative during stick (energy stored in the tangential\n"
        "                    # spring) and during slip (Coulomb dissipation).\n"
        "                    friction_power = friction_power + (\n"
        "                        F_t * (self.velocities[ci] - self.velocities[cj])\n"
        "                    ).sum()\n"
        "\n"
        "                    # Apply tangential force\n"
        "                    force.index_add_(0, ci, F_t)\n"
        "                    force.index_add_(0, cj, -F_t)\n",
    ),
    (
        "                    rr_i, rr_j, torque_i, torque_j = self._rolling_resistance_couple(\n"
        "                        ci, cj, cnormal, contact_mag,\n"
        "                    )\n",
        "                    rr_i, rr_j, torque_i, torque_j = self._rolling_resistance_couple(\n"
        "                        ci, cj, cnormal, contact_mag,\n"
        "                    )\n"
        "                    # The couple is strictly dissipative: P = rr_i . (v_i - v_j)\n"
        "                    # = -mu_r |F_n| |v_t| <= 0.\n"
        "                    rolling_power = rolling_power + (\n"
        "                        rr_i * (self.velocities[ci] - self.velocities[cj])\n"
        "                    ).sum()\n",
    ),
    (
        "                force.index_add_(0, ci, contact_force)\n"
        "                force.index_add_(0, cj, -contact_force)\n"
        "\n"
        "        tool_delta = self.positions - self.tool_position\n",
        "                force.index_add_(0, ci, contact_force)\n"
        "                force.index_add_(0, cj, -contact_force)\n"
        "\n"
        "        if update_history:\n"
        "            self._last_friction_power = float(friction_power)\n"
        "            self._last_rolling_power = float(rolling_power)\n"
        "\n"
        "        tool_delta = self.positions - self.tool_position\n",
    ),
])

patch(ENERGY, [
    (
        "        self.cumulative_fracture_release_J = 0.0\n"
        "        self.reference_mechanical_energy_J = self._mechanical_energy()\n",
        "        self.cumulative_fracture_release_J = 0.0\n"
        "        # Tangential friction work (total work of the Coulomb spring force;\n"
        "        # negative during stick as energy is stored in the tangential spring)\n"
        "        # and rolling-resistance work (always <= 0, pure dissipation).\n"
        "        self.cumulative_friction_work_J = 0.0\n"
        "        self.cumulative_rolling_work_J = 0.0\n"
        "        self.reference_mechanical_energy_J = self._mechanical_energy()\n",
    ),
    (
        "        tool_contact = (0.5 * cfg.contact_stiffness * tool_overlap.square()).sum()\n"
        "        return float(kinetic + bond + particle_contact + tool_contact)\n",
        "        tool_contact = (0.5 * cfg.contact_stiffness * tool_overlap.square()).sum()\n"
        "        # Tangential (Cundall-Strack) spring energy of active, unbonded\n"
        "        # contacts.  The friction force is a spring force: during stick its\n"
        "        # work is recoverable, so the stored energy belongs in the mechanical\n"
        "        # balance.  Linear k_t = k_n convention, same as the normal term above.\n"
        "        tangential = torch.zeros((), dtype=self.dtype, device=self.device)\n"
        "        if pairs.numel() and cfg.friction_coefficient > 0.0:\n"
        "            ci, cj = pairs.T\n"
        "            d = torch.linalg.vector_norm(self.positions[cj] - self.positions[ci], dim=1)\n"
        "            overlap = (2 * cfg.radius - d).clamp_min(0)\n"
        "            active = (overlap > 0) & (~self._alive_bond_for_pairs(pairs))\n"
        "            if bool(active.any()):\n"
        "                cnormal = (self.positions[cj] - self.positions[ci]) / d.clamp_min(\n"
        "                    torch.finfo(self.dtype).eps\n"
        "                )[:, None]\n"
        "                tang_disp = self._get_tangential_state(ci[active], cj[active], cnormal[active])\n"
        "                tangential = (0.5 * cfg.contact_stiffness * tang_disp.square().sum(dim=1)).sum()\n"
        "        return float(kinetic + bond + particle_contact + tool_contact + tangential)\n",
    ),
    (
        "        self.cumulative_fracture_release_J = float(self.failure_energy_J.sum())\n"
        "\n"
        "        self.velocities += self.dt * force / self.config.mass\n",
        "        self.cumulative_fracture_release_J = float(self.failure_energy_J.sum())\n"
        "        self.cumulative_friction_work_J += self._last_friction_power * self.dt\n"
        "        self.cumulative_rolling_work_J += self._last_rolling_power * self.dt\n"
        "\n"
        "        self.velocities += self.dt * force / self.config.mass\n",
    ),
    (
        "        dissipated = (self.cumulative_drag_dissipation_J\n"
        "                      + self.cumulative_particle_damping_J\n"
        "                      + self.cumulative_tool_damping_J)\n"
        "        input_work = self.cumulative_tool_work_J + self.cumulative_external_work_J\n"
        "        self.energy_balance_residual_J = (\n"
        "            energy_after - self.reference_mechanical_energy_J - input_work\n"
        "            + dissipated + self.cumulative_fracture_release_J\n"
        "        )\n",
        "        dissipated = (self.cumulative_drag_dissipation_J\n"
        "                      + self.cumulative_particle_damping_J\n"
        "                      + self.cumulative_tool_damping_J\n"
        "                      + self.cumulative_friction_work_J\n"
        "                      + self.cumulative_rolling_work_J)\n"
        "        input_work = self.cumulative_tool_work_J + self.cumulative_external_work_J\n"
        "        self.energy_balance_residual_J = (\n"
        "            energy_after - self.reference_mechanical_energy_J - input_work\n"
        "            + dissipated + self.cumulative_fracture_release_J\n"
        "        )\n",
    ),
    (
        "        dissipated = (self.cumulative_drag_dissipation_J\n"
        "                      + self.cumulative_particle_damping_J\n"
        "                      + self.cumulative_tool_damping_J)\n"
        "        input_work = self.cumulative_tool_work_J + self.cumulative_external_work_J\n"
        "        self.energy_balance_residual_J = (\n"
        "            energy - self.reference_mechanical_energy_J - input_work\n"
        "            + dissipated + self.cumulative_fracture_release_J\n"
        "        )\n",
        "        dissipated = (self.cumulative_drag_dissipation_J\n"
        "                      + self.cumulative_particle_damping_J\n"
        "                      + self.cumulative_tool_damping_J\n"
        "                      + self.cumulative_friction_work_J\n"
        "                      + self.cumulative_rolling_work_J)\n"
        "        input_work = self.cumulative_tool_work_J + self.cumulative_external_work_J\n"
        "        self.energy_balance_residual_J = (\n"
        "            energy - self.reference_mechanical_energy_J - input_work\n"
        "            + dissipated + self.cumulative_fracture_release_J\n"
        "        )\n",
    ),
    (
        "            \"fracture_release_cumulative_J\": self.cumulative_fracture_release_J,\n"
        "            \"energy_balance_residual_J\": self.energy_balance_residual_J,\n",
        "            \"fracture_release_cumulative_J\": self.cumulative_fracture_release_J,\n"
        "            \"friction_work_cumulative_J\": self.cumulative_friction_work_J,\n"
        "            \"rolling_work_cumulative_J\": self.cumulative_rolling_work_J,\n"
        "            \"energy_balance_residual_J\": self.energy_balance_residual_J,\n",
    ),
    (
        "            \"cumulative_tool_damping_J\": self.cumulative_tool_damping_J,\n"
        "            \"cumulative_fracture_release_J\": self.cumulative_fracture_release_J,\n"
        "            \"energy_balance_residual_J\": self.energy_balance_residual_J,\n"
        "        }\n"
        "        return state\n",
        "            \"cumulative_tool_damping_J\": self.cumulative_tool_damping_J,\n"
        "            \"cumulative_fracture_release_J\": self.cumulative_fracture_release_J,\n"
        "            \"cumulative_friction_work_J\": self.cumulative_friction_work_J,\n"
        "            \"cumulative_rolling_work_J\": self.cumulative_rolling_work_J,\n"
        "            \"energy_balance_residual_J\": self.energy_balance_residual_J,\n"
        "        }\n"
        "        return state\n",
    ),
    (
        "            self.cumulative_tool_damping_J = 0.0\n"
        "            self.cumulative_fracture_release_J = float(self.failure_energy_J.sum())\n"
        "            self.energy_balance_residual_J = 0.0\n"
        "            return\n",
        "            self.cumulative_tool_damping_J = 0.0\n"
        "            self.cumulative_fracture_release_J = float(self.failure_energy_J.sum())\n"
        "            self.cumulative_friction_work_J = 0.0\n"
        "            self.cumulative_rolling_work_J = 0.0\n"
        "            self.energy_balance_residual_J = 0.0\n"
        "            return\n",
    ),
    (
        "            \"cumulative_tool_damping_J\", \"cumulative_fracture_release_J\",\n"
        "            \"energy_balance_residual_J\",\n"
        "        )\n"
        "        staged = {key: float(audit[key]) for key in keys}\n",
        "            \"cumulative_tool_damping_J\", \"cumulative_fracture_release_J\",\n"
        "            \"cumulative_friction_work_J\", \"cumulative_rolling_work_J\",\n"
        "            \"energy_balance_residual_J\",\n"
        "        )\n"
        "        staged = {key: float(audit.get(key, 0.0)) for key in keys}\n",
    ),
])
print("OK")
