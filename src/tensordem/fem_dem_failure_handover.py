"""FEM→DEM failure_faces.json contract for crack geometry handover.

Physical background
-------------------
When a continuous FEM simulation detects crack initiation and propagation in ice,
the crack geometry must be transferred to the DEM discrete model so that the
particle bonds can be broken at the correct locations. This module defines the
contract (data format and validation) for this handover.

The failure_faces.json contract
-------------------------------
The FEM side outputs crack surfaces as a collection of triangular face patches:

{
    "metadata": {
        "format_version": "1.0",
        "source": "TensorFEM",
        "target": "TensorDEM",
        "timestamp": "ISO-8601",
        "mesh_size": <float>,       // FEM mesh characteristic size (m)
        "total_fracture_energy": <float>  // Total energy released by cracking (J)
    },
    "failure_faces": [
        {
            "face_id": <int>,
            "vertices": [[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]],  // Triangle vertices (m)
            "normal": [nx, ny, nz],     // Outward unit normal
            "crack_opening_displacement": <float>,  // COD (m)
            "area": <float>,            // Face area (m^2)
            "fracture_energy": <float>, // Energy for this face (J)
            "stress_state": {           // Stress at crack initiation
                "sigma_n": <float>,     // Normal stress (Pa)
                "sigma_t": <float>,     // Tangential stress (Pa)
            }
        },
        ...
    ],
    "crack_tips": [
        {
            "position": [x, y, z],
            "direction": [dx, dy, dz],
            "stress_intensity_factor": <float>  // K_I (Pa·m^0.5)
        },
        ...
    ]
}

DEM processing
--------------
1. Read failure_faces.json
2. For each face, find DEM bonds that intersect the face
3. Break those bonds (set bond status to "broken")
4. Verify energy conservation: sum of bond break energies ≈ total fracture energy
5. Verify geometric accuracy: crack tip positions match within mesh_size

Validation
----------
- Position accuracy: crack tip position error < mesh_size
- Normal accuracy: face normal angle error < 5°
- Energy conservation: |E_FEM - E_DEM| / E_FEM < 5%
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "FailureFace",
    "CrackTip",
    "FailureFacesContract",
    "ContractMetadata",
    "create_failure_faces_contract",
    "validate_contract",
    "export_contract_json",
    "import_contract_json",
    "find_intersected_bonds",
    "break_bonds_at_faces",
    "compute_bond_break_energy",
    "verify_energy_conservation",
    "verify_geometric_accuracy",
    "face_area_triangle",
    "triangle_normal",
    "point_in_triangle",
    "segment_triangle_intersection",
    "angle_between_normals",
    "FEM_DEM_HANDOVER_VERSION",
]

#: Contract format version
FEM_DEM_HANDOVER_VERSION = "1.0"


@dataclass
class FailureFace:
    """A single triangular crack face from FEM.
    
    Attributes
    ----------
    face_id : int
        Unique identifier for this face.
    vertices : array-like, shape (3, 3)
        Triangle vertex coordinates [[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]] (m).
    normal : array-like, shape (3,)
        Outward unit normal vector.
    crack_opening_displacement : float
        Crack opening displacement (COD) in meters.
    area : float
        Face area (m^2).
    fracture_energy : float
        Energy released by this crack face (J).
    stress_state : dict
        Stress state at crack initiation with keys 'sigma_n', 'sigma_t'.
    """
    face_id: int
    vertices: List[List[float]]
    normal: List[float]
    crack_opening_displacement: float
    area: float
    fracture_energy: float
    stress_state: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.normal = np.asarray(self.normal, dtype=np.float64)
        
        # Validate
        if self.vertices.shape != (3, 3):
            raise ValueError(f"vertices must be (3,3), got {self.vertices.shape}")
        if self.normal.shape != (3,):
            raise ValueError(f"normal must be (3,), got {self.normal.shape}")
        if self.crack_opening_displacement < 0:
            raise ValueError(f"COD must be non-negative, got {self.crack_opening_displacement}")
        if self.area <= 0:
            raise ValueError(f"area must be positive, got {self.area}")
        if self.fracture_energy < 0:
            raise ValueError(f"fracture_energy must be non-negative, got {self.fracture_energy}")
        
        # Normalize normal
        norm = np.linalg.norm(self.normal)
        if norm > 0:
            self.normal = self.normal / norm


@dataclass
class CrackTip:
    """A crack tip position from FEM.
    
    Attributes
    ----------
    position : array-like, shape (3,)
        Crack tip coordinates (m).
    direction : array-like, shape (3,)
        Crack propagation direction (unit vector).
    stress_intensity_factor : float
        Mode-I stress intensity factor K_I (Pa·m^0.5).
    """
    position: List[float]
    direction: List[float]
    stress_intensity_factor: float
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        self.direction = np.asarray(self.direction, dtype=np.float64)
        
        if self.position.shape != (3,):
            raise ValueError(f"position must be (3,), got {self.position.shape}")
        if self.direction.shape != (3,):
            raise ValueError(f"direction must be (3,), got {self.direction.shape}")
        
        # Normalize direction
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction = self.direction / norm


@dataclass
class ContractMetadata:
    """Metadata for the failure_faces contract.
    
    Attributes
    ----------
    format_version : str
        Contract format version.
    source : str
        Source system (e.g., "TensorFEM").
    target : str
        Target system (e.g., "TensorDEM").
    timestamp : str
        ISO-8601 timestamp.
    mesh_size : float
        FEM mesh characteristic size (m).
    total_fracture_energy : float
        Total energy released by all cracks (J).
    """
    format_version: str = FEM_DEM_HANDOVER_VERSION
    source: str = "TensorFEM"
    target: str = "TensorDEM"
    timestamp: str = ""
    mesh_size: float = 0.01
    total_fracture_energy: float = 0.0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
        if self.mesh_size <= 0:
            raise ValueError(f"mesh_size must be positive, got {self.mesh_size}")
        if self.total_fracture_energy < 0:
            raise ValueError(f"total_fracture_energy must be non-negative")


@dataclass
class FailureFacesContract:
    """Complete FEM→DEM failure faces contract.
    
    Attributes
    ----------
    metadata : ContractMetadata
        Contract metadata.
    failure_faces : list of FailureFace
        List of crack faces.
    crack_tips : list of CrackTip
        List of crack tips.
    """
    metadata: ContractMetadata
    failure_faces: List[FailureFace] = field(default_factory=list)
    crack_tips: List[CrackTip] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": {
                "format_version": self.metadata.format_version,
                "source": self.metadata.source,
                "target": self.metadata.target,
                "timestamp": self.metadata.timestamp,
                "mesh_size": self.metadata.mesh_size,
                "total_fracture_energy": self.metadata.total_fracture_energy,
            },
            "failure_faces": [
                {
                    "face_id": f.face_id,
                    "vertices": f.vertices.tolist(),
                    "normal": f.normal.tolist(),
                    "crack_opening_displacement": f.crack_opening_displacement,
                    "area": f.area,
                    "fracture_energy": f.fracture_energy,
                    "stress_state": f.stress_state,
                }
                for f in self.failure_faces
            ],
            "crack_tips": [
                {
                    "position": ct.position.tolist(),
                    "direction": ct.direction.tolist(),
                    "stress_intensity_factor": ct.stress_intensity_factor,
                }
                for ct in self.crack_tips
            ],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureFacesContract":
        """Create from dictionary."""
        meta_data = data["metadata"]
        metadata = ContractMetadata(
            format_version=meta_data.get("format_version", FEM_DEM_HANDOVER_VERSION),
            source=meta_data.get("source", "TensorFEM"),
            target=meta_data.get("target", "TensorDEM"),
            timestamp=meta_data.get("timestamp", ""),
            mesh_size=meta_data.get("mesh_size", 0.01),
            total_fracture_energy=meta_data.get("total_fracture_energy", 0.0),
        )
        
        faces = []
        for f_data in data.get("failure_faces", []):
            faces.append(FailureFace(
                face_id=f_data["face_id"],
                vertices=f_data["vertices"],
                normal=f_data["normal"],
                crack_opening_displacement=f_data["crack_opening_displacement"],
                area=f_data["area"],
                fracture_energy=f_data["fracture_energy"],
                stress_state=f_data.get("stress_state", {}),
            ))
        
        tips = []
        for ct_data in data.get("crack_tips", []):
            tips.append(CrackTip(
                position=ct_data["position"],
                direction=ct_data["direction"],
                stress_intensity_factor=ct_data["stress_intensity_factor"],
            ))
        
        return cls(metadata=metadata, failure_faces=faces, crack_tips=tips)


def create_failure_faces_contract(
    faces: Sequence[Dict[str, Any]],
    crack_tips: Optional[Sequence[Dict[str, Any]]] = None,
    mesh_size: float = 0.01,
) -> FailureFacesContract:
    """Create a failure_faces contract from face data.
    
    Parameters
    ----------
    faces : sequence of dict
        Face data with keys: vertices, normal, crack_opening_displacement,
        fracture_energy, stress_state.
    crack_tips : sequence of dict, optional
        Crack tip data with keys: position, direction, stress_intensity_factor.
    mesh_size : float
        FEM mesh characteristic size (m).
    
    Returns
    -------
    FailureFacesContract
        The constructed contract.
    """
    face_objects = []
    total_energy = 0.0
    
    for i, f_data in enumerate(faces):
        vertices = np.asarray(f_data["vertices"])
        normal = np.asarray(f_data.get("normal", [0, 0, 1]))
        
        # Compute area if not provided
        area = f_data.get("area")
        if area is None:
            area = face_area_triangle(vertices)
        
        # Compute normal if not provided
        if f_data.get("normal") is None:
            normal = triangle_normal(vertices)
        
        fracture_energy = f_data.get("fracture_energy", 0.0)
        total_energy += fracture_energy
        
        face_objects.append(FailureFace(
            face_id=i,
            vertices=vertices.tolist(),
            normal=normal.tolist(),
            crack_opening_displacement=f_data.get("crack_opening_displacement", 0.0),
            area=area,
            fracture_energy=fracture_energy,
            stress_state=f_data.get("stress_state", {}),
        ))
    
    tip_objects = []
    if crack_tips:
        for ct_data in crack_tips:
            tip_objects.append(CrackTip(
                position=ct_data["position"],
                direction=ct_data["direction"],
                stress_intensity_factor=ct_data.get("stress_intensity_factor", 0.0),
            ))
    
    metadata = ContractMetadata(
        mesh_size=mesh_size,
        total_fracture_energy=total_energy,
    )
    
    return FailureFacesContract(
        metadata=metadata,
        failure_faces=face_objects,
        crack_tips=tip_objects,
    )


def validate_contract(contract: FailureFacesContract) -> Dict[str, Any]:
    """Validate a failure_faces contract for internal consistency.
    
    Checks:
    - Face areas are positive
    - Normals are unit vectors
    - Fracture energies are non-negative
    - Total energy matches sum of face energies
    - Crack tips have valid positions
    
    Parameters
    ----------
    contract : FailureFacesContract
        Contract to validate.
    
    Returns
    -------
    dict
        Validation results with keys: 'valid', 'errors', 'warnings'.
    """
    errors = []
    warnings = []
    
    # Check faces
    computed_total_energy = 0.0
    for face in contract.failure_faces:
        if face.area <= 0:
            errors.append(f"Face {face.face_id}: area must be positive")
        
        normal_norm = np.linalg.norm(face.normal)
        if abs(normal_norm - 1.0) > 1e-6:
            errors.append(f"Face {face.face_id}: normal not unit vector (norm={normal_norm})")
        
        if face.fracture_energy < 0:
            errors.append(f"Face {face.face_id}: negative fracture energy")
        
        computed_total_energy += face.fracture_energy
    
    # Check total energy consistency
    energy_diff = abs(computed_total_energy - contract.metadata.total_fracture_energy)
    if contract.metadata.total_fracture_energy > 0:
        relative_diff = energy_diff / contract.metadata.total_fracture_energy
        if relative_diff > 0.01:
            warnings.append(
                f"Total energy mismatch: metadata={contract.metadata.total_fracture_energy}, "
                f"computed={computed_total_energy}, relative_diff={relative_diff:.4f}"
            )
    
    # Check crack tips
    for i, tip in enumerate(contract.crack_tips):
        dir_norm = np.linalg.norm(tip.direction)
        if abs(dir_norm - 1.0) > 1e-6:
            warnings.append(f"Crack tip {i}: direction not unit vector")
    
    valid = len(errors) == 0
    
    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "n_faces": len(contract.failure_faces),
        "n_crack_tips": len(contract.crack_tips),
        "total_fracture_energy": contract.metadata.total_fracture_energy,
    }


def export_contract_json(
    contract: FailureFacesContract,
    filepath: str,
) -> None:
    """Export contract to JSON file.
    
    Parameters
    ----------
    contract : FailureFacesContract
        Contract to export.
    filepath : str
        Output file path.
    """
    data = contract.to_dict()
    Path(filepath).write_text(json.dumps(data, indent=2))


def import_contract_json(filepath: str) -> FailureFacesContract:
    """Import contract from JSON file.
    
    Parameters
    ----------
    filepath : str
        Input file path.
    
    Returns
    -------
    FailureFacesContract
        Loaded contract.
    """
    data = json.loads(Path(filepath).read_text())
    return FailureFacesContract.from_dict(data)


# ============================================================================
# Geometric utilities
# ============================================================================

def face_area_triangle(vertices: np.ndarray) -> float:
    """Compute area of a triangle given its vertices.
    
    Parameters
    ----------
    vertices : ndarray, shape (3, 3)
        Triangle vertices [[x1,y1,z1], [x2,y2,z2], [x3,y3,z3]].
    
    Returns
    -------
    float
        Triangle area.
    """
    v0, v1, v2 = vertices[0], vertices[1], vertices[2]
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = np.cross(edge1, edge2)
    return 0.5 * np.linalg.norm(cross)


def triangle_normal(vertices: np.ndarray) -> np.ndarray:
    """Compute unit normal of a triangle.
    
    Parameters
    ----------
    vertices : ndarray, shape (3, 3)
        Triangle vertices.
    
    Returns
    -------
    ndarray, shape (3,)
        Unit normal vector.
    """
    v0, v1, v2 = vertices[0], vertices[1], vertices[2]
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross = np.cross(edge1, edge2)
    norm = np.linalg.norm(cross)
    if norm > 0:
        return cross / norm
    return np.array([0.0, 0.0, 1.0])


def point_in_triangle(
    point: np.ndarray,
    vertices: np.ndarray,
    tolerance: float = 1e-10,
) -> bool:
    """Check if a point lies inside a triangle (in 3D, projected onto triangle plane).
    
    Parameters
    ----------
    point : ndarray, shape (3,)
        Point to test.
    vertices : ndarray, shape (3, 3)
        Triangle vertices.
    tolerance : float
        Tolerance for boundary checks.
    
    Returns
    -------
    bool
        True if point is inside triangle.
    """
    v0, v1, v2 = vertices[0], vertices[1], vertices[2]
    
    # Compute barycentric coordinates
    edge0 = v1 - v0
    edge1 = v2 - v0
    vp = point - v0
    
    d00 = np.dot(edge0, edge0)
    d01 = np.dot(edge0, edge1)
    d11 = np.dot(edge1, edge1)
    d20 = np.dot(vp, edge0)
    d21 = np.dot(vp, edge1)
    
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-14:
        return False
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    
    return (u >= -tolerance) and (v >= -tolerance) and (w >= -tolerance)


def segment_triangle_intersection(
    p0: np.ndarray,
    p1: np.ndarray,
    vertices: np.ndarray,
    tolerance: float = 1e-10,
) -> Tuple[bool, Optional[np.ndarray]]:
    """Check if a line segment intersects a triangle.
    
    Uses Möller–Trumbore algorithm.
    
    Parameters
    ----------
    p0 : ndarray, shape (3,)
        Segment start point.
    p1 : ndarray, shape (3,)
        Segment end point.
    vertices : ndarray, shape (3, 3)
        Triangle vertices.
    tolerance : float
        Tolerance for intersection checks.
    
    Returns
    -------
    intersects : bool
        True if segment intersects triangle.
    intersection_point : ndarray or None
        Intersection point if intersects, else None.
    """
    v0, v1, v2 = vertices[0], vertices[1], vertices[2]
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    direction = p1 - p0
    h = np.cross(direction, edge2)
    a = np.dot(edge1, h)
    
    if abs(a) < tolerance:
        return False, None
    
    f = 1.0 / a
    s = p0 - v0
    u = f * np.dot(s, h)
    
    if u < -tolerance or u > 1.0 + tolerance:
        return False, None
    
    q = np.cross(s, edge1)
    v = f * np.dot(direction, q)
    
    if v < -tolerance or u + v > 1.0 + tolerance:
        return False, None
    
    t = f * np.dot(edge2, q)
    
    if t < -tolerance or t > 1.0 + tolerance:
        return False, None
    
    intersection_point = p0 + t * direction
    return True, intersection_point


def angle_between_normals(n1: np.ndarray, n2: np.ndarray) -> float:
    """Compute angle between two normal vectors in degrees.
    
    Parameters
    ----------
    n1, n2 : ndarray, shape (3,)
        Normal vectors (need not be unit).
    
    Returns
    -------
    float
        Angle in degrees [0, 180].
    """
    n1 = np.asarray(n1, dtype=np.float64)
    n2 = np.asarray(n2, dtype=np.float64)
    
    norm1 = np.linalg.norm(n1)
    norm2 = np.linalg.norm(n2)
    
    if norm1 < 1e-14 or norm2 < 1e-14:
        return 0.0
    
    cos_angle = np.dot(n1, n2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    return math.degrees(math.acos(cos_angle))


# ============================================================================
# DEM bond operations
# ============================================================================

def find_intersected_bonds(
    bond_positions: np.ndarray,
    bond_directions: np.ndarray,
    faces: Sequence[FailureFace],
    tolerance: float = 1e-10,
) -> List[int]:
    """Find DEM bonds that intersect any failure face.
    
    Parameters
    ----------
    bond_positions : ndarray, shape (N, 3)
        Bond midpoint positions.
    bond_directions : ndarray, shape (N, 3)
        Bond direction vectors (unit vectors).
    faces : sequence of FailureFace
        Failure faces to check against.
    tolerance : float
        Intersection tolerance.
    
    Returns
    -------
    list of int
        Indices of bonds that intersect at least one face.
    """
    intersected = set()
    
    for face in faces:
        vertices = np.asarray(face.vertices)
        
        for i, (pos, direction) in enumerate(zip(bond_positions, bond_directions)):
            # Check if bond segment intersects face triangle
            # Bond extends from pos - 0.5*dir to pos + 0.5*dir
            p0 = pos - 0.5 * direction
            p1 = pos + 0.5 * direction
            
            intersects, _ = segment_triangle_intersection(p0, p1, vertices, tolerance)
            if intersects:
                intersected.add(i)
    
    return sorted(intersected)


def break_bonds_at_faces(
    bond_status: np.ndarray,
    bond_positions: np.ndarray,
    bond_directions: np.ndarray,
    faces: Sequence[FailureFace],
) -> Tuple[np.ndarray, List[int]]:
    """Break DEM bonds at failure faces.
    
    Parameters
    ----------
    bond_status : ndarray, shape (N,)
        Bond status array (1=intact, 0=broken). Modified in place.
    bond_positions : ndarray, shape (N, 3)
        Bond midpoint positions.
    bond_directions : ndarray, shape (N, 3)
        Bond direction vectors.
    faces : sequence of FailureFace
        Failure faces.
    
    Returns
    -------
    bond_status : ndarray
        Updated bond status.
    broken_indices : list of int
        Indices of newly broken bonds.
    """
    intersected = find_intersected_bonds(bond_positions, bond_directions, faces)
    
    broken_indices = []
    for i in intersected:
        if bond_status[i] == 1:  # Was intact
            bond_status[i] = 0  # Break it
            broken_indices.append(i)
    
    return bond_status, broken_indices


def compute_bond_break_energy(
    bond_area: float,
    fracture_energy_density: float,
) -> float:
    """Compute energy to break a single bond.
    
    Parameters
    ----------
    bond_area : float
        Cross-sectional area of the bond (m^2).
    fracture_energy_density : float
        Fracture energy per unit area (J/m^2).
    
    Returns
    -------
    float
        Energy to break the bond (J).
    """
    return bond_area * fracture_energy_density


def verify_energy_conservation(
    fem_fracture_energy: float,
    dem_bond_energies: Sequence[float],
    friction_dissipation: float = 0.0,
    tolerance: float = 0.05,
) -> Dict[str, Any]:
    """Verify energy conservation between FEM and DEM.
    
    Checks: FEM fracture energy ≈ DEM bond break energy + friction dissipation
    
    Parameters
    ----------
    fem_fracture_energy : float
        Total fracture energy from FEM (J).
    dem_bond_energies : sequence of float
        Energies of broken bonds in DEM (J).
    friction_dissipation : float
        Energy dissipated by friction (J).
    tolerance : float
        Relative error tolerance (default 5%).
    
    Returns
    -------
    dict
        Verification results.
    """
    dem_total = sum(dem_bond_energies)
    dem_with_friction = dem_total + friction_dissipation
    
    if fem_fracture_energy > 0:
        relative_error = abs(fem_fracture_energy - dem_with_friction) / fem_fracture_energy
    else:
        relative_error = 0.0 if dem_with_friction == 0 else 1.0
    
    passed = relative_error < tolerance
    
    return {
        "fem_fracture_energy": fem_fracture_energy,
        "dem_bond_energy_total": dem_total,
        "friction_dissipation": friction_dissipation,
        "dem_total_with_friction": dem_with_friction,
        "relative_error": relative_error,
        "tolerance": tolerance,
        "passed": passed,
    }


def verify_geometric_accuracy(
    fem_crack_tips: Sequence[CrackTip],
    dem_crack_positions: Sequence[np.ndarray],
    mesh_size: float,
    fem_normals: Optional[Sequence[np.ndarray]] = None,
    dem_normals: Optional[Sequence[np.ndarray]] = None,
    angle_tolerance: float = 5.0,
) -> Dict[str, Any]:
    """Verify geometric accuracy of FEM→DEM crack transfer.
    
    Checks:
    - Crack tip position error < mesh_size
    - Face normal angle error < angle_tolerance (5° default)
    
    Parameters
    ----------
    fem_crack_tips : sequence of CrackTip
        Crack tip positions from FEM.
    dem_crack_positions : sequence of ndarray
        Crack positions realized in DEM.
    mesh_size : float
        FEM mesh characteristic size (m).
    fem_normals : sequence of ndarray, optional
        Face normals from FEM.
    dem_normals : sequence of ndarray, optional
        Face normals realized in DEM.
    angle_tolerance : float
        Maximum allowed angle error in degrees.
    
    Returns
    -------
    dict
        Verification results.
    """
    results = {
        "position_errors": [],
        "max_position_error": 0.0,
        "position_passed": True,
        "angle_errors": [],
        "max_angle_error": 0.0,
        "angle_passed": True,
        "mesh_size": mesh_size,
    }
    
    # Position check
    for fem_tip, dem_pos in zip(fem_crack_tips, dem_crack_positions):
        fem_pos = np.asarray(fem_tip.position)
        dem_pos = np.asarray(dem_pos)
        error = np.linalg.norm(fem_pos - dem_pos)
        results["position_errors"].append(error)
        results["max_position_error"] = max(results["max_position_error"], error)
    
    results["position_passed"] = results["max_position_error"] < mesh_size
    
    # Angle check
    if fem_normals and dem_normals:
        for fem_n, dem_n in zip(fem_normals, dem_normals):
            angle = angle_between_normals(fem_n, dem_n)
            results["angle_errors"].append(angle)
            results["max_angle_error"] = max(results["max_angle_error"], angle)
        
        results["angle_passed"] = results["max_angle_error"] < angle_tolerance
    
    results["passed"] = results["position_passed"] and results["angle_passed"]
    
    return results
