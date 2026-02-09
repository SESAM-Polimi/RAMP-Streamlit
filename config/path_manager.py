# path_manager.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PathManager:
    """Manages canonical paths for the RAMP application."""
    root: Path

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def inputs_dir(self) -> Path:
        return self.root / "inputs"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"
    
    @property
    def core_dir(self) -> Path:
        return self.root / "core"

    @property
    def archetypes_dir(self) -> Path:
        return self.inputs_dir / "archetypes"

    @property
    def assets_dir(self) -> Path:
        return self.config_dir / "assets"

    @property
    def template_file(self) -> Path:
        return self.config_dir / "ramp_template.xlsx"

    @property
    def full_input_xlsx(self) -> Path:
        return self.inputs_dir / "ramp_input.xlsx"

    @property
    def year_structure_json(self) -> Path:
        return self.inputs_dir / "year_structure.json"

    @property
    def year_structure_yaml(self) -> Path:
        return self.inputs_dir / "year_structure.yaml"

    @property
    def archetype_configs_json(self) -> Path:
        return self.inputs_dir / "archetype_configs.json"

    def ensure_dirs(self) -> None:
        for p in (self.config_dir, self.inputs_dir, self.outputs_dir, self.archetypes_dir):
            p.mkdir(parents=True, exist_ok=True)

# Global, project-wide PM 
PM = PathManager(root=Path(__file__).resolve().parent.parent)
PM.ensure_dirs()