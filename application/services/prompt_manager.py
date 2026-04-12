import yaml
from jinja2 import Environment, FileSystemLoader
from core.config import BASE_DIR

class PromptManager:
    """
    Manages loading and rendering of investigation prompts.
    Uses Jinja2 for templates and YAML for scenario data.
    Standardized for Clean Architecture.
    """
    SCENARIO_UI_LABELS = {
        "general_intelligence": "Trinh sát tổng hợp",
        "drug_trafficking": "Ma túy và đường dây",
        "high_tech_fraud": "Lừa đảo công nghệ cao",
    }

    def __init__(self):
        # Resolve absolute path to prompts directory
        self.base_dir = BASE_DIR / "prompts"
        
        # Include both base_dir and templates/ to allow {% include 'modules/...' %}
        self.env = Environment(loader=FileSystemLoader([str(self.base_dir / "templates"), str(self.base_dir)]))
        
    def load_scenario(self, scenario_name: str) -> dict:
        """Load static data from a scenario YAML file."""
        yaml_path = self.base_dir / "scenarios" / f"{scenario_name}.yaml"
        if not yaml_path.exists():
            return {}
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
        return self._normalize_scenario_data(scenario_name, raw_data)

    def _normalize_scenario_data(self, scenario_name: str, raw_data: dict) -> dict:
        """
        Normalize scenario files so both legacy and current templates can consume them.
        """
        if not isinstance(raw_data, dict):
            return {
                "scenario_name": scenario_name,
                "scenario_context": str(raw_data),
                "dictionaries": {},
            }

        normalized = dict(raw_data)

        if "dictionaries" in normalized:
            dictionaries = normalized.get("dictionaries") or {}
        else:
            scalar_values = normalized and all(isinstance(v, str) for v in normalized.values())
            dictionaries = dict(normalized) if scalar_values else {}
            if scalar_values:
                normalized = {"dictionaries": dictionaries}

        normalized["dictionaries"] = dictionaries
        normalized.setdefault("scenario_name", scenario_name.replace("_", " ").title())
        normalized.setdefault("scenario_context", self._build_scenario_context(normalized))
        return normalized

    def list_scenarios(self) -> list[dict]:
        scenarios_dir = self.base_dir / "scenarios"
        if not scenarios_dir.exists():
            return []

        items = []
        for yaml_path in sorted(scenarios_dir.glob("*.yaml")):
            scenario_id = yaml_path.stem
            scenario_data = self.load_scenario(scenario_id)
            items.append(
                {
                    "value": scenario_id,
                    "label": self.SCENARIO_UI_LABELS.get(
                        scenario_id,
                        scenario_data.get("name") or scenario_data.get("scenario_name") or scenario_id,
                    ),
                    "description": scenario_data.get("description") or "Kịch bản nghiệp vụ cục bộ.",
                }
            )
        return items

    def _build_scenario_context(self, scenario_data: dict) -> str:
        parts = []

        name = scenario_data.get("name")
        if name:
            parts.append(f"Scenario: {name}")

        description = scenario_data.get("description")
        if description:
            parts.append(f"Description: {description}")

        keywords = scenario_data.get("keywords") or []
        if keywords:
            parts.append("Keywords: " + ", ".join(str(keyword) for keyword in keywords))

        focus_areas = scenario_data.get("focus_areas") or []
        if focus_areas:
            parts.append("Focus Areas:")
            parts.extend(f"- {item}" for item in focus_areas)

        dictionaries = scenario_data.get("dictionaries") or {}
        if dictionaries:
            parts.append("Known slang/code words:")
            parts.extend(f'- "{word}": {meaning}' for word, meaning in dictionaries.items())

        return "\n".join(parts)
            
    def render_prompt(self, template_name: str, transcript: str, scenario: str = None) -> str:
        """
        Render a final prompt for the LLM.
        """
        context_data = {"transcript": transcript}
        
        # Inject scenario data if provided
        if scenario:
            scenario_data = self.load_scenario(scenario)
            context_data.update(scenario_data)

        return self.render_template(template_name, **context_data)

    def render_template(self, template_name: str, **context: object) -> str:
        """Render an arbitrary Jinja template from the prompts directory."""
        template = self.env.get_template(template_name)
        return template.render(**context)
