"""Prompt Template Manager for the AI Intelligence Layer.

Provides centralized loading, validation, and rendering of prompt templates
stored as YAML files in the configured PROMPT_TEMPLATES_DIR.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from app.ai.exceptions import MissingVariableError

# Pattern to detect unresolved {variable_name} placeholders in rendered output
_UNRESOLVED_PLACEHOLDER_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


@dataclass
class PromptTemplate:
    """A loaded prompt template with metadata and body content."""

    name: str
    description: str
    required_variables: List[str]
    template: str


@dataclass
class TemplateInfo:
    """Summary information about a template for listing purposes."""

    name: str
    description: str
    required_variables: List[str]


class PromptTemplateManager:
    """Manages prompt templates stored as YAML files.

    Templates are loaded from YAML files in the configured directory. Each file
    must have the following structure:

        name: template_name
        description: What this template does
        required_variables:
          - var1
          - var2
        template: |
          Prompt text with {var1} and {var2} placeholders.
    """

    def __init__(self, templates_dir: str = "app/prompts") -> None:
        """Initialize the manager with the path to the templates directory.

        Args:
            templates_dir: Path to the directory containing YAML template files.
        """
        self._templates_dir = Path(templates_dir)
        self._cache: Dict[str, PromptTemplate] = {}

    def _load_template(self, name: str) -> PromptTemplate:
        """Load a template from its YAML file.

        Args:
            name: The template name (corresponds to the YAML filename without extension).

        Returns:
            The loaded PromptTemplate.

        Raises:
            FileNotFoundError: If the template file does not exist.
            ValueError: If the YAML file is malformed or missing required fields.
        """
        if name in self._cache:
            return self._cache[name]

        template_path = self._templates_dir / f"{name}.yaml"
        if not template_path.exists():
            raise FileNotFoundError(
                f"Template '{name}' not found at {template_path}"
            )

        with open(template_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"Template file '{name}.yaml' must contain a YAML mapping"
            )

        # Validate required fields
        for field in ("name", "description", "required_variables", "template"):
            if field not in data:
                raise ValueError(
                    f"Template file '{name}.yaml' is missing required field: '{field}'"
                )

        if not isinstance(data["required_variables"], list):
            raise ValueError(
                f"Template '{name}' field 'required_variables' must be a list"
            )

        template = PromptTemplate(
            name=data["name"],
            description=data["description"],
            required_variables=data["required_variables"],
            template=data["template"],
        )

        self._cache[name] = template
        return template

    def get_template(self, name: str) -> PromptTemplate:
        """Retrieve a template by name.

        Args:
            name: The template name (YAML filename without extension).

        Returns:
            The PromptTemplate with metadata and content.

        Raises:
            FileNotFoundError: If the template does not exist.
            ValueError: If the template file is malformed.
        """
        return self._load_template(name)

    def validate_variables(self, name: str, variables: Dict[str, Any]) -> None:
        """Validate that all required variables are present and non-empty.

        Args:
            name: The template name.
            variables: Dict of variable name -> value to validate.

        Raises:
            MissingVariableError: If a required variable is missing or empty.
            FileNotFoundError: If the template does not exist.
        """
        template = self._load_template(name)

        for var in template.required_variables:
            if var not in variables:
                raise MissingVariableError(
                    template_name=name, variable_name=var
                )
            value = variables[var]
            # Check for empty strings and None
            if value is None or (isinstance(value, str) and not value.strip()):
                raise MissingVariableError(
                    template_name=name, variable_name=var
                )

    def render(self, template_name: str, **variables: Any) -> str:
        """Render a template with the provided variables.

        Validates all required variables are present and non-empty, performs
        substitution, then verifies no unresolved placeholders remain.

        Args:
            template_name: The template name.
            **variables: Keyword arguments for template variable substitution.

        Returns:
            The rendered prompt string.

        Raises:
            MissingVariableError: If a required variable is missing or empty.
            ValueError: If unresolved placeholders remain after rendering.
            FileNotFoundError: If the template does not exist.
        """
        # Validate all required variables first
        self.validate_variables(template_name, variables)

        template = self._load_template(template_name)

        # Perform substitution using str.format_map with a custom mapping
        # that preserves unknown placeholders for the post-render check
        rendered = template.template.format_map(_SafeFormatDict(variables))

        # Verify no unresolved placeholders remain
        unresolved = _UNRESOLVED_PLACEHOLDER_PATTERN.findall(rendered)
        if unresolved:
            raise ValueError(
                f"Template '{template_name}' has unresolved placeholders after rendering: "
                f"{unresolved}"
            )

        return rendered

    def list_templates(self) -> List[TemplateInfo]:
        """List all available templates in the templates directory.

        Returns:
            List of TemplateInfo objects with name, description, and required variables.
        """
        templates: List[TemplateInfo] = []

        if not self._templates_dir.exists():
            return templates

        for yaml_file in sorted(self._templates_dir.glob("*.yaml")):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                if isinstance(data, dict) and all(
                    k in data for k in ("name", "description", "required_variables")
                ):
                    templates.append(
                        TemplateInfo(
                            name=data["name"],
                            description=data["description"],
                            required_variables=data.get("required_variables", []),
                        )
                    )
            except (yaml.YAMLError, OSError):
                # Skip malformed or unreadable files
                continue

        return templates


class _SafeFormatDict(dict):
    """A dict subclass that raises KeyError for missing keys during format_map.

    This ensures str.format_map raises an error if a placeholder in the template
    doesn't have a corresponding variable, which we then catch in the
    unresolved placeholder check.
    """

    def __missing__(self, key: str) -> str:
        # Re-insert the placeholder so the post-render regex can detect it
        return "{" + key + "}"
