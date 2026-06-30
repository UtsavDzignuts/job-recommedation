"""Unit tests for PromptTemplateManager."""

import pytest

from app.ai.exceptions import MissingVariableError
from app.ai.prompt_manager import PromptTemplateManager, TemplateInfo


@pytest.fixture
def manager(tmp_path):
    """Create a PromptTemplateManager with a temporary templates directory."""
    # Create a sample template file
    template_content = """name: test_template
description: A test template
required_variables:
  - name
  - topic
template: |
  Hello {name}, let's talk about {topic}.
"""
    (tmp_path / "test_template.yaml").write_text(template_content)
    return PromptTemplateManager(templates_dir=str(tmp_path))


@pytest.fixture
def manager_with_prompts():
    """Create a PromptTemplateManager using the real app/prompts/ directory."""
    return PromptTemplateManager(templates_dir="app/prompts")


class TestGetTemplate:
    def test_loads_existing_template(self, manager):
        template = manager.get_template("test_template")
        assert template.name == "test_template"
        assert template.description == "A test template"
        assert template.required_variables == ["name", "topic"]
        assert "{name}" in template.template
        assert "{topic}" in template.template

    def test_raises_file_not_found_for_missing_template(self, manager):
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            manager.get_template("nonexistent")

    def test_caches_loaded_template(self, manager):
        t1 = manager.get_template("test_template")
        t2 = manager.get_template("test_template")
        assert t1 is t2


class TestValidateVariables:
    def test_passes_with_all_required_variables(self, manager):
        # Should not raise
        manager.validate_variables("test_template", {"name": "Alice", "topic": "AI"})

    def test_raises_missing_variable_error_for_absent_variable(self, manager):
        with pytest.raises(MissingVariableError) as exc_info:
            manager.validate_variables("test_template", {"name": "Alice"})
        assert exc_info.value.variable_name == "topic"
        assert exc_info.value.template_name == "test_template"

    def test_raises_missing_variable_error_for_empty_string(self, manager):
        with pytest.raises(MissingVariableError) as exc_info:
            manager.validate_variables("test_template", {"name": "Alice", "topic": ""})
        assert exc_info.value.variable_name == "topic"

    def test_raises_missing_variable_error_for_whitespace_only(self, manager):
        with pytest.raises(MissingVariableError) as exc_info:
            manager.validate_variables(
                "test_template", {"name": "Alice", "topic": "   "}
            )
        assert exc_info.value.variable_name == "topic"

    def test_raises_missing_variable_error_for_none(self, manager):
        with pytest.raises(MissingVariableError) as exc_info:
            manager.validate_variables(
                "test_template", {"name": "Alice", "topic": None}
            )
        assert exc_info.value.variable_name == "topic"


class TestRender:
    def test_renders_with_valid_variables(self, manager):
        result = manager.render("test_template", name="Alice", topic="AI")
        assert "Hello Alice" in result
        assert "talk about AI" in result

    def test_raises_missing_variable_error_on_missing_var(self, manager):
        with pytest.raises(MissingVariableError):
            manager.render("test_template", name="Alice")

    def test_no_unresolved_placeholders_after_render(self, manager):
        result = manager.render("test_template", name="Alice", topic="ML")
        import re
        unresolved = re.findall(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", result)
        assert unresolved == []

    def test_raises_value_error_on_unresolved_placeholders(self, tmp_path):
        """Template with extra placeholder not in required_variables."""
        template_content = """name: extra_placeholder
description: Template with extra unrequired placeholder
required_variables:
  - user_name
template: |
  Hello {user_name}, welcome to {place}.
"""
        (tmp_path / "extra_placeholder.yaml").write_text(template_content)
        mgr = PromptTemplateManager(templates_dir=str(tmp_path))
        with pytest.raises(ValueError, match="unresolved placeholders"):
            mgr.render("extra_placeholder", user_name="Alice")

    def test_renders_real_rag_answer_template(self, manager_with_prompts):
        result = manager_with_prompts.render(
            "rag_answer",
            query="What jobs are available?",
            context_documents="Job 1: Software Engineer\nJob 2: Data Scientist",
        )
        assert "What jobs are available?" in result
        assert "Software Engineer" in result
        assert "Data Scientist" in result


class TestListTemplates:
    def test_lists_templates_in_directory(self, manager, tmp_path):
        # Add another template
        another = """name: another
description: Another template
required_variables:
  - x
template: |
  Value is {x}.
"""
        (tmp_path / "another.yaml").write_text(another)
        # Clear cache so list_templates scans fresh
        templates = manager.list_templates()
        assert len(templates) == 2
        names = [t.name for t in templates]
        assert "another" in names
        assert "test_template" in names

    def test_returns_template_info_objects(self, manager):
        templates = manager.list_templates()
        assert len(templates) == 1
        info = templates[0]
        assert isinstance(info, TemplateInfo)
        assert info.name == "test_template"
        assert info.description == "A test template"
        assert info.required_variables == ["name", "topic"]

    def test_returns_empty_list_for_nonexistent_dir(self):
        mgr = PromptTemplateManager(templates_dir="/nonexistent/path")
        assert mgr.list_templates() == []

    def test_skips_malformed_yaml_files(self, tmp_path):
        (tmp_path / "bad.yaml").write_text("not: valid: yaml: [")
        mgr = PromptTemplateManager(templates_dir=str(tmp_path))
        templates = mgr.list_templates()
        assert templates == []
