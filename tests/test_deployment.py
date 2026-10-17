"""The chat and viewer must address the deployment Osprey actually built."""

from hackring.deployment import identity


def test_identity_prefers_rendered_deployment(tmp_path):
    (tmp_path / "profile.yml").write_text(
        "project_name: hackring\nvirtual_accelerator:\n  port: 5064\n  pva_port: 5075\n"
    )
    assert identity(tmp_path) == ("hackring", 5064, 5075)

    build = tmp_path / "build"
    build.mkdir()
    (build / "config.yml").write_text(
        "project_name: hackring-demo\nservices:\n  virtual_accelerator:\n"
        "    port: 6064\n    pva_port: 6075\n"
    )
    assert identity(tmp_path) == ("hackring-demo", 6064, 6075)
