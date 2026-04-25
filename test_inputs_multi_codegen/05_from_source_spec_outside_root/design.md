# Spec-outside-root Fixture

Intentionally malformed. `source_root` is `./workspace`, but the spec path
resolves to `./stray.spec` (outside the workspace). Upload must reject
this with an error that names both the spec path and the source_root.
