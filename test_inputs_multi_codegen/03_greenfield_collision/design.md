# Collision Fixture

This fixture is intentionally malformed: two spec files share a basename,
and in greenfield mode both resolve to `certora/foo.spec` in the VFS. The
pipeline must refuse this input at upload time.
