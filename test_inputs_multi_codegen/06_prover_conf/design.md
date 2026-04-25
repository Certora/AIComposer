# Counter (per-task prover_conf)

Same trivial `Counter` as scenario 01, with a per-task `prover_conf` section
in the JSON. The agent doesn't need to use the packages alias or any other
config key — this scenario exists to verify the plumbing: that
`InputData.prover_conf` populates `AIComposerContext.prover_conf_overrides`,
which the runner merges into the emitted Certora conf.
