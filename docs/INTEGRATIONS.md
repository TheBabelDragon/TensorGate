# Integrations

All integrations are optional and live under `tensorgate.integrations`.

## WaveBridge

`tensorgate.integrations.wavebridge`

- `to_wavebridge_payload(tensor)` → normalized numerical array + TensorGate metadata
- `from_wavebridge_payload(payload)` → re-describe recovered array

Does **not** implement WAV, optical modulation, or channel simulation.

## MetaField

`tensorgate.integrations.metafield`

- `describe_state(tensor)`
- `prepare_for_operator(tensor, target_spec)`
- `accept_operator_output(tensor, expected_spec)`

Treats arrays as numerical state only. No physics imports.

## Operator ABI

`tensorgate.integrations.operator_abi`

- `validate_inputs(*tensors, specs=...)`
- `validate_output(tensor, expected=...)`

Wraps the numerical boundary around an operator invocation. Wilson–Dirac math stays in `metafield-operator-abi`.
