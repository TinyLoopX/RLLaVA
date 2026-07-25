# CHORD task for obs_mining

- Config: examples/tasks/chord/config_chord.yaml
- Full run: examples/tasks/chord/run_chord.sh
- Debug run: examples/tasks/chord/run_chord_debug.sh

Environment variables (optional): OBS_CHORD_MODEL, OBS_CHORD_TRAIN, OBS_CHORD_VAL, OBS_CHORD_SFT, OBS_CHORD_OUT.

Notes
- Mirrors Trinity MIX-CHORD hyperparameters (mu schedule, expert ratio, phi).
- You can override YAML fields via CLI args when running the scripts.
