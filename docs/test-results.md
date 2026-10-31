# Test results

On October 31, 2026, the course builder and current agent were run in Minecraft Java 26.1 with Fabric and Minescript 5.0 on Linux. The test used an isolated Minecraft client and a local server, so it does **not** verify the official-launcher single-player setup or Windows/macOS.

The agent ran one training episode from fresh weights with `epsilon=1` (random exploration) and seed 7. It crossed the first two gaps and fell at the third. The run logged 28 decisions in 4.824 seconds and saved 56 feature weights. This rate includes the in-game episode loop but excludes world startup and reset; it is one machine's measurement, not a benchmark. A fall is an ordinary episode outcome. This test confirms the code ran and recorded an update; it does not show that the policy learned to complete the course.

A separate run with the same seed fell at the second gap. Seeds control Python's random choices, but they do not make Minecraft physics, communication, and timing deterministic. Evaluate learning over multiple independent runs and compare against baselines.

The Python tests can be run from the project directory with:

```text
python -m unittest discover -s tests -v
```
