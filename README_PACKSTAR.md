# PACK*

Probably Approximately Correct partition-function estimation in OSPREY 3.
PACK* fits a proposal distribution, normalizes and samples it with branch
DP, and corrects samples to a fixed continuously minimized target. K* combines
state intervals into binding-score intervals.

This version includes the default estimator and CPU/CUDA execution. Experimental
preflight workflows, distributed orchestration, alternative fitting policies,
functional-event observables, native SIMD experiments, and timing/audit exporters
are excluded. The original OSPREY algorithms remain available.

## Build

Use JDK 17 and the included Gradle wrapper:

```bash
./gradlew classes writePackStarClasspath
```

The main runtime classpath is written to `build/packstar-classpath.txt`.
Building and running a design does not require compiling tests. To redirect
build output, set `OSPREY_BUILD_ROOT` to a directory you choose; to choose the
classpath file, set `PACKSTAR_CLASSPATH_FILE`.

CPU execution needs no CUDA device. GPU execution requires a CUDA driver and a
GPU compatible with the included kernels. `./gradlew compileCuda` rebuilds the
kernels; `NVCC`, `CUDA_HOME`, and `OSPREY_CUDA_ARCHS` select the compiler and
target architectures.

## Run a design

Supply a prepared PDB and explicit protein/ligand chain selections. Residue IDs
and amino-acid choices below are placeholders for your system:

```bash
java --add-modules jdk.incubator.foreign \
  --add-opens java.base/java.util=ALL-UNNAMED \
  --add-opens java.base/java.lang=ALL-UNNAMED \
  --add-opens java.base/java.lang.invoke=ALL-UNNAMED \
  -Dpackstar.design.pdbPath=input.pdb \
  -Dpackstar.design.proteinChains=A -Dpackstar.design.ligandChains=B \
  '-Dpackstar.design.mutable=A42=ALA,SER' \
  '-Dpackstar.design.flexible=A43;B15' \
  -cp "$(cat build/packstar-classpath.txt)" \
  edu.duke.cs.osprey.packstar.PackStarDesignRunner
```

The default uses two CPU workers and writes results to `packstar-results/`
relative to the working directory. `packstar.design.outputDir` chooses another
directory. No user-specific filesystem location or cluster setup is required.

GPU DP is enabled by default. To run entirely on the CPU, add
`-Dpackstar.dp.gpu=false` before `-cp`. Energy minimization and proposal sampling
use the CPU by default.

To also use one GPU for energy minimization and proposal sampling, add these
JVM options before `-cp`:

```text
-Dpackstar.design.numGpus=1
-Dpackstar.pac.sampling.gpu=true
```

GPU execution retains bounded memory use, child-table slicing, out-of-core
execution, and support for multiple visible GPUs. Small workloads can use the
CPU path when GPU dispatch would add overhead.

For an existing Java conformational space, use `PackStarPartitionFunction`,
call `init(epsilon)`, then `compute(maxSamples)` and `makeResult()`, and close
the partition function after copying its result.
The entry point directly owns the PACK* branch-DP backend and configuration
scope. Results report actual CCD calls and logical sample counts; per-sample
trace callbacks are no longer part of the API. Use this entry point in place of
the removed `PackStarBound` and `getDelegate()` API.

## Defaults and statistical assumptions

| Setting | Default |
| --- | --- |
| `packstar.pac.confidence` | Failure-probability budget delta = 0.05 |
| `packstar.pac.targetEpsilon` | Interval width target, 1 - lower/upper = 0.683 |
| `packstar.pac.randomSeed` | 42, combined with the state identity |
| `packstar.pac.samples` | 1,000, used to size training and pilot batches |
| `packstar.pac.maxEstSamples` | 4,000 independent final draws |
| `packstar.cutoff.residualBudget` | 0.5 kcal/mol |
| `packstar.dp.gpu` | true |
| `packstar.pac.sampling.gpu` | false |

The proposal learns unary and pair corrections with joint residual fitting.
Adaptation preserves the interaction graph and branch decomposition.
Candidate scale is fixed at one and model selection uses two folds; the former
`packstar.pac.frequencySeverity.alphaGrid` and
`packstar.pac.frequencySeverity.folds` settings are removed.

The interval combines a clipped-weight bound with an exceedance-probability
bound. Its upper endpoint assumes a bound on conditional mean excess; the
default value 20 is an assumption, not an experimentally established constant.
A diagnostic can reject that assumption, but passing does not establish it.
Coverage also requires fixed pre-final choices, proposal support, and independent
final draws. A per-state delta is not a design-wide guarantee.

The residual budget changes the interaction model. Report it with results;
intervals concern that finite conformational space and minimization model,
not experimental binding affinity.

## Tests

```bash
./gradlew test \
  --tests 'edu.duke.cs.osprey.packstar.*' \
  --tests 'edu.duke.cs.osprey.branchdp.*'
```

Tests cover the estimator, real CCD/sample accounting, numerical boundaries,
DP storage, and CPU/GPU agreement. CUDA cases skip when no compatible device is
available. Historical experiment drivers and tests of removed modes are excluded.

See [README.md](README.md), [LICENSE.txt](LICENSE.txt), and
[CITING_OSPREY.txt](CITING_OSPREY.txt) for OSPREY installation, licensing, and
citation information.
