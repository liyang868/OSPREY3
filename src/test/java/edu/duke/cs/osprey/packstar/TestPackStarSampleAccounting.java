package edu.duke.cs.osprey.packstar;

import edu.duke.cs.osprey.astar.conf.RCs;
import edu.duke.cs.osprey.branchdp.BranchDpConfig;
import edu.duke.cs.osprey.confspace.SimpleConfSpace;
import edu.duke.cs.osprey.confspace.Strand;
import edu.duke.cs.osprey.ematrix.EnergyMatrix;
import edu.duke.cs.osprey.ematrix.SimplerEnergyMatrixCalculator;
import edu.duke.cs.osprey.energy.ConfEnergyCalculator;
import edu.duke.cs.osprey.energy.EnergyCalculator;
import edu.duke.cs.osprey.energy.forcefield.ForcefieldParams;
import edu.duke.cs.osprey.kstar.pfunc.PartitionFunction;
import edu.duke.cs.osprey.parallelism.Parallelism;
import edu.duke.cs.osprey.structure.PDBIO;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

public class TestPackStarSampleAccounting {

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    public void realCcdAccountingSurvivesCacheHitsAbortAndNoSampling(boolean useGpu) throws Exception {
        if (useGpu) Assumptions.assumeTrue(!edu.duke.cs.osprey.gpu.cuda.Gpus.get().getGpus().isEmpty(),
                "CUDA device required");
        Properties saved = (Properties) System.getProperties().clone();
        try {
            for (String key : new ArrayList<>(System.getProperties().stringPropertyNames())) {
                if (key.startsWith("packstar.") || key.startsWith("branchdp.")
                        || key.startsWith("osprey.packstar.") || key.startsWith("osprey.branchdp.")) {
                    System.clearProperty(key);
                }
            }
            System.setProperty("packstar.dp.gpu", Boolean.toString(useGpu));
            System.setProperty("packstar.dp.parallel.threads", "2");
            System.setProperty("packstar.pac.sampling.threads", "2");
            System.setProperty("packstar.pac.sampling.gpu", Boolean.toString(useGpu));
            System.setProperty("packstar.pac.sampling.progress", "false");
            System.setProperty("packstar.pac.randomSeed", "42");
            System.setProperty("packstar.pac.ccd.cacheBytes", "67108864");
            // PACK* aliases must win throughout construction and compute.
            System.setProperty("packstar.cutoff.residualBudget", "0.5");
            System.setProperty("branchdp.cutoff.residualBudget", "10000");
            Strand strand = new Strand.Builder(PDBIO.readFile("src/test/resources/1CC8.ss.pdb")).build();
            for (String residue : List.of("A16", "A19")) {
                strand.flexibility.get(residue).setLibraryRotamers(Strand.WildType)
                        .addWildTypeRotamers().setContinuous();
            }
            var space = new SimpleConfSpace.Builder().addStrand(strand).build();
            var parallelism = useGpu ? Parallelism.make(2, 1, 1) : Parallelism.makeCpu(2);
            try (var energy = new EnergyCalculator.Builder(space, new ForcefieldParams())
                    .setType(useGpu ? EnergyCalculator.Type.ResidueCudaCCD : EnergyCalculator.Type.Cpu)
                    .setIsMinimizing(true).setParallelism(parallelism).build();
                 var rigid = new EnergyCalculator.SharedBuilder(energy).setIsMinimizing(false).build()) {
                var calculator = new ConfEnergyCalculator.Builder(space, energy).build();
                var min = new SimplerEnergyMatrixCalculator.Builder(calculator).build().calcEnergyMatrix();
                // Avoid an unrelated ~1e-14 lower-bound cancellation at an identical pose.
                for (int pos = 0; pos < min.getNumPos(); pos++) {
                    for (int rc = 0; rc < min.getNumConfAtPos(pos); rc++) {
                        min.setOneBody(pos, rc, min.getOneBody(pos, rc) - 1e-8);
                    }
                }
                var rig = new SimplerEnergyMatrixCalculator.Builder(new ConfEnergyCalculator(calculator, rigid))
                        .build().calcEnergyMatrix();
                PackStarResult defaults = run("defaults", space, calculator, parallelism, min, rig, 0.683);
                assertEquals(PartitionFunction.Status.Estimated, defaults.status,
                        "default estimator settings must produce a usable result");
                System.setProperty("packstar.pac.trainSamples", "200");
                System.setProperty("packstar.pac.frequencySeverity.maxRefits", "2");
                PackStarResult cached = run("cached", space, calculator, parallelism, min, rig, 0.683);
                assertEquals(PartitionFunction.Status.Estimated, cached.status);
                assertTrue(cached.numConfs > 0);
                assertTrue(cached.getStat(PackStarResult.CCD_SAMPLE_RECORDS_STAT) > cached.numConfs,
                        "duplicate draws must remain counted when their target energies are reused");

                try (var scope = BranchDpConfig.enterPackStarAliasScope()) {
                    sameEstimate(cached, run("nested", space, calculator, parallelism, min, rig, 0.683));
                    assertTrue(BranchDpConfig.isPackStarAliasScopeActive());
                }
                assertFalse(BranchDpConfig.isPackStarAliasScopeActive());

                System.setProperty("packstar.pac.ccd.cacheBytes", "0");
                PackStarResult uncached = run("uncached", space, calculator, parallelism, min, rig, 0.683);
                sameEstimate(cached, uncached);
                assertTrue(cached.numConfs < uncached.numConfs, "cache hits must reduce actual CCD calls");
                assertEquals(cached.getStat(PackStarResult.CCD_SAMPLE_RECORDS_STAT),
                        uncached.getStat(PackStarResult.CCD_SAMPLE_RECORDS_STAT));

                System.setProperty("packstar.pac.maxEstSamples", "16");
                System.setProperty("packstar.pac.frequencySeverity.maxRefits", "0");
                PackStarResult aborted = run("aborted", space, calculator, parallelism, min, rig, 1e-8);
                assertEquals(PartitionFunction.Status.Aborted, aborted.status);
                assertTrue(aborted.numConfs > 0);

                try (var pf = new PackStarPartitionFunction(space, rig, min, calculator,
                        new RCs(space), parallelism, "performance-2")) {
                    pf.init(1e-8);
                    System.setProperty("packstar.pac.frequencySeverity.relativeBoundKcal", "0");
                    assertThrows(IllegalArgumentException.class, () -> pf.compute(Integer.MAX_VALUE));
                    assertFalse(BranchDpConfig.isPackStarAliasScopeActive());
                    System.clearProperty("packstar.pac.frequencySeverity.relativeBoundKcal");
                }
                assertFalse(BranchDpConfig.isPackStarAliasScopeActive());
                assertThrows(NullPointerException.class, () -> new PackStarPartitionFunction(
                        space, null, min, calculator, new RCs(space), parallelism));
                assertFalse(BranchDpConfig.isPackStarAliasScopeActive());

                PackStarResult skipped = run("skipped", space, calculator, parallelism, min, min, 0.683);
                assertEquals(PartitionFunction.Status.Estimated, skipped.status);
                assertEquals(0, skipped.numConfs);
                assertEquals(0, skipped.getStat(PackStarResult.CCD_SAMPLE_RECORDS_STAT));
            }
        } finally {
            System.setProperties(saved);
        }
    }

    private PackStarResult run(String name, SimpleConfSpace space, ConfEnergyCalculator calculator,
                               Parallelism parallelism, EnergyMatrix min, EnergyMatrix rig,
                               double epsilon) throws Exception {
        boolean callerScope = BranchDpConfig.isPackStarAliasScopeActive();
        try (var contexts = calculator.ecalc.tasks.contextGroup();
             var pf = new PackStarPartitionFunction(space, rig, min, calculator,
                     new RCs(space), parallelism, "performance-2")) {
            assertEquals(callerScope, BranchDpConfig.isPackStarAliasScopeActive());
            pf.init(epsilon);
            pf.putTaskContexts(contexts);
            pf.compute(Integer.MAX_VALUE);
            assertEquals(callerScope, BranchDpConfig.isPackStarAliasScopeActive());
            PackStarResult result = pf.makeResult();
            assertEquals(pf.getNumConfsEvaluated(), result.numConfs);
            assertEquals(result.numConfs, result.getStat(PackStarResult.FULL_CCD_STAT));
            assertTrue(result.getStat(PackStarResult.CCD_SAMPLE_RECORDS_STAT) >= result.numConfs);
            System.out.println("ACCOUNTING " + name + " status=" + result.status
                    + " CCD=" + result.numConfs
                    + " records=" + result.getStat(PackStarResult.CCD_SAMPLE_RECORDS_STAT)
                    + " lower=" + result.values.calcLowerBound()
                    + " upper=" + result.values.calcUpperBound());
            return result;
        } finally {
            assertEquals(callerScope, BranchDpConfig.isPackStarAliasScopeActive());
        }
    }

    private static void sameEstimate(PackStarResult a, PackStarResult b) {
        assertEquals(a.status, b.status);
        assertEquals(a.values.calcLowerBound(), b.values.calcLowerBound());
        assertEquals(a.values.calcUpperBound(), b.values.calcUpperBound());
    }
}
