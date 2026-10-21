package edu.duke.cs.osprey.packstar;

import edu.duke.cs.osprey.astar.conf.RCs;
import edu.duke.cs.osprey.branchdp.*;
import edu.duke.cs.osprey.confspace.*;
import edu.duke.cs.osprey.ematrix.*;
import edu.duke.cs.osprey.energy.*;
import edu.duke.cs.osprey.energy.forcefield.ForcefieldParams;
import edu.duke.cs.osprey.parallelism.Parallelism;
import edu.duke.cs.osprey.structure.PDBIO;
import edu.duke.cs.osprey.tools.ByteBoundedCache;
import org.junit.jupiter.api.Test;

import java.lang.reflect.*;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

/** Exercise real CCD, ordered sample multiplicities, and proposal-cache invalidation. */
public class TestPackStarPerformance {
    private static Object field(Object value, String name) throws Exception {
        Field field = value.getClass().getDeclaredField(name);
        field.setAccessible(true);
        return field.get(value);
    }

    private static Object invoke(PackStarEstimator estimator, String name,
                                 Class<?>[] types, Object... args) throws Exception {
        Method method = PackStarEstimator.class.getDeclaredMethod(name, types);
        method.setAccessible(true);
        try { return method.invoke(estimator, args); }
        catch (InvocationTargetException ex) { throw new RuntimeException(ex.getCause()); }
    }

    @SuppressWarnings("unchecked")
    private static List<Object> ccd(PackStarEstimator estimator, List<int[]> samples,
                                     EnergyMatrix proposal, double logZ, String id) throws Exception {
        return (List<Object>) invoke(estimator, "runParallelCCD",
                new Class<?>[]{List.class, EnergyMatrix.class, double.class, String.class},
                samples, proposal, logZ, id);
    }

    private static final class PropertiesScope implements AutoCloseable {
        private final Properties saved = (Properties) System.getProperties().clone();
        void set(String key, String value) { System.setProperty(key, value); }
        @Override public void close() { System.setProperties(saved); }
    }

    @Test public void targetCachePreservesMultiplicityAndRefreshesProposalProvenance() throws Exception {
        try (var properties = new PropertiesScope()) {
            properties.set("packstar.pac.ccd.cacheBytes", "1048576");
            properties.set("packstar.pac.ccd.submissionBatchSize", "2");
            Strand strand = new Strand.Builder(PDBIO.readFile("src/test/resources/1CC8.ss.pdb")).build();
            for (String residue : List.of("A16", "A19")) {
                strand.flexibility.get(residue).setLibraryRotamers(Strand.WildType)
                        .addWildTypeRotamers().setContinuous();
            }
            var space = new SimpleConfSpace.Builder().addStrand(strand).build();
            var parallelism = Parallelism.makeCpu(2);
            try (var energy = new EnergyCalculator.Builder(space, new ForcefieldParams())
                    .setType(EnergyCalculator.Type.Cpu).setIsMinimizing(true)
                    .setParallelism(parallelism).build()) {
                var calculator = new ConfEnergyCalculator.Builder(space, energy).build();
                var base = new SimplerEnergyMatrixCalculator.Builder(calculator).build().calcEnergyMatrix();
                var graph = InteractionGraph.buildFromEdges(2, List.of(new int[]{0, 1}));
                var estimator = new PackStarEstimator(null, null, base, base, graph,
                        calculator, new RCs(space));
                List<int[]> samples = List.of(new int[]{0, 0}, new int[]{1, 0}, new int[]{0, 0});
                List<Object> first = ccd(estimator, samples, base, 1.25, "old");
                assertEquals(2, estimator.getTotalCCDCalls());
                assertSame(first.get(0), first.get(2));
                for (Object sample : first) {
                    assertNull(field(sample, "epmol"));
                    assertNotNull(field(sample, "features"));
                }
                var proposal = new EnergyMatrix(base);
                proposal.setOneBody(0, 0, proposal.getOneBody(0, 0) + 0.75);
                List<Object> second = ccd(estimator, samples, proposal, 9.0, "new");
                assertEquals(2, estimator.getTotalCCDCalls(), "cross-batch hits must avoid CCD");
                assertEquals(samples.size(), second.size());
                assertNotSame(first.get(0), second.get(0), "provenance belongs to each batch");
                for (int i = 0; i < samples.size(); i++) {
                    assertEquals(field(first.get(i), "eTrue"), field(second.get(i), "eTrue"));
                    assertEquals("old", field(first.get(i), "sourceProposalId"));
                    assertEquals("new", field(second.get(i), "sourceProposalId"));
                    assertEquals(9.0, field(second.get(i), "sourceProposalLogZ"));
                    assertSame(field(first.get(i), "features"), field(second.get(i), "features"));
                }
                assertEquals((double) field(first.get(0), "sourceProposalEnergy") + 0.75,
                        (double) field(second.get(0), "sourceProposalEnergy"), 1e-12);
                invoke(estimator, "shutdownSamplingResources", new Class<?>[0]);
                assertEquals(0, ((ByteBoundedCache<?, ?>) field(estimator, "ccdEnergyCache")).bytes());
            }
        }
    }

    @Test public void cdfReusePreservesSeededDrawsAndInvalidatesInPlaceRefits() throws Exception {
        for (int threads : new int[]{1, 2}) {
            try (var properties = new PropertiesScope()) {
                properties.set("packstar.pac.sampling.threads", Integer.toString(threads));
                properties.set("packstar.pac.sampling.largeLambdaThreshold", "1");
                properties.set("packstar.pac.sampling.gpu", "false");
                properties.set("packstar.pac.sampling.cdfCacheBytes", "1048576");
                properties.set("branchdp.dp.gpu", "false");
                var rcs = new RCs(new int[][]{{0, 1}, {0, 1}, {0, 1}});
                var base = new EnergyMatrix(3, new int[]{2, 2, 2}, 0);
                var graph = InteractionGraph.buildFromEdges(3, List.of(new int[]{0, 1}, new int[]{1, 2}));
                var decomposition = new BranchDecomposition(graph,
                        BranchDecomposition.Strategy.WEIGHTED_HICKS, new int[]{2, 2, 2}, false);
                decomposition.compute();
                var root = BranchDpBackend.selectConfiguredRoot(
                        decomposition, rcs, true).root;
                RootedTreeEdge.postOrderInitIncremental(root, base, base, graph, 1.0);
                RootedTreeEdge.postOrderComputeFullDP(root);
                var estimator = new PackStarEstimator(root, root.getLeftChild().getChildOfEdge(),
                        base, base, graph, null, rcs);
                try {
                    List<int[]> first = draws(estimator, 123L);
                    List<int[]> second = draws(estimator, 123L);
                    equalDraws(first, second);
                    var cache = (ByteBoundedCache<?, ?>) field(estimator, "conditionalCdfCache");
                    assertTrue(cache.hits() > 0);
                    base.setOneBody(0, 0, 1000.0);
                    invoke(estimator, "recomputeDP", new Class<?>[]{EnergyMatrix.class}, base);
                    assertEquals(0, cache.bytes());
                    List<int[]> changed = draws(estimator, 123L);
                    assertTrue(changed.stream().allMatch(conf -> conf[0] == 1));
                    invoke(estimator, "invalidateSamplingCaches", new Class<?>[0]);
                    equalDraws(changed, draws(estimator, 123L));
                } finally {
                    invoke(estimator, "shutdownSamplingResources", new Class<?>[0]);
                    RootedTreeEdge.postOrderReleaseLargeMemory(root);
                }
            }
        }
    }

    @Test public void sparsePairRefitsRestoreOldCellsAndPreserveTheInteractionModel() throws Exception {
        var rcs = new RCs(new int[][]{{0, 1}, {0, 1}, {0, 1}});
        var base = new EnergyMatrix(3, new int[]{2, 2, 2}, 0);
        base.setOneBody(1, 0, 0.25);
        base.setPairwise(0, 1, 1, 0, 0.5);
        var graph = InteractionGraph.buildFromEdges(3, List.of(new int[]{0, 1}, new int[]{1, 2}));
        var estimator = new PackStarEstimator(null, null, base, base, graph, null, rcs);
        Class<?> etaType = Arrays.stream(PackStarEstimator.class.getDeclaredClasses())
                .filter(type -> type.getSimpleName().equals("EtaCorrections")).findFirst().orElseThrow();
        Constructor<?> constructor = etaType.getDeclaredConstructor(int.class, int[].class);
        constructor.setAccessible(true);
        Method unary = etaType.getDeclaredMethod("addOneBodySample", int.class, int.class, double.class);
        Method pair = etaType.getDeclaredMethod("addPairSample", int.class, int.class, int.class, int.class, double.class);
        unary.setAccessible(true);
        pair.setAccessible(true);
        Field gauge = etaType.getDeclaredField("globalOffsetKcal");
        gauge.setAccessible(true);
        EnergyMatrix previous = null;
        for (int round = 0; round < 3; round++) {
            Object eta = constructor.newInstance(3, new int[]{2, 2, 2});
            var expected = new EnergyMatrix(base);
            double offset = 0.0;
            if (round == 0) {
                unary.invoke(eta, 1, 0, -0.8);
                pair.invoke(eta, 0, 1, 1, 0, 0.35);
                expected.setOneBody(1, 0, expected.getOneBody(1, 0) - 0.8);
                expected.setPairwise(0, 1, 1, 0, expected.getPairwise(0, 1, 1, 0) + 0.35);
                offset = 1.2;
            } else if (round == 1) {
                unary.invoke(eta, 2, 1, 0.6);
                pair.invoke(eta, 1, 1, 2, 0, -0.2);
                expected.setOneBody(2, 1, expected.getOneBody(2, 1) + 0.6);
                expected.setPairwise(1, 1, 2, 0, expected.getPairwise(1, 1, 2, 0) - 0.2);
                offset = -0.7;
            }
            gauge.setDouble(eta, offset);
            for (int rc = 0; rc < 2; rc++)
                expected.setOneBody(0, rc, expected.getOneBody(0, rc) + offset);
            invoke(estimator, "finalizeEtaTermCounts", new Class<?>[]{etaType}, eta);
            var corrected = (EnergyMatrix) invoke(estimator, "buildCorrectedEmat", new Class<?>[]{etaType}, eta);
            if (previous != null) assertSame(previous, corrected);
            assertFalse(corrected.hasHigherOrderTerms());
            assertSame(graph, field(estimator, "interactionGraph"));
            for (int bits = 0; bits < 8; bits++) {
                int[] conf = {bits & 1, (bits >>> 1) & 1, (bits >>> 2) & 1};
                assertEquals(expected.confE(conf), corrected.confE(conf), 1e-12);
            }
            previous = corrected;
        }
        assertEquals(0.25, base.getOneBody(1, 0));
        assertEquals(0.5, base.getPairwise(0, 1, 1, 0));
    }

    @SuppressWarnings("unchecked")
    private static List<int[]> draws(PackStarEstimator estimator, long seed) throws Exception {
        return (List<int[]>) invoke(estimator, "sampleConformationsFromDP",
                new Class<?>[]{int.class, Random.class}, 128, new Random(seed));
    }

    private static void equalDraws(List<int[]> first, List<int[]> second) {
        assertEquals(first.size(), second.size());
        for (int i = 0; i < first.size(); i++) assertArrayEquals(first.get(i), second.get(i));
    }
}
