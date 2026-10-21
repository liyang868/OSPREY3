package edu.duke.cs.osprey.branchdp;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import edu.duke.cs.osprey.astar.conf.RCs;
import edu.duke.cs.osprey.ematrix.EnergyMatrix;
import edu.duke.cs.osprey.gpu.cuda.Gpus;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.math.BigInteger;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Numeric validation of the CUDA GPU full-DP fast path
 * (dp.cu full_dp_n_children + DPGpuFullDP, gated by branchdp.dp.gpu=true)
 * against the Java DP path, on the same synthetic non-leaf edge with N children.
 *
 * The GPU run goes through the real gate (tryComputeFullDPGpu via reflection) so
 * the test asserts the GPU path actually fired; if no CUDA GPU is present the gate
 * returns false and the test is skipped (assumeTrue) rather than vacuously passing.
 * The Java reference uses the LEGACY non-fold path (foldChildren=false,
 * getMstateForFullState), an INDEPENDENT child-index computation from the GPU's
 * ChildFoldPlan, so agreement cross-validates the fold/CSR mapping too.
 *
 * A fully-connected graph guarantees real lambda-M pair terms (lmPairs > 0).
 * Multi-child shapes mix lambda-dependent and lambda-invariant children (as in
 * TestChildFold) to exercise the per-child CSR offsets + table bases.
 *
 * Not bit-identical by design (two-pass log-sum-exp + block-parallel summation +
 * GPU FP), so the comparison is to a relative tolerance; the measured max abs/rel
 * difference is printed for the record.
 */
public class TestGpuFullDP {

    @Test
    public void samplingPreparationIsBoundedAndScopedToAnExplicitProposalEpoch() {
        RootedTreeEdge parent = buildParent(new int[]{2, 2, 3, 2},
                new int[]{0, 1}, new int[]{2, 3}, new int[][]{{1, 2}, {0}});
        assertTrue(parent.canUseGpuSampling());
        var cache = new RootedTreeEdge.SamplingRequestCache(1L << 20);
        var first = parent.prepareGpuSamplingRequest(new long[]{0, 1}, 10, false, cache);
        var second = parent.prepareGpuSamplingRequest(new long[]{1, 0, 1}, 20, true, cache);
        assertSame(first.childUpperAll, second.childUpperAll);
        assertSame(first.lmMin, second.lmMin);
        assertSame(first.modelIdentity, second.modelIdentity);
        assertArrayEquals(new long[]{1, 0, 1}, second.mIdxPerSample);
        assertEquals(20, second.baseSeed);
        assertEquals(1, cache.hits());
        assertTrue(cache.bytes() <= (1L << 20));
        cache.clear();
        var child = parent.getFset().iterator().next();
        child.setLogZ(0, child.getLogZLower(0), child.getLogZUpper(0) + 1.0);
        var changed = parent.prepareGpuSamplingRequest(new long[]{0}, 10, false, cache);
        assertNotSame(first.modelIdentity, changed.modelIdentity);
        assertNotEquals(first.childUpperCacheKey, changed.childUpperCacheKey);
        var disabled = new RootedTreeEdge.SamplingRequestCache(0);
        var uncached = parent.prepareGpuSamplingRequest(new long[]{0}, 10, false, disabled);
        assertNull(uncached.modelIdentity);
        assertArrayEquals(changed.childUpperAll, uncached.childUpperAll);
        cache.clear();
        for (RootedTreeEdge edge : parent.getFset()) edge.releaseLargeMemory();
        parent.releaseLargeMemory();
    }

    private static final double REL_TOL = 1e-6;

    @TempDir
    Path tempDir;

    private static final String[] MUTATED_SYSTEM_PROPERTIES = {
            "branchdp.dp.gpu",
            "branchdp.dp.nativeKernel",
            "branchdp.dp.parallel",
            "branchdp.dp.progress",
            "branchdp.dp.foldChildren",
            "branchdp.dp.gpu.minWork",
            "branchdp.dp.gpu.maxBytes",
            "branchdp.dp.gpu.trace",
            "branchdp.dp.gpu.outputTileMStates",
            "branchdp.dp.gpu.persistentContext",
            "branchdp.dp.gpu.childSlicing",
            "branchdp.dp.gpu.childSlicing.force",
            "branchdp.dp.gpu.childSliceMaxBytes",
            "branchdp.dp.gpu.minMStatesPerGpu",
            "branchdp.dp.gpu.maxGpus",
            "branchdp.dp.gpu.hybridChildTiling",
            "branchdp.dp.gpu.hybridChildTiling.force",
            "branchdp.dp.gpu.outOfCore",
            "branchdp.dp.gpu.outOfCore.force",
            "branchdp.dp.gpu.outOfCore.budgetBytes",
            "branchdp.dp.gpu.outOfCore.outputWorkspaceMaxBytes"
    };

    private final Map<String,String> savedSystemProperties = new HashMap<>();

    @BeforeEach
    public void isolateBranchDpSystemProperties() {
        savedSystemProperties.clear();
        for (String property : MUTATED_SYSTEM_PROPERTIES) {
            String value = System.getProperty(property);
            if (value != null) {
                savedSystemProperties.put(property, value);
            }
            System.clearProperty(property);
        }
    }

    @AfterEach
    public void restoreBranchDpSystemProperties() {
        for (String property : MUTATED_SYSTEM_PROPERTIES) {
            String value = savedSystemProperties.get(property);
            if (value == null) {
                System.clearProperty(property);
            } else {
                System.setProperty(property, value);
            }
        }
    }

    // ---- reflection helpers -----------------------------------------------

    private static void set(Object o, String field, Object val) {
        try {
            Field f = RootedTreeEdge.class.getDeclaredField(field);
            f.setAccessible(true);
            f.set(o, val);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException("set " + field, e);
        }
    }

    private static Object invoke(Object o, String method) {
        try {
            Method m = RootedTreeEdge.class.getDeclaredMethod(method);
            m.setAccessible(true);
            return m.invoke(o);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException("invoke " + method, e);
        }
    }

    private static Object invoke(Object o, String method, long arg) {
        try {
            Method m = RootedTreeEdge.class.getDeclaredMethod(method, long.class);
            m.setAccessible(true);
            return m.invoke(o, arg);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException("invoke " + method, e);
        }
    }

    private static Object field(Object o, String name) {
        try {
            Field f = o.getClass().getDeclaredField(name);
            f.setAccessible(true);
            return f.get(o);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException("field " + name, e);
        }
    }

    private static Object buildChildSlice(DPGpuFullDP.Request req, long unionStart, int unionCount) {
        try {
            Method m = DPGpuFullDP.class.getDeclaredMethod("buildChildSlice",
                    DPGpuFullDP.Request.class, long.class, int.class);
            m.setAccessible(true);
            return m.invoke(null, req, unionStart, unionCount);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException("buildChildSlice", e);
        }
    }

    private MappedDPTable replaceWithMappedTable(RootedTreeEdge edge, int chunkEntries,
                                                  List<Path> paths) {
        DPTable previous = (DPTable)field(edge, "dpTable");
        MappedDPTable mapped = new MappedDPTable(previous.size(), chunkEntries, tempDir);
        for (long i = 0L; i < previous.size(); i++) {
            mapped.set(i, previous.lower(i), previous.upper(i));
        }
        previous.close();
        set(edge, "dpTable", mapped);
        paths.add(mapped.pathForTesting());
        return mapped;
    }

    private static long originalChildIndex(DPGpuFullDP.Request req, int child,
                                           long rowKey, long lambdaKey) {
        long idx = 0L;
        int mOff = req.childMTermOff[child];
        for (int t = 0; t < req.childMTermCnt[child]; t++) {
            long packedStride = req.childMPackedStrideAll[mOff + t];
            long digit = (rowKey / packedStride) % req.childMCountsAll[mOff + t];
            idx += digit * req.childMStrideAll[mOff + t];
        }
        int lOff = req.childLTermOff[child];
        for (int t = 0; t < req.childLTermCnt[child]; t++) {
            long packedStride = req.childLPackedStrideAll[lOff + t];
            long digit = (lambdaKey / packedStride) % req.childLCountsAll[lOff + t];
            idx += digit * req.childLStrideAll[lOff + t];
        }
        return idx;
    }

    // ---- synthetic edge construction (mirrors TestChildFold) ---------------

    private static RCs rcsFor(int[] positions, int[] cardinalities) {
        int maxPos = 0;
        for (int p : positions) maxPos = Math.max(maxPos, p);
        int[][] arr = new int[maxPos + 1][];
        for (int i = 0; i < arr.length; i++) arr[i] = new int[]{0};
        for (int i = 0; i < positions.length; i++) {
            // Identity-map local RC index -> global RC id (0,1,2,...). Leaving this
            // zero-filled (as before) makes RCs.get(pos, localIdx) always return 0
            // regardless of localIdx, which silently makes computeLambdaOnlyEnergy()/
            // computeLambdaMEnergy() constant across every lambda/M local index --
            // masked in child-bearing tests by the child's own (index-arithmetic-based,
            // not RCs.get()-based) contribution, but not masked for leaf edges.
            int card = cardinalities[i];
            int[] ids = new int[card];
            for (int k = 0; k < card; k++) ids[k] = k;
            arr[positions[i]] = ids;
        }
        return new RCs(arr);
    }

    private static EnergyMatrix ematFilled(int numPos, int[] cards, double seed) {
        EnergyMatrix e = new EnergyMatrix(numPos, cards, 0.0);
        for (int p = 0; p < numPos; p++) {
            for (int rc = 0; rc < cards[p]; rc++) {
                e.setOneBody(p, rc, seed + 0.13 * p - 0.07 * rc);
            }
        }
        for (int pi = 0; pi < numPos; pi++) {
            for (int pj = pi + 1; pj < numPos; pj++) {
                for (int ri = 0; ri < cards[pi]; ri++) {
                    for (int rj = 0; rj < cards[pj]; rj++) {
                        e.setPairwise(pi, ri, pj, rj, 0.31 * seed + 0.05 * (ri + 1) * (rj + 2) - 0.02 * (pi + pj));
                    }
                }
            }
        }
        return e;
    }

    private static InteractionGraph fullyConnected(int numPos) {
        boolean[][] adj = new boolean[numPos][numPos];
        for (int i = 0; i < numPos; i++)
            for (int j = 0; j < numPos; j++)
                if (i != j) adj[i][j] = true;
        try {
            Constructor<InteractionGraph> c =
                    InteractionGraph.class.getDeclaredConstructor(int.class, boolean[][].class);
            c.setAccessible(true);
            return c.newInstance(numPos, adj);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException("build InteractionGraph", e);
        }
    }

    private static long product(int[] cards, int[] pos) {
        long c = 1;
        for (int p : pos) c *= cards[p];
        return c;
    }

    private static int visibleGpuCount() {
        try {
            return Gpus.get().getGpus().size();
        } catch (Throwable t) {
            return 0;
        }
    }

    /** A child lambda-edge with the given M positions and a deterministic dense DP table. */
    private static RootedTreeEdge child(RCs rcs, int[] mPos, int[] cards, double base) {
        RootedTreeEdge c = new RootedTreeEdge(null, null, new LinkedHashSet<>(), false, rcs);
        set(c, "mPositionsSorted", mPos.clone());
        set(c, "isLambdaEdge", Boolean.TRUE);
        long mc = product(cards, mPos);
        set(c, "mStateCount", Long.valueOf(mc));
        set(c, "mArraySize", Integer.valueOf((int) mc));
        DenseDPTable t = new DenseDPTable(mc);
        for (long i = 0; i < mc; i++) t.set(i, base + 0.5 * i, base + 50.0 + 0.3 * i);
        set(c, "dpTable", t);
        return c;
    }

    /** Fresh, fully-initialized N-child parent edge ready for computeFullDP. */
    private RootedTreeEdge buildParent(int[] cards, int[] mPos, int[] lambdaPos, int[][] childMPositions) {
        return buildParent(cards, mPos, lambdaPos, childMPositions, false, 1);
    }

    private RootedTreeEdge buildParent(int[] cards, int[] mPos, int[] lambdaPos,
                                      int[][] childMPositions, boolean shardedParent, int shardSize) {
        int numPos = cards.length;
        int[] allPos = new int[numPos];
        for (int i = 0; i < numPos; i++) allPos[i] = i;
        RCs rcs = rcsFor(allPos, cards);

        LinkedHashSet<RootedTreeEdge> fset = new LinkedHashSet<>();
        for (int c = 0; c < childMPositions.length; c++) {
            fset.add(child(rcs, childMPositions[c], cards, 1000.0 * (c + 1)));
        }

        RootedTreeEdge parent = new RootedTreeEdge(null, null, new LinkedHashSet<>(), false, rcs);
        set(parent, "isLambdaEdge", Boolean.TRUE);
        set(parent, "mPositionsSorted", mPos.clone());
        set(parent, "lambdaPositionsSorted", lambdaPos.clone());
        long mc = product(cards, mPos);
        set(parent, "mStateCount", Long.valueOf(mc));
        set(parent, "mArraySize", Integer.valueOf((int) mc));
        set(parent, "totalLambdaStates", Integer.valueOf((int) product(cards, lambdaPos)));
        set(parent, "dpTable", shardedParent
                ? new ShardedDPTable(mc, shardSize)
                : new DenseDPTable(mc));
        set(parent, "Fset", fset);

        // Populates lambdaOnlyRigid/Min + cached emat/G/RT (required by the GPU gate).
        parent.initIncrementalEnumeration(
                ematFilled(numPos, cards, 1.0), ematFilled(numPos, cards, 2.0),
                fullyConnected(numPos), 1.9);
        return parent;
    }

    private static double[][] read(RootedTreeEdge p, long mc) {
        double[] lo = new double[(int) mc];
        double[] up = new double[(int) mc];
        for (int i = 0; i < mc; i++) {
            lo[i] = p.getLogZLower(i);
            up[i] = p.getLogZUpper(i);
        }
        return new double[][]{lo, up};
    }

    private double[][] runJava(int[] cards, int[] mPos, int[] lambdaPos, int[][] childMPositions) {
        System.setProperty("branchdp.dp.gpu", "false");
        System.setProperty("branchdp.dp.nativeKernel", "false");
        System.setProperty("branchdp.dp.parallel", "false");
        System.setProperty("branchdp.dp.progress", "false");
        // Legacy (non-fold) index path: INDEPENDENT child-index from the GPU's
        // ChildFoldPlan, so agreement cross-validates the fold/CSR mapping.
        System.setProperty("branchdp.dp.foldChildren", "false");
        RootedTreeEdge p = buildParent(cards, mPos, lambdaPos, childMPositions);
        p.computeFullDP();
        return read(p, product(cards, mPos));
    }

    private double[][] runGpu(int[] cards, int[] mPos, int[] lambdaPos, int[][] childMPositions, boolean multiGpu) {
        return runGpu(cards, mPos, lambdaPos, childMPositions, multiGpu, false, 1, Integer.MAX_VALUE, false);
    }

    private double[][] runGpu(int[] cards, int[] mPos, int[] lambdaPos, int[][] childMPositions,
                              boolean multiGpu, boolean shardedParent, int shardSize, long outputTileMStates) {
        return runGpu(cards, mPos, lambdaPos, childMPositions, multiGpu,
                shardedParent, shardSize, outputTileMStates, false);
    }

    private double[][] runGpu(int[] cards, int[] mPos, int[] lambdaPos, int[][] childMPositions,
                              boolean multiGpu, boolean shardedParent, int shardSize,
                              long outputTileMStates, boolean forceChildSlicing) {
        System.setProperty("branchdp.dp.gpu", "true");
        System.setProperty("branchdp.dp.gpu.minWork", "1");
        System.setProperty("branchdp.dp.gpu.maxBytes", String.valueOf(6L * 1024 * 1024 * 1024));
        System.setProperty("branchdp.dp.gpu.trace", "true");
        System.setProperty("branchdp.dp.gpu.outputTileMStates", String.valueOf(outputTileMStates));
        System.setProperty("branchdp.dp.gpu.persistentContext", "true");
        System.setProperty("branchdp.dp.gpu.childSlicing", "true");
        System.setProperty("branchdp.dp.gpu.childSlicing.force", String.valueOf(forceChildSlicing));
        System.setProperty("branchdp.dp.gpu.childSliceMaxBytes", "4096");
        System.setProperty("branchdp.dp.nativeKernel", "false");
        System.setProperty("branchdp.dp.parallel", "false");
        System.setProperty("branchdp.dp.foldChildren", "true");
        if (multiGpu) {
            // Force the M-state split across all visible GPUs + log the multi-gpu line.
            System.setProperty("branchdp.dp.progress", "true");
            System.setProperty("branchdp.dp.gpu.minMStatesPerGpu", "1");
            System.setProperty("branchdp.dp.gpu.maxGpus", "0");
        } else {
            System.setProperty("branchdp.dp.progress", "false");
            System.setProperty("branchdp.dp.gpu.minMStatesPerGpu", String.valueOf(Long.MAX_VALUE));
            System.setProperty("branchdp.dp.gpu.maxGpus", "1");
        }
        RootedTreeEdge p = buildParent(cards, mPos, lambdaPos, childMPositions, shardedParent, shardSize);
        invoke(p, "ensureChildFoldPlans");
        boolean fired = (Boolean) invoke(p, "tryComputeFullDPGpu");
        Assumptions.assumeTrue(fired,
                "GPU full-DP path did not fire (no CUDA GPU on this node?) - skipping numeric comparison");
        return read(p, product(cards, mPos));
    }

    private double[][] runGpuOutOfCore(int[] cards, int[] mPos,
                                       int[] lambdaPos,
                                       int[][] childMPositions,
                                       boolean multiGpu) {
        System.setProperty("branchdp.dp.gpu", "true");
        System.setProperty("branchdp.dp.gpu.minWork", "1");
        System.setProperty("branchdp.dp.gpu.maxBytes",
                String.valueOf(1L << 20));
        System.setProperty("branchdp.dp.gpu.trace", "true");
        System.setProperty("branchdp.dp.gpu.outputTileMStates", "17");
        System.setProperty("branchdp.dp.gpu.persistentContext", "true");
        System.setProperty("branchdp.dp.gpu.childSlicing", "true");
        System.setProperty("branchdp.dp.gpu.childSlicing.force", "false");
        // At most four lower+upper child states. This forces both multi-child
        // row tiling and row-internal lambda tiling on the test shape.
        System.setProperty("branchdp.dp.gpu.childSliceMaxBytes", "64");
        System.setProperty("branchdp.dp.gpu.hybridChildTiling.force", "false");
        System.setProperty("branchdp.dp.gpu.outOfCore", "true");
        System.setProperty("branchdp.dp.gpu.outOfCore.force", "true");
        System.setProperty("branchdp.dp.gpu.outOfCore.budgetBytes",
                String.valueOf(1L << 20));
        System.setProperty("branchdp.dp.nativeKernel", "false");
        System.setProperty("branchdp.dp.parallel", "false");
        System.setProperty("branchdp.dp.foldChildren", "true");
        System.setProperty("branchdp.dp.progress", "true");
        if (multiGpu) {
            System.setProperty("branchdp.dp.gpu.minMStatesPerGpu", "1");
            System.setProperty("branchdp.dp.gpu.maxGpus", "2");
        } else {
            System.setProperty("branchdp.dp.gpu.minMStatesPerGpu",
                    String.valueOf(Long.MAX_VALUE));
            System.setProperty("branchdp.dp.gpu.maxGpus", "1");
        }
        RootedTreeEdge p = buildParent(cards, mPos, lambdaPos,
                childMPositions);
        invoke(p, "ensureChildFoldPlans");
        long work = product(cards, mPos) * product(cards, lambdaPos);
        DPGpuFullDP.Request plannedRequest = (DPGpuFullDP.Request)invoke(p,
                "buildGpuFullDPRequest", work);
        assertNotNull(plannedRequest);
        DPGpuOutOfCore.Plan forcedPlan = DPGpuOutOfCore.choosePlan(
                plannedRequest, 1L << 20);
        assertNotNull(forcedPlan);
        assertTrue(forcedPlan.multiChildRowTiling);
        assertTrue(forcedPlan.lambdaTiling);
        int requiredGpus = multiGpu ? 2 : 1;
        Assumptions.assumeTrue(visibleGpuCount() >= requiredGpus,
                "bounded out-of-core validation needs " + requiredGpus
                        + " visible CUDA GPU(s)");
        boolean fired = (Boolean)invoke(p, "tryComputeFullDPGpu");
        assertTrue(fired,
                "bounded out-of-core GPU path failed after CUDA was confirmed visible");
        return read(p, product(cards, mPos));
    }

    private void compareOutOfCoreShape(String name, boolean multiGpu) {
        int[] cards = {3, 4, 5, 6, 7};
        int[] mPos = {0, 1, 2};
        int[] lambdaPos = {3, 4};
        int[][] children = {{0, 3}, {2, 4}};
        double[][] gpu;
        try {
            gpu = runGpuOutOfCore(cards, mPos, lambdaPos, children,
                    multiGpu);
        } finally {
            System.clearProperty("branchdp.dp.gpu.outOfCore.force");
            System.clearProperty("branchdp.dp.gpu.outOfCore.budgetBytes");
        }
        double[][] jav = runJava(cards, mPos, lambdaPos, children);
        double maxRelLower = 0.0;
        double maxRelUpper = 0.0;
        for (int i = 0; i < jav[0].length; i++) {
            assertTrue(Double.isFinite(gpu[0][i]));
            assertTrue(Double.isFinite(gpu[1][i]));
            maxRelLower = Math.max(maxRelLower,
                    Math.abs(gpu[0][i] - jav[0][i])
                            / (1.0 + Math.abs(jav[0][i])));
            maxRelUpper = Math.max(maxRelUpper,
                    Math.abs(gpu[1][i] - jav[1][i])
                            / (1.0 + Math.abs(jav[1][i])));
        }
        System.out.println(String.format(Locale.ROOT,
                "[GPU-DP-OOC-VALIDATE] %s multiGpu=%s maxRelLower=%.3e maxRelUpper=%.3e",
                name, multiGpu, maxRelLower, maxRelUpper));
        assertTrue(maxRelLower <= REL_TOL,
                name + ": lower relative error " + maxRelLower);
        assertTrue(maxRelUpper <= REL_TOL,
                name + ": upper relative error " + maxRelUpper);
    }

    private void compareShape(String name, int[] cards, int[] mPos, int[] lambdaPos, int[][] childMPositions) {
        compareShape(name, cards, mPos, lambdaPos, childMPositions, false);
    }

    private void compareShape(String name, int[] cards, int[] mPos, int[] lambdaPos, int[][] childMPositions, boolean multiGpu) {
        compareShape(name, cards, mPos, lambdaPos, childMPositions, multiGpu, false, 1, Integer.MAX_VALUE);
    }

    private void compareShape(String name, int[] cards, int[] mPos, int[] lambdaPos, int[][] childMPositions,
                              boolean multiGpu, boolean shardedParent, int shardSize, long outputTileMStates) {
        compareShape(name, cards, mPos, lambdaPos, childMPositions, multiGpu,
                shardedParent, shardSize, outputTileMStates, false);
    }

    private void compareShape(String name, int[] cards, int[] mPos, int[] lambdaPos, int[][] childMPositions,
                              boolean multiGpu, boolean shardedParent, int shardSize,
                              long outputTileMStates, boolean forceChildSlicing) {
        double[][] gpu = runGpu(cards, mPos, lambdaPos, childMPositions, multiGpu,
                shardedParent, shardSize, outputTileMStates, forceChildSlicing);   // may skip if no GPU
        double[][] jav = runJava(cards, mPos, lambdaPos, childMPositions);
        int n = jav[0].length;

        double maxAbsLo = 0, maxRelLo = 0, maxAbsUp = 0, maxRelUp = 0;
        for (int i = 0; i < n; i++) {
            assertTrue(Double.isFinite(gpu[0][i]) && Double.isFinite(gpu[1][i]),
                    name + ": GPU produced non-finite bound at mIdx=" + i);
            double aLo = Math.abs(gpu[0][i] - jav[0][i]);
            double aUp = Math.abs(gpu[1][i] - jav[1][i]);
            maxAbsLo = Math.max(maxAbsLo, aLo);
            maxAbsUp = Math.max(maxAbsUp, aUp);
            maxRelLo = Math.max(maxRelLo, aLo / (1.0 + Math.abs(jav[0][i])));
            maxRelUp = Math.max(maxRelUp, aUp / (1.0 + Math.abs(jav[1][i])));
        }

        System.out.println(String.format(Locale.ROOT,
                "[GPU-DP-VALIDATE] %s children=%d mStates=%d lambdaStates=%d | lower maxAbs=%.3e maxRel=%.3e | upper maxAbs=%.3e maxRel=%.3e | java lo[0]=%.10f up[0]=%.10f gpu lo[0]=%.10f up[0]=%.10f",
                name, childMPositions.length, n, product(cards, lambdaPos),
                maxAbsLo, maxRelLo, maxAbsUp, maxRelUp,
                jav[0][0], jav[1][0], gpu[0][0], gpu[1][0]));

        // Check variation across the *whole* array, not just endpoints: for some
        // (mPos, lambdaPos, cards) shapes the linear synthetic emat is symmetric
        // enough that M-state 0 and M-state n-1 coincide by construction (more
        // likely for leaf edges, where there's no child-table term to break the
        // symmetry), which would make an endpoints-only check a false negative.
        double minLo = jav[0][0], maxLo = jav[0][0];
        for (int i = 1; i < n; i++) {
            minLo = Math.min(minLo, jav[0][i]);
            maxLo = Math.max(maxLo, jav[0][i]);
        }
        assertTrue(n > 1 && maxLo - minLo > 1e-9, name + ": Java bounds do not vary across M-states");
        assertTrue(maxRelLo <= REL_TOL, name + ": lower bound rel diff " + maxRelLo + " > " + REL_TOL);
        assertTrue(maxRelUp <= REL_TOL, name + ": upper bound rel diff " + maxRelUp + " > " + REL_TOL);
    }

    @Test
    public void gpuMatchesJava_threeChildren_mixed() {
        // Mirrors TestChildFold: one lambda-dependent child + two lambda-invariant.
        // {1,2}=M+lambda (dep), {0,1}=subset of parent M (invariant), {0}=invariant.
        compareShape("3child-mixed", new int[]{3, 3, 4, 4}, new int[]{0, 1}, new int[]{2, 3},
                new int[][]{{1, 2}, {0, 1}, {0}});
    }

    @Test
    public void gpuMatchesJava_shardedParent_outputTiling() {
        // Parent output is shard-backed and forced through multiple tiny GPU
        // output tiles; child tables remain dense, which is the current GPU gate.
        compareShape("sharded-parent-tiles", new int[]{5, 6, 7, 8}, new int[]{0, 1}, new int[]{2, 3},
                new int[][]{{1, 2}, {0, 3}}, false, true, 11, 7);
    }

    @Test
    public void gpuChildSlicing_matchesJava_twoChildrenDifferentProjections() {
        // Two children use different parent-M projections, so a single contiguous
        // child-table offset slice is not meaningful. The sliced path must pack
        // unique rows separately for each child and scatter parent output.
        compareShape("sliced-2child-different-projections", new int[]{3, 4, 5, 6, 7},
                new int[]{0, 1, 2}, new int[]{3, 4},
                new int[][]{{0, 3}, {2, 4}}, false, false, 1, 8, true);
    }

    @Test
    public void gpuHybridResidentStreamed_matchesJava_twoChildren() {
        System.setProperty("branchdp.dp.gpu.hybridChildTiling.force", "true");
        try {
            compareShape("hybrid-2child", new int[]{3, 4, 5, 6, 7},
                    new int[]{0, 1, 2}, new int[]{3, 4},
                    new int[][]{{0, 3}, {2, 4}}, false, false, 1, 8, false);
        } finally {
            System.clearProperty("branchdp.dp.gpu.hybridChildTiling.force");
        }
    }

    @Test
    public void gpuHybridResidentStreamed_multiGpuMatchesJava() {
        System.setProperty("branchdp.dp.gpu.hybridChildTiling.force", "true");
        try {
            compareShape("hybrid-2child-multigpu", new int[]{4, 5, 6, 7, 8},
                    new int[]{0, 1, 2}, new int[]{3, 4},
                    new int[][]{{0, 3}, {2, 4}}, true, false, 1, 11, false);
        } finally {
            System.clearProperty("branchdp.dp.gpu.hybridChildTiling.force");
        }
    }

    @Test
    public void gpuOutOfCore_bilateralRowAndLambdaTilingMatchesJava() {
        compareOutOfCoreShape("ooc-row-lambda", false);
    }

    @Test
    public void gpuOutOfCore_fileBackedChildrenAndOutputMatchJava() {
        Assumptions.assumeTrue(visibleGpuCount() >= 1,
                "file-backed OOC validation needs one visible CUDA GPU");

        int[] cards = {3, 4, 5, 6, 7};
        int[] mPos = {0, 1, 2};
        int[] lambdaPos = {3, 4};
        int[][] childMPositions = {{0, 3}, {2, 4}};
        double[][] jav = runJava(cards, mPos, lambdaPos, childMPositions);

        System.setProperty("branchdp.dp.gpu", "true");
        System.setProperty("branchdp.dp.gpu.minWork", "1");
        System.setProperty("branchdp.dp.gpu.maxBytes", String.valueOf(1L << 20));
        System.setProperty("branchdp.dp.gpu.outputTileMStates", "17");
        System.setProperty("branchdp.dp.gpu.persistentContext", "true");
        System.setProperty("branchdp.dp.gpu.childSlicing", "true");
        System.setProperty("branchdp.dp.gpu.childSlicing.force", "false");
        System.setProperty("branchdp.dp.gpu.childSliceMaxBytes", "64");
        System.setProperty("branchdp.dp.gpu.outOfCore", "true");
        System.setProperty("branchdp.dp.gpu.outOfCore.force", "false");
        System.setProperty("branchdp.dp.gpu.outOfCore.budgetBytes", String.valueOf(1L << 20));
        System.setProperty("branchdp.dp.nativeKernel", "false");
        System.setProperty("branchdp.dp.parallel", "false");
        System.setProperty("branchdp.dp.foldChildren", "true");
        System.setProperty("branchdp.dp.progress", "true");
        System.setProperty("branchdp.dp.gpu.minMStatesPerGpu", String.valueOf(Long.MAX_VALUE));
        System.setProperty("branchdp.dp.gpu.maxGpus", "1");

        RootedTreeEdge parent = buildParent(cards, mPos, lambdaPos, childMPositions);
        List<RootedTreeEdge> children = new ArrayList<>(parent.getFset());
        List<Path> paths = new ArrayList<>();
        try {
            replaceWithMappedTable(parent, 7, paths);
            for (RootedTreeEdge child : children) {
                replaceWithMappedTable(child, 11, paths);
            }
            invoke(parent, "ensureChildFoldPlans");
            long work = product(cards, mPos) * product(cards, lambdaPos);
            DPGpuFullDP.Request request = (DPGpuFullDP.Request)invoke(
                    parent, "buildGpuFullDPRequest", work);
            assertNotNull(request);
            assertTrue(request.outTable.isFileBacked());
            assertTrue(request.hasFileBackedChildTables);
            assertEquals(0, request.childLowerChunks.length,
                    "file-backed children must not be materialized as Java arrays");

            boolean fired = (Boolean)invoke(parent, "tryComputeFullDPGpu");
            assertTrue(fired);
            double[][] gpu = read(parent, product(cards, mPos));
            for (int bound = 0; bound < 2; bound++) {
                for (int i = 0; i < jav[bound].length; i++) {
                    double rel = Math.abs(gpu[bound][i] - jav[bound][i])
                            / (1.0 + Math.abs(jav[bound][i]));
                    assertTrue(rel <= REL_TOL,
                            "file-backed OOC mismatch bound=" + bound + " mIdx=" + i
                                    + " rel=" + rel);
                }
            }
        } finally {
            for (RootedTreeEdge child : children) child.releaseLargeMemory();
            parent.releaseLargeMemory();
            for (Path path : paths) {
                assertFalse(Files.exists(path), "mapped DP file leaked: " + path);
            }
        }
    }

    @Test
    public void childSliceGatherMatchesOriginalMixedRadixIndices_withoutGpu() {
        int[] cards = {3, 4, 5, 6, 7};
        RootedTreeEdge parent = buildParent(cards, new int[]{0, 1, 2}, new int[]{3, 4},
                new int[][]{{0, 3}, {2, 4}});
        invoke(parent, "ensureChildFoldPlans");
        long work = product(cards, new int[]{0, 1, 2}) * product(cards, new int[]{3, 4});
        DPGpuFullDP.Request req = (DPGpuFullDP.Request)invoke(parent, "buildGpuFullDPRequest", work);
        assertNotNull(req);

        Object slice = buildChildSlice(req, 1L, 7);
        long[] childPackedBase = (long[])field(slice, "childPackedBase");
        long[] childRowKeyBase = (long[])field(slice, "childRowKeyBase");
        int[] childRowKeyCount = (int[])field(slice, "childRowKeyCount");
        long[] childRowKeysAll = (long[])field(slice, "childRowKeysAll");
        double[] lowerPacked = (double[])field(slice, "lowerPacked");
        double[] upperPacked = (double[])field(slice, "upperPacked");

        assertNotNull(req.childLambdaOriginalOffsets);
        for (int c = 0; c < req.numChildren; c++) {
            for (int ri = 0; ri < childRowKeyCount[c]; ri++) {
                long rowKey = childRowKeysAll[(int)childRowKeyBase[c] + ri];
                for (long lambdaKey = 0; lambdaKey < req.childLambdaStates[c]; lambdaKey++) {
                    long original = originalChildIndex(req, c, rowKey, lambdaKey);
                    int packed = (int)(childPackedBase[c]
                            + (long)ri * req.childLambdaStates[c] + lambdaKey);
                    assertEquals(req.childTables[c].lower(original), lowerPacked[packed], 0.0);
                    assertEquals(req.childTables[c].upper(original), upperPacked[packed], 0.0);
                }
            }
        }
    }

    @Test
    public void hybridPlannerGatherAndMEnumeration_withoutGpu() {
        int[] cards = {3, 4, 5, 6, 7};
        RootedTreeEdge parent = buildParent(cards, new int[]{0, 1, 2}, new int[]{3, 4},
                new int[][]{{0, 3}, {2, 4}});
        invoke(parent, "ensureChildFoldPlans");
        long work = product(cards, new int[]{0, 1, 2}) * product(cards, new int[]{3, 4});
        DPGpuFullDP.Request req = (DPGpuFullDP.Request)invoke(parent, "buildGpuFullDPRequest", work);
        assertNotNull(req);
        req.childSlicing = true;
        req.forceChildSlicing = false;
        req.childSliceMaxBytes = 1L << 20;
        System.setProperty("branchdp.dp.gpu.hybridChildTiling", "true");

        DPGpuFullDP.HybridPlan plan = DPGpuFullDP.chooseHybridPlan(req, 1L << 30, 8);
        assertNotNull(plan);
        assertEquals(1, plan.streamedChild,
                "the larger child should stream while the smaller child is replicated");
        assertEquals(5L, plan.streamedRowCount);
        assertEquals(12L, plan.otherMStateCount);

        DPGpuFullDP.StreamedChildBlock block = DPGpuFullDP.buildStreamedChildBlock(
                req, plan, 1L, 3);
        int child = plan.streamedChild;
        for (int row = 0; row < block.rowCount; row++) {
            long rowKey = block.rowStart + row;
            for (long lambdaKey = 0; lambdaKey < req.childLambdaStates[child]; lambdaKey++) {
                long original = originalChildIndex(req, child, rowKey, lambdaKey);
                int packed = (int)((long)row * req.childLambdaStates[child] + lambdaKey);
                assertEquals(req.childTables[child].lower(original), block.lowerPacked[packed], 0.0);
                assertEquals(req.childTables[child].upper(original), block.upperPacked[packed], 0.0);
            }
        }

        DPGpuFullDP.StreamedChildMIdxEnumerator enumerator =
                new DPGpuFullDP.StreamedChildMIdxEnumerator(req, child, 0L,
                        (int)plan.streamedRowCount);
        assertEquals(req.mStateCount, enumerator.outputCount());
        long[] mIndices = new long[(int)req.mStateCount];
        assertEquals(mIndices.length, enumerator.fill(mIndices, mIndices.length));
        Arrays.sort(mIndices);
        for (int i = 0; i < mIndices.length; i++) {
            assertEquals(i, mIndices[i]);
        }
    }

    @Test
    public void minimumChildSlicePreflightDetectsLambdaTilingRequirement_withoutGpu() {
        int[] cards = {3, 4, 5, 6, 7};
        RootedTreeEdge parent = buildParent(cards, new int[]{0, 1, 2}, new int[]{3, 4},
                new int[][]{{0, 3}, {2, 4}});
        invoke(parent, "ensureChildFoldPlans");
        long work = product(cards, new int[]{0, 1, 2}) * product(cards, new int[]{3, 4});
        DPGpuFullDP.Request req = (DPGpuFullDP.Request)invoke(parent,
                "buildGpuFullDPRequest", work);
        assertNotNull(req);
        req.childSlicing = true;

        // Model the 3k3q failure class without allocating its tables: even one row
        // from each child carries more than 30 GiB of complete lambda data.
        req.childLambdaStates = new long[]{1_100_000_000L, 950_000_000L};
        long minimumChildBytes = DPGpuFullDP.estimateMinimumChildSliceBytes(req);
        long minimumDeviceBytes = DPGpuFullDP.estimateMinimumSlicedDeviceBytes(req);
        assertTrue(minimumChildBytes > 30L * 1024L * 1024L * 1024L);
        assertTrue(minimumDeviceBytes >= minimumChildBytes);
        assertTrue(minimumDeviceBytes > 21_500_000_000L,
                "row-only slicing must reject this shape before allocating on an A5000");
        assertTrue(minimumDeviceBytes > 12_000_000_000L,
                "the same preflight must reject it before allocating on a Titan V");
    }

    @Test
    public void outOfCorePlannerAndPackingBoundBothRowAndLambdaAxes_withoutGpu() {
        System.setProperty("branchdp.dp.foldChildren", "true");
        int[] cards = {3, 4, 5, 6, 7};
        RootedTreeEdge parent = buildParent(cards, new int[]{0, 1, 2},
                new int[]{3, 4}, new int[][]{{0, 3}, {2, 4}});
        invoke(parent, "ensureChildFoldPlans");
        long work = product(cards, new int[]{0, 1, 2})
                * product(cards, new int[]{3, 4});
        DPGpuFullDP.Request req = (DPGpuFullDP.Request)invoke(parent,
                "buildGpuFullDPRequest", work);
        assertNotNull(req);

        // Two children need at least two packed states (one row x one lambda
        // state each).  A 64-byte lower+upper workspace permits at most four
        // states total, so neither all child-row projections nor all lambda
        // projections can remain whole.  The pure planner must tile both axes.
        req.childSlicing = true;
        req.childSliceMaxBytes = 64L;
        // The legacy full-DP kernel chunk must not force repeated OOC child
        // gathers.  OOC has its own byte-bounded output workspace.
        req.mStateChunk = 1L;
        long budget = 1L << 20;
        long minimumBudget = DPGpuOutOfCore.estimateMinimumDeviceBytes(req);
        assertNotEquals(Long.MAX_VALUE, minimumBudget);
        assertNull(DPGpuOutOfCore.choosePlan(req, minimumBudget - 1L),
                "one byte below the proven minimum must be rejected");
        assertNotNull(DPGpuOutOfCore.choosePlan(req, minimumBudget),
                "the exact one-state minimum must be feasible");
        DPGpuOutOfCore.Plan plan = DPGpuOutOfCore.choosePlan(req, budget);
        assertNotNull(plan);
        assertTrue(plan.outputStatesPerTile > req.mStateChunk,
                "OOC output capacity must be independent of the legacy full-DP chunk");
        DPGpuOutOfCore.PlanningInput planningInput =
                DPGpuOutOfCore.PlanningInput.fromRequest(req);
        assertNotNull(planningInput);
        DPGpuOutOfCore.Plan shapeOnlyPlan = DPGpuOutOfCore.choosePlan(
                planningInput, budget);
        assertNotNull(shapeOnlyPlan);
        assertArrayEquals(plan.mTileExtents, shapeOnlyPlan.mTileExtents);
        assertArrayEquals(plan.lambdaTileExtents,
                shapeOnlyPlan.lambdaTileExtents);
        assertEquals(plan.estimatedDeviceBytes,
                shapeOnlyPlan.estimatedDeviceBytes);
        BigInteger estimatedTraffic = DPGpuOutOfCore.estimateTrafficBytes(
                planningInput, shapeOnlyPlan);
        assertNotNull(estimatedTraffic);
        assertTrue(estimatedTraffic.signum() > 0);
        assertTrue(plan.multiChildRowTiling,
                "the bounded plan must tile multiple child row projections");
        assertTrue(plan.lambdaTiling,
                "the bounded plan must tile row-internal lambda projections");
        assertTrue(plan.estimatedDeviceBytes <= budget);
        assertTrue(plan.maxChildPackedStates * 2L * Double.BYTES
                        <= req.childSliceMaxBytes,
                "planned packed child workspace must honor its hard cap");

        int[] mCoverage = new int[(int)req.mStateCount];
        for (long mBoxOrdinal = 0; mBoxOrdinal < plan.mBoxCount;
             mBoxOrdinal++) {
            DPGpuOutOfCore.MBox mBox = DPGpuOutOfCore.buildMBox(
                    req, plan, mBoxOrdinal);
            assertTrue(mBox.tile.volume <= plan.maxMBoxStates);
            int freePerBlock = (int)Math.max(1L,
                    plan.outputStatesPerTile / mBox.tile.volume);
            for (long freeStart = 0L; freeStart < req.parentFreeStateCount;
                 freeStart += freePerBlock) {
                int freeCount = (int)Math.min((long)freePerBlock,
                        req.parentFreeStateCount - freeStart);
                DPGpuOutOfCore.OutputBlock output =
                        DPGpuOutOfCore.buildOutputBlock(req, mBox,
                                freeStart, freeCount);
                assertTrue(output.mIndices.length <= plan.outputStatesPerTile);
                for (long mIndex : output.mIndices) {
                    assertTrue(mIndex >= 0L && mIndex < req.mStateCount);
                    mCoverage[(int)mIndex]++;
                }
            }
        }
        for (int mIndex = 0; mIndex < mCoverage.length; mIndex++) {
            assertEquals(1, mCoverage[mIndex],
                    "M tiling must cover output state " + mIndex
                            + " exactly once");
        }

        int[] lambdaCoverage = new int[req.totalLambdaStates];
        DPGpuOutOfCore.LambdaBox[] lambdaBoxes =
                new DPGpuOutOfCore.LambdaBox[(int)plan.lambdaBoxCount];
        for (long lambdaBoxOrdinal = 0;
             lambdaBoxOrdinal < plan.lambdaBoxCount; lambdaBoxOrdinal++) {
            DPGpuOutOfCore.LambdaBox lambdaBox =
                    DPGpuOutOfCore.buildLambdaBox(req, plan,
                            lambdaBoxOrdinal);
            lambdaBoxes[(int)lambdaBoxOrdinal] = lambdaBox;
            assertTrue(lambdaBox.tile.volume <= plan.maxLambdaTileStates);
            for (int lambdaIndex : lambdaBox.lambdaIndices) {
                assertTrue(lambdaIndex >= 0
                        && lambdaIndex < req.totalLambdaStates);
                lambdaCoverage[lambdaIndex]++;
            }
        }
        for (int lambdaIndex = 0; lambdaIndex < lambdaCoverage.length;
             lambdaIndex++) {
            assertEquals(1, lambdaCoverage[lambdaIndex],
                    "lambda tiling must cover state " + lambdaIndex
                            + " exactly once");
        }

        // Exhaust every rectangular M x lambda tile and compare each packed
        // lower/upper value to an independently decoded original-table index.
        for (long mBoxOrdinal = 0; mBoxOrdinal < plan.mBoxCount;
             mBoxOrdinal++) {
            DPGpuOutOfCore.MBox mBox = DPGpuOutOfCore.buildMBox(
                    req, plan, mBoxOrdinal);
            for (DPGpuOutOfCore.LambdaBox lambdaBox : lambdaBoxes) {
                DPGpuOutOfCore.PackedChildBlock packed =
                        DPGpuOutOfCore.buildPackedChildBlock(req, mBox,
                                lambdaBox);
                assertTrue(packed.packedStates
                        <= plan.maxChildPackedStates);
                assertTrue(packed.childRowKeysAll.length
                        <= plan.maxChildRowKeys);
                assertTrue(packed.childLambdaKeysAll.length
                        <= plan.maxChildLambdaKeys);
                for (int c = 0; c < req.numChildren; c++) {
                    int rows = packed.childRowKeyCount[c];
                    int lambdas = packed.childLambdaKeyCount[c];
                    for (int row = 0; row < rows; row++) {
                        long rowKey = packed.childRowKeysAll[
                                (int)packed.childRowKeyBase[c] + row];
                        for (int lambda = 0; lambda < lambdas; lambda++) {
                            long lambdaKey = packed.childLambdaKeysAll[
                                    (int)packed.childLambdaKeyBase[c]
                                            + lambda];
                            long original = originalChildIndex(req, c,
                                    rowKey, lambdaKey);
                            int destination = (int)(packed.childPackedBase[c]
                                    + (long)row * lambdas + lambda);
                            assertEquals(req.childTables[c].lower(original),
                                    packed.lowerPacked[destination], 0.0);
                            assertEquals(req.childTables[c].upper(original),
                                    packed.upperPacked[destination], 0.0);
                        }
                    }
                }
            }
        }
    }

    @Test
    public void outOfCoreStablePartialMergeHandlesExtremeNanAndNegativeInfinity_withoutGpu() {
        double[] values = {
                10_000.0, 9_999.75, -10_000.0,
                Double.NEGATIVE_INFINITY, 9_998.5, -1_000_000.0,
                9_999.125
        };
        double max = Arrays.stream(values).max().orElseThrow();
        double scaled = 0.0;
        for (double value : values) {
            scaled += Math.exp(value - max);
        }
        double expected = max + Math.log(scaled);

        for (int tileWidth = 1; tileWidth <= values.length; tileWidth++) {
            DPGpuOutOfCore.LogSumExpPartial merged =
                    new DPGpuOutOfCore.LogSumExpPartial();
            for (int start = 0; start < values.length; start += tileWidth) {
                DPGpuOutOfCore.LogSumExpPartial tile =
                        new DPGpuOutOfCore.LogSumExpPartial();
                int end = Math.min(values.length, start + tileWidth);
                for (int i = start; i < end; i++) {
                    tile.add(values[i]);
                }
                merged.merge(tile);
            }
            assertEquals(expected, merged.finish(), 1e-12,
                    "stable result must not depend on lambda tile width");
        }

        DPGpuOutOfCore.LogSumExpPartial allNegativeInfinity =
                new DPGpuOutOfCore.LogSumExpPartial();
        allNegativeInfinity.add(Double.NEGATIVE_INFINITY);
        allNegativeInfinity.add(Double.NEGATIVE_INFINITY);
        assertEquals(Double.NEGATIVE_INFINITY,
                allNegativeInfinity.finish());

        DPGpuOutOfCore.LogSumExpPartial finite =
                new DPGpuOutOfCore.LogSumExpPartial();
        finite.add(17.0);
        DPGpuOutOfCore.LogSumExpPartial nan =
                new DPGpuOutOfCore.LogSumExpPartial();
        nan.add(Double.NaN);
        finite.merge(nan);
        assertTrue(Double.isNaN(finite.finish()),
                "a NaN in any lambda tile must poison the output");
    }

    @Test
    public void gpuOutOfCoreExtremeNanNegativeInfinityAndAllocationAudit() {
        Assumptions.assumeTrue(visibleGpuCount() >= 1,
                "bounded out-of-core extreme-value validation needs a CUDA GPU");

        System.setProperty("branchdp.dp.gpu.childSlicing", "true");
        System.setProperty("branchdp.dp.gpu.childSlicing.force", "false");
        System.setProperty("branchdp.dp.gpu.outOfCore", "true");
        System.setProperty("branchdp.dp.gpu.outOfCore.force", "true");
        System.setProperty("branchdp.dp.gpu.outputTileMStates", "2");
        System.setProperty("branchdp.dp.gpu.persistentContext", "true");
        System.setProperty("branchdp.dp.gpu.trace", "true");

        try {
            int[] cards = {2, 7};
            RootedTreeEdge parent = buildParent(cards, new int[]{0},
                    new int[]{1}, new int[][]{});
            invoke(parent, "ensureChildFoldPlans");
            long work = product(cards, new int[]{0})
                    * product(cards, new int[]{1});
            DPGpuFullDP.Request req = (DPGpuFullDP.Request)invoke(parent,
                    "buildGpuFullDPRequest", work);
            assertNotNull(req);

            // Make the CUDA result easy to derive independently: there are no
            // children, every lower lm term is zero, and invRT=1.  The desired
            // lower log terms span a range that would overflow/underflow an
            // unstabilized exp-sum and include an explicit -Infinity term.
            double[] lowerTerms = {
                    10_000.0, 9_999.75, -10_000.0,
                    Double.NEGATIVE_INFINITY, 9_998.5, -1_000_000.0,
                    9_999.125
            };
            req.invRT = 1.0;
            Arrays.fill(req.lmRigid, 0.0);
            Arrays.fill(req.lmMin, 0.0);
            for (int i = 0; i < lowerTerms.length; i++) {
                req.lambdaOnlyRigid[i] = -lowerTerms[i];
            }

            // Upper M-state 0 is all -Infinity.  M-state 1 is also all
            // -Infinity until a NaN in the final lambda state; placing the NaN
            // there proves poisoning survives a later-tile partial merge.
            Arrays.fill(req.lambdaOnlyMin, Double.POSITIVE_INFINITY);
            assertEquals(1, req.lmOffsets.length);
            int lastLambda = cards[1] - 1;
            int nanLmIndex = (int)req.lmOffsets[0]
                    + lastLambda * req.mCounts[0] + 1;
            req.lmMin[nanLmIndex] = Double.NaN;

            long budget = DPGpuOutOfCore.estimateMinimumDeviceBytes(req);
            assertNotEquals(Long.MAX_VALUE, budget);
            DPGpuOutOfCore.Plan plan = DPGpuOutOfCore.choosePlan(req, budget);
            assertNotNull(plan);
            assertTrue(plan.lambdaBoxCount > 1,
                    "extreme values must cross multiple CUDA lambda tiles");
            assertEquals(budget, plan.estimatedDeviceBytes,
                    "the exact-minimum plan should leave no unaudited slack");

            req.maxDeviceBytes = budget;
            req.progress = true;
            req.multiGpu = false;
            req.maxGpus = 1;
            System.setProperty("branchdp.dp.gpu.outOfCore.budgetBytes",
                    String.valueOf(budget));

            boolean fired = DPGpuFullDP.compute(req);
            assertTrue(fired,
                    "bounded out-of-core CUDA path failed after a GPU was confirmed visible");

            DPGpuOutOfCore.LogSumExpPartial expectedLower =
                    new DPGpuOutOfCore.LogSumExpPartial();
            for (double term : lowerTerms) {
                expectedLower.add(term);
            }
            double expected = expectedLower.finish();
            assertEquals(expected, req.outTable.lower(0), 1e-12);
            assertEquals(expected, req.outTable.lower(1), 1e-12);
            assertEquals(Double.NEGATIVE_INFINITY, req.outTable.upper(0));
            assertTrue(Double.isNaN(req.outTable.upper(1)),
                    "a NaN in the final lambda tile must poison the merged output");

            assertEquals(1, req.outOfCoreAllocationAuditCount);
            assertEquals(plan.estimatedDeviceBytes,
                    req.outOfCoreAuditedDeviceBytes);
            assertEquals(plan.estimatedDeviceBytes,
                    req.outOfCorePlannedDeviceBytes);
            assertEquals(budget, req.outOfCoreBudgetBytes);
            assertTrue(req.outOfCoreLargestAllocationBytes > 0L);
            assertTrue(req.outOfCoreLargestAllocationBytes
                            <= req.outOfCoreAuditedDeviceBytes,
                    "no runtime device allocation may sit outside the planned workspace");
        } finally {
            System.clearProperty("branchdp.dp.gpu.outOfCore.force");
            System.clearProperty("branchdp.dp.gpu.outOfCore.budgetBytes");
        }
    }

    // ---- single-node multi-GPU M-state split (requires >=2 visible GPUs) ----

    @Test
    public void gpuMultiGpu_matchesJava_threeChildren() {
        compareShape("multigpu-3child", new int[]{10, 10, 8, 8}, new int[]{0, 1}, new int[]{2, 3},
                new int[][]{{1, 2}, {0, 1}, {0}}, true);
    }

    // ---- leaf edges (no F-set children): GPU gate now fires for numChildren=0 ----
    // child_sum()/child_sum_sliced() already sum 0 children as an empty (zero) sum,
    // so a leaf edge is just local_energy() reduced over lambda per M-state -- the
    // same math as the CPU leaf path (computeFullDPForMState's !hasFsetChildren()
    // branch). These tests exercise tryComputeFullDPGpu()/buildGpuFullDPRequest()
    // with childFoldPlans == null (empty childMPositions => empty Fset => leaf).

    @Test
    public void gpuMatchesJava_leafEdge_noChildren_shardedParentTiles() {
        // Same as gpuMatchesJava_shardedParent_outputTiling but with zero children:
        // huge-mStateCount leaf edges (like the 3bua/3k3q root-split leaf) rely on
        // shard-backed output + small GPU output tiles, not child-table slicing.
        compareShape("leaf-noChildren-sharded-tiles", new int[]{5, 6, 7, 8}, new int[]{0, 1}, new int[]{2, 3},
                new int[][]{}, false, true, 11, 7);
    }

}
