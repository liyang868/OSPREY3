package edu.duke.cs.osprey.branchdp;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import edu.duke.cs.osprey.astar.conf.RCs;
import edu.duke.cs.osprey.ematrix.EnergyMatrix;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.nio.file.Path;
import java.util.LinkedHashSet;

/**
 * End-to-end dense-vs-sharded equivalence for RootedTreeEdge.computeFullDP
 * (design 2.1-2.5). TestDPTable covers the DPTable container in isolation and
 * explicitly leaves "the full DP run through a sharded table" to an
 * integration-level test; this is that test (the remaining [TODO] in the doc:
 * "computeFullDP end-to-end dense-vs-sharded equivalence").
 *
 * It runs the real computeFullDP() on a small synthetic non-leaf edge whose
 * M-state count (12) spans multiple shards, with the DP table injected as either
 * a DenseDPTable or a ShardedDPTable (small shard size -> routing crosses shard
 * boundaries). The per-M-state arithmetic is identical across backings, so
 * results must be BIT-IDENTICAL; any difference is a sharded read/write routing
 * bug. Also exercises the parallel shard scheduler (computeFullDPSharded)
 * writing into a ShardedDPTable.
 */
public class TestComputeFullDPSharded {

    @TempDir
    Path tempDir;

    // ---- reflection / construction helpers (mirror TestChildFold) ----------

    private static void set(Object o, String field, Object val) {
        try {
            Field f = RootedTreeEdge.class.getDeclaredField(field);
            f.setAccessible(true);
            f.set(o, val);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException("set " + field, e);
        }
    }

    private static RCs rcsFor(int[] positions, int[] cardinalities) {
        int maxPos = 0;
        for (int p : positions) maxPos = Math.max(maxPos, p);
        int[][] arr = new int[maxPos + 1][];
        for (int i = 0; i < arr.length; i++) arr[i] = new int[1];
        for (int i = 0; i < positions.length; i++) arr[positions[i]] = new int[cardinalities[i]];
        return new RCs(arr);
    }

    private static EnergyMatrix ematFilled(int numPos, int[] cards, double seed) {
        EnergyMatrix e = new EnergyMatrix(numPos, cards, 0.0);
        for (int p = 0; p < numPos; p++)
            for (int rc = 0; rc < cards[p]; rc++)
                e.setOneBody(p, rc, seed + 0.13 * p - 0.07 * rc);
        for (int pi = 0; pi < numPos; pi++)
            for (int pj = pi + 1; pj < numPos; pj++)
                for (int ri = 0; ri < cards[pi]; ri++)
                    for (int rj = 0; rj < cards[pj]; rj++)
                        e.setPairwise(pi, ri, pj, rj,
                                0.31 * seed + 0.05 * (ri + 1) * (rj + 2) - 0.02 * (pi + pj));
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

    private static long stateCount(RCs rcs, int[] pos) {
        long c = 1;
        for (int p : pos) c *= rcs.getNum(p);
        return c;
    }

    /** A child lambda-edge with given M positions and a deterministic dense DP table. */
    private static RootedTreeEdge child(RCs rcs, int[] mPos, double base) {
        RootedTreeEdge c = new RootedTreeEdge(null, null, new LinkedHashSet<>(), false, rcs);
        set(c, "mPositionsSorted", mPos.clone());
        set(c, "isLambdaEdge", Boolean.TRUE);
        long mc = stateCount(rcs, mPos);
        set(c, "mStateCount", Long.valueOf(mc));
        set(c, "mArraySize", Integer.valueOf((int) mc));
        DenseDPTable t = new DenseDPTable(mc);
        for (long i = 0; i < mc; i++) t.set(i, base + i, base + 100.0 + i);
        set(c, "dpTable", t);
        return c;
    }

    /** Build a fresh parent edge + 3 children, inject the given DP table, run computeFullDP. */
    private double[][] runDP(DPTable table, boolean parallel) {
        System.setProperty("branchdp.dp.foldChildren", "true");
        System.setProperty("branchdp.dp.foldChildren.hoistInvariant", "false");
        System.setProperty("branchdp.dp.nativeKernel", "false");
        System.setProperty("branchdp.dp.progress", "false");
        System.setProperty("branchdp.dp.parallel", String.valueOf(parallel));
        System.setProperty("branchdp.dp.parallel.minMStates", "1");
        System.setProperty("branchdp.dp.computeShardSize", "3");

        int numPos = 4;
        int[] cards = {4, 3, 3, 2};      // pos0..3 cardinalities
        int[] mPos = {0, 1};             // mStateCount = 4*3 = 12 -> spans shards
        int[] lambdaPos = {2, 3};        // 3*2 = 6 lambda states
        RCs rcs = rcsFor(new int[]{0, 1, 2, 3}, cards);

        LinkedHashSet<RootedTreeEdge> fset = new LinkedHashSet<>();
        fset.add(child(rcs, new int[]{1, 2}, 1000.0)); // lambda-dependent
        fset.add(child(rcs, new int[]{0, 1}, 2000.0)); // lambda-invariant (subset of M)
        fset.add(child(rcs, new int[]{0},    3000.0)); // lambda-invariant

        RootedTreeEdge parent = new RootedTreeEdge(null, null, new LinkedHashSet<>(), false, rcs);
        set(parent, "isLambdaEdge", Boolean.TRUE);
        set(parent, "mPositionsSorted", mPos.clone());
        set(parent, "lambdaPositionsSorted", lambdaPos.clone());
        long mc = stateCount(rcs, mPos);
        set(parent, "mStateCount", Long.valueOf(mc));
        set(parent, "mArraySize", Integer.valueOf((int) mc));
        set(parent, "totalLambdaStates", Integer.valueOf((int) stateCount(rcs, lambdaPos)));
        set(parent, "dpTable", table);
        set(parent, "Fset", fset);
        parent.initIncrementalEnumeration(ematFilled(numPos, cards, 1.0),
                ematFilled(numPos, cards, 2.0), fullyConnected(numPos), 1.9);

        parent.computeFullDP();

        int n = (int) mc;
        double[] lo = new double[n];
        double[] up = new double[n];
        for (int i = 0; i < n; i++) { lo[i] = parent.getLogZLower(i); up[i] = parent.getLogZUpper(i); }
        return new double[][]{lo, up};
    }

    private static void assertBitIdentical(double[][] a, double[][] b, String label) {
        assertEquals(a[0].length, b[0].length, label + " length");
        boolean anyFinite = false;
        for (int i = 0; i < a[0].length; i++) {
            assertEquals(Double.doubleToLongBits(a[0][i]), Double.doubleToLongBits(b[0][i]),
                    label + " lower bit-identity at mIdx=" + i);
            assertEquals(Double.doubleToLongBits(a[1][i]), Double.doubleToLongBits(b[1][i]),
                    label + " upper bit-identity at mIdx=" + i);
            if (Double.isFinite(a[0][i])) anyFinite = true;
        }
        assertTrue(anyFinite, label + ": expected some finite DP values (test is non-trivial)");
    }

    @Test
    public void shardedMatchesDense_sequential_bitIdentical() {
        double[][] dense = runDP(new DenseDPTable(12), false);
        double[][] sharded = runDP(new ShardedDPTable(12, 5), false); // shards (5,5,2)
        assertBitIdentical(dense, sharded, "sequential");
    }

    @Test
    public void shardedMatchesDense_parallelScheduler_bitIdentical() {
        double[][] dense = runDP(new DenseDPTable(12), false);
        double[][] sharded = runDP(new ShardedDPTable(12, 4), true); // parallel + shards (4,4,4)
        assertBitIdentical(dense, sharded, "parallel sharded");
    }

    @Test
    public void singleShardShardedMatchesDense_bitIdentical() {
        double[][] dense = runDP(new DenseDPTable(12), false);
        double[][] sharded = runDP(new ShardedDPTable(12, 1000), false); // one shard
        assertBitIdentical(dense, sharded, "single shard");
    }

    @Test
    public void mappedMatchesDenseAcrossMapBoundaries_bitIdentical() {
        double[][] dense = runDP(new DenseDPTable(12), false);
        MappedDPTable mapped = new MappedDPTable(12, 4, tempDir);
        try {
            double[][] actual = runDP(mapped, true);
            assertBitIdentical(dense, actual, "mapped parallel output");
        } finally {
            mapped.close();
        }
    }
}
