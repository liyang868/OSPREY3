/*
** This file is part of OSPREY 3.0
**
** OSPREY Protein Redesign Software Version 3.0
** Copyright (C) 2001-2018 Bruce Donald Lab, Duke University
**
** OSPREY is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License version 2
** as published by the Free Software Foundation.
**
** You should have received a copy of the GNU General Public License
** along with OSPREY.  If not, see <http://www.gnu.org/licenses/>.
*/

package edu.duke.cs.osprey.branchdp;

import edu.duke.cs.osprey.astar.conf.RCs;
import edu.duke.cs.osprey.confspace.HigherTupleFinder;
import edu.duke.cs.osprey.confspace.RCTuple;
import edu.duke.cs.osprey.ematrix.EnergyMatrix;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.LongStream;

/**
 * An edge in the rooted branch decomposition tree with incremental partition function DP.
 *
 * Key design: lambda-states are pre-sorted by energy per M-state. Enumeration is incremental:
 *   - logZ_lower[m] = logSumExp of enumerated rigid-energy terms (valid: omitting positive terms)
 *   - logZ_upper[m] = logSumExp of enumerated min-energy terms + tail bound for remaining
 *
 * "Expanding a node" = enumerating the next lambda-state for a given (edge, mIdx).
 * This is analogous to original MARK*'s A* node expansion with H-score bounds.
 *
 * For leaf edges (no F-set): lambda-state energies come purely from emat, pre-sorted once.
 * For non-leaf edges (has F-set): re-enumeration needed when children change (uses looser tail bound).
 */
public class RootedTreeEdge {

    // ========== Tree structure ==========

    private RootedTreeNode parent;
    private RootedTreeNode child;
    private boolean isRootEdge;
    private boolean isLambdaEdge;

    // ========== Set structure ==========

    private LinkedHashSet<Integer> M;
    private LinkedHashSet<Integer> lambda;
    private LinkedHashSet<Integer> L;

    // F-set: nearest lambda-edge descendants
    private LinkedHashSet<RootedTreeEdge> Fset;

    // Compact tree pointers
    private RootedTreeNode compactLeftChild;
    private RootedTreeNode compactRightChild;

    // ========== M-state indexing ==========

    private int[] mPositionsSorted;
    private int[] lambdaPositionsSorted;
    private int[] mCounts;
    private int[] lambdaCounts;
    private long[] mStrides;
    private long[] lambdaStrides;
    /** Pair slots wholly inside lambda and crossing lambda/M, cached per graph. */
    private int[] lambdaPairFirstSlots;
    private int[] lambdaPairSecondSlots;
    private int[] lambdaMFirstSlots;
    private int[] lambdaMSecondSlots;
    private InteractionGraph interactionTermCacheGraph;
    /** GPU pair-term descriptors: [lambdaSlot, mSlot, lambdaPos, mPos, lambdaCount, mCount]. */
    private int[][] gpuLambdaMTermDescriptors;
    private InteractionGraph gpuLambdaMTermGraph;
    private RCs rcs;
    private long mStateCount;
    private int mArraySize;

    // ========== Log-space partition function bounds per M-state ==========

    private DPTable dpTable;      // log(Z_lower/upper) indexed by M-state

    // Option C (non-leaf DP kernel): precomputed child-fold plans. Built once,
    // before the (possibly parallel) DP loop, then read-only across DP workers.
    // null => fold disabled for this edge (legacy getMstateForFullState path).
    private ChildFoldPlan[] childFoldPlans;
    private boolean childFoldHoistInvariant;
    private boolean childFoldChildrenEnabled;
    private boolean childFoldPlansInitialized;

    // Option A: native kernel toggle (resolved once before the DP loop) + per-thread
    // scratch row buffers reused across edges/M-states to avoid per-call allocation.
    /** Scratch decode buffers for the full-DP worker that never escape the row. */
    private static final ThreadLocal<int[]> FULL_DP_M_DECODE_BUF =
            ThreadLocal.withInitial(() -> new int[0]);
    private static final ThreadLocal<int[]> FULL_DP_LAMBDA_DECODE_BUF =
            ThreadLocal.withInitial(() -> new int[0]);
    private static final ThreadLocal<long[]> FULL_DP_CHILD_BASE_BUF =
            ThreadLocal.withInitial(() -> new long[0]);
    private static final ThreadLocal<LogSumExpAccumulator> FULL_DP_LOWER_ACCUMULATOR =
            ThreadLocal.withInitial(LogSumExpAccumulator::new);
    private static final ThreadLocal<LogSumExpAccumulator> FULL_DP_UPPER_ACCUMULATOR =
            ThreadLocal.withInitial(LogSumExpAccumulator::new);

    private static final double NEG_INF = Double.NEGATIVE_INFINITY;
    private static final String DP_PARALLEL_PROPERTY = "branchdp.dp.parallel";
    private static final String DP_PARALLEL_MIN_M_STATES_PROPERTY = "branchdp.dp.parallel.minMStates";
    private static final String DP_PARALLEL_THREADS_PROPERTY = "branchdp.dp.parallel.threads";
    private static final String DP_COMPUTE_SHARD_SIZE_PROPERTY = "branchdp.dp.computeShardSize";
    private static final String DP_PROGRESS_PROPERTY = "branchdp.dp.progress";
    private static final int DEFAULT_DP_COMPUTE_SHARD_SIZE = 65_536;
    private static final String DP_MAX_M_STATES_PROPERTY = "branchdp.dp.maxMStates";
    private static final String DP_MAX_MATERIALIZED_PAIRS_PROPERTY = "branchdp.dp.maxMaterializedPairs";
    private static final String DP_TABLE_MODE_PROPERTY = "branchdp.dp.tableMode";
    private static final String DP_SHARD_SIZE_PROPERTY = "branchdp.dp.shardSize";
    private static final String DP_MMAP_THRESHOLD_BYTES_PROPERTY = "branchdp.dp.mmap.thresholdBytes";
    private static final String DP_MMAP_DIRECTORY_PROPERTY = "branchdp.dp.mmap.dir";
    private static final String DP_MMAP_CHUNK_ENTRIES_PROPERTY = "branchdp.dp.mmap.chunkEntries";
    private static final String DP_MMAP_SKIP_INITIAL_FILL_PROPERTY = "branchdp.dp.mmap.skipInitialFill";
    private static final int DEFAULT_DP_PARALLEL_MIN_M_STATES = 1024;
    private static final long DEFAULT_DP_MAX_MATERIALIZED_PAIRS = 50_000_000L;
    private static final int DEFAULT_DP_SHARD_SIZE = 1_048_576;
    private static final long DEFAULT_DP_MMAP_THRESHOLD_BYTES = 128L * 1024L * 1024L * 1024L;
    private static final int DEFAULT_DP_MMAP_CHUNK_ENTRIES = 67_108_864; // 512 MiB/map
    // Option C: non-leaf child-table folding (pure Java, no SIMD/JIT-vector risk).
    private static final String DP_FOLD_CHILDREN_PROPERTY = "branchdp.dp.foldChildren";
    private static final String DP_LAMBDA_DECODE_CACHE_BYTES_PROPERTY =
            "branchdp.dp.lambdaDecodeCache.maxBytes";
    private static final long DEFAULT_DP_LAMBDA_DECODE_CACHE_BYTES =
            64L * 1024L * 1024L;
    private static final String DP_CHILD_FOLD_PROJECTION_CACHE_BYTES_PROPERTY =
            "branchdp.dp.childFoldProjectionCache.maxBytes";
    private static final long DEFAULT_DP_CHILD_FOLD_PROJECTION_CACHE_BYTES =
            64L * 1024L * 1024L;
    // Option A: native SIMD log-sum-exp kernel. Default OFF (wired but dormant:
    // libOspreyLogSumExp.so is built/validated separately, see src/main/c/).
    // CUDA full-DP fast path. Enabled by default for PACK* calls. Workload
    // checks require enough m*lambda work to amortize packing/transfer overhead.
    private static final String DP_GPU_PROPERTY = "branchdp.dp.gpu";
    private static final String DP_GPU_MIN_WORK_PROPERTY = "branchdp.dp.gpu.minWork";
    private static final String DP_GPU_MAX_BYTES_PROPERTY = "branchdp.dp.gpu.maxBytes";
    private static final String DP_GPU_FAIL_IF_EXCEEDS_VRAM_PROPERTY = "branchdp.dp.gpu.failIfExceedsVram";
    // GPU DP was requested (branchdp.dp.gpu=true) but this edge is structurally
    // unsupported (e.g. MAX_EDGE_POSITIONS exceeded, or an int-overflow building the
    // request) even though the work is non-trivial: fail the run instead of silently
    // degrading to the single-threaded Java DP path, which can take hours to days on
    // multi-billion-state edges with zero GPU speedup. Benign reasons to skip GPU
    // (GPU not requested, native kernel chosen, leaf edge, trivial/empty edge, or work
    // below DP_GPU_MIN_WORK_PROPERTY) are never affected by this flag.
    private static final String DP_GPU_FAIL_IF_NO_GPU_PATH_PROPERTY = "branchdp.dp.gpu.failIfNoGpuPath";
    private static final boolean DEFAULT_DP_GPU_FAIL_IF_NO_GPU_PATH = true;
    private static final String DP_GPU_BLOCK_THREADS_PROPERTY = "branchdp.dp.gpu.blockThreads";
    private static final String DP_GPU_OUTPUT_TILE_MSTATES_PROPERTY = "branchdp.dp.gpu.outputTileMStates";
    private static final String DP_GPU_PERSISTENT_CONTEXT_PROPERTY = "branchdp.dp.gpu.persistentContext";
    private static final long DEFAULT_DP_GPU_MIN_WORK = 50_000_000L;
    // Single device buffers can now exceed 2 GiB (chunked raw upload in DPGpuFullDP),
    // so 0 = no static cap: the real budget is detected free VRAM (cuMemGetInfo, with
    // headroom) inside DPGpuFullDP. A positive value re-enables an explicit cap.
    private static final long DEFAULT_DP_GPU_MAX_BYTES = 0L;
    private static final int DEFAULT_DP_GPU_BLOCK_THREADS = 256;
    private static final long DEFAULT_DP_GPU_OUTPUT_TILE_MSTATES = 1_048_576L;
    private static final boolean DEFAULT_DP_GPU_PERSISTENT_CONTEXT = true;
    private static final int DP_GPU_MAX_CHILDREN = 64;
    private static final String DP_GPU_MULTI_PROPERTY = "branchdp.dp.gpu.multiGpu";
    private static final String DP_GPU_MAX_GPUS_PROPERTY = "branchdp.dp.gpu.maxGpus";
    private static final String DP_GPU_MIN_MSTATES_PER_GPU_PROPERTY = "branchdp.dp.gpu.minMStatesPerGpu";
    private static final long DEFAULT_DP_GPU_MIN_MSTATES_PER_GPU = 4096L;
    private static final String PAC_SAMPLING_GPU_MULTI_PROPERTY = "branchdp.pac.sampling.gpu.multiGpu";
    private static final String PAC_SAMPLING_GPU_MAX_GPUS_PROPERTY = "branchdp.pac.sampling.gpu.maxGpus";
    private static final String PAC_SAMPLING_GPU_MIN_GROUPS_PER_GPU_PROPERTY = "branchdp.pac.sampling.gpu.minGroupsPerGpu";
    private static final String PAC_SAMPLING_GPU_PERSISTENT_CONTEXT_PROPERTY = "branchdp.pac.sampling.gpu.persistentContext";
    private static final String PAC_SAMPLING_GPU_RESIDENT_CHILD_TABLES_PROPERTY = "branchdp.pac.sampling.gpu.residentChildTables";
    private static final String PAC_SAMPLING_GPU_METHOD_PROPERTY = "branchdp.pac.sampling.gpu.method";
    private static final int DEFAULT_PAC_SAMPLING_GPU_MIN_GROUPS_PER_GPU = 1;
    private static final boolean DEFAULT_PAC_SAMPLING_GPU_PERSISTENT_CONTEXT = true;
    private static final boolean DEFAULT_PAC_SAMPLING_GPU_RESIDENT_CHILD_TABLES = true;
    private static final SamplingGpuPhase1.Method DEFAULT_PAC_SAMPLING_GPU_METHOD =
            SamplingGpuPhase1.Method.GUMBEL;

    // ========== Incremental lambda enumeration state ==========

    /**
     * Per-M-state sorted lambda-states and enumeration progress.
     * sortedLambdaIndices[mIdx] = array of lambda-state indices sorted by E_min ascending.
     * enumeratedCount[mIdx] = how many lambda-states have been enumerated so far (k out of K).
     */
    private int[][] sortedLambdaIndices;  // [mIdx][rank] -> lambdaIdx
    private int[] enumeratedCount;         // [mIdx] -> k
    private int totalLambdaStates;         // K = total number of lambda-states

    // Pre-computed per-lambda-state energies (M-independent part: lambda-only)
    // Full energy per (m, λ) = lambdaOnlyRigid[λ] + lambdaMRigid(m,λ) for rigid
    private double[] lambdaOnlyRigid;      // [lambdaIdx]
    private double[] lambdaOnlyMin;        // [lambdaIdx]
    /** Flat structural lambda-digit cache; null when the configured byte cap is exceeded. */
    private int[] lambdaStateDigits;
    private boolean lambdaStateCacheAttempted;

    // Cached energy matrices and parameters for on-demand computation
    private EnergyMatrix cachedRigidEmat;
    private EnergyMatrix cachedMinEmat;
    private InteractionGraph cachedG;
    private double cachedRT;
    /**
     * The rigid matrix is invariant across PACK* corrected-EMAT refits.  Keep
     * its lambda-only vector tied to the matrix/graph identity so repeated
     * proposal DPs only rebuild the minimizing vector.
     */
    private EnergyMatrix lambdaOnlyRigidSource;
    private InteractionGraph lambdaOnlyRigidGraph;

    // Per (mIdx, lambdaIdx) full energies, lazily computed and cached
    // fullEnergyRigid[mIdx][lambdaIdx], fullEnergyMin[mIdx][lambdaIdx]
    private double[][] fullEnergyRigid;
    private double[][] fullEnergyMin;

    // ========== Constructor ==========

    public RootedTreeEdge(RootedTreeNode parent, RootedTreeNode child,
                           LinkedHashSet<Integer> mSet, boolean isRootEdge, RCs rcs) {
        this.parent = parent;
        this.child = child;
        this.M = new LinkedHashSet<>(mSet);
        this.isRootEdge = isRootEdge;
        this.rcs = rcs;
        this.isLambdaEdge = false;
    }

    // ========== Getters ==========

    public RootedTreeNode getParent() { return parent; }

    public boolean getIsLambdaEdge() { return isLambdaEdge; }
    public LinkedHashSet<Integer> getM() { return M; }

    public LinkedHashSet<Integer> getL() { return L; }
    public double[] getLogZLower() { return dpTable == null ? null : dpTable.lowerArrayUnsafe(); }
    public double[] getLogZUpper() { return dpTable == null ? null : dpTable.upperArrayUnsafe(); }
    /** Like {@link #getLogZLower()}/{@link #getLogZUpper()} for dense and sharded
     *  array-backed tables. File-backed tables must instead be streamed through
     *  DPTable reads; callers must check {@link DPTable#supportsArrayChunks()}. */
    public double[][] getLogZLowerChunks() { return dpTable == null ? null : dpTable.lowerChunks(); }
    public double[][] getLogZUpperChunks() { return dpTable == null ? null : dpTable.upperChunks(); }
    public double getLogZLower(long mIdx) { return dpTable.lower(mIdx); }
    public double getLogZUpper(long mIdx) { return dpTable.upper(mIdx); }
    public void setLogZ(long mIdx, double lower, double upper) { dpTable.set(mIdx, lower, upper); }
    public boolean hasDPTable() { return dpTable != null; }
    public boolean hasDenseDPTable() { return dpTable != null && dpTable.isDenseArrayBacked(); }

    public RootedTreeNode getCompactLeftChild() { return compactLeftChild; }
    public RootedTreeNode getCompactRightChild() { return compactRightChild; }

    public long getMStateCount() { return mStateCount; }
    public int[] getMPositionsSorted() { return mPositionsSorted; }
    public LinkedHashSet<RootedTreeEdge> getFset() { return Fset; }
    public int getTotalLambdaStates() { return totalLambdaStates; }
    public RCs getRCs() { return rcs; }

    public double[][] getFullEnergyMin() { return fullEnergyMin; }
    public double[][] getFullEnergyRigid() { return fullEnergyRigid; }

    /**
     * Drop all large, computation-only storage owned by this rooted edge.
     *
     * <p>PACK* and K* construct a fresh partition function for every state and
     * sequence.  Some production tables are hundreds of GiB, so waiting for a
     * later GC to discover that the whole backend is unreachable is not a safe
     * lifecycle policy.  The scalar partition-function result no longer needs
     * any of these arrays after {@code makeResult()}, so the PACK* facade calls
     * this deterministically before asking the JVM to collect the backend.</p>
     */
    public void releaseLargeMemory() {
        if (dpTable != null) {
            dpTable.close();
        }
        dpTable = null;
        childFoldPlans = null;
        childFoldPlansInitialized = false;
        childFoldChildrenEnabled = false;
        enumeratedCount = null;
        sortedLambdaIndices = null;
        lambdaOnlyRigid = null;
        lambdaOnlyMin = null;
        lambdaStateDigits = null;
        lambdaStateCacheAttempted = false;
        mCounts = null;
        lambdaCounts = null;
        mStrides = null;
        lambdaStrides = null;
        lambdaPairFirstSlots = null;
        lambdaPairSecondSlots = null;
        lambdaMFirstSlots = null;
        lambdaMSecondSlots = null;
        interactionTermCacheGraph = null;
        gpuLambdaMTermDescriptors = null;
        gpuLambdaMTermGraph = null;
        fullEnergyRigid = null;
        fullEnergyMin = null;
        cachedRigidEmat = null;
        cachedMinEmat = null;
        cachedG = null;
        lambdaOnlyRigidSource = null;
        lambdaOnlyRigidGraph = null;

        // Break the compact/F-set graph too. The ordinary rooted tree is still
        // traversable while postOrderReleaseLargeMemory is running.
        if (Fset != null) {
            Fset.clear();
            Fset = null;
        }
        compactLeftChild = null;
        compactRightChild = null;
    }

    // ========== Log-space utility ==========

    public static double logSumExp(double a, double b) {
        if (a == NEG_INF) return b;
        if (b == NEG_INF) return a;
        double max = Math.max(a, b);
        return max + Math.log1p(Math.exp(-Math.abs(a - b)));
    }

    private static String getConfigProperty(String key, String defaultValue) {
        return BranchDpConfig.getBackendProperty(key, defaultValue);
    }

    private static boolean getConfigBoolean(String key, boolean defaultValue) {
        return BranchDpConfig.getBackendBoolean(key, defaultValue);
    }

    private static int getConfigInteger(String key, int defaultValue) {
        return BranchDpConfig.getBackendInteger(key, defaultValue, BranchDpConfig.getBackendLogPrefix());
    }

    private static long getConfigLong(String key, long defaultValue) {
        return BranchDpConfig.getBackendLong(key, defaultValue, BranchDpConfig.getBackendLogPrefix());
    }

    private static long getConfigBytes(String key, long defaultValue) {
        return BranchDpConfig.getBackendBytes(key, defaultValue, BranchDpConfig.getBackendLogPrefix());
    }

    // ========== compLlambda: compute L and lambda sets ==========

    public void compLlambda() {
        compLlambda(true);
    }

    public void compLlambda(boolean initEnumerationArrays) {
        // The decomposition topology can be rebuilt for a proposal-only
        // interaction graph.  Numerical DP arrays are reset separately, but
        // child-fold plans must be rebuilt when this structural method runs.
        childFoldPlans = null;
        childFoldPlansInitialized = false;
        childFoldChildrenEnabled = false;
        childFoldHoistInvariant = false;
        lambdaStateDigits = null;
        lambdaStateCacheAttempted = false;
        lambdaPairFirstSlots = null;
        lambdaPairSecondSlots = null;
        lambdaMFirstSlots = null;
        lambdaMSecondSlots = null;
        interactionTermCacheGraph = null;
        gpuLambdaMTermDescriptors = null;
        gpuLambdaMTermGraph = null;
        lambdaOnlyRigidSource = null;
        lambdaOnlyRigidGraph = null;
        RootedTreeNode clc = child.getLeftChild();

        if (clc == null) {
            L = new LinkedHashSet<>();
            if (child.getV1() >= 0) L.add(child.getV1());
            if (child.getV2() >= 0) L.add(child.getV2());
            L.removeAll(M);
            lambda = new LinkedHashSet<>(L);
        } else {
            RootedTreeEdge leftEdge = clc.getChildOfEdge();
            RootedTreeNode crc = child.getRightChild();
            RootedTreeEdge rightEdge = crc == null ? null : crc.getChildOfEdge();

            LinkedHashSet<Integer> uMc = new LinkedHashSet<>(leftEdge.getM());
            if (rightEdge != null) {
                uMc.addAll(rightEdge.getM());
            }
            lambda = new LinkedHashSet<>(uMc);
            lambda.removeAll(M);

            L = new LinkedHashSet<>(lambda);
            L.addAll(leftEdge.getL());
            if (rightEdge != null) {
                L.addAll(rightEdge.getL());
            }
        }

        if (!lambda.isEmpty()) {
            isLambdaEdge = true;
            Fset = new LinkedHashSet<>();
            computeFset(child);
            initializeIndexing();
            if (initEnumerationArrays) {
                initializeArrays();
            }
        } else {
            isLambdaEdge = false;
        }
    }

    // ========== F-set computation ==========

    private void computeFset(RootedTreeNode tn) {
        if (tn.getLeftChild() != null) {
            RootedTreeEdge leftEdge = tn.getLeftChild().getChildOfEdge();
            if (leftEdge.isLambdaEdge) {
                Fset.add(leftEdge);
            } else {
                computeFset(tn.getLeftChild());
            }
        }
        if (tn.getRightChild() != null) {
            RootedTreeEdge rightEdge = tn.getRightChild().getChildOfEdge();
            if (rightEdge.isLambdaEdge) {
                Fset.add(rightEdge);
            } else {
                computeFset(tn.getRightChild());
            }
        }
    }

    // ========== Array initialization ==========

    private void initializeIndexing() {
        mPositionsSorted = M.stream().mapToInt(Integer::intValue).sorted().toArray();
        lambdaPositionsSorted = lambda.stream().mapToInt(Integer::intValue).sorted().toArray();

        mCounts = countsForPositions(mPositionsSorted);
        lambdaCounts = countsForPositions(lambdaPositionsSorted);
        mStrides = mixedRadixStrides(mCounts);
        lambdaStrides = mixedRadixStrides(lambdaCounts);

        mStateCount = checkedStateCountLong("M", mPositionsSorted, true);
        mArraySize = mStateCount <= Integer.MAX_VALUE ? (int) mStateCount : -1;
        totalLambdaStates = checkedStateCount("lambda", lambdaPositionsSorted, false);
    }

    private int checkedStateCount(String label, int[] positions, boolean isMStateCount) {
        long count = checkedStateCountLong(label, positions, isMStateCount);
        if (count > Integer.MAX_VALUE) {
            throw stateCountException(label, positions, count);
        }
        return (int) count;
    }

    private long checkedStateCountLong(String label, int[] positions, boolean isMStateCount) {
        long count = 1L;
        for (int pos : positions) {
            int n = rcs.getNum(pos);
            if (n < 0 || count > Long.MAX_VALUE / Math.max(1, n)) {
                throw stateCountException(label, positions, Long.MAX_VALUE);
            }
            count *= n;
        }

        long maxStates = isMStateCount
                ? getConfigLong(DP_MAX_M_STATES_PROPERTY, Long.MAX_VALUE)
                : Integer.MAX_VALUE;
        if (count > maxStates) {
            String limit = isMStateCount
                    ? DP_MAX_M_STATES_PROPERTY + "=" + maxStates
                    : "the int-addressable lambda-state limit=" + maxStates;
            String action = isMStateCount
                    ? "Reduce branchwidth/rooting, enable a shard- or file-backed DP table, "
                        + "or raise the limit only if memory allows."
                    : "Choose a root with a smaller lambda separator; lambda indexing and "
                        + "GPU launch metadata are intentionally int-addressable.";
            throw new DPTableTooLargeException(label, count, positions,
                    "DP " + label + "-state count exceeds " + limit + ". " + action);
        }
        return count;
    }

    private DPTableTooLargeException stateCountException(String label, int[] positions, long count) {
        return new DPTableTooLargeException(label, count, positions,
                "DP " + label + "-state count exceeds int indexing limits. "
                        + "This used to overflow into NegativeArraySizeException; "
                        + ("M".equals(label)
                        ? "use a shard- or file-backed DP table."
                        : "choose a root with a smaller lambda separator."));
    }

    private void initializeArrays() {
        dpTable = makeDPTable();
        if (!skipInitialMappedFill(false)) {
            dpTable.fill(NEG_INF, NEG_INF);
        }

        enumeratedCount = mStateCount <= Integer.MAX_VALUE
                ? new int[(int) mStateCount]
                : null;
        // sortedLambdaIndices allocated lazily per M-state in initIncrementalEnumeration
    }

    private DPTable makeDPTable() {
        String mode = getConfigProperty(DP_TABLE_MODE_PROPERTY, "auto").trim().toLowerCase(Locale.ROOT);
        int shardSize = Math.max(1, getConfigInteger(DP_SHARD_SIZE_PROPERTY, DEFAULT_DP_SHARD_SIZE));
        if (shouldUseFileBackedDPTable(mStateCount)) {
            int chunkEntries = Math.max(1, getConfigInteger(
                    DP_MMAP_CHUNK_ENTRIES_PROPERTY, DEFAULT_DP_MMAP_CHUNK_ENTRIES));
            Path directory = Paths.get(getConfigProperty(
                    DP_MMAP_DIRECTORY_PROPERTY, System.getProperty("java.io.tmpdir")));
            DPTable table = new MappedDPTable(mStateCount, chunkEntries, directory);
            System.out.println(BranchDpConfig.getBackendLogPrefix()
                    + " file-backed DP table created, states=" + mStateCount
                    + ", bytes=" + table.estimatedBytes()
                    + ", directory=" + directory);
            return table;
        }
        switch (mode) {
            case "":
            case "auto":
            case "auto_mmap":
            case "mmap_auto":
            case "tiered":
                return mStateCount <= Integer.MAX_VALUE
                        ? new DenseDPTable(mStateCount)
                        : new ShardedDPTable(mStateCount, shardSize);
            case "dense":
                if (mStateCount > Integer.MAX_VALUE) {
                    throw stateCountException("M", mPositionsSorted, mStateCount);
                }
                return new DenseDPTable(mStateCount);
            case "sharded":
            case "shard":
                return new ShardedDPTable(mStateCount, shardSize);
            case "mmap":
            case "mapped":
            case "file":
                // shouldUseFileBackedDPTable() handles unconditional mmap modes.
                throw new IllegalStateException("Unable to create requested mapped DP table");
            default:
                System.err.println(BranchDpConfig.getBackendLogPrefix() + " Unknown DP table mode '" + mode
                        + "', using auto.");
                return mStateCount <= Integer.MAX_VALUE
                        ? new DenseDPTable(mStateCount)
                        : new ShardedDPTable(mStateCount, shardSize);
        }
    }

    public static boolean shouldUseFileBackedDPTable(long mStates) {
        String mode = getConfigProperty(DP_TABLE_MODE_PROPERTY, "auto")
                .trim().toLowerCase(Locale.ROOT);
        if (mode.equals("mmap") || mode.equals("mapped") || mode.equals("file")) {
            return true;
        }
        if (!(mode.equals("auto_mmap") || mode.equals("mmap_auto")
                || mode.equals("tiered"))) {
            return false;
        }
        long threshold = Math.max(1L, getConfigBytes(
                DP_MMAP_THRESHOLD_BYTES_PROPERTY, DEFAULT_DP_MMAP_THRESHOLD_BYTES));
        return estimateDPTableBytes(mStates) >= threshold;
    }

    private boolean skipInitialMappedFill(boolean materializedEnergyTables) {
        return dpTable != null && dpTable.isFileBacked()
                && !materializedEnergyTables
                && getConfigBoolean(DP_MMAP_SKIP_INITIAL_FILL_PROPERTY, true);
    }

    private int requireDenseMStateCount() {
        if (mArraySize < 0) {
            throw new IllegalStateException("DP M-state count " + mStateCount
                    + " exceeds dense int indexing; use getMStateCount() and DPTable access.");
        }
        return mArraySize;
    }

    private static long estimateDPTableBytes(long mStates) {
        if (mStates > Long.MAX_VALUE / (2L * (long) Double.BYTES)) {
            return Long.MAX_VALUE;
        }
        return 2L * mStates * (long) Double.BYTES;
    }

    // ========== M-state indexing ==========

    public long computeIndexInA(int[] mRCIndices) {
        ensureIndexingCaches();
        if (isRootEdge || mRCIndices == null || mPositionsSorted.length == 0) {
            return 0L;
        }

        long index = mRCIndices[mPositionsSorted.length - 1];
        for (int i = mPositionsSorted.length - 2; i >= 0; i--) {
            index += (long) mRCIndices[i] * mStrides[i];
        }
        return index;
    }

    public int[] getMstateForFullState(int[] mRCs, int[] lambdaRCs, RootedTreeEdge childEdge) {
        int[] childMPositions = childEdge.mPositionsSorted;
        int[] result = new int[childMPositions.length];

        for (int i = 0; i < childMPositions.length; i++) {
            int targetPos = childMPositions[i];
            result[i] = -1;

            for (int j = 0; j < mPositionsSorted.length; j++) {
                if (mPositionsSorted[j] == targetPos) {
                    result[i] = mRCs[j];
                    break;
                }
            }
            if (result[i] >= 0) continue;

            for (int j = 0; j < lambdaPositionsSorted.length; j++) {
                if (lambdaPositionsSorted[j] == targetPos) {
                    result[i] = lambdaRCs[j];
                    break;
                }
            }
        }
        return result;
    }

    public int[] decodeMStatePublic(long mIndex) {
        return decodeMState(mIndex);
    }

    private int[] decodeMState(long mIndex) {
        ensureIndexingCaches();
        int[] mRCs = new int[mPositionsSorted.length];
        decodeMStateInto(mIndex, mRCs);
        return mRCs;
    }

    private int[] decodeLambdaState(int lambdaIndex) {
        int[] lambdaRCs = new int[lambdaPositionsSorted.length];
        decodeLambdaStateInto(lambdaIndex, lambdaRCs);
        return lambdaRCs;
    }

    /** Decode an M-state into caller-owned storage without allocating. */
    private void decodeMStateInto(long mIndex, int[] mRCs) {
        ensureIndexingCaches();
        if (mRCs.length != mPositionsSorted.length) {
            throw new IllegalArgumentException(
                    "M-state decode buffer has the wrong length: "
                            + mRCs.length + " != " + mPositionsSorted.length);
        }
        long remaining = mIndex;
        for (int i = mPositionsSorted.length - 1; i >= 0; i--) {
            int numRCs = mCounts[i];
            mRCs[i] = (int) (remaining % numRCs);
            remaining /= numRCs;
        }
    }

    /** Decode a lambda state into caller-owned storage without allocating. */
    private void decodeLambdaStateInto(int lambdaIndex, int[] lambdaRCs) {
        ensureIndexingCaches();
        if (lambdaIndex < 0 || lambdaIndex >= totalLambdaStates) {
            throw new IndexOutOfBoundsException(
                    "lambda state out of range: " + lambdaIndex);
        }
        if (lambdaRCs.length != lambdaPositionsSorted.length) {
            throw new IllegalArgumentException(
                    "lambda-state decode buffer has the wrong length: "
                            + lambdaRCs.length + " != "
                            + lambdaPositionsSorted.length);
        }
        ensureLambdaStateDigits();
        if (lambdaStateDigits != null) {
            System.arraycopy(lambdaStateDigits,
                    lambdaIndex * lambdaPositionsSorted.length,
                    lambdaRCs, 0, lambdaPositionsSorted.length);
            return;
        }
        for (int i = 0; i < lambdaPositionsSorted.length; i++) {
            lambdaRCs[i] = (int) ((lambdaIndex / lambdaStrides[i])
                    % lambdaCounts[i]);
        }
    }

    /** Per-worker M-state buffer used by full DP; the contents are ephemeral. */
    private int[] fullDpMDecodeBuffer() {
        ensureIndexingCaches();
        int[] buffer = FULL_DP_M_DECODE_BUF.get();
        if (buffer.length != mPositionsSorted.length) {
            buffer = new int[mPositionsSorted.length];
            FULL_DP_M_DECODE_BUF.set(buffer);
        }
        return buffer;
    }

    /** Per-worker lambda-state buffer used by full DP; the contents are ephemeral. */
    private int[] fullDpLambdaDecodeBuffer() {
        ensureIndexingCaches();
        int[] buffer = FULL_DP_LAMBDA_DECODE_BUF.get();
        if (buffer.length != lambdaPositionsSorted.length) {
            buffer = new int[lambdaPositionsSorted.length];
            FULL_DP_LAMBDA_DECODE_BUF.set(buffer);
        }
        return buffer;
    }

    /** Per-worker child-index buffer used by full DP; the contents are ephemeral. */
    private long[] fullDpChildBaseBuffer(int length) {
        long[] buffer = FULL_DP_CHILD_BASE_BUF.get();
        if (buffer.length < length) {
            buffer = new long[length];
            FULL_DP_CHILD_BASE_BUF.set(buffer);
        }
        return buffer;
    }

    private void ensureLambdaStateDigits() {
        ensureIndexingCaches();
        if (lambdaStateCacheAttempted) return;
        lambdaStateCacheAttempted = true;
        long entries = (long) totalLambdaStates
                * (long) lambdaPositionsSorted.length;
        long maxBytes = getConfigLong(
                DP_LAMBDA_DECODE_CACHE_BYTES_PROPERTY,
                DEFAULT_DP_LAMBDA_DECODE_CACHE_BYTES);
        if (entries <= 0L || maxBytes <= 0L
                || entries > Integer.MAX_VALUE
                || entries > maxBytes / Integer.BYTES) {
            return;
        }

        int[] digits = new int[(int) entries];
        for (int lambdaIndex = 0; lambdaIndex < totalLambdaStates;
             lambdaIndex++) {
            int base = lambdaIndex * lambdaPositionsSorted.length;
            for (int slot = 0; slot < lambdaPositionsSorted.length; slot++) {
                digits[base + slot] = (int) ((lambdaIndex
                        / lambdaStrides[slot]) % lambdaCounts[slot]);
            }
        }
        lambdaStateDigits = digits;
    }

    private int getLambdaLocalRC(int lambdaIndex, int slot) {
        if (lambdaIndex < 0 || lambdaIndex >= totalLambdaStates
                || slot < 0 || slot >= lambdaPositionsSorted.length) {
            throw new IndexOutOfBoundsException(
                    "lambda state/slot out of range: state=" + lambdaIndex
                            + ", slot=" + slot);
        }
        ensureLambdaStateDigits();
        if (lambdaStateDigits != null) {
            return lambdaStateDigits[
                    lambdaIndex * lambdaPositionsSorted.length + slot];
        }
        return (int) ((lambdaIndex / lambdaStrides[slot])
                % lambdaCounts[slot]);
    }

    /** Copy one lambda state into a full conformation using cached digits. */
    public void writeLambdaStateGlobal(int lambdaIndex, int[] conformation) {
        if (conformation == null) {
            throw new IllegalArgumentException("conformation must be non-null");
        }
        for (int slot = 0; slot < lambdaPositionsSorted.length; slot++) {
            int pos = lambdaPositionsSorted[slot];
            conformation[pos] = rcs.get(pos,
                    getLambdaLocalRC(lambdaIndex, slot));
        }
    }

    // ========== Compact tree ==========

    public void compactTree() {
        compactLeftChild = searchSubtree(child.getLeftChild());
        compactRightChild = searchSubtree(child.getRightChild());

        if (compactLeftChild != null) {
            compactLeftChild.getChildOfEdge().compactTree();
        }
        if (compactRightChild != null) {
            compactRightChild.getChildOfEdge().compactTree();
        }

        if (compactLeftChild == null && compactRightChild != null) {
            compactLeftChild = compactRightChild;
            compactRightChild = null;
        }
    }

    private RootedTreeNode searchSubtree(RootedTreeNode tn) {
        if (tn == null) return null;
        if (tn.getChildOfEdge() != null && tn.getChildOfEdge().isLambdaEdge) {
            return tn;
        }
        if (tn.getIsLeaf()) return null;

        RootedTreeNode left = searchSubtree(tn.getLeftChild());
        RootedTreeNode right = searchSubtree(tn.getRightChild());

        if (left != null && right != null) return tn;
        if (left != null) return left;
        return right;
    }

    // ========== TESS computation ==========

    public long computeTESS() {
        long leftTESS = 0;
        if (compactLeftChild != null) {
            leftTESS = compactLeftChild.getChildOfEdge().computeTESS();
        }
        long rightTESS = 0;
        if (compactRightChild != null) {
            rightTESS = compactRightChild.getChildOfEdge().computeTESS();
        }

        long selfTESS = 1;
        if (mPositionsSorted != null) {
            for (int pos : mPositionsSorted) selfTESS *= rcs.getNum(pos);
        }
        if (lambdaPositionsSorted != null) {
            for (int pos : lambdaPositionsSorted) selfTESS *= rcs.getNum(pos);
        }

        return leftTESS + rightTESS + selfTESS;
    }

    public double computeLogTESS() {
        List<Double> logTerms = new ArrayList<>();
        collectLogTESSTerms(logTerms);

        if (logTerms.isEmpty()) return 0.0;

        double maxLog = Collections.max(logTerms);
        double sumExp = 0.0;
        for (double logTerm : logTerms) {
            sumExp += Math.exp(logTerm - maxLog);
        }
        return maxLog + Math.log(sumExp);
    }

    private void collectLogTESSTerms(List<Double> logTerms) {
        if (compactLeftChild != null) {
            compactLeftChild.getChildOfEdge().collectLogTESSTerms(logTerms);
        }
        if (compactRightChild != null) {
            compactRightChild.getChildOfEdge().collectLogTESSTerms(logTerms);
        }

        double logSelf = 0.0;
        if (mPositionsSorted != null) {
            for (int pos : mPositionsSorted) logSelf += Math.log(rcs.getNum(pos));
        }
        if (lambdaPositionsSorted != null) {
            for (int pos : lambdaPositionsSorted) logSelf += Math.log(rcs.getNum(pos));
        }
        logTerms.add(logSelf);
    }

    // ========== Energy computation helpers ==========

    private double computeLambdaOnlyEnergy(int[] lambdaRCs, EnergyMatrix emat, InteractionGraph G) {
        double energy = 0.0;
        for (int i = 0; i < lambdaPositionsSorted.length; i++) {
            int pos = lambdaPositionsSorted[i];
            int rc = rcs.get(pos, lambdaRCs[i]);
            energy += emat.getOneBody(pos, rc);
        }
        for (int i = 0; i < lambdaPositionsSorted.length; i++) {
            int posI = lambdaPositionsSorted[i];
            int rcI = rcs.get(posI, lambdaRCs[i]);
            for (int j = i + 1; j < lambdaPositionsSorted.length; j++) {
                int posJ = lambdaPositionsSorted[j];
                if (G.hasEdge(posI, posJ)) {
                    int rcJ = rcs.get(posJ, lambdaRCs[j]);
                    energy += emat.getPairwise(posI, rcI, posJ, rcJ);
                }
            }
        }
        energy += computeHigherOrderEnergy(
                lambdaPositionsSorted, lambdaRCs, emat);
        return energy;
    }

    /**
     * Pairwise-only lambda energy directly from a mixed-radix index.  This
     * keeps the full-DP and ancestral-sampling hot paths from allocating a
     * temporary lambda digit array for every state.  Higher-order matrices
     * deliberately use the array-based path because their tuple lookup needs
     * the complete assignment.
     */
    private double computeLambdaOnlyEnergyForIndex(
            int lambdaIndex, EnergyMatrix emat, InteractionGraph G) {
        if (emat.hasHigherOrderTerms()) {
            return computeLambdaOnlyEnergy(
                    decodeLambdaState(lambdaIndex), emat, G);
        }

        ensureInteractionTermCaches(G);
        double energy = 0.0;
        for (int i = 0; i < lambdaPositionsSorted.length; i++) {
            int pos = lambdaPositionsSorted[i];
            int rc = rcs.get(pos, getLambdaLocalRC(lambdaIndex, i));
            energy += emat.getOneBody(pos, rc);
        }
        for (int pair = 0; pair < lambdaPairFirstSlots.length; pair++) {
            int firstSlot = lambdaPairFirstSlots[pair];
            int secondSlot = lambdaPairSecondSlots[pair];
            int firstPos = lambdaPositionsSorted[firstSlot];
            int secondPos = lambdaPositionsSorted[secondSlot];
            int firstRC = rcs.get(firstPos,
                    getLambdaLocalRC(lambdaIndex, firstSlot));
            int secondRC = rcs.get(secondPos,
                    getLambdaLocalRC(lambdaIndex, secondSlot));
            energy += emat.getPairwise(
                    firstPos, firstRC, secondPos, secondRC);
        }
        return energy;
    }

    private double computeLambdaMEnergy(int[] mRCs, int[] lambdaRCs,
                                         EnergyMatrix emat, InteractionGraph G) {
        double energy = 0.0;
        for (int i = 0; i < lambdaPositionsSorted.length; i++) {
            int posI = lambdaPositionsSorted[i];
            int rcI = rcs.get(posI, lambdaRCs[i]);
            for (int j = 0; j < mPositionsSorted.length; j++) {
                int posJ = mPositionsSorted[j];
                if (G.hasEdge(posI, posJ)) {
                    int rcJ = rcs.get(posJ, mRCs[j]);
                    energy += emat.getPairwise(posI, rcI, posJ, rcJ);
                }
            }
        }
        if (emat.hasHigherOrderTerms()) {
            // Higher-order factors wholly inside lambda were charged by
            // computeLambdaOnlyEnergy.  Factors wholly inside M belong to an
            // ancestor/descendant elimination edge.  The remainder is exactly
            // the set of local factors crossing M and lambda.
            int[] localPositions = mergeSortedPositions(
                    mPositionsSorted, lambdaPositionsSorted);
            int[] localRCs = mergeLocalRCs(
                    localPositions, mPositionsSorted, mRCs,
                    lambdaPositionsSorted, lambdaRCs);
            energy += computeHigherOrderEnergy(
                    localPositions, localRCs, emat);
            energy -= computeHigherOrderEnergy(
                    mPositionsSorted, mRCs, emat);
            energy -= computeHigherOrderEnergy(
                    lambdaPositionsSorted, lambdaRCs, emat);
        }
        return energy;
    }

    /**
     * Return only the n-body (n &gt; 2) contribution for one partial assignment.
     * RC arrays use the local RC indices employed by the branch DP; the energy
     * matrix stores global RC identifiers, so translate through {@link RCs}.
     *
     * <p>The branch decomposition is exact for a higher-order factor whenever
     * its primal positions form a clique in the interaction graph.  In that
     * case the running-intersection property places the full factor in a bag,
     * and the M/lambda subtraction above charges it exactly once, when its
     * first position is eliminated.  Proposal builders must reject factors
     * that are not graph cliques.</p>
     */
    private double computeHigherOrderEnergy(
            int[] positions, int[] localRCs, EnergyMatrix emat) {
        if (!emat.hasHigherOrderTerms() || positions.length < 3) {
            return 0.0;
        }
        if (positions.length != localRCs.length) {
            throw new IllegalArgumentException(
                    "higher-order local position/RC lengths differ");
        }
        RCTuple tuple = new RCTuple();
        double pairwise = 0.0;
        for (int i = 0; i < positions.length; i++) {
            int pos = positions[i];
            int rc = rcs.get(pos, localRCs[i]);
            tuple.pos.add(pos);
            tuple.RCs.add(rc);
            pairwise += emat.getOneBody(pos, rc);
            for (int j = 0; j < i; j++) {
                int pos2 = positions[j];
                int rc2 = rcs.get(pos2, localRCs[j]);
                pairwise += emat.getPairwise(pos, rc, pos2, rc2);
            }
        }
        return emat.getInternalEnergy(tuple) - pairwise;
    }

    private static int[] mergeSortedPositions(int[] left, int[] right) {
        int[] merged = new int[left.length + right.length];
        int i = 0;
        int j = 0;
        int out = 0;
        while (i < left.length || j < right.length) {
            if (j >= right.length
                    || (i < left.length && left[i] < right[j])) {
                merged[out++] = left[i++];
            } else if (i >= left.length || right[j] < left[i]) {
                merged[out++] = right[j++];
            } else {
                throw new IllegalArgumentException(
                        "M and lambda position sets overlap at " + left[i]);
            }
        }
        return merged;
    }

    private static int[] mergeLocalRCs(
            int[] mergedPositions,
            int[] leftPositions, int[] leftRCs,
            int[] rightPositions, int[] rightRCs) {
        if (leftPositions.length != leftRCs.length
                || rightPositions.length != rightRCs.length) {
            throw new IllegalArgumentException(
                    "local position/RC lengths differ");
        }
        int[] mergedRCs = new int[mergedPositions.length];
        int left = 0;
        int right = 0;
        for (int out = 0; out < mergedPositions.length; out++) {
            int pos = mergedPositions[out];
            if (left < leftPositions.length
                    && leftPositions[left] == pos) {
                mergedRCs[out] = leftRCs[left++];
            } else if (right < rightPositions.length
                    && rightPositions[right] == pos) {
                mergedRCs[out] = rightRCs[right++];
            } else {
                throw new IllegalArgumentException(
                        "merged position is absent from both inputs: " + pos);
            }
        }
        return mergedRCs;
    }

    private double computeLocalEnergy(int[] mRCs, int[] lambdaRCs,
                                       EnergyMatrix emat, InteractionGraph G) {
        return computeLambdaOnlyEnergy(lambdaRCs, emat, G)
                + computeLambdaMEnergy(mRCs, lambdaRCs, emat, G);
    }

    /** Exact local proposal energy used by ancestral samplers. */

    /**
     * Exact local energy for a lambda index, reusing the cached lambda-only
     * portion prepared during DP initialization.  This is the hot path used by
     * PACK* ancestral sampling; it avoids rebuilding a lambda digit array and
     * avoids recomputing all lambda-lambda terms for every (M,lambda) query.
     */
    public double computeLocalEnergyForLambdaIndex(
            int[] mRCs, int lambdaIndex,
            EnergyMatrix emat, InteractionGraph graph) {
        if (mRCs == null) {
            throw new IllegalArgumentException("M-state digits must be non-null");
        }
        if (lambdaIndex < 0 || lambdaIndex >= totalLambdaStates) {
            throw new IndexOutOfBoundsException(
                    "lambda state out of range: " + lambdaIndex);
        }
        double[] cachedLambdaOnly = null;
        if (graph == cachedG) {
            if (emat == cachedMinEmat) {
                cachedLambdaOnly = lambdaOnlyMin;
            } else if (emat == cachedRigidEmat) {
                cachedLambdaOnly = lambdaOnlyRigid;
            }
        }
        if (cachedLambdaOnly != null) {
            return cachedLambdaOnly[lambdaIndex]
                    + computeLambdaMEnergyForIndex(
                    mRCs, lambdaIndex, emat, graph);
        }
        int[] lambdaRCs = decodeLambdaState(lambdaIndex);
        return computeLocalEnergy(mRCs, lambdaRCs, emat, graph);
    }

    private double computeLambdaMEnergyForIndex(
            int[] mRCs, int lambdaIndex,
            EnergyMatrix emat, InteractionGraph graph) {
        if (emat.hasHigherOrderTerms()) {
            int[] lambdaRCs = decodeLambdaState(lambdaIndex);
            return computeLambdaMEnergy(mRCs, lambdaRCs, emat, graph);
        }

        ensureInteractionTermCaches(graph);
        double energy = 0.0;
        for (int pair = 0; pair < lambdaMFirstSlots.length; pair++) {
            int lambdaSlot = lambdaMFirstSlots[pair];
            int mSlot = lambdaMSecondSlots[pair];
            int lambdaPos = lambdaPositionsSorted[lambdaSlot];
            int mPos = mPositionsSorted[mSlot];
            int lambdaRC = rcs.get(lambdaPos,
                    getLambdaLocalRC(lambdaIndex, lambdaSlot));
            int mRC = rcs.get(mPos, mRCs[mSlot]);
            energy += emat.getPairwise(
                    lambdaPos, lambdaRC, mPos, mRC);
        }
        return energy;
    }

    /**
     * Exact log conditional weight for one lambda state.  Child folding uses
     * the same mixed-radix plans as full DP, so CPU sampling and DP address the
     * child tables identically without temporary child-M arrays.
     */
    public double computeLambdaLogWeightPublic(
            long mIdx, int[] mRCs, int lambdaIndex,
            EnergyMatrix emat, InteractionGraph graph, double RT) {
        if (!hasFsetChildren()) {
            // Leaf edges already own the exact M x lambda minimum-energy
            // accessor.  Keep using it here so streamed/file-backed tables do
            // not require decoding an M state just for the leaf conditional.
            return -getFullEnergyMin(mIdx, lambdaIndex) / RT;
        }
        // The PACK* sampler queries this once per lambda state.  The backend
        // properties are run-scoped, so do not re-read them for every query;
        // structural rebuilds reset childFoldPlansInitialized.
        if (!childFoldPlansInitialized) {
            ensureChildFoldPlans();
        }
        if (childFoldPlans != null) {
            double localEnergy = computeLocalEnergyForLambdaIndex(
                    mRCs, lambdaIndex, emat, graph);
            double childLogZ = 0.0;
            for (ChildFoldPlan plan : childFoldPlans) {
                long childIndex = plan.baseIndex(mRCs)
                        + lambdaIndexForPlan(plan, lambdaIndex);
                childLogZ += plan.child.getLogZUpper(childIndex);
            }
            return -localEnergy / RT + childLogZ;
        }

        int[] lambdaRCs = decodeLambdaState(lambdaIndex);
        double localEnergy = computeLocalEnergy(
                mRCs, lambdaRCs, emat, graph);
        double childLogZ = 0.0;
        for (RootedTreeEdge fEdge : Fset) {
            int[] childM = getMstateForFullState(
                    mRCs, lambdaRCs, fEdge);
            childLogZ += fEdge.getLogZUpper(
                    fEdge.computeIndexInA(childM));
        }
        return -localEnergy / RT + childLogZ;
    }

    // ========== Incremental lambda enumeration ==========

    /**
     * Initialize incremental enumeration for all edges in the tree (post-order).
     * Pre-computes lambda-only energies and per-M-state sorted lambda indices.
     * Initial logZ bounds are loose (k=0 for all M-states):
     *   logZ_lower = -inf (nothing enumerated)
     *   logZ_upper = tail bound (all K lambda-states in tail)
     */
    public void initIncrementalEnumeration(EnergyMatrix rigidEmat, EnergyMatrix minEmat,
                                            InteractionGraph G, double RT) {
        initIncrementalEnumeration(rigidEmat, minEmat, G, RT, false);
    }

    public void initIncrementalEnumeration(EnergyMatrix rigidEmat, EnergyMatrix minEmat,
                                            InteractionGraph G, double RT,
                                            boolean materializeFullEnergyTables) {
        if (!isLambdaEdge) return;

        boolean reuseRigidLambdaOnly = lambdaOnlyRigid != null
                && lambdaOnlyRigidSource == rigidEmat
                && lambdaOnlyRigidGraph == G
                && lambdaOnlyRigid.length == totalLambdaStates;
        this.cachedRigidEmat = rigidEmat;
        this.cachedMinEmat = minEmat;
        this.cachedG = G;
        this.cachedRT = RT;

        // Pre-compute lambda-only energies (M-independent)
        if (!reuseRigidLambdaOnly) {
            lambdaOnlyRigid = new double[totalLambdaStates];
            lambdaOnlyRigidSource = rigidEmat;
            lambdaOnlyRigidGraph = G;
        }
        lambdaOnlyMin = new double[totalLambdaStates];
        for (int lIdx = 0; lIdx < totalLambdaStates; lIdx++) {
            if (!reuseRigidLambdaOnly) {
                lambdaOnlyRigid[lIdx] = computeLambdaOnlyEnergyForIndex(
                        lIdx, rigidEmat, G);
            }
            lambdaOnlyMin[lIdx] = computeLambdaOnlyEnergyForIndex(
                    lIdx, minEmat, G);
        }

        fullEnergyRigid = null;
        fullEnergyMin = null;
        sortedLambdaIndices = null;

        if (materializeFullEnergyTables) {
            requireDenseMStateCount();
            checkMaterializedPairCount();

            // Allocate per-(mIdx, lambdaIdx) energy cache only for legacy paths
            // that mutate or sort per-entry energies (region-atom DP integration,
            // incremental enumeration). Full DP uses streaming and does not need it.
            fullEnergyRigid = new double[mArraySize][totalLambdaStates];
            fullEnergyMin = new double[mArraySize][totalLambdaStates];
            for (double[] row : fullEnergyRigid) Arrays.fill(row, Double.NaN);
            for (double[] row : fullEnergyMin) Arrays.fill(row, Double.NaN);

            // For leaf edges (no F-set): pre-compute full energies and sort per M-state.
            // For non-leaf edges: sorting depends on child logZ, handled differently.
            sortedLambdaIndices = new int[mArraySize][];

            if (!hasFsetChildren()) {
                // Leaf edge: full energy = lambdaOnly + lambdaM (no child logZ)
                for (int mIdx = 0; mIdx < mArraySize; mIdx++) {
                    int[] mRCs = decodeMState(mIdx);
                    double[] minEnergies = new double[totalLambdaStates];
                    for (int lIdx = 0; lIdx < totalLambdaStates; lIdx++) {
                        int[] lambdaRCs = decodeLambdaState(lIdx);
                        double lmRigid = computeLambdaMEnergy(mRCs, lambdaRCs, rigidEmat, G);
                        double lmMin = computeLambdaMEnergy(mRCs, lambdaRCs, minEmat, G);
                        fullEnergyRigid[mIdx][lIdx] = lambdaOnlyRigid[lIdx] + lmRigid;
                        fullEnergyMin[mIdx][lIdx] = lambdaOnlyMin[lIdx] + lmMin;
                        minEnergies[lIdx] = fullEnergyMin[mIdx][lIdx];
                    }
                    // Sort lambda indices by E_min ascending (highest Boltzmann first)
                    sortedLambdaIndices[mIdx] = sortIndicesBy(minEnergies);
                }
            } else {
                // Non-leaf edge: use identity order initially (will recompute on demand)
                for (int mIdx = 0; mIdx < mArraySize; mIdx++) {
                    sortedLambdaIndices[mIdx] = new int[totalLambdaStates];
                    for (int i = 0; i < totalLambdaStates; i++) {
                        sortedLambdaIndices[mIdx][i] = i;
                    }
                }
            }
        }

        // Set initial logZ bounds for materialized incremental callers.  A
        // non-materialized full-DP sweep overwrites every M-state immediately;
        // skipping this whole-table pre-write avoids another pass over every
        // corrected proposal table.
        if (materializeFullEnergyTables
                && !skipInitialMappedFill(true)) {
            dpTable.fill(NEG_INF, NEG_INF);
        }
        if (materializeFullEnergyTables) {
            // logZ_upper = tail bound for all K lambda-states in incremental mode.
            for (int mIdx = 0; mIdx < mArraySize; mIdx++) {
                dpTable.set(mIdx, NEG_INF, computeTailBound(mIdx, 0));
            }
        }
    }

    private void checkMaterializedPairCount() {
        long pairs = mStateCount * (long) totalLambdaStates;
        long maxPairs = getConfigLong(DP_MAX_MATERIALIZED_PAIRS_PROPERTY,
                DEFAULT_DP_MAX_MATERIALIZED_PAIRS);
        if (pairs > maxPairs) {
            throw new IllegalStateException(BranchDpConfig.getBackendLogPrefix() + " Refusing to materialize DP full-energy table with "
                    + pairs + " (M,lambda) entries for M=" + Arrays.toString(mPositionsSorted)
                    + ", lambda=" + Arrays.toString(lambdaPositionsSorted)
                    + ". Full DP now streams this table; region-atom/incremental paths can raise "
                    + DP_MAX_MATERIALIZED_PAIRS_PROPERTY + " only if memory allows.");
        }
    }

    /**
     * Sort indices [0..n-1] by values[i] ascending. Returns sorted index array.
     */
    private static int[] sortIndicesBy(double[] values) {
        Integer[] indices = new Integer[values.length];
        for (int i = 0; i < indices.length; i++) indices[i] = i;
        Arrays.sort(indices, Comparator.comparingDouble(i -> values[i]));
        int[] result = new int[indices.length];
        for (int i = 0; i < indices.length; i++) result[i] = indices[i];
        return result;
    }

    /**
     * Compute tail bound for un-enumerated lambda-states.
     *
     * For leaf edges: remaining (K-k) states each have E_min >= E_min of the k-th sorted state.
     *   tail = log(K-k) + (-E_min(sorted[k]) / RT)
     *
     * For non-leaf edges: uses a looser bound based on best possible local energy
     *   + best child logZ_upper values.
     *
     * @param mIdx  M-state index
     * @param k     number already enumerated
     * @return log-space tail bound (upper)
     */
    private double computeTailBound(int mIdx, int k) {
        int remaining = totalLambdaStates - k;
        if (remaining <= 0) return NEG_INF;

        if (!hasFsetChildren()) {
            // Leaf edge: sorted by E_min, so sorted[k] is the best remaining
            int bestRemainingLambdaIdx = sortedLambdaIndices[mIdx][k];
            double bestRemainingEMin = fullEnergyMin[mIdx][bestRemainingLambdaIdx];
            return Math.log(remaining) + (-bestRemainingEMin / cachedRT);
        } else {
            // Non-leaf edge: use conservative bound
            // Best local energy among all lambda-states
            double bestLocalEMin = Double.MAX_VALUE;
            for (int lIdx = 0; lIdx < totalLambdaStates; lIdx++) {
                int[] mRCs = fullDpMDecodeBuffer();
                decodeMStateInto(mIdx, mRCs);
                int[] lambdaRCs = decodeLambdaState(lIdx);
                double localE = computeLocalEnergy(mRCs, lambdaRCs, cachedMinEmat, cachedG);
                bestLocalEMin = Math.min(bestLocalEMin, localE);
            }
            // Best child logZ_upper for any M-state mapping
            double bestFSumUpper = 0.0;
            if (Fset != null) {
                for (RootedTreeEdge fEdge : Fset) {
                    double bestChildUpper = NEG_INF;
                    for (long fi = 0; fi < fEdge.getMStateCount(); fi++) {
                        bestChildUpper = Math.max(bestChildUpper, fEdge.getLogZUpper(fi));
                    }
                    bestFSumUpper += bestChildUpper;
                }
            }
            return Math.log(remaining) + (-bestLocalEMin / cachedRT) + bestFSumUpper;
        }
    }

    /** Whether this edge has F-set children (non-leaf in compact tree). */
    public boolean hasFsetChildren() {
        return Fset != null && !Fset.isEmpty();
    }

    /**
     * Get the rigid and min energies for a specific (mIdx, lambdaIdx) on a leaf edge.
     * Used by BranchDpBackend for correction computation.
     */
    public double getFullEnergyRigid(long mIdx, int lambdaIdx) {
        if (fullEnergyRigid != null) {
            return fullEnergyRigid[(int) mIdx][lambdaIdx];
        }
        return computeFullEnergy(mIdx, lambdaIdx, cachedRigidEmat, lambdaOnlyRigid);
    }

    public double getFullEnergyMin(long mIdx, int lambdaIdx) {
        if (fullEnergyMin != null) {
            return fullEnergyMin[(int) mIdx][lambdaIdx];
        }
        return computeFullEnergy(mIdx, lambdaIdx, cachedMinEmat, lambdaOnlyMin);
    }

    private double computeFullEnergy(long mIdx, int lambdaIdx,
                                     EnergyMatrix emat, double[] lambdaOnlyEnergies) {
        int[] mRCs = decodeMState(mIdx);
        if (!emat.hasHigherOrderTerms() && lambdaOnlyEnergies != null) {
            return lambdaOnlyEnergies[lambdaIdx]
                    + computeLambdaMEnergyForIndex(
                    mRCs, lambdaIdx, emat, cachedG);
        }
        int[] lambdaRCs = decodeLambdaState(lambdaIdx);
        double lambdaOnly = lambdaOnlyEnergies != null
                ? lambdaOnlyEnergies[lambdaIdx]
                : computeLambdaOnlyEnergy(lambdaRCs, emat, cachedG);
        return lambdaOnly + computeLambdaMEnergy(mRCs, lambdaRCs, emat, cachedG);
    }

    // ========== One-shot bottom-up DP ==========

    /**
     * Compute logZ bounds for all M-states by enumerating ALL lambda-states at once.
     * Enumerates all lambda-states in one pass.
     *
     * Must be called in post-order: all F-set children must have their logZ computed first.
     *
     * After this call:
     *   logZLower[mIdx] = logSumExp over all λ-states of (-eRigid/RT + fSumLower)
     *   logZUpper[mIdx] = logSumExp over all λ-states of (-eMin/RT + fSumUpper)
     * No tail bound needed — all λ-states are enumerated.
     */
    public void computeFullDP() {
        if (!isLambdaEdge) return;

        // Build the read-only child-fold plan before any parallel dispatch.
        ensureChildFoldPlans();

        if (tryComputeFullDPGpu()) {
            if (enumeratedCount != null) {
                Arrays.fill(enumeratedCount, totalLambdaStates);
            }
            return;
        }

        if (shouldParallelizeFullDP()) {
            computeFullDPSharded();
        } else {
            LongStream.range(0, mStateCount).forEach(this::computeFullDPForMState);
        }

        // Mark all M-states as fully enumerated (for Phase 2 conformation generation)
        if (enumeratedCount != null) {
            Arrays.fill(enumeratedCount, totalLambdaStates);
        }
    }

    private boolean shouldParallelizeFullDP() {
        return getConfigBoolean(DP_PARALLEL_PROPERTY, true)
                && mStateCount >= getConfigInteger(DP_PARALLEL_MIN_M_STATES_PROPERTY,
                        DEFAULT_DP_PARALLEL_MIN_M_STATES);
    }

    /**
     * Shard-scheduled parallel full DP (design 2.5).
     *
     * The M-state range [0, mStateCount) is split into contiguous shards of
     * {@code computeShardSize} M-states. A bounded, fixed-size thread pool pulls
     * shards off a shared atomic cursor (dynamic load balancing, so uneven shard
     * cost does not let one shard monopolize a worker). Contiguous shards keep
     * each worker's reads/writes local to a few backing arrays, which is friendly
     * to both cache and the ShardedDPTable layout.
     *
     * Safety: each M-state writes a distinct dpTable index, child edges are
     * already fully computed (post-order) and read-only here, and the log-sum-exp
     * accumulators are per-M-state locals, so contiguous-range partitioning has no
     * data races.
     */
    private void computeFullDPSharded() {
        int threads = resolveDPThreads();
        long shardSize = resolveComputeShardSize();
        long numShards = (mStateCount + shardSize - 1L) / shardSize;
        boolean progress = getConfigBoolean(DP_PROGRESS_PROPERTY, true);

        if (numShards <= 1 || threads <= 1) {
            LongStream.range(0, mStateCount).forEach(this::computeFullDPForMState);
            return;
        }

        int workers = (int) Math.min(threads, numShards);
        ExecutorService pool = Executors.newFixedThreadPool(workers,
                daemonThreadFactory(BranchDpConfig.getBackendThreadNamePrefix() + "-dp"));
        AtomicLong nextShard = new AtomicLong(0);
        AtomicLong completedShards = new AtomicLong(0);
        long startNanos = System.nanoTime();
        long logEvery = Math.max(1L, numShards / 20L);

        if (progress) {
            System.out.println(BranchDpConfig.getBackendLogPrefix() + " DP shard schedule start, mStates=" + mStateCount
                    + ", shards=" + numShards + ", shardSize=" + shardSize
                    + ", workers=" + workers);
        }

        List<Future<?>> futures = new ArrayList<>(workers);
        for (int w = 0; w < workers; w++) {
            futures.add(pool.submit(() -> {
                long s;
                while ((s = nextShard.getAndIncrement()) < numShards) {
                    long shardStartNanos = System.nanoTime();
                    long mStart = s * shardSize;
                    long mEnd = Math.min(mStart + shardSize, mStateCount);
                    for (long mIdx = mStart; mIdx < mEnd; mIdx++) {
                        computeFullDPForMState(mIdx);
                    }
                    long done = completedShards.incrementAndGet();
                    if (progress && (done % logEvery == 0 || done == numShards)) {
                        double shardMs = (System.nanoTime() - shardStartNanos) / 1e6;
                        double elapsedS = (System.nanoTime() - startNanos) / 1e9;
                        System.out.println(BranchDpConfig.getBackendLogPrefix() + " DP shards " + done + "/" + numShards
                                + " (" + String.format(Locale.ROOT, "%.1f", 100.0 * done / numShards)
                                + "%), lastShard=" + String.format(Locale.ROOT, "%.1f", shardMs)
                                + "ms, elapsed=" + String.format(Locale.ROOT, "%.1f", elapsedS) + "s");
                    }
                }
            }));
        }

        try {
            for (Future<?> f : futures) {
                f.get();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("DP shard scheduling interrupted", e);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause();
            if (cause instanceof RuntimeException) {
                throw (RuntimeException) cause;
            }
            if (cause instanceof Error) {
                throw (Error) cause;
            }
            throw new RuntimeException("DP shard computation failed", cause);
        } finally {
            pool.shutdownNow();
        }

        if (progress) {
            double totalS = (System.nanoTime() - startNanos) / 1e9;
            System.out.println(BranchDpConfig.getBackendLogPrefix() + " DP shard schedule done, shards=" + numShards
                    + ", elapsed=" + String.format(Locale.ROOT, "%.1f", totalS) + "s");
        }
    }

    private int resolveDPThreads() {
        int configured = getConfigInteger(DP_PARALLEL_THREADS_PROPERTY, 0);
        if (configured > 0) {
            return configured;
        }
        return Math.max(1, Runtime.getRuntime().availableProcessors());
    }

    private long resolveComputeShardSize() {
        long shardSize = getConfigInteger(DP_COMPUTE_SHARD_SIZE_PROPERTY, DEFAULT_DP_COMPUTE_SHARD_SIZE);
        return Math.max(1L, shardSize);
    }

    /**
     * An edge is structurally unsupported by the GPU full-DP path even though GPU DP
     * was requested and the edge's work is non-trivial. Fails fast by default
     * (branchdp.dp.gpu.failIfNoGpuPath=true) rather than silently falling through to
     * computeFullDPSharded()/the plain Java path, which is single-threaded-per-shard
     * and can take hours to days on multi-billion-state edges with zero GPU speedup.
     * Set the property to false to restore the old silent-fallback behavior.
     */
    private boolean failNoGpuPath(String reason) {
        String msg = "GPU DP unsupported for this edge (mStates=" + mStateCount
                + ", lambdaStates=" + totalLambdaStates + "): " + reason;
        if (getConfigBoolean(DP_GPU_FAIL_IF_NO_GPU_PATH_PROPERTY, DEFAULT_DP_GPU_FAIL_IF_NO_GPU_PATH)) {
            throw new DPGpuFullDP.GpuUnsupportedEdgeException(msg
                    + " -- set -D" + DP_GPU_FAIL_IF_NO_GPU_PATH_PROPERTY + "=false to fall back to the"
                    + " (much slower) Java DP path for this edge instead of aborting");
        }
        System.out.println(BranchDpConfig.getBackendLogPrefix() + " GPU DP skipped (falling back to Java), " + msg);
        return false;
    }

    private boolean tryComputeFullDPGpu() {
        if (!getConfigBoolean(DP_GPU_PROPERTY, BranchDpConfig.isPackStarAliasScopeActive())) {
            return false;
        }
        {}
        // Leaf edges (no F-set children) have nothing to fold, but the GPU kernel
        // already handles numChildren=0 as a trivial empty sum (child_sum() loops
        // 0 times), so a leaf edge with enough raw (mState x lambdaState) work is
        // still worth dispatching to GPU -- it just skips every childFoldPlans-only
        // check below. Only non-leaf edges require a populated fold plan.
        boolean isLeafEdge = !hasFsetChildren();
        if (!isLeafEdge && (childFoldPlans == null || childFoldPlans.length < 1)) {
            return false; // non-leaf edge with no fold plan (yet): benign, not ready
        }
        if (dpTable == null) {
            return false;
        }
        if (mStateCount <= 0 || totalLambdaStates <= 0) {
            return false; // trivial/empty edge
        }
        if (mStateCount > Long.MAX_VALUE/(long)totalLambdaStates) {
            // Work itself overflows a long; no amount of hardware fixes this structurally.
            return failNoGpuPath("mStateCount*totalLambdaStates overflows a 64-bit long");
        }

        long work = mStateCount*(long)totalLambdaStates;
        long minWork = getConfigLong(DP_GPU_MIN_WORK_PROPERTY, DEFAULT_DP_GPU_MIN_WORK);
        if (work < minWork) {
            return false; // deliberately not worth GPU dispatch overhead
        }

        if (!isLeafEdge) {
            if (childFoldPlans.length > DP_GPU_MAX_CHILDREN) {
                return failNoGpuPath("edge has " + childFoldPlans.length + " children, exceeds"
                        + " DP_GPU_MAX_CHILDREN=" + DP_GPU_MAX_CHILDREN);
            }
        }
        if (mPositionsSorted.length > DPGpuFullDP.MAX_EDGE_POSITIONS
                || lambdaPositionsSorted.length > DPGpuFullDP.MAX_EDGE_POSITIONS) {
            return failNoGpuPath("edge has mPositions=" + mPositionsSorted.length
                    + ", lambdaPositions=" + lambdaPositionsSorted.length
                    + ", exceeds MAX_EDGE_POSITIONS=" + DPGpuFullDP.MAX_EDGE_POSITIONS);
        }
        if (!isLeafEdge) {
            for (ChildFoldPlan plan : childFoldPlans) {
                if (plan == null || plan.child == null) {
                    return false; // missing fold plan: benign/not-ready-yet, not a capacity limit
                }
                // NOTE: no dense-table requirement on the child anymore -- child tables are
                // uploaded chunk-by-chunk (getLogZLowerChunks()/getLogZUpperChunks()), so a
                // sharded (>Integer.MAX_VALUE-element) child is fully supported on GPU.
                if (plan.mSrcIdx.length > DPGpuFullDP.MAX_EDGE_POSITIONS
                        || plan.lamSrcIdx.length > DPGpuFullDP.MAX_EDGE_POSITIONS) {
                    return failNoGpuPath("a child fold plan exceeds MAX_EDGE_POSITIONS="
                            + DPGpuFullDP.MAX_EDGE_POSITIONS);
                }
            }
        }
        if (lambdaOnlyRigid == null || lambdaOnlyMin == null
                || cachedRigidEmat == null || cachedMinEmat == null
                || cachedG == null || cachedRT == 0.0 || !Double.isFinite(cachedRT)) {
            return false; // precomputed state not ready yet: benign
        }

        DPGpuFullDP.Request req;
        try {
            req = buildGpuFullDPRequest(work);
        } catch (GpuHigherOrderPackingException ex) {
            return failNoGpuPath(ex.getMessage());
        }
        if (req == null) {
            return failNoGpuPath("lm cross-term count (or a child CSR index array) overflows a 32-bit int"
                    + " while building the GPU request");
        }

        long maxBytes = getConfigLong(DP_GPU_MAX_BYTES_PROPERTY, DEFAULT_DP_GPU_MAX_BYTES);
        req.maxDeviceBytes = maxBytes;
        req.estimatedFullDeviceBytes = DPGpuFullDP.estimateDeviceBytes(req);
        req.estimatedSlicedDeviceBytes = DPGpuFullDP.estimateSlicedDeviceBytes(req);
        req.estimatedOutOfCoreMinimumDeviceBytes =
                DPGpuOutOfCore.estimateMinimumDeviceBytes(req);
        req.estimatedDeviceBytes = req.childSlicing
                ? Math.min(req.estimatedFullDeviceBytes,
                Math.min(req.estimatedSlicedDeviceBytes,
                        req.estimatedOutOfCoreMinimumDeviceBytes))
                : req.estimatedFullDeviceBytes;
        boolean progress = getConfigBoolean(DP_PROGRESS_PROPERTY, true);
        req.progress = progress;
        // maxBytes <= 0 means "no static cap": defer to the detected-VRAM gate in
        // DPGpuFullDP. A positive cap is an explicit operator limit.
        if (maxBytes > 0 && req.estimatedDeviceBytes > maxBytes) {
            String msg = "GPU DP footprint " + req.estimatedDeviceBytes + " B exceeds "
                    + DP_GPU_MAX_BYTES_PROPERTY + "=" + maxBytes + " B (lambdaStates="
                    + totalLambdaStates + ", childTableElems=" + req.childTableTotal + ")";
            if (getConfigBoolean(DP_GPU_FAIL_IF_EXCEEDS_VRAM_PROPERTY, true)) {
                throw new DPGpuFullDP.GpuMemoryExceededException(msg
                        + " -- resubmit on a larger-VRAM GPU or raise " + DP_GPU_MAX_BYTES_PROPERTY);
            }
            if (progress) {
                System.out.println(BranchDpConfig.getBackendLogPrefix() + " GPU DP skipped, " + msg);
            }
            return false;
        }

        if (progress) {
            System.out.println(BranchDpConfig.getBackendLogPrefix() + " GPU DP start, mStates=" + mStateCount
                    + ", lambdaStates=" + totalLambdaStates
                    + ", children=" + req.numChildren
                    + ", work=" + work
                    + ", lmPairs=" + req.lmLamSlots.length
                    + ", lmTerms=" + req.lmRigid.length
                    + ", tripleFactors=" + req.tripleOffsets.length
                    + ", tripleTerms=" + req.tripleRigid.length
                    + ", outputTileMStates=" + req.mStateChunk
                    + ", outputTableDense=" + dpTable.isDenseArrayBacked()
                    + ", childSlicing=" + req.childSlicing
                    + ", childSlicingForce=" + req.forceChildSlicing
                    + ", persistentContext=" + req.persistentContext
                    + ", estimatedFullDeviceBytes=" + req.estimatedFullDeviceBytes
                    + ", estimatedSlicedDeviceBytes=" + req.estimatedSlicedDeviceBytes
                    + ", estimatedOutOfCoreMinimumDeviceBytes="
                    + req.estimatedOutOfCoreMinimumDeviceBytes
                    + ", estimatedDeviceBytes=" + req.estimatedDeviceBytes);
        }
        return DPGpuFullDP.compute(req);
    }

    private DPGpuFullDP.Request buildGpuFullDPRequest(long work) {
        long lmTermCount = 0L;
        int[][] lmPairs = getGpuLambdaMTermDescriptors();
        for (int[] pair : lmPairs) {
            long terms = (long)pair[4] * (long)pair[5];
            if (terms < 0 || lmTermCount > Integer.MAX_VALUE - terms) {
                return null;
            }
            lmTermCount += terms;
        }

        DPGpuFullDP.Request req = new DPGpuFullDP.Request();
        req.mStateCount = mStateCount;
        req.totalLambdaStates = totalLambdaStates;
        req.blockThreads = Math.max(1, getConfigInteger(DP_GPU_BLOCK_THREADS_PROPERTY,
                DEFAULT_DP_GPU_BLOCK_THREADS));
        req.invRT = 1.0/cachedRT;
        req.work = work;
        ensureIndexingCaches();
        req.mCounts = mCounts;
        req.lambdaCounts = lambdaCounts;
        req.lambdaOnlyRigid = lambdaOnlyRigid;
        req.lambdaOnlyMin = lambdaOnlyMin;
        req.childSlicing = DPGpuFullDP.defaultChildSlicingEnabled();
        req.forceChildSlicing = DPGpuFullDP.forceChildSlicingEnabled();
        req.childSliceMaxBytes = DPGpuFullDP.configuredChildSliceMaxBytes();
        req.outOfCoreOutputWorkspaceMaxBytes =
                DPGpuFullDP.configuredOutOfCoreOutputWorkspaceMaxBytes();
        // Concatenate the per-child fold plans CSR-style for full_dp_n_children.
        // Leaf edges have childFoldPlans == null (nothing to fold): treat as 0 children.
        int nChildren = (childFoldPlans != null) ? childFoldPlans.length : 0;
        req.numChildren = nChildren;
        req.mStateChunk = resolveGpuOutputTileMStates();
        req.outTable = dpTable;
        req.persistentContext = getConfigBoolean(DP_GPU_PERSISTENT_CONTEXT_PROPERTY,
                DEFAULT_DP_GPU_PERSISTENT_CONTEXT);
        req.multiGpu = getConfigBoolean(DP_GPU_MULTI_PROPERTY, true);
        req.maxGpus = getConfigInteger(DP_GPU_MAX_GPUS_PROPERTY, 0);
        req.minMStatesPerGpu = getConfigLong(DP_GPU_MIN_MSTATES_PER_GPU_PROPERTY,
                DEFAULT_DP_GPU_MIN_MSTATES_PER_GPU);

        long mTermTotal = 0L, lTermTotal = 0L, tableTotal = 0L;
        if (childFoldPlans != null) {
            for (ChildFoldPlan plan : childFoldPlans) {
                mTermTotal += plan.mSrcIdx.length;
                lTermTotal += plan.lamSrcIdx.length;
                // getMStateCount() (not getLogZLower().length) so a sharded (>2^31-element)
                // child's size can be read without ever throwing/materializing one array.
                tableTotal += plan.child.getMStateCount();
            }
        }
        // The CSR index arrays ARE concatenated into single Java int[] buffers, so they
        // must each fit in an int (in practice tiny: ~#positions). The child *tables* are
        // no longer concatenated on the Java heap -- each child's chunks (one chunk for a
        // dense child, one per shard for a sharded child) are copied straight into their
        // device-buffer slots -- so their combined length (tableTotal) may exceed
        // Integer.MAX_VALUE; only device VRAM bounds it (enforced at upload, where an
        // over-budget edge fails the run instead of crawling on CPU).
        if (mTermTotal > Integer.MAX_VALUE || lTermTotal > Integer.MAX_VALUE) {
            return null;
        }

        req.childMSrcAll = new int[(int)mTermTotal];
        req.childMStrideAll = new long[(int)mTermTotal];
        req.childMTermOff = new int[nChildren];
        req.childMTermCnt = new int[nChildren];
        req.childLSrcAll = new int[(int)lTermTotal];
        req.childLStrideAll = new long[(int)lTermTotal];
        req.childLTermOff = new int[nChildren];
        req.childLTermCnt = new int[nChildren];
        req.childTableBase = new long[nChildren];
        req.childTableTotal = tableTotal;
        req.childTables = new DPTable[nChildren];
        req.childMCountsAll = new int[(int)mTermTotal];
        req.childMPackedStrideAll = new long[(int)mTermTotal];
        req.childLCountsAll = new int[(int)lTermTotal];
        req.childLPackedStrideAll = new long[(int)lTermTotal];
        req.childMRowStates = new long[nChildren];
        req.childLambdaStates = new long[nChildren];
        buildGpuChildSliceDimensions(req);

        // Upload-side chunk lists: one entry per physical chunk (dense child => 1 chunk,
        // sharded child => 1 chunk per shard), NOT one entry per logical child. Sized
        // dynamically since the chunk count per child isn't known until we ask.
        List<double[]> lowerChunkList = new ArrayList<>();
        List<double[]> upperChunkList = new ArrayList<>();
        List<Long> chunkOffsetList = new ArrayList<>();

        int mOff = 0, lOff = 0;
        long tBase = 0L;
        for (int c = 0; c < nChildren; c++) {
            ChildFoldPlan plan = childFoldPlans[c];
            int childMOff = mOff;
            req.childMTermOff[c] = mOff;
            req.childMTermCnt[c] = plan.mSrcIdx.length;
            System.arraycopy(plan.mSrcIdx, 0, req.childMSrcAll, mOff, plan.mSrcIdx.length);
            System.arraycopy(plan.mStride, 0, req.childMStrideAll, mOff, plan.mStride.length);
            long childMStates = 1L;
            for (int t = plan.mSrcIdx.length - 1; t >= 0; t--) {
                int dst = childMOff + t;
                req.childMPackedStrideAll[dst] = childMStates;
                int count = req.mCounts[plan.mSrcIdx[t]];
                req.childMCountsAll[dst] = count;
                childMStates = saturatingMultiply(childMStates, count);
            }
            req.childMRowStates[c] = childMStates;
            mOff += plan.mSrcIdx.length;

            int childLOff = lOff;
            req.childLTermOff[c] = lOff;
            req.childLTermCnt[c] = plan.lamSrcIdx.length;
            System.arraycopy(plan.lamSrcIdx, 0, req.childLSrcAll, lOff, plan.lamSrcIdx.length);
            System.arraycopy(plan.lamStride, 0, req.childLStrideAll, lOff, plan.lamStride.length);
            long childLStates = 1L;
            for (int t = plan.lamSrcIdx.length - 1; t >= 0; t--) {
                int dst = childLOff + t;
                req.childLPackedStrideAll[dst] = childLStates;
                int count = req.lambdaCounts[plan.lamSrcIdx[t]];
                req.childLCountsAll[dst] = count;
                childLStates = saturatingMultiply(childLStates, count);
            }
            req.childLambdaStates[c] = childLStates;
            lOff += plan.lamSrcIdx.length;

            // Each child's chunks land contiguously in the device buffer starting at
            // childTableBase[c] (a long element offset): consecutive chunks of the same
            // child get consecutive offsets, so the kernel's flat childTableBase[c]-based
            // indexing is unaffected by how many host-side chunks filled that region.
            req.childTableBase[c] = tBase;
            req.childTables[c] = plan.child.dpTable;
            if (req.childTables[c].supportsArrayChunks()) {
                double[][] lowerChunks = plan.child.getLogZLowerChunks();
                double[][] upperChunks = plan.child.getLogZUpperChunks();
                long chunkBase = tBase;
                for (int k = 0; k < lowerChunks.length; k++) {
                    lowerChunkList.add(lowerChunks[k]);
                    upperChunkList.add(upperChunks[k]);
                    chunkOffsetList.add(chunkBase);
                    chunkBase += lowerChunks[k] == null ? 0 : lowerChunks[k].length;
                }
            } else {
                req.hasFileBackedChildTables = true;
            }
            tBase += plan.child.getMStateCount();
        }

        req.childLowerChunks = lowerChunkList.toArray(new double[0][]);
        req.childUpperChunks = upperChunkList.toArray(new double[0][]);
        req.childChunkOffsets = new long[chunkOffsetList.size()];
        for (int i = 0; i < req.childChunkOffsets.length; i++) {
            req.childChunkOffsets[i] = chunkOffsetList.get(i);
        }

        int nPairs = lmPairs.length;
        req.lmLamSlots = new int[nPairs];
        req.lmMSlots = new int[nPairs];
        req.lmMCounts = new int[nPairs];
        req.lmOffsets = new long[nPairs];
        req.lmRigid = new double[(int)lmTermCount];
        req.lmMin = new double[(int)lmTermCount];

        int offset = 0;
        for (int p = 0; p < nPairs; p++) {
            int[] pair = lmPairs[p];
            int lambdaSlot = pair[0];
            int mSlot = pair[1];
            int lambdaPos = pair[2];
            int mPos = pair[3];
            int lambdaCount = pair[4];
            int mCount = pair[5];

            req.lmLamSlots[p] = lambdaSlot;
            req.lmMSlots[p] = mSlot;
            req.lmMCounts[p] = mCount;
            req.lmOffsets[p] = offset;

            for (int lrc = 0; lrc < lambdaCount; lrc++) {
                int globalLrc = rcs.get(lambdaPos, lrc);
                for (int mrc = 0; mrc < mCount; mrc++) {
                    int globalMrc = rcs.get(mPos, mrc);
                    int idx = offset + lrc*mCount + mrc;
                    req.lmRigid[idx] = cachedRigidEmat.getPairwise(lambdaPos, globalLrc, mPos, globalMrc);
                    req.lmMin[idx] = cachedMinEmat.getPairwise(lambdaPos, globalLrc, mPos, globalMrc);
                }
            }
            offset += (int)((long)lambdaCount*(long)mCount);
        }

        populateGpuTripleFactors(req);

        return req;
    }

    /** A structurally unsupported higher-order request, safe to route through failNoGpuPath(). */
    private static final class GpuHigherOrderPackingException extends RuntimeException {
        GpuHigherOrderPackingException(String message) {
            super(message);
        }
    }

    /** Canonical (descending-position) identity of one three-position factor table. */
    private static final class GpuTriplePositions implements Comparable<GpuTriplePositions> {
        final int high;
        final int middle;
        final int low;

        GpuTriplePositions(int high, int middle, int low) {
            if (!(high > middle && middle > low)) {
                throw new IllegalArgumentException("triple positions are not strictly descending");
            }
            this.high = high;
            this.middle = middle;
            this.low = low;
        }

        @Override
        public int compareTo(GpuTriplePositions other) {
            int cmp = Integer.compare(high, other.high);
            if (cmp != 0) return cmp;
            cmp = Integer.compare(middle, other.middle);
            if (cmp != 0) return cmp;
            return Integer.compare(low, other.low);
        }

        @Override
        public boolean equals(Object other) {
            if (this == other) return true;
            if (!(other instanceof GpuTriplePositions)) return false;
            GpuTriplePositions rhs = (GpuTriplePositions)other;
            return high == rhs.high && middle == rhs.middle && low == rhs.low;
        }

        @Override
        public int hashCode() {
            return Objects.hash(high, middle, low);
        }
    }

    /**
     * Pack exactly the n=3 factors owned by this M/lambda edge.  Lambda-only
     * factors are already present in lambdaOnlyRigid/lambdaOnlyMin; M-only
     * factors belong to another elimination edge.  Therefore only factors
     * crossing M and lambda are uploaded here.
     */
    private void populateGpuTripleFactors(DPGpuFullDP.Request req) {
        if (!cachedRigidEmat.hasHigherOrderTerms()
                && !cachedMinEmat.hasHigherOrderTerms()) {
            return;
        }

        int[] localPositions = mergeSortedPositions(
                mPositionsSorted, lambdaPositionsSorted);
        SortedSet<GpuTriplePositions> candidates = new TreeSet<>();
        collectGpuTriplePositions(cachedRigidEmat, localPositions, candidates);
        collectGpuTriplePositions(cachedMinEmat, localPositions, candidates);

        List<GpuTriplePositions> crossing = new ArrayList<>();
        long totalTerms = 0L;
        for (GpuTriplePositions triple : candidates) {
            int slot0 = gpuLocalSlot(triple.high);
            int slot1 = gpuLocalSlot(triple.middle);
            int slot2 = gpuLocalSlot(triple.low);
            boolean hasM = slot0 < mPositionsSorted.length
                    || slot1 < mPositionsSorted.length
                    || slot2 < mPositionsSorted.length;
            boolean hasLambda = slot0 >= mPositionsSorted.length
                    || slot1 >= mPositionsSorted.length
                    || slot2 >= mPositionsSorted.length;
            if (!hasM || !hasLambda) {
                continue;
            }
            long terms = (long)rcs.getNum(triple.high)
                    * (long)rcs.getNum(triple.middle)
                    * (long)rcs.getNum(triple.low);
            if (terms < 0L || totalTerms > Integer.MAX_VALUE - terms) {
                throw new GpuHigherOrderPackingException(
                        "exact triple-factor table exceeds the Java/CUDA 32-bit dense-table limit");
            }
            crossing.add(triple);
            totalTerms += terms;
        }

        int factorCount = crossing.size();
        req.tripleSlots = new int[factorCount * 3];
        req.tripleStrides = new long[factorCount * 3];
        req.tripleOffsets = new long[factorCount];
        req.tripleRigid = new double[(int)totalTerms];
        req.tripleMin = new double[(int)totalTerms];

        int tableOffset = 0;
        for (int factor = 0; factor < factorCount; factor++) {
            GpuTriplePositions triple = crossing.get(factor);
            int highCount = rcs.getNum(triple.high);
            int middleCount = rcs.getNum(triple.middle);
            int lowCount = rcs.getNum(triple.low);
            int meta = factor * 3;
            req.tripleSlots[meta] = gpuLocalSlot(triple.high);
            req.tripleSlots[meta + 1] = gpuLocalSlot(triple.middle);
            req.tripleSlots[meta + 2] = gpuLocalSlot(triple.low);
            req.tripleStrides[meta] = (long)middleCount * (long)lowCount;
            req.tripleStrides[meta + 1] = lowCount;
            req.tripleStrides[meta + 2] = 1L;
            req.tripleOffsets[factor] = tableOffset;

            for (int highRc = 0; highRc < highCount; highRc++) {
                for (int middleRc = 0; middleRc < middleCount; middleRc++) {
                    for (int lowRc = 0; lowRc < lowCount; lowRc++) {
                        int index = tableOffset
                                + (highRc * middleCount + middleRc) * lowCount
                                + lowRc;
                        req.tripleRigid[index] = gpuTripleValue(cachedRigidEmat,
                                triple, highRc, middleRc, lowRc);
                        req.tripleMin[index] = gpuTripleValue(cachedMinEmat,
                                triple, highRc, middleRc, lowRc);
                    }
                }
            }
            // totalTerms was checked in long arithmetic above. Keep this
            // increment in the same domain so an individual dense factor
            // cannot overflow an intermediate int multiplication.
            tableOffset += (int)((long)highCount
                    * (long)middleCount * (long)lowCount);
        }
    }

    /** Discover canonical direct triples and reject cross-edge factors of order n&gt;3. */
    private void collectGpuTriplePositions(
            EnergyMatrix emat,
            int[] localPositions,
            SortedSet<GpuTriplePositions> out) {
        if (!emat.hasHigherOrderTerms()) return;

        for (int highIndex = 2; highIndex < localPositions.length; highIndex++) {
            int high = localPositions[highIndex];
            for (int middleIndex = 1; middleIndex < highIndex; middleIndex++) {
                int middle = localPositions[middleIndex];
                for (int highRc = 0; highRc < rcs.getNum(high); highRc++) {
                    int globalHighRc = rcs.get(high, highRc);
                    for (int middleRc = 0; middleRc < rcs.getNum(middle); middleRc++) {
                        int globalMiddleRc = rcs.get(middle, middleRc);
                        HigherTupleFinder<Double> finder = emat.getHigherOrderTerms(
                                high, globalHighRc, middle, globalMiddleRc);
                        if (finder == null) continue;

                        for (int lowIndex = 0; lowIndex < middleIndex; lowIndex++) {
                            int low = localPositions[lowIndex];
                            if (!finder.getInteractingPos().contains(low)) continue;
                            boolean hasDirectTriple = false;
                            for (int lowRc = 0; lowRc < rcs.getNum(low); lowRc++) {
                                int globalLowRc = rcs.get(low, lowRc);
                                Double value = finder.getInteraction(low, globalLowRc);
                                if (value != null && value.doubleValue() != 0.0) {
                                    hasDirectTriple = true;
                                }
                                if (finder.getHigherInteractions(low, globalLowRc) != null) {
                                    throw new GpuHigherOrderPackingException(
                                            "CUDA exact DP supports three-position factors, but this edge"
                                                    + " contains a factor of order greater than three");
                                }
                            }
                            if (hasDirectTriple) {
                                out.add(new GpuTriplePositions(high, middle, low));
                            }
                        }
                    }
                }
            }
        }
    }

    /** Unified GPU local slot: M first, followed by lambda. */
    private int gpuLocalSlot(int position) {
        int slot = Arrays.binarySearch(mPositionsSorted, position);
        if (slot >= 0) return slot;
        slot = Arrays.binarySearch(lambdaPositionsSorted, position);
        if (slot >= 0) return mPositionsSorted.length + slot;
        throw new IllegalArgumentException("position is outside this edge: " + position);
    }

    private double gpuTripleValue(
            EnergyMatrix emat,
            GpuTriplePositions triple,
            int highLocalRc,
            int middleLocalRc,
            int lowLocalRc) {
        HigherTupleFinder<Double> finder = emat.getHigherOrderTerms(
                triple.high, rcs.get(triple.high, highLocalRc),
                triple.middle, rcs.get(triple.middle, middleLocalRc));
        if (finder == null) return 0.0;
        Double value = finder.getInteraction(
                triple.low, rcs.get(triple.low, lowLocalRc));
        return value == null ? 0.0 : value;
    }

    private long resolveGpuOutputTileMStates() {
        long requested = getConfigLong(DP_GPU_OUTPUT_TILE_MSTATES_PROPERTY,
                DEFAULT_DP_GPU_OUTPUT_TILE_MSTATES);
        long tile = Math.max(1L, requested);
        tile = Math.min(tile, (long)Integer.MAX_VALUE);
        return Math.min(tile, mStateCount);
    }

    // ===== PAC Phase-1 GPU sampling (Gumbel-max over the DP upper-bound weight) =====

    /** Structural gate: this lambda edge can be sampled by the GPU Gumbel kernel. */
    public boolean canUseGpuSampling() {
        if (!isLambdaEdge) {
            return false;
        }
        ensureChildFoldPlans();
        if (!hasFsetChildren() || childFoldPlans == null
                || childFoldPlans.length < 1
                || childFoldPlans.length > DP_GPU_MAX_CHILDREN) {
            return false;
        }
        if (totalLambdaStates <= 0) {
            return false;
        }
        if (mPositionsSorted.length > SamplingGpuPhase1.MAX_EDGE_POSITIONS
                || lambdaPositionsSorted.length > SamplingGpuPhase1.MAX_EDGE_POSITIONS) {
            return false;
        }
        for (ChildFoldPlan plan : childFoldPlans) {
            if (plan == null || plan.child == null || !plan.child.hasDenseDPTable()
                    || plan.mSrcIdx.length > SamplingGpuPhase1.MAX_EDGE_POSITIONS
                    || plan.lamSrcIdx.length > SamplingGpuPhase1.MAX_EDGE_POSITIONS) {
                return false;
            }
        }
        return lambdaOnlyMin != null && cachedMinEmat != null && cachedG != null
                && cachedRT != 0.0 && Double.isFinite(cachedRT);
    }

    /**
     * Draw one lambda-state index per sample on the GPU via Gumbel-max, for the
     * parent M-states in mIdxPerSample. Returns null on any failure so the caller
     * falls back to the Java sampler. Statistically (not bit-) equivalent.
     */
    public int[] sampleLambdaStatesGpu(long[] mIdxPerSample, long baseSeed, boolean progress) {
        return sampleLambdaStatesGpu(mIdxPerSample, baseSeed, progress, null);
    }

    /**
     * One byte-bounded cache per estimator/proposal epoch. The owner MUST clear
     * it before changing any DP table or EMAT, including in-place corrections.
     * No static cache retains a rooted tree or a sample batch.
     */
    public static final class SamplingRequestCache {
        private final edu.duke.cs.osprey.tools.ByteBoundedCache<RootedTreeEdge,
                SamplingGpuPhase1.Request> prepared;
        private final Object owner = new Object();

        public SamplingRequestCache(long maximumBytes) {
            prepared = new edu.duke.cs.osprey.tools.ByteBoundedCache<>(maximumBytes);
        }

        public void clear() {
            prepared.clear();
            SamplingGpuPhase1.releasePreparedModel(owner);
        }
        public long hits() { return prepared.hits(); }
        public long bytes() { return prepared.bytes(); }
    }

    public int[] sampleLambdaStatesGpu(long[] mIdxPerSample, long baseSeed, boolean progress,
                                       SamplingRequestCache cache) {
        if (mIdxPerSample == null || mIdxPerSample.length == 0) {
            return new int[0];
        }
        SamplingGpuPhase1.Request req = prepareGpuSamplingRequest(
                mIdxPerSample, baseSeed, progress, cache);
        if (req == null) {
            return null;
        }
        return SamplingGpuPhase1.sample(req);
    }

    SamplingGpuPhase1.Request prepareGpuSamplingRequest(long[] mIdxPerSample,
            long baseSeed, boolean progress, SamplingRequestCache cache) {
        SamplingGpuPhase1.Request cached = cache == null ? null : cache.prepared.get(this);
        if (cached != null) return cached.forSamples(mIdxPerSample, baseSeed, progress);
        SamplingGpuPhase1.Request req = buildGpuSamplingRequest(mIdxPerSample, baseSeed, progress);
        if (req != null && cache != null && cache.prepared.canStore(req.retainedBytes())) {
            req.modelIdentity = new Object();
            req.modelOwner = cache.owner;
            cache.prepared.put(this, req.forSamples(null, 0L, false), req.retainedBytes());
        }
        return req;
    }

    /** Build the sampling request: the DP upper/min half + per-sample mIdx (no rigid/lower/dpTable). */
    private SamplingGpuPhase1.Request buildGpuSamplingRequest(long[] mIdxPerSample, long baseSeed, boolean progress) {
        long lmTermCount = 0L;
        int[][] lmPairs = getGpuLambdaMTermDescriptors();
        for (int[] pair : lmPairs) {
            long terms = (long)pair[4] * (long)pair[5];
            if (terms < 0 || lmTermCount > Integer.MAX_VALUE - terms) {
                return null;
            }
            lmTermCount += terms;
        }

        SamplingGpuPhase1.Request req = new SamplingGpuPhase1.Request();
        req.totalLambdaStates = totalLambdaStates;
        req.blockThreads = Math.max(1, getConfigInteger(DP_GPU_BLOCK_THREADS_PROPERTY,
                DEFAULT_DP_GPU_BLOCK_THREADS));
        req.invRT = 1.0/cachedRT;
        ensureIndexingCaches();
        req.mCounts = mCounts;
        req.lambdaCounts = lambdaCounts;
        req.lambdaOnlyMin = lambdaOnlyMin;
        req.mIdxPerSample = mIdxPerSample;
        req.baseSeed = baseSeed;
        req.progress = progress;
        req.multiGpu = getConfigBoolean(PAC_SAMPLING_GPU_MULTI_PROPERTY, true);
        req.maxGpus = getConfigInteger(PAC_SAMPLING_GPU_MAX_GPUS_PROPERTY, 0);
        req.minGroupsPerGpu = Math.max(1, getConfigInteger(PAC_SAMPLING_GPU_MIN_GROUPS_PER_GPU_PROPERTY,
                DEFAULT_PAC_SAMPLING_GPU_MIN_GROUPS_PER_GPU));
        req.persistentContext = getConfigBoolean(PAC_SAMPLING_GPU_PERSISTENT_CONTEXT_PROPERTY,
                DEFAULT_PAC_SAMPLING_GPU_PERSISTENT_CONTEXT);
        req.residentChildTables = getConfigBoolean(PAC_SAMPLING_GPU_RESIDENT_CHILD_TABLES_PROPERTY,
                DEFAULT_PAC_SAMPLING_GPU_RESIDENT_CHILD_TABLES);
        req.method = SamplingGpuPhase1.Method.fromProperty(
                getConfigProperty(PAC_SAMPLING_GPU_METHOD_PROPERTY,
                        DEFAULT_PAC_SAMPLING_GPU_METHOD.propertyValue),
                DEFAULT_PAC_SAMPLING_GPU_METHOD);

        int nChildren = childFoldPlans.length;
        req.numChildren = nChildren;

        long mTermTotal = 0L, lTermTotal = 0L, tableTotal = 0L;
        for (ChildFoldPlan plan : childFoldPlans) {
            mTermTotal += plan.mSrcIdx.length;
            lTermTotal += plan.lamSrcIdx.length;
            tableTotal += plan.child.getLogZUpper().length;
        }
        if (mTermTotal > Integer.MAX_VALUE || lTermTotal > Integer.MAX_VALUE
                || tableTotal > Integer.MAX_VALUE) {
            return null;
        }

        req.childMSrcAll = new int[(int)mTermTotal];
        req.childMStrideAll = new long[(int)mTermTotal];
        req.childMTermOff = new int[nChildren];
        req.childMTermCnt = new int[nChildren];
        req.childLSrcAll = new int[(int)lTermTotal];
        req.childLStrideAll = new long[(int)lTermTotal];
        req.childLTermOff = new int[nChildren];
        req.childLTermCnt = new int[nChildren];
        req.childTableBase = new long[nChildren];
        req.childUpperAll = new double[(int)tableTotal];

        int mOff = 0, lOff = 0, tBase = 0;
        long childUpperKey = 0x6a09e667f3bcc909L;
        childUpperKey = mixGpuSamplingKey(childUpperKey, nChildren);
        childUpperKey = mixGpuSamplingKey(childUpperKey, tableTotal);
        for (int c = 0; c < nChildren; c++) {
            ChildFoldPlan plan = childFoldPlans[c];
            req.childMTermOff[c] = mOff;
            req.childMTermCnt[c] = plan.mSrcIdx.length;
            System.arraycopy(plan.mSrcIdx, 0, req.childMSrcAll, mOff, plan.mSrcIdx.length);
            System.arraycopy(plan.mStride, 0, req.childMStrideAll, mOff, plan.mStride.length);
            mOff += plan.mSrcIdx.length;

            req.childLTermOff[c] = lOff;
            req.childLTermCnt[c] = plan.lamSrcIdx.length;
            System.arraycopy(plan.lamSrcIdx, 0, req.childLSrcAll, lOff, plan.lamSrcIdx.length);
            System.arraycopy(plan.lamStride, 0, req.childLStrideAll, lOff, plan.lamStride.length);
            lOff += plan.lamSrcIdx.length;

            double[] cu = plan.child.getLogZUpper();
            req.childTableBase[c] = tBase;
            childUpperKey = mixGpuSamplingKey(childUpperKey, c);
            childUpperKey = mixGpuSamplingKey(childUpperKey, cu.length);
            for (int i = 0; i < cu.length; i++) {
                double value = cu[i];
                req.childUpperAll[tBase + i] = value;
                childUpperKey = mixGpuSamplingKey(childUpperKey, Double.doubleToRawLongBits(value));
            }
            tBase += cu.length;
        }
        req.childUpperCacheKey = childUpperKey;

        int nPairs = lmPairs.length;
        req.lmLamSlots = new int[nPairs];
        req.lmMSlots = new int[nPairs];
        req.lmMCounts = new int[nPairs];
        req.lmOffsets = new long[nPairs];
        req.lmMin = new double[(int)lmTermCount];

        int offset = 0;
        for (int p = 0; p < nPairs; p++) {
            int[] pair = lmPairs[p];
            int lambdaSlot = pair[0];
            int mSlot = pair[1];
            int lambdaPos = pair[2];
            int mPos = pair[3];
            int lambdaCount = pair[4];
            int mCount = pair[5];

            req.lmLamSlots[p] = lambdaSlot;
            req.lmMSlots[p] = mSlot;
            req.lmMCounts[p] = mCount;
            req.lmOffsets[p] = offset;

            for (int lrc = 0; lrc < lambdaCount; lrc++) {
                int globalLrc = rcs.get(lambdaPos, lrc);
                for (int mrc = 0; mrc < mCount; mrc++) {
                    int globalMrc = rcs.get(mPos, mrc);
                    int idx = offset + lrc*mCount + mrc;
                    req.lmMin[idx] = cachedMinEmat.getPairwise(lambdaPos, globalLrc, mPos, globalMrc);
                }
            }
            offset += (int)((long)lambdaCount*(long)mCount);
        }

        return req;
    }

    private static long mixGpuSamplingKey(long h, long x) {
        long z = x + 0x9E3779B97F4A7C15L + (h << 6) + (h >>> 2);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return h ^ (z ^ (z >>> 31));
    }

    private void buildGpuChildSliceDimensions(DPGpuFullDP.Request req) {
        boolean[] used = new boolean[req.mCounts.length];
        // Leaf edges (childFoldPlans == null, nothing to fold) reference no child M
        // slots at all, so every M position is "free": union stays empty and the
        // whole M-state space is the parent-free dimension (no slicing needed/possible).
        if (childFoldPlans != null) {
            for (ChildFoldPlan plan : childFoldPlans) {
                for (int src : plan.mSrcIdx) {
                    used[src] = true;
                }
            }
        }
        long[] parentStrides = mixedRadixStrides(req.mCounts);
        int unionCount = 0;
        int freeCount = 0;
        for (boolean u : used) {
            if (u) {
                unionCount++;
            } else {
                freeCount++;
            }
        }
        req.unionMSlots = new int[unionCount];
        req.unionMCounts = new int[unionCount];
        req.unionMParentStrides = new long[unionCount];
        req.freeMCounts = new int[freeCount];
        req.freeMParentStrides = new long[freeCount];
        req.unionStateCount = 1L;
        req.parentFreeStateCount = 1L;

        int uOff = 0;
        int fOff = 0;
        for (int i = 0; i < req.mCounts.length; i++) {
            if (used[i]) {
                req.unionMSlots[uOff] = i;
                req.unionMCounts[uOff] = req.mCounts[i];
                req.unionMParentStrides[uOff] = parentStrides[i];
                req.unionStateCount = saturatingMultiply(req.unionStateCount, req.mCounts[i]);
                uOff++;
            } else {
                req.freeMCounts[fOff] = req.mCounts[i];
                req.freeMParentStrides[fOff] = parentStrides[i];
                req.parentFreeStateCount = saturatingMultiply(req.parentFreeStateCount, req.mCounts[i]);
                fOff++;
            }
        }
    }

    private static long[] mixedRadixStrides(int[] counts) {
        long[] strides = new long[counts.length];
        long stride = 1L;
        for (int i = counts.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride = saturatingMultiply(stride, counts[i]);
        }
        return strides;
    }

    private static long saturatingMultiply(long a, long b) {
        if (a == 0L || b == 0L) {
            return 0L;
        }
        if (a > Long.MAX_VALUE / b) {
            return Long.MAX_VALUE;
        }
        return a * b;
    }

    private int[] countsForPositions(int[] positions) {
        int[] counts = new int[positions.length];
        for (int i = 0; i < positions.length; i++) {
            counts[i] = rcs.getNum(positions[i]);
        }
        return counts;
    }

    private int[][] getGpuLambdaMTermDescriptors() {
        if (gpuLambdaMTermGraph == cachedG
                && gpuLambdaMTermDescriptors != null) {
            return gpuLambdaMTermDescriptors;
        }
        List<int[]> descriptors = new ArrayList<>();
        for (int lambdaSlot = 0;
             lambdaSlot < lambdaPositionsSorted.length; lambdaSlot++) {
            int lambdaPos = lambdaPositionsSorted[lambdaSlot];
            int lambdaCount = rcs.getNum(lambdaPos);
            for (int mSlot = 0; mSlot < mPositionsSorted.length; mSlot++) {
                int mPos = mPositionsSorted[mSlot];
                if (!cachedG.hasEdge(lambdaPos, mPos)) continue;
                descriptors.add(new int[]{
                        lambdaSlot, mSlot, lambdaPos, mPos,
                        lambdaCount, rcs.getNum(mPos)});
            }
        }
        gpuLambdaMTermDescriptors = descriptors.toArray(new int[0][]);
        gpuLambdaMTermGraph = cachedG;
        return gpuLambdaMTermDescriptors;
    }

    /**
     * Cache the graph-dependent pair-slot lists used by pairwise DP energy
     * evaluation.  The old nested loops asked the interaction graph about
     * every possible slot pair for every lambda state.  The graph and the
     * position sets are immutable for an edge during a DP sweep, so resolving
     * the sparse topology once removes that lookup from the hot loop while
     * retaining the original summation order.
     */
    private void ensureInteractionTermCaches(InteractionGraph G) {
        ensureIndexingCaches();
        if (interactionTermCacheGraph == G
                && lambdaPairFirstSlots != null
                && lambdaMFirstSlots != null) {
            return;
        }

        long lambdaPairCapacity = (long) lambdaPositionsSorted.length
                * Math.max(0, lambdaPositionsSorted.length - 1) / 2L;
        long lambdaMCapacity = (long) lambdaPositionsSorted.length
                * (long) mPositionsSorted.length;
        if (lambdaPairCapacity > Integer.MAX_VALUE
                || lambdaMCapacity > Integer.MAX_VALUE) {
            throw new IllegalStateException(
                    "edge interaction slot count exceeds int indexing limits");
        }

        int pairCount = 0;
        for (int first = 0; first < lambdaPositionsSorted.length; first++) {
            int firstPos = lambdaPositionsSorted[first];
            for (int second = first + 1;
                 second < lambdaPositionsSorted.length; second++) {
                if (!G.hasEdge(firstPos, lambdaPositionsSorted[second])) {
                    continue;
                }
                pairCount++;
            }
        }
        int[] pairFirst = new int[pairCount];
        int[] pairSecond = new int[pairCount];
        int pairOut = 0;
        for (int first = 0; first < lambdaPositionsSorted.length; first++) {
            int firstPos = lambdaPositionsSorted[first];
            for (int second = first + 1;
                 second < lambdaPositionsSorted.length; second++) {
                if (!G.hasEdge(firstPos, lambdaPositionsSorted[second])) {
                    continue;
                }
                pairFirst[pairOut] = first;
                pairSecond[pairOut] = second;
                pairOut++;
            }
        }

        int crossCount = 0;
        for (int lambdaSlot = 0;
             lambdaSlot < lambdaPositionsSorted.length; lambdaSlot++) {
            int lambdaPos = lambdaPositionsSorted[lambdaSlot];
            for (int mSlot = 0; mSlot < mPositionsSorted.length; mSlot++) {
                if (!G.hasEdge(lambdaPos, mPositionsSorted[mSlot])) {
                    continue;
                }
                crossCount++;
            }
        }
        int[] crossLambda = new int[crossCount];
        int[] crossM = new int[crossCount];
        int crossOut = 0;
        for (int lambdaSlot = 0;
             lambdaSlot < lambdaPositionsSorted.length; lambdaSlot++) {
            int lambdaPos = lambdaPositionsSorted[lambdaSlot];
            for (int mSlot = 0; mSlot < mPositionsSorted.length; mSlot++) {
                if (!G.hasEdge(lambdaPos, mPositionsSorted[mSlot])) {
                    continue;
                }
                crossLambda[crossOut] = lambdaSlot;
                crossM[crossOut] = mSlot;
                crossOut++;
            }
        }

        lambdaPairFirstSlots = pairFirst;
        lambdaPairSecondSlots = pairSecond;
        lambdaMFirstSlots = crossLambda;
        lambdaMSecondSlots = crossM;
        interactionTermCacheGraph = G;
    }

    /**
     * Reflection-heavy branch-DP tests and a few legacy callers construct an
     * edge by populating the position arrays directly instead of invoking
     * compLlambda().  Keep the cached indexing fast path compatible with that
     * construction style by rebuilding only missing structural arrays lazily.
     */
    private void ensureIndexingCaches() {
        if (mPositionsSorted == null) {
            mPositionsSorted = M.stream().mapToInt(Integer::intValue)
                    .sorted().toArray();
        }
        if (lambdaPositionsSorted == null) {
            lambdaPositionsSorted = lambda == null
                    ? new int[0]
                    : lambda.stream().mapToInt(Integer::intValue)
                    .sorted().toArray();
        }
        if (mCounts == null) {
            mCounts = countsForPositions(mPositionsSorted);
        }
        if (lambdaCounts == null) {
            lambdaCounts = countsForPositions(lambdaPositionsSorted);
        }
        if (mStrides == null) {
            mStrides = mixedRadixStrides(mCounts);
        }
        if (lambdaStrides == null) {
            lambdaStrides = mixedRadixStrides(lambdaCounts);
        }
    }

    private static ThreadFactory daemonThreadFactory(String namePrefix) {
        AtomicLong counter = new AtomicLong(0);
        return runnable -> {
            Thread t = new Thread(runnable, namePrefix + "-" + counter.getAndIncrement());
            t.setDaemon(true);
            return t;
        };
    }

    private void computeFullDPForMState(long mIdx) {
        {}

        LogSumExpAccumulator lower = FULL_DP_LOWER_ACCUMULATOR.get();
        LogSumExpAccumulator upper = FULL_DP_UPPER_ACCUMULATOR.get();
        lower.reset();
        upper.reset();
        boolean pairwiseEnergy = cachedRigidEmat != null
                && cachedMinEmat != null
                && !cachedRigidEmat.hasHigherOrderTerms()
                && !cachedMinEmat.hasHigherOrderTerms();

        if (!hasFsetChildren()) {
            // Leaf edge: stream lambda states without materializing [M x lambda].
            if (fullEnergyRigid != null && fullEnergyMin != null) {
                // Materialized: read the contiguous per-M rows directly.
                double[] rigidRow = fullEnergyRigid[(int) mIdx];
                double[] minRow = fullEnergyMin[(int) mIdx];
                for (int lIdx = 0; lIdx < totalLambdaStates; lIdx++) {
                    lower.add(-rigidRow[lIdx] / cachedRT);
                    upper.add(-minRow[lIdx] / cachedRT);
                }
            } else {
                // Streamed: hoist the M-state decode (constant across the lambda loop) and
                // decode each lambda-state ONCE for both rigid and min energy. The previous
                // getFullEnergyRigid/getFullEnergyMin path re-decoded the M-state twice and
                // the lambda-state twice per element. This is bit-for-bit identical (same
                // operations, same order) — it only removes redundant decode work/allocations.
                int[] mRCs = fullDpMDecodeBuffer();
                decodeMStateInto(mIdx, mRCs);
                for (int lIdx = 0; lIdx < totalLambdaStates; lIdx++) {
                    double eRigid;
                    double eMin;
                    if (pairwiseEnergy) {
                        eRigid = lambdaOnlyRigid[lIdx]
                                + computeLambdaMEnergyForIndex(
                                mRCs, lIdx, cachedRigidEmat, cachedG);
                        eMin = lambdaOnlyMin[lIdx]
                                + computeLambdaMEnergyForIndex(
                                mRCs, lIdx, cachedMinEmat, cachedG);
                    } else {
                        int[] lambdaRCs = fullDpLambdaDecodeBuffer();
                        decodeLambdaStateInto(lIdx, lambdaRCs);
                        eRigid = (lambdaOnlyRigid != null
                                ? lambdaOnlyRigid[lIdx]
                                : computeLambdaOnlyEnergy(
                                lambdaRCs, cachedRigidEmat, cachedG))
                                + computeLambdaMEnergy(
                                mRCs, lambdaRCs, cachedRigidEmat, cachedG);
                        eMin = (lambdaOnlyMin != null
                                ? lambdaOnlyMin[lIdx]
                                : computeLambdaOnlyEnergy(
                                lambdaRCs, cachedMinEmat, cachedG))
                                + computeLambdaMEnergy(
                                mRCs, lambdaRCs, cachedMinEmat, cachedG);
                    }
                    lower.add(-eRigid / cachedRT);
                    upper.add(-eMin / cachedRT);
                }
            }
        } else if (childFoldPlans != null) {
            // Non-leaf edge (Option C): fold in child bounds with a precomputed plan.
            // The plan replaces the per-element getMstateForFullState() int[]
            // allocation + linear search with a direct mixed-radix index, and
            // hoists each child's M-only ("base") index out of the lambda loop.
            int[] mRCs = fullDpMDecodeBuffer();
            decodeMStateInto(mIdx, mRCs);
            int nChildren = childFoldPlans.length;

            long[] baseIdx = fullDpChildBaseBuffer(nChildren);
            double constLower = 0.0;
            double constUpper = 0.0;
            for (int c = 0; c < nChildren; c++) {
                ChildFoldPlan plan = childFoldPlans[c];
                baseIdx[c] = plan.baseIndex(mRCs);
                {}
            }

            for (int lIdx = 0; lIdx < totalLambdaStates; lIdx++) {
                int[] lambdaRCs = null;
                double eRigid;
                double eMin;
                if (pairwiseEnergy) {
                    eRigid = lambdaOnlyRigid[lIdx]
                            + computeLambdaMEnergyForIndex(
                            mRCs, lIdx, cachedRigidEmat, cachedG);
                    eMin = lambdaOnlyMin[lIdx]
                            + computeLambdaMEnergyForIndex(
                            mRCs, lIdx, cachedMinEmat, cachedG);
                } else {
                    lambdaRCs = fullDpLambdaDecodeBuffer();
                    decodeLambdaStateInto(lIdx, lambdaRCs);
                    eRigid = computeLocalEnergy(
                            mRCs, lambdaRCs, cachedRigidEmat, cachedG);
                    eMin = computeLocalEnergy(
                            mRCs, lambdaRCs, cachedMinEmat, cachedG);
                }

                double fSumLower = constLower;
                double fSumUpper = constUpper;
                for (int c = 0; c < nChildren; c++) {
                    ChildFoldPlan plan = childFoldPlans[c];
                    {}
                    long lambdaIdx = pairwiseEnergy
                            ? lambdaIndexForPlan(plan, lIdx)
                            : plan.lambdaIndex(lambdaRCs);
                    long fIdx = baseIdx[c] + lambdaIdx;
                    fSumLower += plan.child.getLogZLower(fIdx);
                    fSumUpper += plan.child.getLogZUpper(fIdx);
                }

                lower.add(-eRigid / cachedRT + fSumLower);
                upper.add(-eMin / cachedRT + fSumUpper);
            }
        } else {
            // Non-leaf edge (legacy path): fold disabled or unmapped child slot.
            int[] mRCs = fullDpMDecodeBuffer();
            decodeMStateInto(mIdx, mRCs);
            for (int lIdx = 0; lIdx < totalLambdaStates; lIdx++) {
                int[] lambdaRCs = fullDpLambdaDecodeBuffer();
                decodeLambdaStateInto(lIdx, lambdaRCs);
                double eRigid = computeLocalEnergy(mRCs, lambdaRCs, cachedRigidEmat, cachedG);
                double eMin = computeLocalEnergy(mRCs, lambdaRCs, cachedMinEmat, cachedG);

                double fSumLower = 0.0;
                double fSumUpper = 0.0;
                for (RootedTreeEdge fEdge : Fset) {
                    int[] fM = getMstateForFullState(mRCs, lambdaRCs, fEdge);
                    long fIdx = fEdge.computeIndexInA(fM);
                    fSumLower += fEdge.getLogZLower(fIdx);
                    fSumUpper += fEdge.getLogZUpper(fIdx);
                }

                lower.add(-eRigid / cachedRT + fSumLower);
                upper.add(-eMin / cachedRT + fSumUpper);
            }
        }

        dpTable.set(mIdx, lower.value(), upper.value());
    }

    /**
     * JNA binding for the Option A native kernel (libOspreyLogSumExp). Registered
     * lazily and defensively via {@link #ensureNativeKernelLoaded()} so a missing
     * .so cannot poison class init — it just disables the native path.
     */

    private static class LogSumExpAccumulator {
        private double max = NEG_INF;
        private double scaledSum = 0.0;

        void reset() {
            max = NEG_INF;
            scaledSum = 0.0;
        }

        void add(double value) {
            if (Double.isNaN(value) || Double.isNaN(max)) {
                max = Double.NaN;
                scaledSum = Double.NaN;
                return;
            }
            if (value == NEG_INF) return;
            if (max == NEG_INF) {
                max = value;
                scaledSum = 1.0;
            } else if (value > max) {
                scaledSum = scaledSum * Math.exp(max - value) + 1.0;
                max = value;
            } else {
                scaledSum += Math.exp(value - max);
            }
        }

        double value() {
            if (max == NEG_INF) return NEG_INF;
            return max + Math.log(scaledSum);
        }
    }

    /**
     * Direct independent-position DP for a 0-edge interaction graph (no pairwise
     * terms). Z then factorizes per position, so
     *   logZ = sum_pos logsumexp_rc( -E(pos, rc) / RT ),
     * computed exactly in O(sum cardinalities) instead of the O(prod cardinalities)
     * generic joint leaf DP. Returns {logZLower (rigid emat), logZUpper (min emat)},
     * matching the leaf-DP convention (lower uses rigid, upper uses min).
     *
     * Validated against brute-force joint enumeration in TestZeroEdgeDP. Wired
     * behind {@code branchdp.dp.zeroEdgeDirect} (default off) at the 0-edge
     * fallback in BranchDpBackend.
     */

    /**
     * Precomputed plan to fold one child edge's logZ bounds into this (non-leaf)
     * edge's DP. Replaces the per-element getMstateForFullState() linear search +
     * int[] allocation with a direct mixed-radix index built from the parent's
     * already-decoded M/lambda RC arrays.
     *
     * For child M-slot i, computeIndexInA() weights it by
     *   stride[i] = product_{k>i} rcs.getNum(child.mPositionsSorted[k]).
     * By the branch-decomposition invariant each child M-position is one of the
     * parent's M-positions or lambda-positions, so the index splits into an
     * M-only "base" (constant across the lambda loop) plus a lambda-dependent
     * part. A child with no lambda-sourced slots has a constant index across the
     * whole lambda loop ("λ-invariant") and can be hoisted out entirely.
     */
    private static final class ChildFoldPlan {
        final RootedTreeEdge child;
        final int[] mSrcIdx;    // parent mRCs index for each M-sourced child slot
        final long[] mStride;   // mixed-radix stride for each M-sourced child slot
        final int[] lamSrcIdx;  // parent lambdaRCs index for each λ-sourced child slot
        final long[] lamStride; // mixed-radix stride for each λ-sourced child slot
        final boolean lambdaDependent;
        long[] lambdaIndexByState;

        ChildFoldPlan(RootedTreeEdge child, int[] mSrcIdx, long[] mStride,
                      int[] lamSrcIdx, long[] lamStride) {
            this.child = child;
            this.mSrcIdx = mSrcIdx;
            this.mStride = mStride;
            this.lamSrcIdx = lamSrcIdx;
            this.lamStride = lamStride;
            this.lambdaDependent = lamStride.length > 0;
        }

        long baseIndex(int[] mRCs) {
            long idx = 0L;
            for (int t = 0; t < mStride.length; t++) {
                idx += (long) mRCs[mSrcIdx[t]] * mStride[t];
            }
            return idx;
        }

        long lambdaIndex(int[] lambdaRCs) {
            long idx = 0L;
            for (int t = 0; t < lamStride.length; t++) {
                idx += (long) lambdaRCs[lamSrcIdx[t]] * lamStride[t];
            }
            return idx;
        }

        long lambdaIndexForState(RootedTreeEdge edge, int lambdaIndex) {
            if (lambdaIndexByState != null) {
                return lambdaIndexByState[lambdaIndex];
            }
            long idx = 0L;
            for (int t = 0; t < lamStride.length; t++) {
                idx += (long) edge.getLambdaLocalRC(
                        lambdaIndex, lamSrcIdx[t]) * lamStride[t];
            }
            return idx;
        }
    }

    private long lambdaIndexForPlan(
            ChildFoldPlan plan, int lambdaIndex) {
        return plan.lambdaIndexForState(this, lambdaIndex);
    }

    /**
     * Build the child-fold plans once, before the (possibly parallel) DP loop.
     * The result is read-only afterwards, so it is safe to share across DP
     * worker threads. Disabled (null plans => legacy path) when
     * {@code branchdp.dp.foldChildren=false}, when there are no F-set
     * children, or when any child M-position is not covered by this edge's
     * M ∪ lambda (in which case the legacy getMstateForFullState() path is used
     * unchanged, preserving behavior on malformed/edge-case decompositions).
     */
    private void ensureChildFoldPlans() {
        boolean foldEnabled = getConfigBoolean(
                DP_FOLD_CHILDREN_PROPERTY, true);
        boolean hoistInvariant = false;
        if (childFoldPlansInitialized
                && childFoldChildrenEnabled == foldEnabled
                && childFoldHoistInvariant == hoistInvariant) {
            return;
        }
        childFoldPlansInitialized = true;
        childFoldChildrenEnabled = foldEnabled;
        childFoldHoistInvariant = hoistInvariant;
        if (!foldEnabled || !hasFsetChildren()) {
            childFoldPlans = null;
            return;
        }

        ChildFoldPlan[] plans = new ChildFoldPlan[Fset.size()];
        int ci = 0;
        for (RootedTreeEdge fEdge : Fset) {
            ChildFoldPlan plan = buildChildFoldPlan(fEdge);
            if (plan == null) {
                childFoldPlans = null; // unmapped slot -> fall back to legacy path
                return;
            }
            plans[ci++] = plan;
        }
        prepareChildFoldProjectionCache(plans);
        childFoldPlans = plans;

        if (getConfigBoolean(DP_PROGRESS_PROPERTY, true)) {
            int invariant = 0;
            for (ChildFoldPlan p : plans) {
                if (!p.lambdaDependent) invariant++;
            }
            System.out.println(BranchDpConfig.getBackendLogPrefix() + " childFold lambdaStates=" + totalLambdaStates
                    + ", mStates=" + mStateCount + ", children=" + plans.length
                    + ", lambdaInvariant=" + invariant
                    + ", lambdaDependent=" + (plans.length - invariant)
                    + ", hoist=" + childFoldHoistInvariant);
        }
    }

    /**
     * Cache each child projection's lambda-only mixed-radix index when the
     * bounded table is small enough.  A DP row revisits the same projection
     * for every parent M-state; precomputing it removes another inner-loop
     * walk over lambda slots without making large edges pay an unbounded heap
     * cost.  If the cap is exceeded, the direct cached-digit path remains.
     */
    private void prepareChildFoldProjectionCache(ChildFoldPlan[] plans) {
        long maxBytes = getConfigLong(
                DP_CHILD_FOLD_PROJECTION_CACHE_BYTES_PROPERTY,
                DEFAULT_DP_CHILD_FOLD_PROJECTION_CACHE_BYTES);
        if (maxBytes <= 0L || totalLambdaStates <= 0) {
            return;
        }
        long dependentPlans = 0L;
        for (ChildFoldPlan plan : plans) {
            if (plan.lambdaDependent) dependentPlans++;
        }
        long entries = dependentPlans * (long) totalLambdaStates;
        if (entries <= 0L
                || entries > Integer.MAX_VALUE
                || entries > maxBytes / Long.BYTES) {
            return;
        }
        for (ChildFoldPlan plan : plans) {
            if (!plan.lambdaDependent) continue;
            long[] projection = new long[totalLambdaStates];
            for (int lambdaIndex = 0;
                 lambdaIndex < totalLambdaStates; lambdaIndex++) {
                long projected = 0L;
                for (int t = 0; t < plan.lamStride.length; t++) {
                    projected += (long) getLambdaLocalRC(
                            lambdaIndex, plan.lamSrcIdx[t])
                            * plan.lamStride[t];
                }
                projection[lambdaIndex] = projected;
            }
            plan.lambdaIndexByState = projection;
        }
    }

    private ChildFoldPlan buildChildFoldPlan(RootedTreeEdge child) {
        int[] childM = child.mPositionsSorted;
        int len = childM.length;

        // Mixed-radix stride per child M-slot (matches child.computeIndexInA):
        // stride[len-1] = 1; stride[i] = product_{k>i} rcs.getNum(childM[k]).
        long[] stride = new long[len];
        if (len > 0) {
            stride[len - 1] = 1L;
            for (int i = len - 2; i >= 0; i--) {
                stride[i] = stride[i + 1] * child.rcs.getNum(childM[i + 1]);
            }
        }

        int[] mSrcTmp = new int[len];
        long[] mStrideTmp = new long[len];
        int mCount = 0;
        int[] lamSrcTmp = new int[len];
        long[] lamStrideTmp = new long[len];
        int lamCount = 0;

        for (int i = 0; i < len; i++) {
            int targetPos = childM[i];
            // Search parent M first, then parent lambda (matches getMstateForFullState).
            int j = indexOf(mPositionsSorted, targetPos);
            if (j >= 0) {
                mSrcTmp[mCount] = j;
                mStrideTmp[mCount] = stride[i];
                mCount++;
                continue;
            }
            j = indexOf(lambdaPositionsSorted, targetPos);
            if (j >= 0) {
                lamSrcTmp[lamCount] = j;
                lamStrideTmp[lamCount] = stride[i];
                lamCount++;
                continue;
            }
            return null; // child M-position outside parent M ∪ lambda -> bail to legacy
        }

        return new ChildFoldPlan(child,
                Arrays.copyOf(mSrcTmp, mCount), Arrays.copyOf(mStrideTmp, mCount),
                Arrays.copyOf(lamSrcTmp, lamCount), Arrays.copyOf(lamStrideTmp, lamCount));
    }

    private static int indexOf(int[] arr, int value) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == value) return i;
        }
        return -1;
    }

    /**
     * Post-order traversal: compute full DP for all lambda-edges.
     * Leaves are computed first, then their parents fold in the leaf bounds.
     */
    public static void postOrderComputeFullDP(RootedTreeNode node) {
        if (node == null) return;
        postOrderComputeFullDP(node.getLeftChild());
        postOrderComputeFullDP(node.getRightChild());
        if (node.getChildOfEdge() != null && node.getChildOfEdge().isLambdaEdge) {
            node.getChildOfEdge().computeFullDP();
        }
    }

    // ========== Post-order traversal helpers ==========

    public static void postOrderCompLlambda(RootedTreeNode node) {
        postOrderCompLlambda(node, true);
    }

    public static void postOrderCompLlambda(RootedTreeNode node, boolean initEnumerationArrays) {
        if (node == null) return;
        postOrderCompLlambda(node.getLeftChild(), initEnumerationArrays);
        postOrderCompLlambda(node.getRightChild(), initEnumerationArrays);
        if (node.getChildOfEdge() != null) {
            node.getChildOfEdge().compLlambda(initEnumerationArrays);
        }
    }

    /**
     * Post-order traversal: initialize incremental enumeration for all lambda-edges.
     * After this call, all edges have k=0 with loose initial bounds.
     */
    public static void postOrderInitIncremental(RootedTreeNode node,
                                                  EnergyMatrix rigidEmat, EnergyMatrix minEmat,
                                                  InteractionGraph G, double RT) {
        postOrderInitIncremental(node, rigidEmat, minEmat, G, RT, false);
    }

    public static void postOrderInitIncremental(RootedTreeNode node,
                                                  EnergyMatrix rigidEmat, EnergyMatrix minEmat,
                                                  InteractionGraph G, double RT,
                                                  boolean materializeFullEnergyTables) {
        if (node == null) return;
        postOrderInitIncremental(node.getLeftChild(), rigidEmat, minEmat, G, RT,
                materializeFullEnergyTables);
        postOrderInitIncremental(node.getRightChild(), rigidEmat, minEmat, G, RT,
                materializeFullEnergyTables);

        if (node.getChildOfEdge() != null && node.getChildOfEdge().isLambdaEdge) {
            node.getChildOfEdge().initIncrementalEnumeration(rigidEmat, minEmat, G, RT,
                    materializeFullEnergyTables);
        }
    }

    /** Release all large edge storage in a rooted tree, children first. */
    public static void postOrderReleaseLargeMemory(RootedTreeNode node) {
        if (node == null) return;
        postOrderReleaseLargeMemory(node.getLeftChild());
        postOrderReleaseLargeMemory(node.getRightChild());
        if (node.getChildOfEdge() != null) {
            node.getChildOfEdge().releaseLargeMemory();
        }
    }

    /**
     * Collect all lambda-edges in the tree (post-order).
     */
    public static void collectLambdaEdges(RootedTreeNode node, List<RootedTreeEdge> result) {
        if (node == null) return;
        collectLambdaEdges(node.getLeftChild(), result);
        collectLambdaEdges(node.getRightChild(), result);
        if (node.getChildOfEdge() != null && node.getChildOfEdge().isLambdaEdge) {
            result.add(node.getChildOfEdge());
        }
    }
}
