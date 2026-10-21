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

import edu.duke.cs.osprey.gpu.cuda.CUBuffer;
import edu.duke.cs.osprey.gpu.cuda.Context;
import edu.duke.cs.osprey.gpu.cuda.Gpu;
import edu.duke.cs.osprey.gpu.cuda.GpuStream;
import edu.duke.cs.osprey.gpu.cuda.Gpus;
import edu.duke.cs.osprey.gpu.cuda.Kernel;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUresult;
import jcuda.driver.JCudaDriver;

import java.nio.DoubleBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;
import java.util.function.LongConsumer;

/**
 * CUDA fast path for branch-DP full-DP: one non-leaf edge with one or more
 * dense child DP tables and full lambda enumeration. Parent output is written
 * tile-by-tile into DPTable, so it may be dense or shard-backed.
 * The caller owns shape checks and falls back to the Java DP path otherwise.
 */
final class DPGpuFullDP {

    static final int MAX_EDGE_POSITIONS = 32;

    // When an edge's GPU footprint exceeds device memory, fail the whole run instead
    // of silently falling back to the (single-threaded, hours-long) Java DP path.
    private static final String DP_GPU_FAIL_IF_EXCEEDS_VRAM = "branchdp.dp.gpu.failIfExceedsVram";
    private static final String DP_GPU_VRAM_HEADROOM_PERCENT = "branchdp.dp.gpu.vramHeadroomPercent";
    private static final String DP_GPU_CHILD_SLICING = "branchdp.dp.gpu.childSlicing";
    private static final String DP_GPU_CHILD_SLICING_FORCE = "branchdp.dp.gpu.childSlicing.force";
    private static final String DP_GPU_CHILD_SLICE_MAX_BYTES = "branchdp.dp.gpu.childSliceMaxBytes";
    private static final String DP_GPU_HYBRID_CHILD_TILING = "branchdp.dp.gpu.hybridChildTiling";
    private static final String DP_GPU_HYBRID_CHILD_TILING_FORCE = "branchdp.dp.gpu.hybridChildTiling.force";
    private static final String DP_GPU_OUT_OF_CORE = "branchdp.dp.gpu.outOfCore";
    private static final String DP_GPU_OUT_OF_CORE_FORCE = "branchdp.dp.gpu.outOfCore.force";
    private static final String DP_GPU_OUT_OF_CORE_BUDGET_BYTES = "branchdp.dp.gpu.outOfCore.budgetBytes";
    private static final String DP_GPU_OUT_OF_CORE_OUTPUT_WORKSPACE_MAX_BYTES =
            "branchdp.dp.gpu.outOfCore.outputWorkspaceMaxBytes";
    private static final int DEFAULT_DP_GPU_VRAM_HEADROOM_PERCENT = 92;
    private static final boolean DEFAULT_DP_GPU_CHILD_SLICING = true;
    private static final boolean DEFAULT_DP_GPU_HYBRID_CHILD_TILING = true;
    private static final boolean DEFAULT_DP_GPU_OUT_OF_CORE = true;
    private static final long DEFAULT_DP_GPU_CHILD_SLICE_MAX_BYTES = 2L * 1024L * 1024L * 1024L;
    /** Combined m-index, partial-accumulator, and final-output workspace.  This
     * cap is independent of the legacy full-DP output chunk: OOC needs enough
     * outputs resident to reuse one gathered child tile across all free-M
     * states.  Four GiB is large enough for the production 3bua/3k3q free-M
     * ranges while keeping each individual Java direct buffer well below its
     * signed-int byte limit. */
    private static final long DEFAULT_DP_GPU_OUT_OF_CORE_OUTPUT_WORKSPACE_MAX_BYTES =
            4L * 1024L * 1024L * 1024L;
    private static final int CHILD_GATHER_CHUNK_STATES = 16 * 1024;
    private static final int CHILD_GATHER_ACTION_LEAF_CHUNKS = 32;
    /** All GPU workers share one bounded pool instead of each flooding the JVM's
     *  process-wide common pool with an independent parallel stream. */
    private static final ForkJoinPool CHILD_GATHER_POOL = new ForkJoinPool(
            Math.max(1, Math.min(32, Runtime.getRuntime().availableProcessors())));

    private static final class GatherChunksAction extends RecursiveAction {
        private final long start;
        private final long end;
        private final LongConsumer body;

        GatherChunksAction(long start, long end, LongConsumer body) {
            this.start = start;
            this.end = end;
            this.body = body;
        }

        @Override
        protected void compute() {
            long count = end - start;
            if (count <= CHILD_GATHER_ACTION_LEAF_CHUNKS) {
                for (long chunk = start; chunk < end; chunk++) {
                    body.accept(chunk);
                }
                return;
            }
            long mid = start + count / 2L;
            invokeAll(new GatherChunksAction(start, mid, body),
                    new GatherChunksAction(mid, end, body));
        }
    }

    static void parallelGatherChunks(long chunkCount, LongConsumer body) {
        if (chunkCount <= 0L) {
            return;
        }
        if (chunkCount <= CHILD_GATHER_ACTION_LEAF_CHUNKS
                || CHILD_GATHER_POOL.getParallelism() <= 1) {
            for (long chunk = 0L; chunk < chunkCount; chunk++) {
                body.accept(chunk);
            }
            return;
        }
        CHILD_GATHER_POOL.invoke(new GatherChunksAction(0L, chunkCount, body));
    }

    /** Edge GPU footprint exceeds device memory; propagated as fatal (no CPU fallback). */
    static final class GpuMemoryExceededException extends RuntimeException {
        GpuMemoryExceededException(String message) {
            super(message);
        }
    }

    /**
     * An edge is structurally unsupported by the GPU full-DP path (e.g. a
     * required limit like {@code MAX_EDGE_POSITIONS} is exceeded) even though
     * GPU DP was requested. Propagated as fatal by default
     * ({@code branchdp.dp.gpu.failIfNoGpuPath=true}) instead of silently
     * degrading to the single-threaded Java DP path, which can take hours to
     * days on multi-billion-state edges with no GPU speedup at all.
     */
    static final class GpuUnsupportedEdgeException extends RuntimeException {
        GpuUnsupportedEdgeException(String message) {
            super(message);
        }
    }

    private static volatile boolean unavailable = false;
    private static volatile boolean unavailableLogged = false;
    // Per-edge (non-sticky) fallback: a single oversized/failed edge falls back to
    // Java without disabling GPU for the whole run; only a genuinely broken GPU
    // (MAX_CONSECUTIVE_EDGE_FAILURES in a row) flips the sticky unavailable flag.
    private static volatile int consecutiveEdgeFailures = 0;
    private static final int MAX_CONSECUTIVE_EDGE_FAILURES = 5;
    private static final Object persistentLock = new Object();
    private static PersistentPool persistentPool = null;
    private static boolean persistentShutdownHook = false;

    private DPGpuFullDP() {
    }

    /**
     * Cheap upfront sanity check (no allocation, no kernel launch): usable VRAM bytes on
     * one GPU, i.e. what a single participating device could hold. Since child tables are
     * fully replicated onto every GPU used for an edge (see the class doc), this -- not the
     * sum of all GPUs' memory -- is the right number to compare a design's worst-case
     * child-table requirement against before committing to an expensive run. Assumes a
     * homogeneous GPU allocation (true for all current SLURM configs used by this project:
     * one GPU model per job), so only the first visible device is queried. Returns -1 if
     * GPU discovery/query fails for any reason (caller should treat that as "could not
     * determine" and skip the check rather than abort).
     */
    static long queryMinUsableVramBytes() {
        try {
            List<Gpu> gpus = Gpus.get().getGpus();
            if (gpus.isEmpty()) {
                return -1L;
            }
            Context context = new Context(gpus.get(0));
            try {
                context.attachCurrentThread();
                long[] freeMem = {0L};
                long[] totalMem = {0L};
                if (JCudaDriver.cuMemGetInfo(freeMem, totalMem) != CUresult.CUDA_SUCCESS) {
                    return -1L;
                }
                int headroomPct = Math.max(1, Math.min(100,
                        getConfigInteger(DP_GPU_VRAM_HEADROOM_PERCENT, DEFAULT_DP_GPU_VRAM_HEADROOM_PERCENT)));
                return (long)((double)freeMem[0]*headroomPct/100.0);
            } finally {
                context.cleanup();
            }
        } catch (Throwable t) {
            return -1L;
        }
    }

    private static boolean getConfigBoolean(String key, boolean defaultValue) {
        return BranchDpConfig.getBackendBoolean(key, defaultValue);
    }

    private static long getConfigLong(String key, long defaultValue) {
        return BranchDpConfig.getBackendLong(key, defaultValue, BranchDpConfig.getBackendLogPrefix());
    }

    private static int getConfigInteger(String key, int defaultValue) {
        return BranchDpConfig.getBackendInteger(key, defaultValue, BranchDpConfig.getBackendLogPrefix());
    }

    /** True if t (or any cause) is a device-OOM / footprint-too-big condition. */
    private static boolean isGpuMemoryFailure(Throwable t) {
        for (Throwable c = t; c != null; c = c.getCause()) {
            if (c instanceof GpuMemoryExceededException) {
                return true;
            }
            String m = c.getMessage();
            if (m != null && m.contains("OUT_OF_MEMORY")) {
                return true;
            }
        }
        return false;
    }

    private static String rootMessage(Throwable t) {
        Throwable c = t;
        while (c.getCause() != null) {
            c = c.getCause();
        }
        return c.getMessage();
    }

    static final class Request {
        long mStateCount;     // global parent M-state count
        long mStateChunk;     // max M-states per output tile/kernel launch (= numBlocks)
        int totalLambdaStates;
        int blockThreads;
        int numChildren;
        double invRT;

        int[] mCounts;
        int[] lambdaCounts;

        double[] lambdaOnlyRigid;
        double[] lambdaOnlyMin;
        double[] lmRigid;
        double[] lmMin;
        int[] lmLamSlots;
        int[] lmMSlots;
        int[] lmMCounts;
        long[] lmOffsets;

        // Exact three-position factors local to this edge. Slots use one
        // namespace: [0,mCounts.length) is M and the remainder is lambda.
        // Each factor owns one dense table in tripleRigid/tripleMin.
        int[] tripleSlots = new int[0];       // 3 entries per factor
        long[] tripleStrides = new long[0];  // 3 entries per factor
        long[] tripleOffsets = new long[0];  // one table base per factor
        double[] tripleRigid = new double[0];
        double[] tripleMin = new double[0];

        // Per-child fold plans, concatenated CSR-style across numChildren.
        int[] childMSrcAll;
        long[] childMStrideAll;
        int[] childMTermOff;
        int[] childMTermCnt;
        int[] childLSrcAll;
        long[] childLStrideAll;
        int[] childLTermOff;
        int[] childLTermCnt;
        long[] childTableBase;     // per-child element offset into the device buffer (long: may exceed int);
                                   // used by the kernel-side childTableBase device buffer, so it stays
                                   // indexed one-per-logical-child regardless of how each child is chunked.
        long childTableTotal;      // total element count across all child tables (may exceed Integer.MAX_VALUE)
        // Upload-side view: one entry per physical chunk (a dense child contributes
        // exactly one chunk; a sharded child contributes one entry per shard, in
        // index order). childChunkOffsets[i] is the absolute element offset of
        // childLowerChunks[i]/childUpperChunks[i] within the single concatenated
        // device buffer, so consecutive chunks belonging to the same child land
        // contiguously starting at that child's childTableBase[c]. Never
        // concatenated on the Java heap, so the combined length may exceed
        // Integer.MAX_VALUE; only device memory (a 64-bit byte count) bounds it.
        double[][] childLowerChunks;
        double[][] childUpperChunks;
        long[] childChunkOffsets;
        DPTable[] childTables;
        boolean hasFileBackedChildTables;

        boolean childSlicing;
        boolean forceChildSlicing;
        long childSliceMaxBytes;
        long outOfCoreOutputWorkspaceMaxBytes;
        int[] unionMSlots;
        int[] unionMCounts;
        long[] unionMParentStrides;
        int[] freeMCounts;
        long[] freeMParentStrides;
        long unionStateCount;
        long parentFreeStateCount;
        int[] childMCountsAll;
        long[] childMPackedStrideAll;
        int[] childLCountsAll;
        long[] childLPackedStrideAll;
        long[] childMRowStates;
        long[] childLambdaStates;
        volatile long[][] childLambdaOriginalOffsets;

        DPTable outTable;     // parent table written tile-by-tile

        boolean progress;
        long work;
        long estimatedDeviceBytes;
        long estimatedFullDeviceBytes;
        long estimatedSlicedDeviceBytes;
        long estimatedOutOfCoreMinimumDeviceBytes;
        long maxDeviceBytes; // <=0 means no operator cap
        boolean persistentContext;

        // Multi-GPU (single-node) M-state split controls.
        boolean multiGpu;
        int maxGpus;          // 0 = all available
        long minMStatesPerGpu;

        // Filled by the bounded out-of-core executor after its device
        // workspaces have actually been allocated.  These are deliberately on
        // the request (rather than in a process-global test hook) so concurrent
        // GPU workers can report against the exact edge they execute.
        volatile int outOfCoreAllocationAuditCount;
        volatile long outOfCoreAuditedDeviceBytes;
        volatile long outOfCoreLargestAllocationBytes;
        volatile long outOfCorePlannedDeviceBytes;
        volatile long outOfCoreBudgetBytes;
    }

    static final class HybridPlan {
        final int streamedChild;
        final long streamedRowCount;
        final long streamedRowsPerBlock;
        final long otherMStateCount;
        final long residentTableStates;
        final long estimatedAggregateTrafficBytes;
        final long[] residentTableBase;

        HybridPlan(int streamedChild, long streamedRowCount, long streamedRowsPerBlock,
                   long otherMStateCount, long residentTableStates,
                   long estimatedAggregateTrafficBytes, long[] residentTableBase) {
            this.streamedChild = streamedChild;
            this.streamedRowCount = streamedRowCount;
            this.streamedRowsPerBlock = streamedRowsPerBlock;
            this.otherMStateCount = otherMStateCount;
            this.residentTableStates = residentTableStates;
            this.estimatedAggregateTrafficBytes = estimatedAggregateTrafficBytes;
            this.residentTableBase = residentTableBase;
        }
    }

    static synchronized boolean compute(Request req) {
        if (unavailable) {
            return false;
        }

        List<Gpu> gpus;
        try {
            gpus = Gpus.get().getGpus();
        } catch (Throwable t) {
            // GPU discovery itself failed => genuinely no usable CUDA; sticky.
            markUnavailable("GPU discovery failed: " + t.getMessage(), t);
            return false;
        }
        if (gpus.isEmpty()) {
            markUnavailable("no CUDA GPU with double support", null);
            return false;
        }

        int nGpus = chooseGpuCount(req, gpus.size());
        GpuExecutor[] executors = null;
        try {
            long usableBytes = queryMinUsableVramBytes();
            boolean forceHybrid = getConfigBoolean(DP_GPU_HYBRID_CHILD_TILING_FORCE, false);
            boolean forceOutOfCore = getConfigBoolean(DP_GPU_OUT_OF_CORE_FORCE, false)
                    || req.hasFileBackedChildTables;
            boolean hybridEnabled = forceHybrid || getConfigBoolean(DP_GPU_HYBRID_CHILD_TILING,
                    DEFAULT_DP_GPU_HYBRID_CHILD_TILING);
            boolean outOfCoreEnabled = forceOutOfCore || getConfigBoolean(DP_GPU_OUT_OF_CORE,
                    DEFAULT_DP_GPU_OUT_OF_CORE);
            if (req.hasFileBackedChildTables && req.progress) {
                System.out.println(BranchDpConfig.getBackendLogPrefix()
                        + " GPU DP forcing bounded out-of-core gather for file-backed child tables.");
            }
            long fullNeed = estimateDeviceBytes(req);
            long deviceBudget = usableBytes;
            if (req.maxDeviceBytes > 0L) {
                deviceBudget = deviceBudget > 0L
                        ? Math.min(deviceBudget, req.maxDeviceBytes)
                        : req.maxDeviceBytes;
            }
            boolean fullExceedsBudget = deviceBudget > 0L
                    && fullNeed > deviceBudget;
            // forceChildSlicing is an execution-policy override, not a property of
            // the edge shape. Keep that gate here so chooseHybridPlan() remains a
            // deterministic shape/budget planner. In particular, root scoring and
            // CPU-only planner tests must not change when another test or alias
            // scope mutates the process-wide child-slicing property.
            HybridPlan hybridPlan = !forceOutOfCore && !req.forceChildSlicing
                    && hybridEnabled && deviceBudget > 0L
                    && (forceHybrid || fullExceedsBudget)
                    ? chooseHybridPlan(req, deviceBudget, nGpus)
                    : null;
            long configuredOutOfCoreBudget = Math.max(0L,
                    getConfigLong(DP_GPU_OUT_OF_CORE_BUDGET_BYTES, 0L));
            long outOfCoreBudget = deviceBudget;
            if (configuredOutOfCoreBudget > 0L) {
                outOfCoreBudget = usableBytes > 0L
                        ? Math.min(usableBytes, configuredOutOfCoreBudget)
                        : configuredOutOfCoreBudget;
            }
            DPGpuOutOfCore.Plan outOfCorePlan = outOfCoreEnabled
                    && outOfCoreBudget > 0L
                    && (forceOutOfCore || (!req.forceChildSlicing
                    && hybridPlan == null && fullExceedsBudget))
                    ? DPGpuOutOfCore.choosePlan(req, outOfCoreBudget)
                    : null;
            if (forceOutOfCore && outOfCorePlan == null) {
                throw new GpuMemoryExceededException("Forced bounded out-of-core GPU DP has no feasible "
                        + "one-state tile under budget " + outOfCoreBudget + " B; minimum="
                        + DPGpuOutOfCore.estimateMinimumDeviceBytes(req) + " B");
            }

            // A one-state M domain can still have a very large lambda domain.
            // The ordinary GPU-count admission is intentionally M-oriented,
            // but OOC has a safe host merge for disjoint lambda boxes in this
            // exact shape, so let that path recruit otherwise idle devices.
            if (outOfCorePlan != null
                    && outOfCorePlan.mBoxCount == 1L
                    && req.parentFreeStateCount == 1L
                    && outOfCorePlan.lambdaBoxCount > 1L) {
                nGpus = Math.max(nGpus, chooseLambdaBoxGpuCount(
                        req, gpus.size(), outOfCorePlan.lambdaBoxCount));
            }
            executors = makeExecutors(req, gpus, nGpus);
            boolean useChildSlicing = hybridPlan == null && outOfCorePlan == null
                    && canUseChildSlicing(req)
                    && (req.forceChildSlicing || fullExceedsBudget);
            if (hybridPlan != null || useChildSlicing) {
                ensureChildSliceIndexMaps(req);
            }
            if (hybridPlan != null && nGpus <= 1) {
                executors[0].runHybridRowRange(req, hybridPlan, 0L,
                        hybridPlan.streamedRowCount);
            } else if (hybridPlan != null) {
                runMultiGpuHybrid(req, hybridPlan, executors, nGpus);
            } else if (outOfCorePlan != null && nGpus <= 1) {
                executors[0].runOutOfCore(req, outOfCorePlan, 0L,
                        outOfCorePlan.mBoxCount, 0L, req.parentFreeStateCount);
            } else if (outOfCorePlan != null
                    && canSplitOutOfCoreLambdaBoxes(
                    req, outOfCorePlan, nGpus)) {
                runMultiGpuOutOfCoreLambdaBoxes(
                        req, outOfCorePlan, executors, nGpus);
            } else if (outOfCorePlan != null) {
                runMultiGpuOutOfCore(req, outOfCorePlan, executors, nGpus);
            } else if (nGpus <= 1) {
                runTiled(req, executors[0], 0L, req.mStateCount);
            } else if (useChildSlicing) {
                runMultiGpuSliced(req, executors, nGpus);
            } else {
                runMultiGpu(req, executors, nGpus);
            }
            consecutiveEdgeFailures = 0;
            return true;
        } catch (Throwable t) {
            // Device out-of-memory: the only CPU fallback for these huge edges is the
            // single-threaded Java DP path, which can take hours. Fail the run fast so
            // the job can be resubmitted on a larger-VRAM GPU (a6000/rtx_pro_6000),
            // unless the operator explicitly opts back into fallback.
            if (getConfigBoolean(DP_GPU_FAIL_IF_EXCEEDS_VRAM, true) && isGpuMemoryFailure(t)) {
                System.out.println(BranchDpConfig.getBackendLogPrefix()
                        + " GPU DP out of device memory; failing fast (no CPU fallback): "
                        + rootMessage(t));
                if (t instanceof RuntimeException) {
                    throw (RuntimeException)t;
                }
                throw new RuntimeException(t);
            }
            // Per-edge failure (e.g. a buffer too big even for chunked upload, or a
            // device OOM on one oversized edge): fall back to Java for THIS edge and
            // keep GPU enabled. Disable globally only after a run of failures.
            consecutiveEdgeFailures++;
            if (consecutiveEdgeFailures >= MAX_CONSECUTIVE_EDGE_FAILURES) {
                markUnavailable("disabled after " + consecutiveEdgeFailures
                        + " consecutive edge failures; last="
                        + t.getClass().getSimpleName() + ": " + t.getMessage(), t);
            } else {
                edgeFallback(t);
            }
            return false;
        } finally {
            cleanupExecutors(executors);
        }
    }

    /** Number of GPUs to split the M-state range across (>=1). */
    private static int chooseGpuCount(Request req, int available) {
        if (!req.multiGpu) {
            return 1;
        }
        int cap = available;
        if (req.maxGpus > 0) {
            cap = Math.min(cap, req.maxGpus);
        }
        if (req.mStateCount > 0 && req.mStateCount < cap) {
            cap = (int)req.mStateCount;
        }
        if (req.minMStatesPerGpu > 0) {
            long byWork = req.mStateCount/req.minMStatesPerGpu;
            cap = (int)Math.min((long)cap, Math.max(1L, byWork));
        }
        return Math.max(1, cap);
    }

    private static int chooseLambdaBoxGpuCount(
            Request req, int available, long lambdaBoxCount) {
        if (!req.multiGpu) {
            return 1;
        }
        int cap = Math.min(available,
                (int)Math.min((long)Integer.MAX_VALUE, lambdaBoxCount));
        if (req.maxGpus > 0) {
            cap = Math.min(cap, req.maxGpus);
        }
        return Math.max(1, cap);
    }

    /** Split [0, mStateCount) into nGpus contiguous ranges; each GPU then tiles its range. */
    private static void runMultiGpu(Request req, GpuExecutor[] executors, int nGpus) throws InterruptedException {
        long total = req.mStateCount;
        long base = total/nGpus;
        long rem = total%nGpus;
        Thread[] threads = new Thread[nGpus];
        Throwable[] errors = new Throwable[nGpus];
        long offset = 0;
        long startNanos = System.nanoTime();
        for (int g=0; g<nGpus; g++) {
            long chunk = base + (g < rem ? 1 : 0);
            final long off = offset;
            final long sz = chunk;
            final GpuExecutor executor = executors[g];
            final int gi = g;
            threads[g] = new Thread(() -> {
                try {
                    runTiled(req, executor, off, sz);
                } catch (Throwable t) {
                    errors[gi] = t;
                }
            }, "bms-gpu-dp-" + g);
            threads[g].start();
            offset += chunk;
        }
        for (Thread th : threads) {
            th.join();
        }
        for (Throwable e : errors) {
            if (e != null) {
                throw new RuntimeException("multi-GPU DP failed: " + e.getMessage(), e);
            }
        }
        if (req.progress) {
            double ms = (System.nanoTime() - startNanos)/1e6;
            System.out.println(BranchDpConfig.getBackendLogPrefix() + " GPU DP multi-gpu done, gpus=" + nGpus
                    + ", mStates=" + total
                    + ", lambdaStates=" + req.totalLambdaStates
                    + ", children=" + req.numChildren
                    + ", outputTileMStates=" + req.mStateChunk
                    + ", persistentContext=" + req.persistentContext
                    + ", elapsedMs=" + String.format(java.util.Locale.ROOT, "%.1f", ms));
        }
    }

    /** Split the child-row union-key space across GPUs for projection-aware child slicing. */
    private static void runMultiGpuSliced(Request req, GpuExecutor[] executors, int nGpus) throws InterruptedException {
        long total = req.unionStateCount;
        long base = total/nGpus;
        long rem = total%nGpus;
        Thread[] threads = new Thread[nGpus];
        Throwable[] errors = new Throwable[nGpus];
        long offset = 0;
        long startNanos = System.nanoTime();
        for (int g=0; g<nGpus; g++) {
            long chunk = base + (g < rem ? 1 : 0);
            final long off = offset;
            final long sz = chunk;
            final GpuExecutor executor = executors[g];
            final int gi = g;
            threads[g] = new Thread(() -> {
                try {
                    executor.runUnionRange(req, off, sz);
                } catch (Throwable t) {
                    errors[gi] = t;
                }
            }, "bms-gpu-dp-slice-" + g);
            threads[g].start();
            offset += chunk;
        }
        for (Thread th : threads) {
            th.join();
        }
        for (Throwable e : errors) {
            if (e != null) {
                throw new RuntimeException("multi-GPU sliced DP failed: " + e.getMessage(), e);
            }
        }
        if (req.progress) {
            double ms = (System.nanoTime() - startNanos)/1e6;
            System.out.println(BranchDpConfig.getBackendLogPrefix() + " GPU DP sliced multi-gpu done, gpus=" + nGpus
                    + ", unionStates=" + total
                    + ", parentFreeStates=" + req.parentFreeStateCount
                    + ", mStates=" + req.mStateCount
                    + ", lambdaStates=" + req.totalLambdaStates
                    + ", children=" + req.numChildren
                    + ", outputTileMStates=" + req.mStateChunk
                    + ", persistentContext=" + req.persistentContext
                    + ", elapsedMs=" + String.format(java.util.Locale.ROOT, "%.1f", ms));
        }
    }

    private static void runMultiGpuHybrid(Request req, HybridPlan plan,
                                          GpuExecutor[] executors, int nGpus)
            throws InterruptedException {
        long total = plan.streamedRowCount;
        long base = total / nGpus;
        long rem = total % nGpus;
        Thread[] threads = new Thread[nGpus];
        Throwable[] errors = new Throwable[nGpus];
        long offset = 0L;
        long startNanos = System.nanoTime();
        for (int g = 0; g < nGpus; g++) {
            long count = base + (g < rem ? 1L : 0L);
            final long rowStart = offset;
            final long rowCount = count;
            final GpuExecutor executor = executors[g];
            final int gi = g;
            threads[g] = new Thread(() -> {
                try {
                    executor.runHybridRowRange(req, plan, rowStart, rowCount);
                } catch (Throwable t) {
                    errors[gi] = t;
                }
            }, "bms-gpu-dp-hybrid-" + g);
            threads[g].start();
            offset += count;
        }
        for (Thread thread : threads) {
            thread.join();
        }
        for (Throwable error : errors) {
            if (error != null) {
                throw new RuntimeException("multi-GPU hybrid DP failed: " + error.getMessage(), error);
            }
        }
        if (req.progress) {
            double ms = (System.nanoTime() - startNanos) / 1e6;
            System.out.println(BranchDpConfig.getBackendLogPrefix()
                    + " GPU DP hybrid multi-gpu done, gpus=" + nGpus
                    + ", streamedChild=" + plan.streamedChild
                    + ", streamedRows=" + plan.streamedRowCount
                    + ", residentStates=" + plan.residentTableStates
                    + ", estimatedAggregateTrafficBytes=" + plan.estimatedAggregateTrafficBytes
                    + ", elapsedMs=" + String.format(java.util.Locale.ROOT, "%.1f", ms));
        }
    }

    /** Give each GPU a disjoint output region. Prefer splitting the free-M
     * dimension (all GPUs reuse the same rectangular union-M boxes); if there
     * are too few free states, split the M-box ordinal range instead. */
    private static void runMultiGpuOutOfCore(Request req, DPGpuOutOfCore.Plan plan,
                                             GpuExecutor[] executors, int nGpus)
            throws InterruptedException {
        final boolean splitFree = req.parentFreeStateCount >= nGpus;
        final long total = splitFree ? req.parentFreeStateCount : plan.mBoxCount;
        long base = total / nGpus;
        long rem = total % nGpus;
        Thread[] threads = new Thread[nGpus];
        Throwable[] errors = new Throwable[nGpus];
        long offset = 0L;
        long startNanos = System.nanoTime();
        for (int g = 0; g < nGpus; g++) {
            long count = base + (g < rem ? 1L : 0L);
            final long boxStart = splitFree ? 0L : offset;
            final long boxCount = splitFree ? plan.mBoxCount : count;
            final long freeStart = splitFree ? offset : 0L;
            final long freeCount = splitFree ? count : req.parentFreeStateCount;
            final GpuExecutor executor = executors[g];
            final int gi = g;
            threads[g] = new Thread(() -> {
                try {
                    executor.runOutOfCore(req, plan, boxStart, boxCount,
                            freeStart, freeCount);
                } catch (Throwable t) {
                    errors[gi] = t;
                }
            }, "bms-gpu-dp-ooc-" + g);
            threads[g].start();
            offset += count;
        }
        for (Thread thread : threads) {
            thread.join();
        }
        for (Throwable error : errors) {
            if (error != null) {
                throw new RuntimeException("multi-GPU bounded out-of-core DP failed: "
                        + error.getMessage(), error);
            }
        }
        if (req.progress) {
            double ms = (System.nanoTime() - startNanos) / 1e6;
            System.out.println(BranchDpConfig.getBackendLogPrefix()
                    + " GPU DP bounded out-of-core multi-gpu done, gpus=" + nGpus
                    + ", split=" + (splitFree ? "free-M" : "M-box")
                    + ", mBoxes=" + plan.mBoxCount
                    + ", lambdaBoxes=" + plan.lambdaBoxCount
                    + ", peakDeviceBytes=" + plan.estimatedDeviceBytes
                    + ", elapsedMs="
                    + String.format(java.util.Locale.ROOT, "%.1f", ms));
        }
    }

    /**
     * Lambda-box parallelism for the regime where M/free-M splitting cannot
     * expose useful independent output work.  Each GPU computes a disjoint
     * lambda range for the same single output block; the host merges the
     * resulting log-sum-exp values before committing the table.
     *
     * <p>The one-output-block restriction is deliberate: it keeps the merge
     * storage bounded by one DP output tile and avoids a global concurrent
     * accumulator for large production tables.  Larger M domains continue to
     * use the existing M/free-M decomposition.</p>
     */
    private static boolean canSplitOutOfCoreLambdaBoxes(
            Request req, DPGpuOutOfCore.Plan plan, int nGpus) {
        return nGpus > 1
                && plan.lambdaBoxCount > 1L
                && plan.mBoxCount == 1L
                && req.parentFreeStateCount == 1L
                && plan.maxMBoxStates > 0L
                && plan.maxMBoxStates <= plan.outputStatesPerTile;
    }

    private static void runMultiGpuOutOfCoreLambdaBoxes(
            Request req, DPGpuOutOfCore.Plan plan,
            GpuExecutor[] executors, int nGpus)
            throws InterruptedException {
        int workers = (int)Math.min((long)nGpus, plan.lambdaBoxCount);
        OutOfCorePartial[] partials = new OutOfCorePartial[workers];
        Throwable[] errors = new Throwable[workers];
        Thread[] threads = new Thread[workers];
        long base = plan.lambdaBoxCount / workers;
        long remainder = plan.lambdaBoxCount % workers;
        long offset = 0L;
        long startNanos = System.nanoTime();
        for (int worker = 0; worker < workers; worker++) {
            long count = base + (worker < remainder ? 1L : 0L);
            final long lambdaStart = offset;
            final long lambdaCount = count;
            final int index = worker;
            threads[worker] = new Thread(() -> {
                try {
                    partials[index] = executors[index]
                            .runOutOfCoreLambdaRange(
                                    req, plan, lambdaStart, lambdaCount);
                } catch (Throwable t) {
                    errors[index] = t;
                }
            }, "bms-gpu-dp-ooc-lambda-" + worker);
            threads[worker].start();
            offset += count;
        }
        for (Thread thread : threads) {
            thread.join();
        }
        for (Throwable error : errors) {
            if (error != null) {
                throw new RuntimeException(
                        "multi-GPU OOC lambda-box split failed: "
                                + error.getMessage(), error);
            }
        }

        OutOfCorePartial merged = partials[0];
        if (merged == null || merged.indices == null) {
            throw new IllegalStateException(
                    "multi-GPU OOC lambda split returned no output");
        }
        for (int worker = 1; worker < workers; worker++) {
            OutOfCorePartial partial = partials[worker];
            if (partial == null || partial.indices == null
                    || partial.indices.length != merged.indices.length) {
                throw new IllegalStateException(
                        "multi-GPU OOC lambda split returned mismatched output");
            }
            for (int i = 0; i < merged.indices.length; i++) {
                if (partial.indices[i] != merged.indices[i]) {
                    throw new IllegalStateException(
                            "multi-GPU OOC lambda split changed output ordering");
                }
                merged.lower[i] = mergeLogValues(
                        merged.lower[i], partial.lower[i]);
                merged.upper[i] = mergeLogValues(
                        merged.upper[i], partial.upper[i]);
            }
        }
        req.outTable.copyFromIndexed(merged.indices,
                DoubleBuffer.wrap(merged.lower),
                DoubleBuffer.wrap(merged.upper), merged.indices.length);

        if (req.progress) {
            double ms = (System.nanoTime() - startNanos) / 1e6;
            System.out.println(BranchDpConfig.getBackendLogPrefix()
                    + " GPU DP bounded out-of-core lambda-box multi-gpu done"
                    + ", gpus=" + workers
                    + ", lambdaBoxes=" + plan.lambdaBoxCount
                    + ", outputStates=" + merged.indices.length
                    + ", peakDeviceBytes=" + plan.estimatedDeviceBytes
                    + ", elapsedMs=" + String.format(
                    java.util.Locale.ROOT, "%.1f", ms));
        }
    }

    private static double mergeLogValues(double first, double second) {
        if (Double.isNaN(first) || Double.isNaN(second)) {
            return Double.NaN;
        }
        if (first == Double.POSITIVE_INFINITY
                || second == Double.POSITIVE_INFINITY) {
            return Double.POSITIVE_INFINITY;
        }
        if (first == Double.NEGATIVE_INFINITY) return second;
        if (second == Double.NEGATIVE_INFINITY) return first;
        double max = Math.max(first, second);
        return max + Math.log(Math.exp(first - max)
                + Math.exp(second - max));
    }

    private static void runTiled(Request req, GpuExecutor executor, long rangeStart, long rangeSize) {
        if (rangeSize <= 0) {
            return;
        }
        if (req.mStateChunk <= 0 || req.mStateChunk > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("GPU DP output tile exceeds CUDA grid int limit: " + req.mStateChunk);
        }
        executor.runRange(req, rangeStart, rangeSize);
    }

    private static GpuExecutor[] makeExecutors(Request req, List<Gpu> gpus, int nGpus) {
        if (req.persistentContext) {
            return persistentExecutors(gpus, nGpus);
        }
        GpuExecutor[] execs = new GpuExecutor[nGpus];
        for (int i = 0; i < nGpus; i++) {
            execs[i] = new GpuExecutor(gpus.get(i), true);
        }
        return execs;
    }

    private static GpuExecutor[] persistentExecutors(List<Gpu> gpus, int nGpus) {
        synchronized (persistentLock) {
            if (persistentPool == null || !persistentPool.matches(gpus, nGpus)) {
                if (persistentPool != null) {
                    persistentPool.cleanup();
                }
                persistentPool = new PersistentPool(gpus, nGpus);
                if (!persistentShutdownHook) {
                    Runtime.getRuntime().addShutdownHook(new Thread(DPGpuFullDP::cleanupPersistentPool,
                            "bms-gpu-dp-persistent-cleanup"));
                    persistentShutdownHook = true;
                }
            }
            return persistentPool.executors(nGpus);
        }
    }

    private static void cleanupExecutors(GpuExecutor[] executors) {
        if (executors == null) {
            return;
        }
        for (GpuExecutor executor : executors) {
            if (executor != null) {
                executor.cleanupIfOwned();
            }
        }
    }

    private static final class PersistentPool {
        private final Gpu[] gpus;
        private final GpuExecutor[] executors;

        PersistentPool(List<Gpu> available, int nGpus) {
            this.gpus = new Gpu[nGpus];
            this.executors = new GpuExecutor[nGpus];
            for (int i = 0; i < nGpus; i++) {
                this.gpus[i] = available.get(i);
                this.executors[i] = new GpuExecutor(this.gpus[i], false);
            }
        }

        boolean matches(List<Gpu> available, int nGpus) {
            if (executors.length < nGpus) {
                return false;
            }
            for (int i = 0; i < nGpus; i++) {
                if (gpus[i] != available.get(i)) {
                    return false;
                }
            }
            return true;
        }

        GpuExecutor[] executors(int nGpus) {
            GpuExecutor[] out = new GpuExecutor[nGpus];
            System.arraycopy(executors, 0, out, 0, nGpus);
            return out;
        }

        void cleanup() {
            for (int i = executors.length - 1; i >= 0; i--) {
                executors[i].cleanupPersistent();
            }
        }
    }

    private static final class GpuExecutor {
        private final Context context;
        private final GpuStream stream;
        private final boolean owned;

        GpuExecutor(Gpu gpu, boolean owned) {
            this.owned = owned;
            this.context = new Context(gpu);
            this.context.attachCurrentThread();
            this.stream = new GpuStream(context);
        }

        synchronized void runRange(Request req, long rangeStart, long rangeSize) {
            context.attachCurrentThread();
            runOnGpuRange(req, stream, rangeStart, rangeSize);
        }

        synchronized void runUnionRange(Request req, long unionStart, long unionCount) {
            context.attachCurrentThread();
            runOnGpuUnionRangeSliced(req, stream, unionStart, unionCount);
        }

        synchronized void runHybridRowRange(Request req, HybridPlan plan,
                                            long rowStart, long rowCount) {
            context.attachCurrentThread();
            runOnGpuHybridRowRange(req, plan, stream, rowStart, rowCount);
        }

        synchronized void runOutOfCore(Request req, DPGpuOutOfCore.Plan plan,
                                       long boxStart, long boxCount,
                                       long freeStart, long freeCount) {
            context.attachCurrentThread();
            runOnGpuOutOfCore(req, plan, stream, boxStart, boxCount,
                    freeStart, freeCount);
        }

        synchronized OutOfCorePartial runOutOfCoreLambdaRange(
                Request req, DPGpuOutOfCore.Plan plan,
                long lambdaStart, long lambdaCount) {
            context.attachCurrentThread();
            OutOfCorePartial partial = new OutOfCorePartial();
            runOnGpuOutOfCore(req, plan, stream,
                    0L, plan.mBoxCount, 0L, req.parentFreeStateCount,
                    lambdaStart, lambdaCount, partial);
            return partial;
        }

        void cleanupIfOwned() {
            if (owned) {
                cleanup();
            }
        }

        void cleanupPersistent() {
            cleanup();
        }

        private void cleanup() {
            try {
                stream.cleanup();
            } catch (Throwable t) {
                t.printStackTrace(System.err);
            }
            try {
                context.cleanup();
            } catch (Throwable t) {
                t.printStackTrace(System.err);
            }
        }
    }

    /** One GPU's log-space partial for a disjoint lambda-box range. */
    private static final class OutOfCorePartial {
        long[] indices;
        double[] lower;
        double[] upper;

        void record(long[] outputIndices, DoubleBuffer lowerSource,
                    DoubleBuffer upperSource, int count) {
            if (indices != null) {
                throw new IllegalStateException(
                        "lambda-box split produced more than one output block");
            }
            indices = Arrays.copyOf(outputIndices, count);
            lower = new double[count];
            upper = new double[count];
            lowerSource.rewind();
            upperSource.rewind();
            lowerSource.get(lower);
            upperSource.get(upper);
        }
    }

    private static void runOnGpuRange(Request req, GpuStream stream, long rangeStart, long rangeSize) {
        stream.getContext().attachCurrentThread();
        long usable = queryUsableVramInCurrentContext();
        long fullNeed = estimateDeviceBytes(req);
        if (!req.forceChildSlicing && (usable < 0L || fullNeed <= usable)) {
            runOnGpuRangeFull(req, stream, rangeStart, rangeSize, usable, fullNeed);
            return;
        }
        if (canUseChildSlicing(req)) {
            runOnGpuUnionRangeSliced(req, stream, 0L, req.unionStateCount, usable);
            return;
        }
        throw new GpuMemoryExceededException("GPU DP footprint ~" + fullNeed
                + " B exceeds usable VRAM " + usable + " B and child slicing is disabled/unavailable; "
                + "lambdaStates=" + req.totalLambdaStates
                + ", childTableElems=" + req.childTableTotal);
    }

    private static void runOnGpuRangeFull(Request req, GpuStream stream, long rangeStart, long rangeSize,
            long usable, long need) {
        List<CUBuffer<?>> buffers = new ArrayList<>();
        // Large double inputs (child tables, lm/lambda energy tables) can exceed the
        // 2 GiB hard cap of a single java.nio direct buffer, so they bypass CUBuffer
        // and are uploaded straight into raw cuMemAlloc'd device memory (size is long;
        // the device side / cuMemcpyHtoD ByteCount are long, only NIO was int-capped).
        List<CUdeviceptr> rawBufs = new ArrayList<>();

        try {
            stream.getContext().attachCurrentThread();

            // Proactive VRAM gate: child tables must be fully resident (their indexing
            // is scattered, so they cannot be streamed in slices). If the footprint
            // does not fit this GPU, stop here rather than allocate-then-OOM.
            if (usable >= 0L) {
                if (need > usable) {
                    throw new GpuMemoryExceededException("GPU DP footprint ~" + need
                            + " B exceeds usable VRAM " + usable + " B; "
                            + "lambdaStates=" + req.totalLambdaStates
                            + ", childTableElems=" + req.childTableTotal
                            + " -- resubmit on a larger-VRAM GPU or enable child table slicing");
                }
            }

            int blockThreads = resolveBlockThreads(req, stream);
            Kernel kernel = new Kernel(stream, "dp");
            Kernel.Function func = kernel.makeFunction("full_dp_n_children");
            func.blockThreads = blockThreads;
            func.sharedMemCalc = bt -> 4*bt*Double.BYTES;

            CUBuffer<IntBuffer> mCounts = uploadInts(stream, req.mCounts, buffers);
            CUBuffer<IntBuffer> lambdaCounts = uploadInts(stream, req.lambdaCounts, buffers);
            CUdeviceptr lambdaOnlyRigid = uploadDoublesBig(req.lambdaOnlyRigid, rawBufs);
            CUdeviceptr lambdaOnlyMin = uploadDoublesBig(req.lambdaOnlyMin, rawBufs);
            CUdeviceptr lmRigid = uploadDoublesBig(req.lmRigid, rawBufs);
            CUdeviceptr lmMin = uploadDoublesBig(req.lmMin, rawBufs);
            CUBuffer<IntBuffer> lmLamSlots = uploadInts(stream, req.lmLamSlots, buffers);
            CUBuffer<IntBuffer> lmMSlots = uploadInts(stream, req.lmMSlots, buffers);
            CUBuffer<IntBuffer> lmMCounts = uploadInts(stream, req.lmMCounts, buffers);
            CUBuffer<LongBuffer> lmOffsets = uploadLongs(stream, req.lmOffsets, buffers);
            GpuTripleBuffers triple = uploadTripleBuffers(req, stream, buffers, rawBufs);
            CUBuffer<IntBuffer> childMSrcAll = uploadInts(stream, req.childMSrcAll, buffers);
            CUBuffer<LongBuffer> childMStrideAll = uploadLongs(stream, req.childMStrideAll, buffers);
            CUBuffer<IntBuffer> childMTermOff = uploadInts(stream, req.childMTermOff, buffers);
            CUBuffer<IntBuffer> childMTermCnt = uploadInts(stream, req.childMTermCnt, buffers);
            CUBuffer<IntBuffer> childLSrcAll = uploadInts(stream, req.childLSrcAll, buffers);
            CUBuffer<LongBuffer> childLStrideAll = uploadLongs(stream, req.childLStrideAll, buffers);
            CUBuffer<IntBuffer> childLTermOff = uploadInts(stream, req.childLTermOff, buffers);
            CUBuffer<IntBuffer> childLTermCnt = uploadInts(stream, req.childLTermCnt, buffers);
            CUBuffer<LongBuffer> childTableBase = uploadLongs(stream, req.childTableBase, buffers);
            // Chunk-granularity upload: childLowerChunks/childUpperChunks may have more
            // entries than there are logical children (a sharded child contributes one
            // entry per shard), so this uses childChunkOffsets rather than
            // childTableBase (which stays indexed one-per-logical-child for the kernel).
            CUdeviceptr childLowerAll = uploadChildTables(req.childLowerChunks, req.childChunkOffsets,
                    req.childTableTotal, rawBufs);
            CUdeviceptr childUpperAll = uploadChildTables(req.childUpperChunks, req.childChunkOffsets,
                    req.childTableTotal, rawBufs);
            int maxTile = (int)Math.min(req.mStateChunk, rangeSize);
            CUBuffer<DoubleBuffer> outLower = makeDoubles(stream, maxTile, buffers);
            CUBuffer<DoubleBuffer> outUpper = makeDoubles(stream, maxTile, buffers);

            long[] mStateCountArg = { req.mStateCount };
            long[] mStateOffsetArg = { rangeStart };
            int[] totalLambdaStatesArg = { req.totalLambdaStates };
            int[] mCountArg = { req.mCounts.length };
            int[] lambdaCountArg = { req.lambdaCounts.length };
            int[] lmPairCountArg = { req.lmLamSlots.length };
            int[] tripleFactorCountArg = { req.tripleOffsets.length };
            int[] numChildrenArg = { req.numChildren };
            double[] invRTArg = { req.invRT };

            long remaining = rangeSize;
            long offset = rangeStart;
            while (remaining > 0) {
                int chunk = (int)Math.min(remaining, req.mStateChunk);
                func.numBlocks = chunk;
                mStateOffsetArg[0] = offset;

                func.setArgs(Pointer.to(
                    mCounts.getDevicePointer(),
                    lambdaCounts.getDevicePointer(),
                    Pointer.to(lambdaOnlyRigid),
                    Pointer.to(lambdaOnlyMin),
                    Pointer.to(lmRigid),
                    Pointer.to(lmMin),
                    lmLamSlots.getDevicePointer(),
                    lmMSlots.getDevicePointer(),
                    lmMCounts.getDevicePointer(),
                    lmOffsets.getDevicePointer(),
                    triple.slots.getDevicePointer(),
                    triple.strides.getDevicePointer(),
                    triple.offsets.getDevicePointer(),
                    Pointer.to(triple.rigid),
                    Pointer.to(triple.min),
                    childMSrcAll.getDevicePointer(),
                    childMStrideAll.getDevicePointer(),
                    childMTermOff.getDevicePointer(),
                    childMTermCnt.getDevicePointer(),
                    childLSrcAll.getDevicePointer(),
                    childLStrideAll.getDevicePointer(),
                    childLTermOff.getDevicePointer(),
                    childLTermCnt.getDevicePointer(),
                    childTableBase.getDevicePointer(),
                    Pointer.to(childLowerAll),
                    Pointer.to(childUpperAll),
                    outLower.getDevicePointer(),
                    outUpper.getDevicePointer(),
                    Pointer.to(mStateCountArg),
                    Pointer.to(mStateOffsetArg),
                    Pointer.to(totalLambdaStatesArg),
                    Pointer.to(mCountArg),
                    Pointer.to(lambdaCountArg),
                    Pointer.to(lmPairCountArg),
                    Pointer.to(tripleFactorCountArg),
                    Pointer.to(numChildrenArg),
                    Pointer.to(invRTArg)
                ));

                long startNanos = System.nanoTime();
                func.runAsync();
                outLower.downloadAsync();
                outUpper.downloadAsync();
                stream.waitForGpu();

                req.outTable.copyFrom(offset, outLower.getHostBuffer(),
                        outUpper.getHostBuffer(), chunk);

                if (req.progress) {
                    double ms = (System.nanoTime() - startNanos)/1e6;
                    System.out.println(BranchDpConfig.getBackendLogPrefix() + " GPU DP done, mStates=" + req.mStateCount
                            + " (chunk=" + chunk + "@" + offset + ")"
                            + ", lambdaStates=" + req.totalLambdaStates
                            + ", children=" + req.numChildren
                            + ", work=" + req.work
                            + ", blockThreads=" + blockThreads
                            + ", persistentContext=" + req.persistentContext
                            + ", elapsedMs=" + String.format(java.util.Locale.ROOT, "%.1f", ms));
                }

                offset += chunk;
                remaining -= chunk;
            }
        } finally {
            for (int i=buffers.size() - 1; i>=0; i--) {
                try {
                    buffers.get(i).cleanup();
                } catch (Throwable t) {
                    t.printStackTrace(System.err);
                }
            }
            for (int i=rawBufs.size() - 1; i>=0; i--) {
                try {
                    JCudaDriver.cuMemFree(rawBufs.get(i));
                } catch (Throwable t) {
                    t.printStackTrace(System.err);
                }
            }
        }
    }

    private static void runOnGpuHybridRowRange(Request req, HybridPlan plan, GpuStream stream,
                                                long rowStart, long rowCount) {
        if (rowCount <= 0L) {
            return;
        }
        stream.getContext().attachCurrentThread();
        if (req.mStateChunk <= 0L || req.mStateChunk > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("GPU DP output tile exceeds CUDA grid int limit: "
                    + req.mStateChunk);
        }

        List<CUBuffer<?>> buffers = new ArrayList<>();
        List<CUdeviceptr> rawBufs = new ArrayList<>();
        GpuStream uploadStream = null;
        HybridStreamWorkspace[] streamWorkspaces = null;
        ForkJoinTask<PreparedHybridBlock> pendingPrefetch = null;
        try {
            uploadStream = new GpuStream(stream.getContext());
            streamWorkspaces = new HybridStreamWorkspace[]{
                    new HybridStreamWorkspace(uploadStream),
                    new HybridStreamWorkspace(uploadStream)
            };
            int blockThreads = resolveBlockThreads(req, stream);
            Kernel kernel = new Kernel(stream, "dp");
            Kernel.Function func = kernel.makeFunction("full_dp_n_children_hybrid");
            func.blockThreads = blockThreads;
            func.sharedMemCalc = bt -> 4 * bt * Double.BYTES;

            CUBuffer<IntBuffer> mCounts = uploadInts(stream, req.mCounts, buffers);
            CUBuffer<IntBuffer> lambdaCounts = uploadInts(stream, req.lambdaCounts, buffers);
            CUdeviceptr lambdaOnlyRigid = uploadDoublesBig(req.lambdaOnlyRigid, rawBufs);
            CUdeviceptr lambdaOnlyMin = uploadDoublesBig(req.lambdaOnlyMin, rawBufs);
            CUdeviceptr lmRigid = uploadDoublesBig(req.lmRigid, rawBufs);
            CUdeviceptr lmMin = uploadDoublesBig(req.lmMin, rawBufs);
            CUBuffer<IntBuffer> lmLamSlots = uploadInts(stream, req.lmLamSlots, buffers);
            CUBuffer<IntBuffer> lmMSlots = uploadInts(stream, req.lmMSlots, buffers);
            CUBuffer<IntBuffer> lmMCounts = uploadInts(stream, req.lmMCounts, buffers);
            CUBuffer<LongBuffer> lmOffsets = uploadLongs(stream, req.lmOffsets, buffers);
            GpuTripleBuffers triple = uploadTripleBuffers(req, stream, buffers, rawBufs);
            CUBuffer<IntBuffer> childMSrcAll = uploadInts(stream, req.childMSrcAll, buffers);
            CUBuffer<LongBuffer> childMStrideAll = uploadLongs(stream, req.childMStrideAll, buffers);
            CUBuffer<LongBuffer> childMPackedStrideAll = uploadLongs(stream,
                    req.childMPackedStrideAll, buffers);
            CUBuffer<IntBuffer> childMTermOff = uploadInts(stream, req.childMTermOff, buffers);
            CUBuffer<IntBuffer> childMTermCnt = uploadInts(stream, req.childMTermCnt, buffers);
            CUBuffer<IntBuffer> childLSrcAll = uploadInts(stream, req.childLSrcAll, buffers);
            CUBuffer<LongBuffer> childLStrideAll = uploadLongs(stream, req.childLStrideAll, buffers);
            CUBuffer<LongBuffer> childLPackedStrideAll = uploadLongs(stream,
                    req.childLPackedStrideAll, buffers);
            CUBuffer<IntBuffer> childLTermOff = uploadInts(stream, req.childLTermOff, buffers);
            CUBuffer<IntBuffer> childLTermCnt = uploadInts(stream, req.childLTermCnt, buffers);
            CUBuffer<LongBuffer> residentTableBase = uploadLongs(stream, plan.residentTableBase, buffers);
            CUBuffer<LongBuffer> childLambdaStates = uploadLongs(stream, req.childLambdaStates, buffers);
            CUdeviceptr residentLower = uploadResidentChildTables(req, plan, true, rawBufs);
            CUdeviceptr residentUpper = uploadResidentChildTables(req, plan, false, rawBufs);

            int maxTile = (int)Math.min(req.mStateChunk, req.mStateCount);
            CUBuffer<DoubleBuffer> outLower = makeDoubles(stream, maxTile, buffers);
            CUBuffer<DoubleBuffer> outUpper = makeDoubles(stream, maxTile, buffers);
            CUBuffer<LongBuffer> mIdxList = stream.makeLongBuffer(maxTile);
            buffers.add(mIdxList);
            long[] mIdxListHost = new long[maxTile];

            long[] mStateCountArg = { req.mStateCount };
            int[] totalLambdaStatesArg = { req.totalLambdaStates };
            int[] mCountArg = { req.mCounts.length };
            int[] lambdaCountArg = { req.lambdaCounts.length };
            int[] lmPairCountArg = { req.lmLamSlots.length };
            int[] tripleFactorCountArg = { req.tripleOffsets.length };
            int[] numChildrenArg = { req.numChildren };
            int[] streamedChildArg = { plan.streamedChild };
            double[] invRTArg = { req.invRT };
            long rowEnd = rowStart + rowCount;
            long processedOutputs = 0L;
            long startNanos = System.nanoTime();

            if (req.progress) {
                System.out.println(BranchDpConfig.getBackendLogPrefix()
                        + " GPU DP hybrid start, streamedChild=" + plan.streamedChild
                        + ", rowRange=[" + rowStart + "," + rowEnd + ")"
                        + ", rowsPerBlock=" + plan.streamedRowsPerBlock
                        + ", otherMStates=" + plan.otherMStateCount
                        + ", residentStates=" + plan.residentTableStates
                        + ", doubleBuffered=true"
                        + ", estimatedAggregateTrafficBytes="
                        + plan.estimatedAggregateTrafficBytes);
            }

            long unscheduledRowStart = rowStart;
            int firstBlockRows = (int)Math.min(plan.streamedRowsPerBlock,
                    rowEnd - unscheduledRowStart);
            PreparedHybridBlock prepared = prepareHybridBlock(req, plan,
                    unscheduledRowStart, firstBlockRows, streamWorkspaces[0]);
            unscheduledRowStart += firstBlockRows;
            int currentWorkspace = 0;

            while (prepared != null) {
                StreamedChildBlock block = prepared.block;
                long blockRowStart = block.rowStart;
                int blockRows = block.rowCount;

                int nextWorkspace = 1 - currentWorkspace;
                if (unscheduledRowStart < rowEnd) {
                    final long nextRowStart = unscheduledRowStart;
                    final int nextRows = (int)Math.min(plan.streamedRowsPerBlock,
                            rowEnd - nextRowStart);
                    final HybridStreamWorkspace nextSlot = streamWorkspaces[nextWorkspace];
                    pendingPrefetch = CHILD_GATHER_POOL.submit(() ->
                            prepareHybridBlock(req, plan, nextRowStart, nextRows, nextSlot));
                    unscheduledRowStart += nextRows;
                }

                StreamedChildMIdxEnumerator mIdxs = new StreamedChildMIdxEnumerator(
                        req, plan.streamedChild, blockRowStart, blockRows);
                long outputCount = mIdxs.outputCount();
                long emitted = 0L;
                double gpuAndDownloadMs = 0.0;
                while (emitted < outputCount) {
                    int chunk = (int)Math.min((long)maxTile, outputCount - emitted);
                    int filled = mIdxs.fill(mIdxListHost, chunk);
                    if (filled != chunk) {
                        throw new IllegalStateException("Hybrid M-index enumerator emitted "
                                + filled + " states, expected " + chunk);
                    }
                    LongBuffer mIdxHostBuffer = mIdxList.getHostBuffer();
                    mIdxHostBuffer.clear();
                    mIdxHostBuffer.put(mIdxListHost, 0, chunk);
                    mIdxList.uploadAsync((long)chunk * Long.BYTES);
                    long[] streamedRowStartArg = { blockRowStart };
                    int[] streamedRowCountArg = { blockRows };
                    func.numBlocks = chunk;
                    func.setArgs(Pointer.to(
                                mCounts.getDevicePointer(),
                                lambdaCounts.getDevicePointer(),
                                Pointer.to(lambdaOnlyRigid),
                                Pointer.to(lambdaOnlyMin),
                                Pointer.to(lmRigid),
                                Pointer.to(lmMin),
                                lmLamSlots.getDevicePointer(),
                                lmMSlots.getDevicePointer(),
                                lmMCounts.getDevicePointer(),
                                lmOffsets.getDevicePointer(),
                                triple.slots.getDevicePointer(),
                                triple.strides.getDevicePointer(),
                                triple.offsets.getDevicePointer(),
                                Pointer.to(triple.rigid),
                                Pointer.to(triple.min),
                                childMSrcAll.getDevicePointer(),
                                childMStrideAll.getDevicePointer(),
                                childMPackedStrideAll.getDevicePointer(),
                                childMTermOff.getDevicePointer(),
                                childMTermCnt.getDevicePointer(),
                                childLSrcAll.getDevicePointer(),
                                childLStrideAll.getDevicePointer(),
                                childLPackedStrideAll.getDevicePointer(),
                                childLTermOff.getDevicePointer(),
                                childLTermCnt.getDevicePointer(),
                                residentTableBase.getDevicePointer(),
                                childLambdaStates.getDevicePointer(),
                                Pointer.to(residentLower),
                                Pointer.to(residentUpper),
                                streamWorkspaces[currentWorkspace].lowerDevicePointer(),
                                streamWorkspaces[currentWorkspace].upperDevicePointer(),
                                mIdxList.getDevicePointer(),
                                outLower.getDevicePointer(),
                                outUpper.getDevicePointer(),
                                Pointer.to(mStateCountArg),
                                Pointer.to(totalLambdaStatesArg),
                                Pointer.to(mCountArg),
                                Pointer.to(lambdaCountArg),
                                Pointer.to(lmPairCountArg),
                                Pointer.to(tripleFactorCountArg),
                                Pointer.to(numChildrenArg),
                                Pointer.to(streamedChildArg),
                                Pointer.to(streamedRowStartArg),
                                Pointer.to(streamedRowCountArg),
                                Pointer.to(invRTArg)
                    ));
                    long gpuStartNanos = System.nanoTime();
                    func.runAsync();
                    long outputBytes = (long)chunk * Double.BYTES;
                    outLower.downloadAsync(outputBytes);
                    outUpper.downloadAsync(outputBytes);
                    stream.waitForGpu();
                    req.outTable.copyFromIndexed(mIdxListHost, outLower.getHostBuffer(),
                            outUpper.getHostBuffer(), chunk);
                    gpuAndDownloadMs += (System.nanoTime() - gpuStartNanos) / 1e6;
                    emitted += chunk;
                }
                processedOutputs += outputCount;
                if (req.progress) {
                    double ms = (System.nanoTime() - startNanos) / 1e6;
                    System.out.println(BranchDpConfig.getBackendLogPrefix()
                            + " GPU DP hybrid block done, rows=[" + blockRowStart + ","
                            + (blockRowStart + blockRows) + ")"
                            + ", packedStates=" + block.packedStates
                            + ", outputs=" + outputCount
                            + ", processedOutputs=" + processedOutputs
                            + ", gatherMs="
                            + String.format(java.util.Locale.ROOT, "%.1f", prepared.gatherMs)
                            + ", uploadMs="
                            + String.format(java.util.Locale.ROOT, "%.1f", prepared.uploadMs)
                            + ", gpuAndDownloadMs="
                            + String.format(java.util.Locale.ROOT, "%.1f", gpuAndDownloadMs)
                            + ", elapsedMs="
                            + String.format(java.util.Locale.ROOT, "%.1f", ms));
                }

                if (pendingPrefetch != null) {
                    prepared = pendingPrefetch.join();
                    pendingPrefetch = null;
                    currentWorkspace = nextWorkspace;
                } else {
                    prepared = null;
                }
            }
        } finally {
            // If kernel/output handling failed while the producer was filling the
            // other slot, let it finish before unpinning/freeing that slot.
            if (pendingPrefetch != null) {
                try {
                    pendingPrefetch.join();
                } catch (Throwable ignored) {
                    // Preserve the original failure; cleanup below is best effort.
                }
            }
            if (streamWorkspaces != null) {
                for (int i = streamWorkspaces.length - 1; i >= 0; i--) {
                    try {
                        streamWorkspaces[i].cleanup();
                    } catch (Throwable t) {
                        t.printStackTrace(System.err);
                    }
                }
            }
            if (uploadStream != null) {
                uploadStream.cleanup();
            }
            for (int i = buffers.size() - 1; i >= 0; i--) {
                try {
                    buffers.get(i).cleanup();
                } catch (Throwable t) {
                    t.printStackTrace(System.err);
                }
            }
            for (int i = rawBufs.size() - 1; i >= 0; i--) {
                try {
                    JCudaDriver.cuMemFree(rawBufs.get(i));
                } catch (Throwable t) {
                    t.printStackTrace(System.err);
                }
            }
        }
    }

    private static void runOnGpuUnionRangeSliced(Request req, GpuStream stream, long unionStart, long unionCount) {
        stream.getContext().attachCurrentThread();
        runOnGpuUnionRangeSliced(req, stream, unionStart, unionCount, queryUsableVramInCurrentContext());
    }

    private static void runOnGpuUnionRangeSliced(Request req, GpuStream stream, long unionStart, long unionCount,
            long usable) {
        if (unionCount <= 0) {
            return;
        }
        if (!canUseChildSlicing(req)) {
            throw new GpuMemoryExceededException("GPU child-table slicing requested but the request has no slice plan");
        }
        ensureChildSliceIndexMaps(req);
        if (req.mStateChunk <= 0 || req.mStateChunk > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("GPU DP output tile exceeds CUDA grid int limit: " + req.mStateChunk);
        }
        // Do this before loading the kernel or allocating any of the fixed device
        // buffers. A one-union-state slice is the smallest representation supported
        // by the row slicer; if that cannot fit, allocation order must not decide
        // whether the user gets this diagnostic or a raw CUDA OOM (as happened on
        // smaller-VRAM Titan V nodes for 3k3q).
        long minimumSlicedNeed = estimateMinimumSlicedDeviceBytes(req);
        if (usable >= 0L && minimumSlicedNeed > usable) {
            long minimumChildBytes = estimateMinimumChildSliceBytes(req);
            throw new GpuMemoryExceededException("One child-table slice still exceeds usable VRAM; need="
                    + minimumSlicedNeed + " B, usable=" + usable
                    + " B, minimumSliceChildBytes=" + minimumChildBytes
                    + " B. This shape requires lambda tiling in addition to child-row slicing.");
        }
        List<CUBuffer<?>> buffers = new ArrayList<>();
        List<CUdeviceptr> rawBufs = new ArrayList<>();
        ChildSliceWorkspace sliceWorkspace = new ChildSliceWorkspace();

        try {
            stream.getContext().attachCurrentThread();

            int blockThreads = resolveBlockThreads(req, stream);
            Kernel kernel = new Kernel(stream, "dp");
            Kernel.Function func = kernel.makeFunction("full_dp_n_children_sliced");
            func.blockThreads = blockThreads;
            func.sharedMemCalc = bt -> 4*bt*Double.BYTES;

            CUBuffer<IntBuffer> mCounts = uploadInts(stream, req.mCounts, buffers);
            CUBuffer<IntBuffer> lambdaCounts = uploadInts(stream, req.lambdaCounts, buffers);
            CUdeviceptr lambdaOnlyRigid = uploadDoublesBig(req.lambdaOnlyRigid, rawBufs);
            CUdeviceptr lambdaOnlyMin = uploadDoublesBig(req.lambdaOnlyMin, rawBufs);
            CUdeviceptr lmRigid = uploadDoublesBig(req.lmRigid, rawBufs);
            CUdeviceptr lmMin = uploadDoublesBig(req.lmMin, rawBufs);
            CUBuffer<IntBuffer> lmLamSlots = uploadInts(stream, req.lmLamSlots, buffers);
            CUBuffer<IntBuffer> lmMSlots = uploadInts(stream, req.lmMSlots, buffers);
            CUBuffer<IntBuffer> lmMCounts = uploadInts(stream, req.lmMCounts, buffers);
            CUBuffer<LongBuffer> lmOffsets = uploadLongs(stream, req.lmOffsets, buffers);
            GpuTripleBuffers triple = uploadTripleBuffers(req, stream, buffers, rawBufs);
            CUBuffer<IntBuffer> childMSrcAll = uploadInts(stream, req.childMSrcAll, buffers);
            CUBuffer<LongBuffer> childMPackedStrideAll = uploadLongs(stream, req.childMPackedStrideAll, buffers);
            CUBuffer<IntBuffer> childMTermOff = uploadInts(stream, req.childMTermOff, buffers);
            CUBuffer<IntBuffer> childMTermCnt = uploadInts(stream, req.childMTermCnt, buffers);
            CUBuffer<IntBuffer> childLSrcAll = uploadInts(stream, req.childLSrcAll, buffers);
            CUBuffer<LongBuffer> childLPackedStrideAll = uploadLongs(stream, req.childLPackedStrideAll, buffers);
            CUBuffer<IntBuffer> childLTermOff = uploadInts(stream, req.childLTermOff, buffers);
            CUBuffer<IntBuffer> childLTermCnt = uploadInts(stream, req.childLTermCnt, buffers);
            CUBuffer<LongBuffer> childLambdaStates = uploadLongs(stream, req.childLambdaStates, buffers);
            int maxTile = (int)Math.min(req.mStateChunk, req.mStateCount);
            CUBuffer<DoubleBuffer> outLower = makeDoubles(stream, maxTile, buffers);
            CUBuffer<DoubleBuffer> outUpper = makeDoubles(stream, maxTile, buffers);
            CUBuffer<LongBuffer> mIdxList = stream.makeLongBuffer(maxTile);
            buffers.add(mIdxList);
            long[] mIdxListHost = new long[maxTile];

            long childSliceMaxBytes = resolveChildSliceMaxBytes(req, usable);
            long unionStatesPerSlice = chooseUnionStatesPerSlice(req, childSliceMaxBytes);
            long unionEnd = unionStart + unionCount;
            if (req.progress) {
                System.out.println(BranchDpConfig.getBackendLogPrefix() + " GPU DP child slicing start"
                        + ", unionRange=[" + unionStart + "," + unionEnd + ")"
                        + ", unionStatesPerSlice=" + unionStatesPerSlice
                        + ", parentFreeStates=" + req.parentFreeStateCount
                        + ", childSliceMaxBytes=" + childSliceMaxBytes
                        + ", fullDeviceBytes=" + estimateDeviceBytes(req)
                        + ", slicedDeviceBytes~=" + estimateSlicedDeviceBytes(req));
            }

            long[] mStateCountArg = { req.mStateCount };
            int[] totalLambdaStatesArg = { req.totalLambdaStates };
            int[] mCountArg = { req.mCounts.length };
            int[] lambdaCountArg = { req.lambdaCounts.length };
            int[] lmPairCountArg = { req.lmLamSlots.length };
            int[] tripleFactorCountArg = { req.tripleOffsets.length };
            int[] numChildrenArg = { req.numChildren };
            double[] invRTArg = { req.invRT };
            long processedOutputs = 0L;
            long startAllNanos = System.nanoTime();

            for (long sliceStart = unionStart; sliceStart < unionEnd; ) {
                int sliceUnionCount = (int)Math.min(Math.min(unionStatesPerSlice, unionEnd - sliceStart),
                        (long)Integer.MAX_VALUE);
                ChildSlice slice = buildChildSlice(req, sliceStart, sliceUnionCount, sliceWorkspace);
                long sliceBytes = slice.packedStates * 2L * (long)Double.BYTES;
                if (usable >= 0L && estimateSlicedDeviceBytesForSlice(req, slice) > usable) {
                    throw new GpuMemoryExceededException("One child-table slice still exceeds usable VRAM; need="
                            + estimateSlicedDeviceBytesForSlice(req, slice) + " B, usable=" + usable
                            + " B, sliceChildBytes=" + sliceBytes
                            + " B. This shape likely needs lambda tiling in addition to child-row slicing.");
                }
                runChildSlice(req, stream, func, slice, sliceWorkspace, maxTile,
                        mCounts, lambdaCounts,
                        lambdaOnlyRigid, lambdaOnlyMin, lmRigid, lmMin,
                        lmLamSlots, lmMSlots, lmMCounts, lmOffsets,
                        triple,
                        childMSrcAll, childMPackedStrideAll, childMTermOff, childMTermCnt,
                        childLSrcAll, childLPackedStrideAll, childLTermOff, childLTermCnt,
                        childLambdaStates, outLower, outUpper, mIdxList, mIdxListHost,
                        mStateCountArg, totalLambdaStatesArg, mCountArg, lambdaCountArg,
                        lmPairCountArg, tripleFactorCountArg, numChildrenArg, invRTArg);
                processedOutputs += slice.outputCount;
                if (req.progress) {
                    double ms = (System.nanoTime() - startAllNanos)/1e6;
                    System.out.println(BranchDpConfig.getBackendLogPrefix() + " GPU DP child slice done"
                            + ", union=[" + sliceStart + "," + (sliceStart + sliceUnionCount) + ")"
                            + ", outputs=" + slice.outputCount
                            + ", packedStates=" + slice.packedStates
                            + ", packedBytes=" + sliceBytes
                            + ", processedOutputs=" + processedOutputs
                            + ", elapsedMs=" + String.format(java.util.Locale.ROOT, "%.1f", ms));
                }
                sliceStart += sliceUnionCount;
            }
        } finally {
            sliceWorkspace.cleanup();
            for (int i=buffers.size() - 1; i>=0; i--) {
                try {
                    buffers.get(i).cleanup();
                } catch (Throwable t) {
                    t.printStackTrace(System.err);
                }
            }
            for (int i=rawBufs.size() - 1; i>=0; i--) {
                try {
                    JCudaDriver.cuMemFree(rawBufs.get(i));
                } catch (Throwable t) {
                    t.printStackTrace(System.err);
                }
            }
        }
    }

    private static void runChildSlice(Request req, GpuStream stream, Kernel.Function func, ChildSlice slice,
            ChildSliceWorkspace sliceWorkspace, int maxTile,
            CUBuffer<IntBuffer> mCounts,
            CUBuffer<IntBuffer> lambdaCounts,
            CUdeviceptr lambdaOnlyRigid,
            CUdeviceptr lambdaOnlyMin,
            CUdeviceptr lmRigid,
            CUdeviceptr lmMin,
            CUBuffer<IntBuffer> lmLamSlots,
            CUBuffer<IntBuffer> lmMSlots,
            CUBuffer<IntBuffer> lmMCounts,
            CUBuffer<LongBuffer> lmOffsets,
            GpuTripleBuffers triple,
            CUBuffer<IntBuffer> childMSrcAll,
            CUBuffer<LongBuffer> childMPackedStrideAll,
            CUBuffer<IntBuffer> childMTermOff,
            CUBuffer<IntBuffer> childMTermCnt,
            CUBuffer<IntBuffer> childLSrcAll,
            CUBuffer<LongBuffer> childLPackedStrideAll,
            CUBuffer<IntBuffer> childLTermOff,
            CUBuffer<IntBuffer> childLTermCnt,
            CUBuffer<LongBuffer> childLambdaStates,
            CUBuffer<DoubleBuffer> outLower,
            CUBuffer<DoubleBuffer> outUpper,
            CUBuffer<LongBuffer> mIdxList,
            long[] mIdxListHost,
            long[] mStateCountArg,
            int[] totalLambdaStatesArg,
            int[] mCountArg,
            int[] lambdaCountArg,
            int[] lmPairCountArg,
            int[] tripleFactorCountArg,
            int[] numChildrenArg,
            double[] invRTArg) {

        List<CUBuffer<?>> sliceBuffers = new ArrayList<>();
        try {
            CUBuffer<LongBuffer> childPackedBase = uploadLongs(stream, slice.childPackedBase, sliceBuffers);
            CUBuffer<LongBuffer> childRowKeyBase = uploadLongs(stream, slice.childRowKeyBase, sliceBuffers);
            CUBuffer<IntBuffer> childRowKeyCount = uploadInts(stream, slice.childRowKeyCount, sliceBuffers);
            CUBuffer<LongBuffer> childRowKeysAll = uploadLongs(stream, slice.childRowKeysAll, sliceBuffers);
            sliceWorkspace.upload(slice.packedStates);
            CUdeviceptr childLowerPacked = sliceWorkspace.lowerDevice;
            CUdeviceptr childUpperPacked = sliceWorkspace.upperDevice;

            MIdxTileEnumerator mIdxs = new MIdxTileEnumerator(req, slice.unionStart, slice.unionCount);
            long emitted = 0L;
            while (emitted < slice.outputCount) {
                int chunk = (int)Math.min((long)maxTile, slice.outputCount - emitted);
                int filled = mIdxs.fill(mIdxListHost, chunk);
                if (filled != chunk) {
                    throw new IllegalStateException("M-index enumerator emitted " + filled
                            + " states, expected " + chunk);
                }
                LongBuffer mIdxHostBuffer = mIdxList.getHostBuffer();
                mIdxHostBuffer.clear();
                mIdxHostBuffer.put(mIdxListHost, 0, chunk);
                mIdxList.uploadAsync((long)chunk * Long.BYTES);
                func.numBlocks = chunk;
                func.setArgs(Pointer.to(
                        mCounts.getDevicePointer(),
                        lambdaCounts.getDevicePointer(),
                        Pointer.to(lambdaOnlyRigid),
                        Pointer.to(lambdaOnlyMin),
                        Pointer.to(lmRigid),
                        Pointer.to(lmMin),
                        lmLamSlots.getDevicePointer(),
                        lmMSlots.getDevicePointer(),
                        lmMCounts.getDevicePointer(),
                        lmOffsets.getDevicePointer(),
                        triple.slots.getDevicePointer(),
                        triple.strides.getDevicePointer(),
                        triple.offsets.getDevicePointer(),
                        Pointer.to(triple.rigid),
                        Pointer.to(triple.min),
                        childMSrcAll.getDevicePointer(),
                        childMPackedStrideAll.getDevicePointer(),
                        childMTermOff.getDevicePointer(),
                        childMTermCnt.getDevicePointer(),
                        childLSrcAll.getDevicePointer(),
                        childLPackedStrideAll.getDevicePointer(),
                        childLTermOff.getDevicePointer(),
                        childLTermCnt.getDevicePointer(),
                        childPackedBase.getDevicePointer(),
                        childRowKeyBase.getDevicePointer(),
                        childRowKeyCount.getDevicePointer(),
                        childRowKeysAll.getDevicePointer(),
                        childLambdaStates.getDevicePointer(),
                        Pointer.to(childLowerPacked),
                        Pointer.to(childUpperPacked),
                        mIdxList.getDevicePointer(),
                        outLower.getDevicePointer(),
                        outUpper.getDevicePointer(),
                        Pointer.to(mStateCountArg),
                        Pointer.to(totalLambdaStatesArg),
                        Pointer.to(mCountArg),
                        Pointer.to(lambdaCountArg),
                        Pointer.to(lmPairCountArg),
                        Pointer.to(tripleFactorCountArg),
                        Pointer.to(numChildrenArg),
                        Pointer.to(invRTArg)
                ));

                long startNanos = System.nanoTime();
                func.runAsync();
                long outputBytes = (long)chunk * Double.BYTES;
                outLower.downloadAsync(outputBytes);
                outUpper.downloadAsync(outputBytes);
                stream.waitForGpu();
                req.outTable.copyFromIndexed(mIdxListHost, outLower.getHostBuffer(),
                        outUpper.getHostBuffer(), chunk);
                if (req.progress) {
                    double ms = (System.nanoTime() - startNanos)/1e6;
                    System.out.println(BranchDpConfig.getBackendLogPrefix() + " GPU DP sliced tile done"
                            + ", mStates=" + req.mStateCount
                            + " (chunk=" + chunk + ", emitted=" + emitted + "/" + slice.outputCount + ")"
                            + ", lambdaStates=" + req.totalLambdaStates
                            + ", children=" + req.numChildren
                            + ", work=" + req.work
                            + ", blockThreads=" + func.blockThreads
                            + ", elapsedMs=" + String.format(java.util.Locale.ROOT, "%.1f", ms));
                }
                emitted += chunk;
            }
        } finally {
            for (int i=sliceBuffers.size() - 1; i>=0; i--) {
                try {
                    sliceBuffers.get(i).cleanup();
                } catch (Throwable t) {
                    t.printStackTrace(System.err);
                }
            }
        }
    }

    private static void runOnGpuOutOfCore(Request req, DPGpuOutOfCore.Plan plan,
                                          GpuStream stream,
                                          long boxStart, long boxCount,
                                          long freeStart, long freeCount) {
        runOnGpuOutOfCore(req, plan, stream, boxStart, boxCount,
                freeStart, freeCount, 0L, plan.lambdaBoxCount, null);
    }

    /**
     * Execute a bounded OOC range, optionally returning one output block as a
     * log-space partial.  The partial form is used only when multiple GPUs
     * split lambda boxes for one shared output block; the ordinary form writes
     * directly into the destination table as before.
     */
    private static void runOnGpuOutOfCore(Request req,
                                          DPGpuOutOfCore.Plan plan,
                                          GpuStream stream,
                                          long boxStart, long boxCount,
                                          long freeStart, long freeCount,
                                          long lambdaStart, long lambdaCount,
                                          OutOfCorePartial partialOutput) {
        if (boxCount <= 0L || freeCount <= 0L) {
            return;
        }
        if (boxStart < 0L || boxStart + boxCount > plan.mBoxCount
                || freeStart < 0L
                || freeStart + freeCount > req.parentFreeStateCount) {
            throw new IndexOutOfBoundsException("Invalid bounded out-of-core range: boxes=["
                    + boxStart + "," + (boxStart + boxCount) + ")/"
                    + plan.mBoxCount + ", free=[" + freeStart + ","
                    + (freeStart + freeCount) + ")/" + req.parentFreeStateCount);
        }
        if (lambdaStart < 0L || lambdaCount <= 0L
                || lambdaStart + lambdaCount > plan.lambdaBoxCount) {
            throw new IndexOutOfBoundsException(
                    "Invalid bounded out-of-core lambda range: ["
                            + lambdaStart + ","
                            + (lambdaStart + lambdaCount) + ")/"
                            + plan.lambdaBoxCount);
        }

        stream.getContext().attachCurrentThread();
        long usable = queryUsableVramInCurrentContext();
        if (usable >= 0L && plan.estimatedDeviceBytes > usable) {
            throw new GpuMemoryExceededException("Bounded out-of-core plan needs "
                    + plan.estimatedDeviceBytes + " B but current usable VRAM is "
                    + usable + " B");
        }

        List<CUBuffer<?>> buffers = new ArrayList<>();
        List<CUdeviceptr> rawBufs = new ArrayList<>();
        try {
            int blockThreads = resolveBlockThreads(req, stream);
            Kernel kernel = new Kernel(stream, "dp");
            Kernel.Function func = kernel.makeFunction(
                    "full_dp_n_children_out_of_core");
            func.blockThreads = blockThreads;
            func.sharedMemCalc = bt -> 4 * bt * Double.BYTES;

            CUBuffer<IntBuffer> mCounts = uploadInts(stream, req.mCounts, buffers);
            CUBuffer<IntBuffer> lambdaCounts = uploadInts(stream,
                    req.lambdaCounts, buffers);
            CUdeviceptr lmRigid = uploadDoublesBig(req.lmRigid, rawBufs);
            CUdeviceptr lmMin = uploadDoublesBig(req.lmMin, rawBufs);
            CUBuffer<IntBuffer> lmLamSlots = uploadInts(stream,
                    req.lmLamSlots, buffers);
            CUBuffer<IntBuffer> lmMSlots = uploadInts(stream,
                    req.lmMSlots, buffers);
            CUBuffer<IntBuffer> lmMCounts = uploadInts(stream,
                    req.lmMCounts, buffers);
            CUBuffer<LongBuffer> lmOffsets = uploadLongs(stream,
                    req.lmOffsets, buffers);
            GpuTripleBuffers triple = uploadTripleBuffers(req, stream, buffers, rawBufs);
            CUBuffer<IntBuffer> childMSrcAll = uploadInts(stream,
                    req.childMSrcAll, buffers);
            CUBuffer<LongBuffer> childMPackedStrideAll = uploadLongs(stream,
                    req.childMPackedStrideAll, buffers);
            CUBuffer<IntBuffer> childMTermOff = uploadInts(stream,
                    req.childMTermOff, buffers);
            CUBuffer<IntBuffer> childMTermCnt = uploadInts(stream,
                    req.childMTermCnt, buffers);
            CUBuffer<IntBuffer> childLSrcAll = uploadInts(stream,
                    req.childLSrcAll, buffers);
            CUBuffer<LongBuffer> childLPackedStrideAll = uploadLongs(stream,
                    req.childLPackedStrideAll, buffers);
            CUBuffer<IntBuffer> childLTermOff = uploadInts(stream,
                    req.childLTermOff, buffers);
            CUBuffer<IntBuffer> childLTermCnt = uploadInts(stream,
                    req.childLTermCnt, buffers);

            int children = Math.max(1, req.numChildren);
            int maxRows = checkedBufferSize(plan.maxChildRowKeys,
                    "child row keys");
            int maxLambdas = checkedBufferSize(plan.maxChildLambdaKeys,
                    "child lambda keys");
            int maxPacked = checkedBufferSize(plan.maxChildPackedStates,
                    "packed child states");
            int maxLambdaTile = checkedBufferSize(plan.maxLambdaTileStates,
                    "lambda tile states");
            int maxOutput = Math.max(1, plan.outputStatesPerTile);

            CUBuffer<LongBuffer> childPackedBase = makeLongs(stream,
                    children, buffers);
            CUBuffer<LongBuffer> childRowKeyBase = makeLongs(stream,
                    children, buffers);
            CUBuffer<IntBuffer> childRowKeyCount = makeInts(stream,
                    children, buffers);
            CUBuffer<LongBuffer> childRowKeysAll = makeLongs(stream,
                    maxRows, buffers);
            CUBuffer<LongBuffer> childLambdaKeyBase = makeLongs(stream,
                    children, buffers);
            CUBuffer<IntBuffer> childLambdaKeyCount = makeInts(stream,
                    children, buffers);
            CUBuffer<LongBuffer> childLambdaKeysAll = makeLongs(stream,
                    maxLambdas, buffers);
            CUBuffer<DoubleBuffer> childLowerPacked = makeDoubles(stream,
                    maxPacked, buffers);
            CUBuffer<DoubleBuffer> childUpperPacked = makeDoubles(stream,
                    maxPacked, buffers);
            CUBuffer<IntBuffer> lambdaIdxList = makeInts(stream,
                    maxLambdaTile, buffers);
            CUBuffer<DoubleBuffer> lambdaOnlyRigidTile = makeDoubles(stream,
                    maxLambdaTile, buffers);
            CUBuffer<DoubleBuffer> lambdaOnlyMinTile = makeDoubles(stream,
                    maxLambdaTile, buffers);
            CUBuffer<LongBuffer> mIdxList = makeLongs(stream, maxOutput,
                    buffers);
            CUBuffer<DoubleBuffer> accLowerMax = makeDoubles(stream,
                    maxOutput, buffers);
            CUBuffer<DoubleBuffer> accLowerSum = makeDoubles(stream,
                    maxOutput, buffers);
            CUBuffer<DoubleBuffer> accUpperMax = makeDoubles(stream,
                    maxOutput, buffers);
            CUBuffer<DoubleBuffer> accUpperSum = makeDoubles(stream,
                    maxOutput, buffers);
            CUBuffer<DoubleBuffer> outLower = makeDoubles(stream, maxOutput,
                    buffers);
            CUBuffer<DoubleBuffer> outUpper = makeDoubles(stream, maxOutput,
                    buffers);

            long auditedDeviceBytes = Math.max(1L, req.lmRigid.length)
                    * (long)Double.BYTES
                    + Math.max(1L, req.lmMin.length) * (long)Double.BYTES
                    + Math.max(1L, req.tripleRigid.length) * (long)Double.BYTES
                    + Math.max(1L, req.tripleMin.length) * (long)Double.BYTES;
            long largestAuditedAllocation = Math.max(
                    Math.max(
                            Math.max(1L, req.lmRigid.length) * (long)Double.BYTES,
                            Math.max(1L, req.lmMin.length) * (long)Double.BYTES),
                    Math.max(
                            Math.max(1L, req.tripleRigid.length) * (long)Double.BYTES,
                            Math.max(1L, req.tripleMin.length) * (long)Double.BYTES));
            for (CUBuffer<?> buffer : buffers) {
                auditedDeviceBytes = Math.addExact(auditedDeviceBytes,
                        buffer.getNumBytes());
                largestAuditedAllocation = Math.max(largestAuditedAllocation,
                        buffer.getNumBytes());
            }
            if (auditedDeviceBytes != plan.estimatedDeviceBytes) {
                throw new IllegalStateException("Out-of-core runtime buffers allocate "
                        + auditedDeviceBytes + " B, but the planner declared "
                        + plan.estimatedDeviceBytes + " B");
            }
            if (largestAuditedAllocation > auditedDeviceBytes
                    || largestAuditedAllocation > plan.budgetBytes) {
                throw new IllegalStateException("Out-of-core runtime allocation "
                        + largestAuditedAllocation + " B is outside the declared "
                        + "workspace/budget (workspace=" + auditedDeviceBytes
                        + " B, budget=" + plan.budgetBytes + " B)");
            }
            synchronized (req) {
                req.outOfCoreAllocationAuditCount++;
                req.outOfCoreAuditedDeviceBytes = Math.max(
                        req.outOfCoreAuditedDeviceBytes, auditedDeviceBytes);
                req.outOfCoreLargestAllocationBytes = Math.max(
                        req.outOfCoreLargestAllocationBytes,
                        largestAuditedAllocation);
                req.outOfCorePlannedDeviceBytes = Math.max(
                        req.outOfCorePlannedDeviceBytes,
                        plan.estimatedDeviceBytes);
                req.outOfCoreBudgetBytes = Math.max(req.outOfCoreBudgetBytes,
                        plan.budgetBytes);
            }

            long[] mStateCountArg = {req.mStateCount};
            int[] mCountArg = {req.mCounts.length};
            int[] lambdaCountArg = {req.lambdaCounts.length};
            int[] lmPairCountArg = {req.lmLamSlots.length};
            int[] tripleFactorCountArg = {req.tripleOffsets.length};
            int[] numChildrenArg = {req.numChildren};
            double[] invRTArg = {req.invRT};
            long boxEnd = boxStart + boxCount;
            long freeEnd = freeStart + freeCount;
            long startNanos = System.nanoTime();
            long completedOutputs = 0L;
            long outputBlocks = 0L;
            long childTileGathers = 0L;
            long kernelLaunches = 0L;
            long childStatesGathered = 0L;
            long gatherNanos = 0L;
            long enqueueNanos = 0L;
            long gpuWaitNanos = 0L;
            long copyOutNanos = 0L;

            if (req.progress) {
                System.out.println(BranchDpConfig.getBackendLogPrefix()
                        + " GPU DP bounded out-of-core start"
                        + ", boxRange=[" + boxStart + "," + boxEnd + ")"
                        + ", freeRange=[" + freeStart + "," + freeEnd + ")"
                        + ", lambdaBoxRange=[" + lambdaStart + ","
                        + (lambdaStart + lambdaCount) + ")"
                        + ", mBoxes=" + plan.mBoxCount
                        + ", lambdaBoxes=" + plan.lambdaBoxCount
                        + ", mTileExtents=" + Arrays.toString(plan.mTileExtents)
                        + ", lambdaTileExtents="
                        + Arrays.toString(plan.lambdaTileExtents)
                        + ", outputStatesPerTile="
                        + plan.outputStatesPerTile
                        + ", childSliceMaxBytes="
                        + req.childSliceMaxBytes
                        + ", outputWorkspaceMaxBytes="
                        + req.outOfCoreOutputWorkspaceMaxBytes
                        + ", peakDeviceBytes=" + plan.estimatedDeviceBytes
                        + ", auditedDeviceBytes=" + auditedDeviceBytes
                        + ", largestAllocationBytes="
                        + largestAuditedAllocation
                        + ", budgetBytes=" + plan.budgetBytes);
            }

            for (long boxOrdinal = boxStart; boxOrdinal < boxEnd;
                 boxOrdinal++) {
                DPGpuOutOfCore.MBox mBox = DPGpuOutOfCore.buildMBox(req,
                        plan, boxOrdinal);
                int freePerBlock = (int)Math.max(1L,
                        plan.outputStatesPerTile / mBox.tile.volume);
                for (long blockFreeStart = freeStart;
                     blockFreeStart < freeEnd;
                     blockFreeStart += freePerBlock) {
                    int blockFreeCount = (int)Math.min((long)freePerBlock,
                            freeEnd - blockFreeStart);
                    DPGpuOutOfCore.OutputBlock output =
                            DPGpuOutOfCore.buildOutputBlock(req, mBox,
                                    blockFreeStart, blockFreeCount);
                    int outputCount = output.mIndices.length;
                    outputBlocks++;
                    putLongs(mIdxList, output.mIndices);
                    mIdxList.uploadAsync((long)outputCount * Long.BYTES);

                    for (long lambdaOrdinal = lambdaStart;
                         lambdaOrdinal < lambdaStart + lambdaCount;
                         lambdaOrdinal++) {
                        DPGpuOutOfCore.LambdaBox lambdaBox =
                                DPGpuOutOfCore.buildLambdaBox(req, plan,
                                        lambdaOrdinal);
                        long gatherStartNanos = System.nanoTime();
                        DPGpuOutOfCore.PackedChildBlock packed =
                                DPGpuOutOfCore.buildPackedChildBlock(req,
                                        mBox, lambdaBox);
                        gatherNanos += System.nanoTime() - gatherStartNanos;
                        childTileGathers++;
                        childStatesGathered = Math.addExact(
                                childStatesGathered, packed.packedStates);
                        if (packed.packedStates > plan.maxChildPackedStates
                                || packed.childRowKeysAll.length
                                > plan.maxChildRowKeys
                                || packed.childLambdaKeysAll.length
                                > plan.maxChildLambdaKeys) {
                            throw new IllegalStateException("Out-of-core tile exceeded "
                                    + "its planned workspace maxima");
                        }

                        long enqueueStartNanos = System.nanoTime();
                        uploadOutOfCoreChildTile(packed, childPackedBase,
                                childRowKeyBase, childRowKeyCount,
                                childRowKeysAll, childLambdaKeyBase,
                                childLambdaKeyCount, childLambdaKeysAll,
                                childLowerPacked, childUpperPacked);
                        uploadOutOfCoreLambdaTile(req, lambdaBox,
                                lambdaIdxList, lambdaOnlyRigidTile,
                                lambdaOnlyMinTile);
                        enqueueNanos += System.nanoTime() - enqueueStartNanos;

                        int lambdaStates = lambdaBox.lambdaIndices.length;
                        int[] lambdaTileStatesArg = {lambdaStates};
                        int[] firstLambdaTileArg = {
                                lambdaOrdinal == lambdaStart ? 1 : 0};
                        int[] lastLambdaTileArg = {
                                lambdaOrdinal + 1L == lambdaStart + lambdaCount
                                        ? 1 : 0};
                        func.numBlocks = outputCount;
                        func.setArgs(Pointer.to(
                                mCounts.getDevicePointer(),
                                lambdaCounts.getDevicePointer(),
                                lambdaOnlyRigidTile.getDevicePointer(),
                                lambdaOnlyMinTile.getDevicePointer(),
                                Pointer.to(lmRigid),
                                Pointer.to(lmMin),
                                lmLamSlots.getDevicePointer(),
                                lmMSlots.getDevicePointer(),
                                lmMCounts.getDevicePointer(),
                                lmOffsets.getDevicePointer(),
                                triple.slots.getDevicePointer(),
                                triple.strides.getDevicePointer(),
                                triple.offsets.getDevicePointer(),
                                Pointer.to(triple.rigid),
                                Pointer.to(triple.min),
                                childMSrcAll.getDevicePointer(),
                                childMPackedStrideAll.getDevicePointer(),
                                childMTermOff.getDevicePointer(),
                                childMTermCnt.getDevicePointer(),
                                childLSrcAll.getDevicePointer(),
                                childLPackedStrideAll.getDevicePointer(),
                                childLTermOff.getDevicePointer(),
                                childLTermCnt.getDevicePointer(),
                                childPackedBase.getDevicePointer(),
                                childRowKeyBase.getDevicePointer(),
                                childRowKeyCount.getDevicePointer(),
                                childRowKeysAll.getDevicePointer(),
                                childLambdaKeyBase.getDevicePointer(),
                                childLambdaKeyCount.getDevicePointer(),
                                childLambdaKeysAll.getDevicePointer(),
                                childLowerPacked.getDevicePointer(),
                                childUpperPacked.getDevicePointer(),
                                lambdaIdxList.getDevicePointer(),
                                mIdxList.getDevicePointer(),
                                accLowerMax.getDevicePointer(),
                                accLowerSum.getDevicePointer(),
                                accUpperMax.getDevicePointer(),
                                accUpperSum.getDevicePointer(),
                                outLower.getDevicePointer(),
                                outUpper.getDevicePointer(),
                                Pointer.to(mStateCountArg),
                                Pointer.to(lambdaTileStatesArg),
                                Pointer.to(mCountArg),
                                Pointer.to(lambdaCountArg),
                                Pointer.to(lmPairCountArg),
                                Pointer.to(tripleFactorCountArg),
                                Pointer.to(numChildrenArg),
                                Pointer.to(firstLambdaTileArg),
                                Pointer.to(lastLambdaTileArg),
                                Pointer.to(invRTArg)
                        ));
                        func.runAsync();
                        kernelLaunches++;
                        if (lastLambdaTileArg[0] != 0) {
                            long outputBytes = (long)outputCount
                                    * Double.BYTES;
                            outLower.downloadAsync(outputBytes);
                            outUpper.downloadAsync(outputBytes);
                        }
                        // All tile input buffers are reusable. Synchronising here
                        // is the correctness boundary before the host overwrites
                        // their pinned memory for the next tile. A later pipeline
                        // may alternate two workspaces without changing the plan.
                        long gpuWaitStartNanos = System.nanoTime();
                        stream.waitForGpu();
                        gpuWaitNanos += System.nanoTime() - gpuWaitStartNanos;
                        if (lastLambdaTileArg[0] != 0) {
                            long copyOutStartNanos = System.nanoTime();
                            if (partialOutput != null) {
                                partialOutput.record(output.mIndices,
                                        outLower.getHostBuffer(),
                                        outUpper.getHostBuffer(), outputCount);
                            } else {
                                req.outTable.copyFromIndexed(output.mIndices,
                                        outLower.getHostBuffer(),
                                        outUpper.getHostBuffer(), outputCount);
                            }
                            copyOutNanos += System.nanoTime() - copyOutStartNanos;
                        }
                    }
                    completedOutputs += outputCount;
                }
            }
            if (req.progress) {
                double ms = (System.nanoTime() - startNanos) / 1e6;
                System.out.println(BranchDpConfig.getBackendLogPrefix()
                        + " GPU DP bounded out-of-core done"
                        + ", outputs=" + completedOutputs
                        + ", outputBlocks=" + outputBlocks
                        + ", childTileGathers=" + childTileGathers
                        + ", kernelLaunches=" + kernelLaunches
                        + ", childBytesGathered="
                        + Math.multiplyExact(childStatesGathered,
                        2L * Double.BYTES)
                        + ", gatherMs=" + String.format(
                        java.util.Locale.ROOT, "%.1f", gatherNanos / 1e6)
                        + ", enqueueMs=" + String.format(
                        java.util.Locale.ROOT, "%.1f", enqueueNanos / 1e6)
                        + ", gpuWaitMs=" + String.format(
                        java.util.Locale.ROOT, "%.1f", gpuWaitNanos / 1e6)
                        + ", copyOutMs=" + String.format(
                        java.util.Locale.ROOT, "%.1f", copyOutNanos / 1e6)
                        + ", elapsedMs="
                        + String.format(java.util.Locale.ROOT, "%.1f", ms));
            }
        } finally {
            for (int i = buffers.size() - 1; i >= 0; i--) {
                try {
                    buffers.get(i).cleanup();
                } catch (Throwable t) {
                    t.printStackTrace(System.err);
                }
            }
            for (int i = rawBufs.size() - 1; i >= 0; i--) {
                try {
                    JCudaDriver.cuMemFree(rawBufs.get(i));
                } catch (Throwable t) {
                    t.printStackTrace(System.err);
                }
            }
        }
    }

    private static int checkedBufferSize(long states, String description) {
        long size = Math.max(1L, states);
        if (size > Integer.MAX_VALUE) {
            throw new GpuMemoryExceededException("Out-of-core " + description
                    + " exceeds Java direct-buffer index limit: " + size);
        }
        return (int)size;
    }

    private static CUBuffer<IntBuffer> makeInts(GpuStream stream, int size,
                                                 List<CUBuffer<?>> buffers) {
        CUBuffer<IntBuffer> buffer = stream.makeIntBuffer(Math.max(1, size));
        buffers.add(buffer);
        return buffer;
    }

    private static CUBuffer<LongBuffer> makeLongs(GpuStream stream, int size,
                                                   List<CUBuffer<?>> buffers) {
        CUBuffer<LongBuffer> buffer = stream.makeLongBuffer(Math.max(1, size));
        buffers.add(buffer);
        return buffer;
    }

    private static void putInts(CUBuffer<IntBuffer> destination, int[] values) {
        IntBuffer host = destination.getHostBuffer();
        host.clear();
        host.put(values);
    }

    private static void putLongs(CUBuffer<LongBuffer> destination,
                                 long[] values) {
        LongBuffer host = destination.getHostBuffer();
        host.clear();
        host.put(values);
    }

    private static void putDoubles(CUBuffer<DoubleBuffer> destination,
                                   double[] values) {
        DoubleBuffer host = destination.getHostBuffer();
        host.clear();
        host.put(values);
    }

    private static void uploadOutOfCoreChildTile(
            DPGpuOutOfCore.PackedChildBlock packed,
            CUBuffer<LongBuffer> childPackedBase,
            CUBuffer<LongBuffer> childRowKeyBase,
            CUBuffer<IntBuffer> childRowKeyCount,
            CUBuffer<LongBuffer> childRowKeysAll,
            CUBuffer<LongBuffer> childLambdaKeyBase,
            CUBuffer<IntBuffer> childLambdaKeyCount,
            CUBuffer<LongBuffer> childLambdaKeysAll,
            CUBuffer<DoubleBuffer> childLowerPacked,
            CUBuffer<DoubleBuffer> childUpperPacked) {
        putLongs(childPackedBase, packed.childPackedBase);
        putLongs(childRowKeyBase, packed.childRowKeyBase);
        putInts(childRowKeyCount, packed.childRowKeyCount);
        putLongs(childRowKeysAll, packed.childRowKeysAll);
        putLongs(childLambdaKeyBase, packed.childLambdaKeyBase);
        putInts(childLambdaKeyCount, packed.childLambdaKeyCount);
        putLongs(childLambdaKeysAll, packed.childLambdaKeysAll);
        putDoubles(childLowerPacked, packed.lowerPacked);
        putDoubles(childUpperPacked, packed.upperPacked);
        childPackedBase.uploadAsync((long)packed.childPackedBase.length
                * Long.BYTES);
        childRowKeyBase.uploadAsync((long)packed.childRowKeyBase.length
                * Long.BYTES);
        childRowKeyCount.uploadAsync((long)packed.childRowKeyCount.length
                * Integer.BYTES);
        childRowKeysAll.uploadAsync((long)packed.childRowKeysAll.length
                * Long.BYTES);
        childLambdaKeyBase.uploadAsync((long)packed.childLambdaKeyBase.length
                * Long.BYTES);
        childLambdaKeyCount.uploadAsync((long)packed.childLambdaKeyCount.length
                * Integer.BYTES);
        childLambdaKeysAll.uploadAsync((long)packed.childLambdaKeysAll.length
                * Long.BYTES);
        childLowerPacked.uploadAsync(packed.packedStates * Double.BYTES);
        childUpperPacked.uploadAsync(packed.packedStates * Double.BYTES);
    }

    private static void uploadOutOfCoreLambdaTile(
            Request req, DPGpuOutOfCore.LambdaBox lambdaBox,
            CUBuffer<IntBuffer> lambdaIdxList,
            CUBuffer<DoubleBuffer> lambdaOnlyRigidTile,
            CUBuffer<DoubleBuffer> lambdaOnlyMinTile) {
        IntBuffer indices = lambdaIdxList.getHostBuffer();
        DoubleBuffer rigid = lambdaOnlyRigidTile.getHostBuffer();
        DoubleBuffer min = lambdaOnlyMinTile.getHostBuffer();
        indices.clear();
        rigid.clear();
        min.clear();
        for (int globalLambda : lambdaBox.lambdaIndices) {
            indices.put(globalLambda);
            rigid.put(req.lambdaOnlyRigid[globalLambda]);
            min.put(req.lambdaOnlyMin[globalLambda]);
        }
        long count = lambdaBox.lambdaIndices.length;
        lambdaIdxList.uploadAsync(count * Integer.BYTES);
        lambdaOnlyRigidTile.uploadAsync(count * Double.BYTES);
        lambdaOnlyMinTile.uploadAsync(count * Double.BYTES);
    }

    private static final class ChildSliceWorkspace {
        double[] lowerPacked = new double[0];
        double[] upperPacked = new double[0];
        CUdeviceptr lowerDevice;
        CUdeviceptr upperDevice;
        long deviceCapacity;

        void ensureHostCapacity(int states) {
            if (lowerPacked.length >= states) {
                return;
            }
            lowerPacked = new double[states];
            upperPacked = new double[states];
        }

        void upload(long states) {
            ensureDeviceCapacity(states);
            long bytes = states * (long)Double.BYTES;
            if (bytes > 0L) {
                cudaCheck(JCudaDriver.cuMemcpyHtoD(lowerDevice, Pointer.to(lowerPacked), bytes),
                        "cuMemcpyHtoD(child lower slice)");
                cudaCheck(JCudaDriver.cuMemcpyHtoD(upperDevice, Pointer.to(upperPacked), bytes),
                        "cuMemcpyHtoD(child upper slice)");
            }
        }

        void cleanup() {
            freeDevice();
        }

        private void ensureDeviceCapacity(long states) {
            if (lowerDevice != null && deviceCapacity >= states) {
                return;
            }
            freeDevice();
            long bytes = Math.max(1L, states) * (long)Double.BYTES;
            CUdeviceptr newLower = new CUdeviceptr();
            CUdeviceptr newUpper = new CUdeviceptr();
            cudaCheck(JCudaDriver.cuMemAlloc(newLower, bytes),
                    "cuMemAlloc(child lower workspace, " + bytes + " bytes)");
            try {
                cudaCheck(JCudaDriver.cuMemAlloc(newUpper, bytes),
                        "cuMemAlloc(child upper workspace, " + bytes + " bytes)");
            } catch (Throwable t) {
                JCudaDriver.cuMemFree(newLower);
                throw t;
            }
            lowerDevice = newLower;
            upperDevice = newUpper;
            deviceCapacity = states;
        }

        private void freeDevice() {
            if (upperDevice != null) {
                JCudaDriver.cuMemFree(upperDevice);
                upperDevice = null;
            }
            if (lowerDevice != null) {
                JCudaDriver.cuMemFree(lowerDevice);
                lowerDevice = null;
            }
            deviceCapacity = 0L;
        }
    }

    /** One pinned host/device slot for the hybrid streamed child. Two instances
     *  are alternated so the next block can gather and upload on a transfer stream
     *  while the current block's kernel runs on the compute stream. */
    private static final class HybridStreamWorkspace {
        private static final int MAX_DIRECT_DOUBLE_STATES = Integer.MAX_VALUE / Double.BYTES;

        final GpuStream uploadStream;
        CUBuffer<DoubleBuffer> lower;
        CUBuffer<DoubleBuffer> upper;

        HybridStreamWorkspace(GpuStream uploadStream) {
            this.uploadStream = uploadStream;
        }

        void ensureCapacity(int states) {
            if (states < 0 || states > MAX_DIRECT_DOUBLE_STATES) {
                throw new GpuMemoryExceededException("Streamed child block exceeds direct-buffer limits: "
                        + states + " states (max " + MAX_DIRECT_DOUBLE_STATES + ")");
            }
            lower = uploadStream.makeOrExpandDoubleBuffer(lower, Math.max(1, states));
            upper = uploadStream.makeOrExpandDoubleBuffer(upper, Math.max(1, states));
        }

        DoubleBuffer lowerHost() {
            return lower.getHostBuffer();
        }

        DoubleBuffer upperHost() {
            return upper.getHostBuffer();
        }

        Pointer lowerDevicePointer() {
            return lower.getDevicePointer();
        }

        Pointer upperDevicePointer() {
            return upper.getDevicePointer();
        }

        void uploadAndWait(long states) {
            long bytes = multiplyLong(states, Double.BYTES);
            lower.uploadAsync(bytes);
            upper.uploadAsync(bytes);
            uploadStream.waitForGpu();
        }

        void cleanup() {
            if (upper != null) {
                upper.cleanup();
                upper = null;
            }
            if (lower != null) {
                lower.cleanup();
                lower = null;
            }
        }
    }

    private static final class PreparedHybridBlock {
        final StreamedChildBlock block;
        final double gatherMs;
        final double uploadMs;

        PreparedHybridBlock(StreamedChildBlock block, double gatherMs, double uploadMs) {
            this.block = block;
            this.gatherMs = gatherMs;
            this.uploadMs = uploadMs;
        }
    }

    private static PreparedHybridBlock prepareHybridBlock(Request req, HybridPlan plan,
                                                           long rowStart, int rowCount,
                                                           HybridStreamWorkspace workspace) {
        // A shared CUDA context may be current on multiple worker threads, but every
        // thread must bind it before allocating or enqueueing a transfer.
        workspace.uploadStream.getContext().attachCurrentThread();
        long gatherStartNanos = System.nanoTime();
        StreamedChildBlock block = buildStreamedChildBlock(req, plan, rowStart, rowCount,
                workspace);
        double gatherMs = (System.nanoTime() - gatherStartNanos) / 1e6;
        long uploadStartNanos = System.nanoTime();
        workspace.uploadAndWait(block.packedStates);
        double uploadMs = (System.nanoTime() - uploadStartNanos) / 1e6;
        return new PreparedHybridBlock(block, gatherMs, uploadMs);
    }

    private static final class ChildSlice {
        final long unionStart;
        final int unionCount;
        final long outputCount;
        final long[] childPackedBase;
        final long[] childRowKeyBase;
        final int[] childRowKeyCount;
        final long[] childRowKeysAll;
        final long packedStates;
        final double[] lowerPacked;
        final double[] upperPacked;

        ChildSlice(long unionStart, int unionCount, long outputCount, int numChildren,
                long rowKeyTotal, long packedStates, ChildSliceWorkspace workspace) {
            this.unionStart = unionStart;
            this.unionCount = unionCount;
            this.outputCount = outputCount;
            this.childPackedBase = new long[numChildren];
            this.childRowKeyBase = new long[numChildren];
            this.childRowKeyCount = new int[numChildren];
            if (rowKeyTotal > Integer.MAX_VALUE || packedStates > Integer.MAX_VALUE) {
                throw new GpuMemoryExceededException("Child slice exceeds Java array limits: rowKeys="
                        + rowKeyTotal + ", packedStates=" + packedStates
                        + ". Lower branchdp.dp.gpu.childSliceMaxBytes.");
            }
            this.childRowKeysAll = new long[(int)rowKeyTotal];
            this.packedStates = packedStates;
            workspace.ensureHostCapacity((int)packedStates);
            this.lowerPacked = workspace.lowerPacked;
            this.upperPacked = workspace.upperPacked;
        }
    }

    private static ChildSlice buildChildSlice(Request req, long unionStart, int unionCount) {
        return buildChildSlice(req, unionStart, unionCount, new ChildSliceWorkspace());
    }

    private static ChildSlice buildChildSlice(Request req, long unionStart, int unionCount,
                                              ChildSliceWorkspace workspace) {
        ensureChildSliceIndexMaps(req);
        long[][] rowKeysByChild = new long[req.numChildren][];
        long rowKeyTotal = 0L;
        long packedStates = 0L;
        for (int c = 0; c < req.numChildren; c++) {
            long[] keys = collectUniqueRowKeys(req, c, unionStart, unionCount);
            rowKeysByChild[c] = keys;
            rowKeyTotal = addLong(rowKeyTotal, keys.length);
            packedStates = addLong(packedStates, multiplyLong(keys.length, req.childLambdaStates[c]));
        }
        long outputCount = multiplyLong(unionCount, req.parentFreeStateCount);
        ChildSlice slice = new ChildSlice(unionStart, unionCount, outputCount,
                req.numChildren, rowKeyTotal, packedStates, workspace);

        int rowKeyOff = 0;
        long packedBase = 0L;
        for (int c = 0; c < req.numChildren; c++) {
            long[] rowKeys = rowKeysByChild[c];
            slice.childRowKeyBase[c] = rowKeyOff;
            slice.childRowKeyCount[c] = rowKeys.length;
            System.arraycopy(rowKeys, 0, slice.childRowKeysAll, rowKeyOff, rowKeys.length);
            rowKeyOff += rowKeys.length;

            slice.childPackedBase[c] = packedBase;
            long lambdaStates = req.childLambdaStates[c];
            long childTotal = multiplyLong(rowKeys.length, lambdaStates);
            // Gather-pack this child's (rowKey x lambdaKey) block into the flat
            // lowerPacked/upperPacked arrays. Each element is an independent read
            // from the child's (already fully computed) DP table plus a write to
            // its own distinct slot, so this fans out safely across all CPU cores
            // instead of a single thread doing tens of millions of sequential
            // lower()/upper() calls -- for edges with huge child tables, this
            // single-threaded gather (not the GPU kernel or the PCIe transfer)
            // was the dominant per-slice cost.
            final long fLambdaStates = lambdaStates;
            final long fPackedBase = packedBase;
            final int fc = c;
            final long[] rowOriginalBases = new long[rowKeys.length];
            for (int ri = 0; ri < rowKeys.length; ri++) {
                rowOriginalBases[ri] = childOriginalRowBase(req, fc, rowKeys[ri]);
            }
            final long[] lambdaOriginalOffsets = req.childLambdaOriginalOffsets[fc];
            final DPTable childTable = req.childTables[fc];
            long gatherChunks = (childTotal + CHILD_GATHER_CHUNK_STATES - 1L)
                    / CHILD_GATHER_CHUNK_STATES;
            parallelGatherChunks(gatherChunks, chunkIndex -> {
                long start = chunkIndex * CHILD_GATHER_CHUNK_STATES;
                long end = Math.min(childTotal, start + CHILD_GATHER_CHUNK_STATES);
                long ri = start / fLambdaStates;
                int lamKey = (int)(start - ri * fLambdaStates);
                for (long linear = start; linear < end; linear++) {
                    long originalIdx = rowOriginalBases[(int)ri] + lambdaOriginalOffsets[lamKey];
                    int dst = (int)(fPackedBase + linear);
                    childTable.readPair(originalIdx, slice.lowerPacked, slice.upperPacked, dst);
                    lamKey++;
                    if (lamKey == fLambdaStates) {
                        lamKey = 0;
                        ri++;
                    }
                }
            });
            packedBase += childTotal;
        }
        return slice;
    }

    private static long[] collectUniqueRowKeys(Request req, int child, long unionStart, int unionCount) {
        if (req.childMTermCnt[child] == 0) {
            return new long[] { 0L };
        }
        long[] keys = new long[unionCount];
        int[] unionLocal = new int[req.unionMCounts.length];
        int[] mLocal = new int[req.mCounts.length];
        for (int i = 0; i < unionCount; i++) {
            Arrays.fill(mLocal, 0);
            decodeStateHost(unionStart + (long)i, req.unionMCounts, unionLocal);
            for (int u = 0; u < req.unionMSlots.length; u++) {
                mLocal[req.unionMSlots[u]] = unionLocal[u];
            }
            keys[i] = childRowIndex(req, child, mLocal);
        }
        Arrays.sort(keys);
        int unique = 0;
        long prev = Long.MIN_VALUE;
        for (long key : keys) {
            if (unique == 0 || key != prev) {
                keys[unique++] = key;
                prev = key;
            }
        }
        return Arrays.copyOf(keys, unique);
    }

    private static long childRowIndex(Request req, int child, int[] mLocal) {
        long row = 0L;
        int off = req.childMTermOff[child];
        int cnt = req.childMTermCnt[child];
        for (int t = 0; t < cnt; t++) {
            row += (long)mLocal[req.childMSrcAll[off + t]] * req.childMPackedStrideAll[off + t];
        }
        return row;
    }

    private static void ensureChildSliceIndexMaps(Request req) {
        if (req.childLambdaOriginalOffsets != null) {
            return;
        }
        synchronized (req) {
            if (req.childLambdaOriginalOffsets != null) {
                return;
            }
            long[][] offsets = new long[req.numChildren][];
            for (int c = 0; c < req.numChildren; c++) {
                long lambdaStates = req.childLambdaStates[c];
                if (lambdaStates > Integer.MAX_VALUE) {
                    throw new GpuMemoryExceededException("Child lambda projection exceeds Java index-map limits: "
                            + lambdaStates + " states for child " + c);
                }
                long[] childOffsets = new long[(int)lambdaStates];
                final int fc = c;
                long offsetChunks = (lambdaStates + CHILD_GATHER_CHUNK_STATES - 1L)
                        / CHILD_GATHER_CHUNK_STATES;
                parallelGatherChunks(offsetChunks, chunkIndex -> {
                    long start = chunkIndex * CHILD_GATHER_CHUNK_STATES;
                    long end = Math.min(lambdaStates, start + CHILD_GATHER_CHUNK_STATES);
                    for (long lambdaKey = start; lambdaKey < end; lambdaKey++) {
                        childOffsets[(int)lambdaKey] = childOriginalLambdaOffset(req, fc, lambdaKey);
                    }
                });
                offsets[c] = childOffsets;
            }
            req.childLambdaOriginalOffsets = offsets;
        }
    }

    private static long childOriginalRowBase(Request req, int child, long rowKey) {
        long idx = 0L;
        int mOff = req.childMTermOff[child];
        int mCnt = req.childMTermCnt[child];
        for (int t = 0; t < mCnt; t++) {
            long stride = req.childMPackedStrideAll[mOff + t];
            int digit = (int)((rowKey / stride) % (long)req.childMCountsAll[mOff + t]);
            idx += (long)digit * req.childMStrideAll[mOff + t];
        }
        return idx;
    }

    private static long childOriginalLambdaOffset(Request req, int child, long lambdaKey) {
        long idx = 0L;
        int lOff = req.childLTermOff[child];
        int lCnt = req.childLTermCnt[child];
        for (int t = 0; t < lCnt; t++) {
            long stride = req.childLPackedStrideAll[lOff + t];
            int digit = (int)((lambdaKey / stride) % (long)req.childLCountsAll[lOff + t]);
            idx += (long)digit * req.childLStrideAll[lOff + t];
        }
        return idx;
    }

    static final class StreamedChildBlock {
        final long rowStart;
        final int rowCount;
        final long packedStates;
        final double[] lowerPacked;
        final double[] upperPacked;
        final DoubleBuffer lowerDirect;
        final DoubleBuffer upperDirect;

        StreamedChildBlock(long rowStart, int rowCount, long packedStates,
                           ChildSliceWorkspace workspace) {
            if (packedStates > Integer.MAX_VALUE) {
                throw new GpuMemoryExceededException("Streamed child block exceeds Java array limits: "
                        + packedStates + " states");
            }
            this.rowStart = rowStart;
            this.rowCount = rowCount;
            this.packedStates = packedStates;
            workspace.ensureHostCapacity((int)packedStates);
            this.lowerPacked = workspace.lowerPacked;
            this.upperPacked = workspace.upperPacked;
            this.lowerDirect = null;
            this.upperDirect = null;
        }

        StreamedChildBlock(long rowStart, int rowCount, long packedStates,
                           HybridStreamWorkspace workspace) {
            if (packedStates > Integer.MAX_VALUE) {
                throw new GpuMemoryExceededException("Streamed child block exceeds Java index limits: "
                        + packedStates + " states");
            }
            this.rowStart = rowStart;
            this.rowCount = rowCount;
            this.packedStates = packedStates;
            workspace.ensureCapacity((int)packedStates);
            this.lowerPacked = null;
            this.upperPacked = null;
            this.lowerDirect = workspace.lowerHost();
            this.upperDirect = workspace.upperHost();
        }

        void readPair(DPTable table, long original, int destination) {
            if (lowerPacked != null) {
                table.readPair(original, lowerPacked, upperPacked, destination);
            } else {
                table.readPair(original, lowerDirect, upperDirect, destination);
            }
        }
    }

    static StreamedChildBlock buildStreamedChildBlock(Request req, HybridPlan plan,
                                                       long rowStart, int rowCount) {
        return buildStreamedChildBlock(req, plan, rowStart, rowCount,
                new ChildSliceWorkspace());
    }

    private static StreamedChildBlock buildStreamedChildBlock(Request req, HybridPlan plan,
                                                               long rowStart, int rowCount,
                                                               ChildSliceWorkspace workspace) {
        ensureChildSliceIndexMaps(req);
        int child = plan.streamedChild;
        long lambdaStates = req.childLambdaStates[child];
        long packedStates = multiplyLong(rowCount, lambdaStates);
        StreamedChildBlock block = new StreamedChildBlock(rowStart, rowCount, packedStates,
                workspace);
        return fillStreamedChildBlock(req, plan, block);
    }

    private static StreamedChildBlock buildStreamedChildBlock(Request req, HybridPlan plan,
                                                               long rowStart, int rowCount,
                                                               HybridStreamWorkspace workspace) {
        ensureChildSliceIndexMaps(req);
        int child = plan.streamedChild;
        long lambdaStates = req.childLambdaStates[child];
        long packedStates = multiplyLong(rowCount, lambdaStates);
        StreamedChildBlock block = new StreamedChildBlock(rowStart, rowCount, packedStates,
                workspace);
        return fillStreamedChildBlock(req, plan, block);
    }

    private static StreamedChildBlock fillStreamedChildBlock(Request req, HybridPlan plan,
                                                              StreamedChildBlock block) {
        int child = plan.streamedChild;
        long rowStart = block.rowStart;
        int rowCount = block.rowCount;
        long lambdaStates = req.childLambdaStates[child];
        long packedStates = block.packedStates;

        long[] rowOriginalBases = new long[rowCount];
        for (int row = 0; row < rowCount; row++) {
            rowOriginalBases[row] = childOriginalRowBase(req, child, rowStart + row);
        }
        long[] lambdaOriginalOffsets = req.childLambdaOriginalOffsets[child];
        DPTable childTable = req.childTables[child];
        long gatherChunks = (packedStates + CHILD_GATHER_CHUNK_STATES - 1L)
                / CHILD_GATHER_CHUNK_STATES;
        parallelGatherChunks(gatherChunks, chunkIndex -> {
            long start = chunkIndex * CHILD_GATHER_CHUNK_STATES;
            long end = Math.min(packedStates, start + CHILD_GATHER_CHUNK_STATES);
            long row = start / lambdaStates;
            int lambdaKey = (int)(start - row * lambdaStates);
            for (long linear = start; linear < end; linear++) {
                long original = rowOriginalBases[(int)row] + lambdaOriginalOffsets[lambdaKey];
                block.readPair(childTable, original, (int)linear);
                lambdaKey++;
                if (lambdaKey == lambdaStates) {
                    lambdaKey = 0;
                    row++;
                }
            }
        });
        return block;
    }

    static final class StreamedChildMIdxEnumerator {
        private static final long OFFSET_PRECOMPUTE_LIMIT = 1_000_000L;

        private final Request req;
        private final int child;
        private final int rowCount;
        private final long otherStateCount;
        private final long[] parentStrides;
        private final int[] otherSlots;
        private final int[] otherCounts;
        private final long[] otherStrides;
        private final long[] rowParentBases;
        private final long[] otherOffsets;
        private final int[] otherLocal;
        private long rowOffset;
        private long otherOffset;

        StreamedChildMIdxEnumerator(Request req, int child, long rowStart, int rowCount) {
            this.req = req;
            this.child = child;
            this.rowCount = rowCount;
            this.parentStrides = mixedRadixStridesHost(req.mCounts);

            boolean[] streamedSlots = new boolean[req.mCounts.length];
            int childMOff = req.childMTermOff[child];
            for (int t = 0; t < req.childMTermCnt[child]; t++) {
                streamedSlots[req.childMSrcAll[childMOff + t]] = true;
            }
            int otherCount = 0;
            for (boolean streamed : streamedSlots) {
                if (!streamed) {
                    otherCount++;
                }
            }
            this.otherSlots = new int[otherCount];
            this.otherCounts = new int[otherCount];
            this.otherStrides = new long[otherCount];
            long otherStates = 1L;
            int off = 0;
            for (int slot = 0; slot < req.mCounts.length; slot++) {
                if (!streamedSlots[slot]) {
                    otherSlots[off] = slot;
                    otherCounts[off] = req.mCounts[slot];
                    otherStrides[off] = parentStrides[slot];
                    otherStates = multiplyLong(otherStates, req.mCounts[slot]);
                    off++;
                }
            }
            this.otherStateCount = otherStates;
            this.otherLocal = new int[otherCount];

            this.rowParentBases = new long[rowCount];
            for (int row = 0; row < rowCount; row++) {
                rowParentBases[row] = parentIndexForChildRow(rowStart + row);
            }
            if (otherStateCount <= OFFSET_PRECOMPUTE_LIMIT) {
                this.otherOffsets = new long[(int)otherStateCount];
                for (int i = 0; i < otherOffsets.length; i++) {
                    otherOffsets[i] = decodeOtherOffset(i);
                }
            } else {
                this.otherOffsets = null;
            }
        }

        long outputCount() {
            return multiplyLong(rowCount, otherStateCount);
        }

        int fill(long[] out, int count) {
            int written = 0;
            while (written < count && rowOffset < rowCount) {
                long other = otherOffsets != null
                        ? otherOffsets[(int)otherOffset]
                        : decodeOtherOffset(otherOffset);
                out[written++] = rowParentBases[(int)rowOffset] + other;
                otherOffset++;
                if (otherOffset == otherStateCount) {
                    otherOffset = 0L;
                    rowOffset++;
                }
            }
            return written;
        }

        private long parentIndexForChildRow(long rowKey) {
            long parent = 0L;
            int off = req.childMTermOff[child];
            for (int t = 0; t < req.childMTermCnt[child]; t++) {
                long packedStride = req.childMPackedStrideAll[off + t];
                int digit = (int)((rowKey / packedStride)
                        % (long)req.childMCountsAll[off + t]);
                int parentSlot = req.childMSrcAll[off + t];
                parent += (long)digit * parentStrides[parentSlot];
            }
            return parent;
        }

        private long decodeOtherOffset(long state) {
            decodeStateHost(state, otherCounts, otherLocal);
            long offset = 0L;
            for (int i = 0; i < otherSlots.length; i++) {
                offset += (long)otherLocal[i] * otherStrides[i];
            }
            return offset;
        }
    }

    private static long[] mixedRadixStridesHost(int[] counts) {
        long[] strides = new long[counts.length];
        long stride = 1L;
        for (int i = counts.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride = multiplyLong(stride, counts[i]);
        }
        return strides;
    }

    private static final class MIdxTileEnumerator {
        private static final long FREE_OFFSET_PRECOMPUTE_LIMIT = 1_000_000L;

        private final Request req;
        private final long unionStart;
        private final long unionCount;
        private final long[] freeOffsets;
        private final int[] unionLocal;
        private final int[] freeLocal;
        private long unionOffset = 0L;
        private long freeOffset = 0L;
        private long cachedUnionOffset = -1L;
        private long cachedUnionBase = 0L;

        MIdxTileEnumerator(Request req, long unionStart, long unionCount) {
            this.req = req;
            this.unionStart = unionStart;
            this.unionCount = unionCount;
            this.unionLocal = new int[req.unionMCounts.length];
            this.freeLocal = new int[req.freeMCounts.length];
            this.freeOffsets = buildFreeOffsets(req);
        }

        int fill(long[] out, int max) {
            int n = 0;
            while (n < max && unionOffset < unionCount) {
                long unionBase = unionBase(unionOffset);
                while (n < max && freeOffset < req.parentFreeStateCount) {
                    out[n++] = unionBase + freeContribution(freeOffset);
                    freeOffset++;
                }
                if (freeOffset >= req.parentFreeStateCount) {
                    unionOffset++;
                    freeOffset = 0L;
                }
            }
            return n;
        }

        private long unionBase(long offset) {
            if (cachedUnionOffset == offset) {
                return cachedUnionBase;
            }
            decodeStateHost(unionStart + offset, req.unionMCounts, unionLocal);
            long base = 0L;
            for (int u = 0; u < req.unionMParentStrides.length; u++) {
                base += (long)unionLocal[u] * req.unionMParentStrides[u];
            }
            cachedUnionOffset = offset;
            cachedUnionBase = base;
            return base;
        }

        private long freeContribution(long key) {
            if (freeOffsets != null) {
                return freeOffsets[(int)key];
            }
            decodeStateHost(key, req.freeMCounts, freeLocal);
            long idx = 0L;
            for (int f = 0; f < req.freeMParentStrides.length; f++) {
                idx += (long)freeLocal[f] * req.freeMParentStrides[f];
            }
            return idx;
        }

        private static long[] buildFreeOffsets(Request req) {
            if (req.parentFreeStateCount > FREE_OFFSET_PRECOMPUTE_LIMIT
                    || req.parentFreeStateCount > Integer.MAX_VALUE) {
                return null;
            }
            long[] offsets = new long[(int)req.parentFreeStateCount];
            int[] local = new int[req.freeMCounts.length];
            for (int i = 0; i < offsets.length; i++) {
                decodeStateHost(i, req.freeMCounts, local);
                long idx = 0L;
                for (int f = 0; f < req.freeMParentStrides.length; f++) {
                    idx += (long)local[f] * req.freeMParentStrides[f];
                }
                offsets[i] = idx;
            }
            return offsets;
        }
    }

    private static void decodeStateHost(long idx, int[] counts, int[] out) {
        for (int i = counts.length - 1; i >= 0; i--) {
            int n = counts[i];
            out[i] = (int)(idx % (long)n);
            idx /= (long)n;
        }
    }

    private static long queryUsableVramInCurrentContext() {
        long[] freeMem = {0L};
        long[] totalMem = {0L};
        if (JCudaDriver.cuMemGetInfo(freeMem, totalMem) != CUresult.CUDA_SUCCESS) {
            return -1L;
        }
        int headroomPct = Math.max(1, Math.min(100,
                getConfigInteger(DP_GPU_VRAM_HEADROOM_PERCENT, DEFAULT_DP_GPU_VRAM_HEADROOM_PERCENT)));
        return (long)((double)freeMem[0]*headroomPct/100.0);
    }

    private static boolean canUseChildSlicing(Request req) {
        return req.childSlicing
                && req.childTables != null
                && req.childTables.length == req.numChildren
                && req.unionMCounts != null
                && req.freeMCounts != null
                && req.childMPackedStrideAll != null
                && req.childLPackedStrideAll != null
                && req.childLambdaStates != null
                && req.unionStateCount > 0L
                && req.parentFreeStateCount > 0L;
    }

    static HybridPlan chooseHybridPlan(Request req, long usableBytes, int gpuCount) {
        if (req.numChildren < 2 || !canUseChildSlicing(req)
                || usableBytes <= 0L) {
            return null;
        }

        HybridPlan best = null;
        for (int streamed = 0; streamed < req.numChildren; streamed++) {
            long streamedRows = req.childMRowStates[streamed];
            long streamedLambdaStates = req.childLambdaStates[streamed];
            if (streamedRows <= 0L || streamedLambdaStates <= 0L
                    || req.mStateCount % streamedRows != 0L) {
                continue;
            }

            long[] residentBase = new long[req.numChildren];
            long residentStates = 0L;
            for (int c = 0; c < req.numChildren; c++) {
                if (c == streamed) {
                    continue;
                }
                residentBase[c] = residentStates;
                residentStates = addLong(residentStates, childTableStates(req, c));
            }

            long fixedBytes = estimateDeviceBytesForShape(
                    req.mCounts.length,
                    req.lambdaCounts.length,
                    req.totalLambdaStates,
                    req.lmRigid.length,
                    req.lmLamSlots.length,
                    req.childMSrcAll.length,
                    req.childLSrcAll.length,
                    req.numChildren,
                    residentStates,
                    req.mStateChunk);
            fixedBytes = addTripleDeviceBytes(fixedBytes, req);
            fixedBytes = addBytes(fixedBytes, req.mStateChunk, Long.BYTES);
            fixedBytes = addBytes(fixedBytes, req.numChildren, Long.BYTES);
            fixedBytes = addBytes(fixedBytes, req.numChildren, Long.BYTES);
            if (fixedBytes >= usableBytes) {
                continue;
            }

            long rowBytes = multiplyLong(2L * (long)Double.BYTES, streamedLambdaStates);
            // The hybrid executor alternates two pinned host/device slots. Budget
            // each slot from half the VRAM left after resident/fixed allocations.
            long streamBudget = Math.min(Math.max(1L, req.childSliceMaxBytes),
                    (usableBytes - fixedBytes) / 2L);
            if (rowBytes <= 0L || rowBytes > streamBudget) {
                continue;
            }
            long rowsByMemory = streamBudget / rowBytes;
            long directDoubleStates = Integer.MAX_VALUE / (long)Double.BYTES;
            long rowsByJavaArrays = directDoubleStates / streamedLambdaStates;
            long rowsPerBlock = Math.min(streamedRows,
                    Math.min(rowsByMemory, rowsByJavaArrays));
            if (rowsPerBlock <= 0L) {
                continue;
            }

            long streamedTableBytes = multiplyLong(childTableStates(req, streamed),
                    2L * (long)Double.BYTES);
            long replicatedResidentBytes = multiplyLong(residentStates,
                    2L * (long)Double.BYTES);
            replicatedResidentBytes = multiplyLong(replicatedResidentBytes,
                    Math.max(1, gpuCount));
            long aggregateTraffic = addLong(streamedTableBytes, replicatedResidentBytes);
            HybridPlan candidate = new HybridPlan(streamed, streamedRows, rowsPerBlock,
                    req.mStateCount / streamedRows, residentStates, aggregateTraffic,
                    residentBase);
            if (best == null
                    || candidate.estimatedAggregateTrafficBytes < best.estimatedAggregateTrafficBytes
                    || (candidate.estimatedAggregateTrafficBytes == best.estimatedAggregateTrafficBytes
                    && candidate.streamedRowsPerBlock > best.streamedRowsPerBlock)) {
                best = candidate;
            }
        }
        return best;
    }

    private static long childTableStates(Request req, int child) {
        long start = req.childTableBase[child];
        long end = child + 1 < req.numChildren
                ? req.childTableBase[child + 1]
                : req.childTableTotal;
        if (start < 0L || end < start) {
            throw new IllegalArgumentException("Invalid child table range for child " + child
                    + ": [" + start + "," + end + ")");
        }
        return end - start;
    }

    static boolean defaultChildSlicingEnabled() {
        return getConfigBoolean(DP_GPU_CHILD_SLICING, DEFAULT_DP_GPU_CHILD_SLICING);
    }

    static boolean forceChildSlicingEnabled() {
        return getConfigBoolean(DP_GPU_CHILD_SLICING_FORCE, false);
    }

    static long configuredChildSliceMaxBytes() {
        return Math.max(1L, getConfigLong(DP_GPU_CHILD_SLICE_MAX_BYTES,
                DEFAULT_DP_GPU_CHILD_SLICE_MAX_BYTES));
    }

    static long configuredOutOfCoreOutputWorkspaceMaxBytes() {
        return Math.max(1L, getConfigLong(
                DP_GPU_OUT_OF_CORE_OUTPUT_WORKSPACE_MAX_BYTES,
                DEFAULT_DP_GPU_OUT_OF_CORE_OUTPUT_WORKSPACE_MAX_BYTES));
    }

    private static long resolveChildSliceMaxBytes(Request req, long usable) {
        long configured = Math.max(1L, req.childSliceMaxBytes);
        if (usable < 0L) {
            return configured;
        }
        long fixed = estimateDeviceBytesForShape(
                req.mCounts.length,
                req.lambdaCounts.length,
                req.totalLambdaStates,
                req.lmRigid.length,
                req.lmLamSlots.length,
                req.childMSrcAll.length,
                req.childLSrcAll.length,
                req.childTableBase.length,
                0L,
                req.mStateChunk);
        fixed = addTripleDeviceBytes(fixed, req);
        fixed = addBytes(fixed, req.mStateChunk, Long.BYTES); // mIdxList
        long available = usable > fixed ? usable - fixed : 1L;
        return Math.max(1L, Math.min(configured, available));
    }

    private static long chooseUnionStatesPerSlice(Request req, long childSliceMaxBytes) {
        long lambdaStatesPerUnion = 0L;
        for (long childLambda : req.childLambdaStates) {
            lambdaStatesPerUnion = addLong(lambdaStatesPerUnion, childLambda);
        }
        long bytesPerUnion = multiplyLong(2L * (long)Double.BYTES, Math.max(1L, lambdaStatesPerUnion));
        long count = Math.max(1L, childSliceMaxBytes / Math.max(1L, bytesPerUnion));
        count = Math.min(count, req.unionStateCount);
        count = Math.min(count, (long)Integer.MAX_VALUE);
        return Math.max(1L, count);
    }

    static long estimateSlicedDeviceBytes(Request req) {
        if (!canUseChildSlicing(req)) {
            return estimateDeviceBytes(req);
        }
        long unionStatesPerSlice = chooseUnionStatesPerSlice(req,
                Math.max(1L, req.childSliceMaxBytes));
        long childTableStates = 0L;
        for (long childLambda : req.childLambdaStates) {
            childTableStates = addLong(childTableStates, multiplyLong(unionStatesPerSlice, childLambda));
        }
        return estimateSlicedDeviceBytes(req, childTableStates,
                multiplyLong(unionStatesPerSlice, req.numChildren));
    }

    /**
     * Device bytes for the smallest slice representable by the current row-only
     * fallback: one row from every child, with each row's complete projected lambda
     * domain. If this exceeds the per-device budget, reducing the union/M slice width
     * cannot help; the edge needs lambda tiling and partial log-sum-exp merging.
     */
    static long estimateMinimumSlicedDeviceBytes(Request req) {
        if (!canUseChildSlicing(req)) {
            return estimateDeviceBytes(req);
        }
        long childTableStates = 0L;
        for (long childLambda : req.childLambdaStates) {
            childTableStates = addLong(childTableStates, childLambda);
        }
        return estimateSlicedDeviceBytes(req, childTableStates, req.numChildren);
    }

    private static long estimateSlicedDeviceBytes(Request req, long childTableStates, long rowKeyCount) {
        long bytes = estimateDeviceBytesForShape(
                req.mCounts.length,
                req.lambdaCounts.length,
                req.totalLambdaStates,
                req.lmRigid.length,
                req.lmLamSlots.length,
                req.childMSrcAll.length,
                req.childLSrcAll.length,
                req.childTableBase.length,
                childTableStates,
                req.mStateChunk);
        bytes = addTripleDeviceBytes(bytes, req);
        bytes = addBytes(bytes, req.mStateChunk, Long.BYTES); // mIdxList
        bytes = addBytes(bytes, rowKeyCount, Long.BYTES);
        bytes = addBytes(bytes, req.numChildren, Long.BYTES); // childPackedBase
        bytes = addBytes(bytes, req.numChildren, Long.BYTES); // childRowKeyBase
        bytes = addBytes(bytes, req.numChildren, Integer.BYTES); // childRowKeyCount
        bytes = addBytes(bytes, req.numChildren, Long.BYTES); // childLambdaStates
        return bytes;
    }

    static long estimateMinimumChildSliceBytes(Request req) {
        if (!canUseChildSlicing(req)) {
            return req.childTableTotal == 0L
                    ? 0L
                    : multiplyLong(req.childTableTotal, 2L * (long)Double.BYTES);
        }
        long states = 0L;
        for (long childLambda : req.childLambdaStates) {
            states = addLong(states, childLambda);
        }
        return multiplyLong(states, 2L * (long)Double.BYTES);
    }

    private static long estimateSlicedDeviceBytesForSlice(Request req, ChildSlice slice) {
        long bytes = estimateDeviceBytesForShape(
                req.mCounts.length,
                req.lambdaCounts.length,
                req.totalLambdaStates,
                req.lmRigid.length,
                req.lmLamSlots.length,
                req.childMSrcAll.length,
                req.childLSrcAll.length,
                req.childTableBase.length,
                slice.packedStates,
                req.mStateChunk);
        bytes = addTripleDeviceBytes(bytes, req);
        bytes = addBytes(bytes, req.mStateChunk, Long.BYTES);
        bytes = addBytes(bytes, slice.childRowKeysAll.length, Long.BYTES);
        bytes = addBytes(bytes, req.numChildren, Long.BYTES);
        bytes = addBytes(bytes, req.numChildren, Long.BYTES);
        bytes = addBytes(bytes, req.numChildren, Integer.BYTES);
        bytes = addBytes(bytes, req.numChildren, Long.BYTES);
        return bytes;
    }

    private static long addLong(long a, long b) {
        if (a > Long.MAX_VALUE - b) {
            return Long.MAX_VALUE;
        }
        return a + b;
    }

    private static long multiplyLong(long a, long b) {
        if (a == 0L || b == 0L) {
            return 0L;
        }
        if (a > Long.MAX_VALUE / b) {
            return Long.MAX_VALUE;
        }
        return a * b;
    }

    private static int resolveBlockThreads(Request req, GpuStream stream) {
        int requested = Math.max(1, req.blockThreads);
        int capped = Math.min(requested, stream.getContext().getGpu().getMaxBlockThreads());
        int powerOfTwo = Integer.highestOneBit(capped);
        return Math.max(1, powerOfTwo);
    }

    private static final class GpuTripleBuffers {
        final CUBuffer<IntBuffer> slots;
        final CUBuffer<LongBuffer> strides;
        final CUBuffer<LongBuffer> offsets;
        final CUdeviceptr rigid;
        final CUdeviceptr min;

        GpuTripleBuffers(CUBuffer<IntBuffer> slots,
                         CUBuffer<LongBuffer> strides,
                         CUBuffer<LongBuffer> offsets,
                         CUdeviceptr rigid, CUdeviceptr min) {
            this.slots = slots;
            this.strides = strides;
            this.offsets = offsets;
            this.rigid = rigid;
            this.min = min;
        }
    }

    private static GpuTripleBuffers uploadTripleBuffers(
            Request req, GpuStream stream,
            List<CUBuffer<?>> buffers, List<CUdeviceptr> rawBufs) {
        return new GpuTripleBuffers(
                uploadInts(stream, req.tripleSlots, buffers),
                uploadLongs(stream, req.tripleStrides, buffers),
                uploadLongs(stream, req.tripleOffsets, buffers),
                uploadDoublesBig(req.tripleRigid, rawBufs),
                uploadDoublesBig(req.tripleMin, rawBufs));
    }

    private static CUBuffer<IntBuffer> uploadInts(GpuStream stream, int[] values, List<CUBuffer<?>> buffers) {
        CUBuffer<IntBuffer> buf = stream.makeIntBuffer(Math.max(1, values.length));
        IntBuffer host = buf.getHostBuffer();
        host.clear();
        if (values.length > 0) {
            host.put(values);
        }
        while (host.position() < host.capacity()) {
            host.put(0);
        }
        buf.uploadAsync();
        buffers.add(buf);
        return buf;
    }

    private static CUBuffer<LongBuffer> uploadLongs(GpuStream stream, long[] values, List<CUBuffer<?>> buffers) {
        CUBuffer<LongBuffer> buf = stream.makeLongBuffer(Math.max(1, values.length));
        LongBuffer host = buf.getHostBuffer();
        host.clear();
        if (values.length > 0) {
            host.put(values);
        }
        while (host.position() < host.capacity()) {
            host.put(0L);
        }
        buf.uploadAsync();
        buffers.add(buf);
        return buf;
    }

    private static CUBuffer<DoubleBuffer> makeDoubles(GpuStream stream, int size, List<CUBuffer<?>> buffers) {
        CUBuffer<DoubleBuffer> buf = stream.makeDoubleBuffer(Math.max(1, size));
        buffers.add(buf);
        return buf;
    }

    /**
     * Upload a double[] straight into raw device memory, bypassing the 2 GiB cap of a
     * single java.nio direct buffer. cuMemAlloc takes a long byte count and a single
     * cuMemcpyHtoD from the Java array copies the whole (possibly > 2 GiB) buffer.
     * The caller must cuMemFree every returned pointer (tracked in rawBufs).
     */
    private static CUdeviceptr uploadDoublesBig(double[] values, List<CUdeviceptr> rawBufs) {
        long count = Math.max(1L, values.length);
        long bytes = count*(long)Double.BYTES;
        CUdeviceptr dptr = new CUdeviceptr();
        cudaCheck(JCudaDriver.cuMemAlloc(dptr, bytes), "cuMemAlloc(" + bytes + " bytes)");
        rawBufs.add(dptr);
        if (values.length > 0) {
            cudaCheck(JCudaDriver.cuMemcpyHtoD(dptr, Pointer.to(values),
                    (long)values.length*(long)Double.BYTES), "cuMemcpyHtoD");
        } else {
            cudaCheck(JCudaDriver.cuMemsetD8(dptr, (byte)0, bytes), "cuMemsetD8");
        }
        return dptr;
    }

    private static CUdeviceptr uploadResidentChildTables(Request req, HybridPlan plan,
                                                          boolean lower,
                                                          List<CUdeviceptr> rawBufs) {
        long count = Math.max(1L, plan.residentTableStates);
        long bytes = count * (long)Double.BYTES;
        CUdeviceptr dptr = new CUdeviceptr();
        cudaCheck(JCudaDriver.cuMemAlloc(dptr, bytes),
                "cuMemAlloc(resident child tables, " + bytes + " bytes)");
        rawBufs.add(dptr);
        if (plan.residentTableStates == 0L) {
            cudaCheck(JCudaDriver.cuMemsetD8(dptr, (byte)0, bytes),
                    "cuMemsetD8(resident child tables)");
            return dptr;
        }

        for (int c = 0; c < req.numChildren; c++) {
            if (c == plan.streamedChild) {
                continue;
            }
            double[][] chunks = lower
                    ? req.childTables[c].lowerChunks()
                    : req.childTables[c].upperChunks();
            long childOffset = 0L;
            for (double[] chunk : chunks) {
                if (chunk.length > 0) {
                    long deviceOffset = plan.residentTableBase[c] + childOffset;
                    CUdeviceptr slot = dptr.withByteOffset(deviceOffset * (long)Double.BYTES);
                    cudaCheck(JCudaDriver.cuMemcpyHtoD(slot, Pointer.to(chunk),
                                    (long)chunk.length * (long)Double.BYTES),
                            "cuMemcpyHtoD(resident child " + c + ")");
                }
                childOffset += chunk.length;
            }
            if (childOffset != req.childTables[c].size()) {
                throw new IllegalStateException("Resident child chunk size mismatch for child " + c
                        + ": chunks=" + childOffset + ", table=" + req.childTables[c].size());
            }
        }
        return dptr;
    }

    /**
     * Allocate one device buffer for all concatenated child tables and copy each
     * chunk straight into its slot at {@code offsets[i]} (element offset). A dense
     * child contributes exactly one chunk; a sharded child contributes one chunk
     * per shard, with consecutive shards given consecutive offsets so they land
     * contiguously (i.e. the device-side layout is indistinguishable from a
     * hypothetical single dense upload). The chunks are NEVER concatenated on the
     * Java heap, so their combined length may exceed Integer.MAX_VALUE; only
     * device memory (long byte count via cuMemAlloc) bounds it. Device layout is
     * byte-identical to the old single concatenated upload, so the kernel and
     * childTableBase semantics are unchanged.
     */
    private static CUdeviceptr uploadChildTables(double[][] chunks, long[] offsets, long total,
            List<CUdeviceptr> rawBufs) {
        long count = Math.max(1L, total);
        long bytes = count*(long)Double.BYTES;
        CUdeviceptr dptr = new CUdeviceptr();
        cudaCheck(JCudaDriver.cuMemAlloc(dptr, bytes), "cuMemAlloc(" + bytes + " bytes)");
        rawBufs.add(dptr);
        for (int i = 0; i < chunks.length; i++) {
            double[] chunk = chunks[i];
            if (chunk == null || chunk.length == 0) {
                continue;
            }
            CUdeviceptr slot = dptr.withByteOffset(offsets[i]*(long)Double.BYTES);
            cudaCheck(JCudaDriver.cuMemcpyHtoD(slot, Pointer.to(chunk),
                    (long)chunk.length*(long)Double.BYTES), "cuMemcpyHtoD(chunk " + i + ")");
        }
        return dptr;
    }

    private static void cudaCheck(int result, String what) {
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("CUDA " + what + " failed: " + CUresult.stringFor(result));
        }
    }

    private static void edgeFallback(Throwable t) {
        System.out.println(BranchDpConfig.getBackendLogPrefix() + " GPU DP edge fallback ("
                + t.getClass().getSimpleName() + ": " + t.getMessage()
                + "); this edge on Java DP path");
        if (getConfigBoolean("branchdp.dp.gpu.trace", false)) {
            t.printStackTrace(System.err);
        }
    }

    static long estimateDeviceBytes(Request req) {
        long bytes = estimateDeviceBytesForShape(
                req.mCounts.length,
                req.lambdaCounts.length,
                req.totalLambdaStates,
                req.lmRigid.length,
                req.lmLamSlots.length,
                req.childMSrcAll.length,
                req.childLSrcAll.length,
                req.childTableBase.length,
                req.childTableTotal,
                req.mStateChunk);
        return addTripleDeviceBytes(bytes, req);
    }

    private static long addTripleDeviceBytes(long bytes, Request req) {
        bytes = addBytes(bytes, req.tripleSlots.length, Integer.BYTES);
        bytes = addBytes(bytes, req.tripleStrides.length, Long.BYTES);
        bytes = addBytes(bytes, req.tripleOffsets.length, Long.BYTES);
        bytes = addBytes(bytes, req.tripleRigid.length, Double.BYTES);
        bytes = addBytes(bytes, req.tripleMin.length, Double.BYTES);
        return bytes;
    }

    static long estimateDeviceBytesForShape(int mCountLen, int lambdaCountLen,
                                            long totalLambdaStates, long lmTermCount,
                                            long lmPairCount, long childMTermTotal,
                                            long childLTermTotal, long numChildren,
                                            long childTableTotal, long mStateChunk) {
        long bytes = 0L;
        bytes = addBytes(bytes, mCountLen, Integer.BYTES);
        bytes = addBytes(bytes, lambdaCountLen, Integer.BYTES);
        bytes = addBytes(bytes, totalLambdaStates, Double.BYTES);
        bytes = addBytes(bytes, totalLambdaStates, Double.BYTES);
        bytes = addBytes(bytes, lmTermCount, Double.BYTES);
        bytes = addBytes(bytes, lmTermCount, Double.BYTES);
        bytes = addBytes(bytes, lmPairCount, Integer.BYTES);
        bytes = addBytes(bytes, lmPairCount, Integer.BYTES);
        bytes = addBytes(bytes, lmPairCount, Integer.BYTES);
        bytes = addBytes(bytes, lmPairCount, Long.BYTES);
        bytes = addBytes(bytes, childMTermTotal, Integer.BYTES);
        bytes = addBytes(bytes, childMTermTotal, Long.BYTES);
        bytes = addBytes(bytes, numChildren, Integer.BYTES);
        bytes = addBytes(bytes, numChildren, Integer.BYTES);
        bytes = addBytes(bytes, childLTermTotal, Integer.BYTES);
        bytes = addBytes(bytes, childLTermTotal, Long.BYTES);
        bytes = addBytes(bytes, numChildren, Integer.BYTES);
        bytes = addBytes(bytes, numChildren, Integer.BYTES);
        bytes = addBytes(bytes, numChildren, Long.BYTES);
        bytes = addBytes(bytes, childTableTotal, Double.BYTES);
        bytes = addBytes(bytes, childTableTotal, Double.BYTES);
        bytes = addBytes(bytes, mStateChunk, Double.BYTES);
        bytes = addBytes(bytes, mStateChunk, Double.BYTES);
        return bytes;
    }

    private static long addBytes(long current, int count, int elemBytes) {
        return addBytes(current, (long)count, elemBytes);
    }

    private static long addBytes(long current, long count, int elemBytes) {
        long c = Math.max(1L, count);
        if (c > Long.MAX_VALUE/(long)elemBytes) {
            return Long.MAX_VALUE;
        }
        long add = c*(long)elemBytes;
        if (current > Long.MAX_VALUE - add) {
            return Long.MAX_VALUE;
        }
        return current + add;
    }

    private static void markUnavailable(String reason, Throwable t) {
        unavailable = true;
        if (!unavailableLogged) {
            unavailableLogged = true;
            System.out.println(BranchDpConfig.getBackendLogPrefix() + " GPU DP unavailable (" + reason
                    + "); falling back to Java DP path");
            if (t != null && getConfigBoolean("branchdp.dp.gpu.trace", false)) {
                t.printStackTrace(System.err);
            }
        }
    }
private static void cleanupPersistentPool() {
        synchronized (persistentLock) {
            if (persistentPool != null) {
                persistentPool.cleanup();
                persistentPool = null;
            }
        }
    }
}
