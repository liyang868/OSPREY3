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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * CUDA fast path for PAC Phase-1 ancestral sampling on one non-leaf lambda edge.
 * Supports the validated per-sample Gumbel-max kernel and an experimental
 * deduplicated CDF path that reuses log-weight vectors for repeated parent
 * M-states.
 */
final class SamplingGpuPhase1 {

    static final int MAX_EDGE_POSITIONS = 32;

    private static volatile boolean unavailable = false;
    private static volatile boolean unavailableLogged = false;
    // Per-edge (non-sticky) fallback: a single oversized/failed edge falls back to the
    // Java sampler without disabling GPU for the whole run; only a run of failures
    // (MAX_CONSECUTIVE_EDGE_FAILURES) flips the sticky unavailable flag.
    private static volatile int consecutiveEdgeFailures = 0;
    private static final int MAX_CONSECUTIVE_EDGE_FAILURES = 5;
    private static final Object persistentLock = new Object();
    private static PersistentPool persistentPool = null;
    private static boolean persistentShutdownHook = false;
    private static final long SAMPLE_SEED_STRIDE =
            0x9E3779B97F4A7C15L;

    private SamplingGpuPhase1() {
    }

    private static boolean getConfigBoolean(String key, boolean defaultValue) {
        return BranchDpConfig.getBackendBoolean(key, defaultValue);
    }

    static synchronized void resetForTesting() {
        unavailable = false;
        unavailableLogged = false;
        cleanupPersistentPool();
    }

    /** Inputs for the sampling kernels: the DP upper/min half + per-sample mIdx. */
    static final class Request implements Cloneable {
        /** Only prepared, immutable model arrays receive an identity. */
        Object modelIdentity;
        Object modelOwner;
        int totalLambdaStates;
        int blockThreads;
        int numChildren;
        double invRT;

        int[] mCounts;
        int[] lambdaCounts;

        double[] lambdaOnlyMin;
        double[] lmMin;
        int[] lmLamSlots;
        int[] lmMSlots;
        int[] lmMCounts;
        long[] lmOffsets;

        int[] childMSrcAll;
        long[] childMStrideAll;
        int[] childMTermOff;
        int[] childMTermCnt;
        int[] childLSrcAll;
        long[] childLStrideAll;
        int[] childLTermOff;
        int[] childLTermCnt;
        long[] childTableBase;
        double[] childUpperAll;
        long childUpperCacheKey;

        long[] mIdxPerSample;   // one parent M-state per sample (length = numSamples)
        long baseSeed;          // per-edge seed for the CDF sampler
        boolean progress;

        boolean multiGpu;
        int maxGpus;            // 0 = all available
        int minGroupsPerGpu;
        boolean persistentContext;
        boolean residentChildTables;
        Method method = Method.GUMBEL;

        Request forSamples(long[] assignments, long seed, boolean reportProgress) {
            try {
                Request copy = (Request) clone();
                copy.mIdxPerSample = assignments;
                copy.baseSeed = seed;
                copy.progress = reportProgress;
                return copy;
            } catch (CloneNotSupportedException ex) {
                throw new AssertionError(ex);
            }
        }

        long retainedBytes() {
            return 1024L + 4L * (mCounts.length + (long) lambdaCounts.length
                    + lmLamSlots.length + lmMSlots.length + lmMCounts.length
                    + childMSrcAll.length + childMTermOff.length + childMTermCnt.length
                    + childLSrcAll.length + childLTermOff.length + childLTermCnt.length)
                    + 8L * (lambdaOnlyMin.length + (long) lmMin.length + lmOffsets.length
                    + childMStrideAll.length + childLStrideAll.length
                    + childTableBase.length + childUpperAll.length);
        }
    }

    static synchronized void releasePreparedModel(Object owner) {
        synchronized (persistentLock) {
            if (persistentPool != null) {
                for (GpuExecutor executor : persistentPool.executors) {
                    executor.releasePreparedModel(owner);
                }
            }
        }
    }

    enum Method {
        GUMBEL("gumbel"),
        CDF_DEDUP("cdfDedup"),
        AUTO("auto");

        final String propertyValue;

        Method(String propertyValue) {
            this.propertyValue = propertyValue;
        }

        static Method fromProperty(String value, Method defaultValue) {
            if (value == null || value.trim().isEmpty()) {
                return defaultValue;
            }
            String normalized = value.trim();
            for (Method method : values()) {
                if (method.propertyValue.equalsIgnoreCase(normalized)
                        || method.name().equalsIgnoreCase(normalized)) {
                    return method;
                }
            }
            System.err.println(BranchDpConfig.getBackendLogPrefix() + " Invalid GPU sampling method '" + value
                    + "', using " + defaultValue.propertyValue + ".");
            return defaultValue;
        }
    }

    /** Returns the sampled lambda-state index per sample, or null on fallback. */
    static synchronized int[] sample(Request req) {
        if (unavailable) {
            return null;
        }
        if (req == null || req.mIdxPerSample == null) {
            return null;
        }
        if (req.mIdxPerSample.length == 0) {
            return new int[0];
        }
        if (req.totalLambdaStates <= 0) {
            return null;
        }

        try {
            GroupedSamples grouped = groupSamples(req);

            List<Gpu> gpus = Gpus.get().getGpus();
            if (gpus.isEmpty()) {
                markUnavailable("no CUDA GPU with double support", null);
                return null;
            }

            Method method = resolveMethod(req, grouped);
            if (method == Method.CDF_DEDUP && !cdfFits(req, grouped)) {
                return null;
            }
            if (method == Method.GUMBEL) {
                int nGpus = chooseGpuCountForSamples(
                        req, gpus.size(), grouped.numSamples);
                int[] g;
                long startNanos = System.nanoTime();
                GpuExecutor[] gumbelExecutors = makeExecutors(req, gpus, nGpus);
                try {
                    if (nGpus <= 1) {
                        g = runGumbel(req, gumbelExecutors[0], grouped);
                    } else {
                        g = new int[grouped.numSamples];
                        runMultiGpuGumbel(req, grouped, gumbelExecutors, nGpus, g);
                        if (req.progress) {
                            double ms = (System.nanoTime() - startNanos) / 1e6;
                            System.out.println(BranchDpConfig.getBackendLogPrefix()
                                    + " GPU sampling done, samples="
                                    + grouped.numSamples
                                    + ", method=" + Method.GUMBEL.propertyValue
                                    + ", distinctMIdx=" + grouped.numGroups
                                    + ", cdfReuse=0"
                                    + ", gpus=" + nGpus
                                    + ", lambdaStates=" + req.totalLambdaStates
                                    + ", children=" + req.numChildren
                                    + ", blockThreads="
                                    + Math.max(1, req.blockThreads)
                                    + ", residentChildTables=" + req.residentChildTables
                                    + ", elapsedMs="
                                    + String.format(java.util.Locale.ROOT,
                                    "%.1f", ms));
                        }
                    }
                    consecutiveEdgeFailures = 0;
                    return g;
                } finally {
                    cleanupExecutors(gumbelExecutors);
                }
            }

            int nGpus = chooseGpuCount(req, gpus.size(), grouped.numGroups);
            GpuExecutor[] executors = makeExecutors(req, gpus, nGpus);
            long startNanos = System.nanoTime();
            int[] result = new int[grouped.numSamples];
            try {
                if (nGpus <= 1) {
                    SliceResult slice = executors[0].runSlice(req,
                            makeSlice(grouped, 0, grouped.numGroups));
                    copySliceResult(result, slice);
                } else {
                    runMultiGpu(req, grouped, executors, nGpus, result);
                }
            } finally {
                cleanupExecutors(executors);
            }

            if (req.progress) {
                double ms = (System.nanoTime() - startNanos)/1e6;
                System.out.println(BranchDpConfig.getBackendLogPrefix() + " GPU sampling done, samples=" + grouped.numSamples
                        + ", method=" + method.propertyValue
                        + ", distinctMIdx=" + grouped.numGroups
                        + ", cdfReuse=" + (grouped.numSamples - grouped.numGroups)
                        + ", gpus=" + nGpus
                        + ", lambdaStates=" + req.totalLambdaStates
                        + ", children=" + req.numChildren
                        + ", blockThreads=" + Math.max(1, req.blockThreads)
                        + ", residentChildTables=" + req.residentChildTables
                        + ", elapsedMs=" + String.format(java.util.Locale.ROOT, "%.1f", ms));
            }
            consecutiveEdgeFailures = 0;
            return result;
        } catch (Throwable t) {
            // Per-edge failure: fall back to Java sampling for THIS edge, keep GPU on.
            // Disable globally only after a run of consecutive failures.
            consecutiveEdgeFailures++;
            if (consecutiveEdgeFailures >= MAX_CONSECUTIVE_EDGE_FAILURES) {
                markUnavailable("disabled after " + consecutiveEdgeFailures
                        + " consecutive edge failures; last="
                        + t.getClass().getSimpleName() + ": " + t.getMessage(), t);
            } else {
                System.out.println(BranchDpConfig.getBackendLogPrefix() + " GPU sampling edge fallback ("
                        + t.getClass().getSimpleName() + ": " + t.getMessage()
                        + "); this edge on Java sampling path");
                if (getConfigBoolean("branchdp.pac.sampling.gpu.trace", false)) {
                    t.printStackTrace(System.err);
                }
            }
            return null;
        }
    }

    private static Method resolveMethod(Request req, GroupedSamples grouped) {
        Method method = req.method == null ? Method.GUMBEL : req.method;
        if (method != Method.AUTO) {
            return method;
        }
        return grouped.numGroups < grouped.numSamples && cdfFits(req, grouped)
                ? Method.CDF_DEDUP
                : Method.GUMBEL;
    }

    private static GroupedSamples groupSamples(Request req) {
        int numSamples = req.mIdxPerSample.length;
        Map<Long, Integer> groupByMIdx = new LinkedHashMap<>();
        int[] groupForSample = new int[numSamples];
        for (int s = 0; s < numSamples; s++) {
            long mIdx = req.mIdxPerSample[s];
            Integer group = groupByMIdx.get(mIdx);
            if (group == null) {
                group = groupByMIdx.size();
                groupByMIdx.put(mIdx, group);
            }
            groupForSample[s] = group;
        }

        int numGroups = groupByMIdx.size();
        long[] mIdxByGroup = new long[numGroups];
        for (Map.Entry<Long, Integer> entry : groupByMIdx.entrySet()) {
            mIdxByGroup[entry.getValue()] = entry.getKey();
        }
        return new GroupedSamples(numSamples, numGroups, mIdxByGroup, groupForSample);
    }

    private static boolean cdfFits(Request req, GroupedSamples grouped) {
        long cdfValues = (long)grouped.numGroups*(long)req.totalLambdaStates;
        return cdfValues > 0L && cdfValues <= Integer.MAX_VALUE;
    }

    private static int[] runGumbel(Request req, GpuExecutor executor, GroupedSamples grouped) {
        return runGumbel(req, executor, grouped, 0,
                req.mIdxPerSample.length, req.baseSeed, true);
    }

    /** Run Gumbel for one contiguous sample slice. */
    private static int[] runGumbel(Request req, GpuExecutor executor,
                                   GroupedSamples grouped,
                                   int sampleStart, int numSamples,
                                   long baseSeed, boolean reportProgress) {
        long sampleEnd = (long) sampleStart + numSamples;
        if (sampleStart < 0 || numSamples < 0
                || sampleEnd > req.mIdxPerSample.length) {
            throw new IndexOutOfBoundsException(
                    "invalid Gumbel sample slice [" + sampleStart + ","
                            + sampleEnd + ")/" + req.mIdxPerSample.length);
        }
        long[] mIdxSlice = Arrays.copyOfRange(
                req.mIdxPerSample, sampleStart, (int) sampleEnd);
        GpuStream stream = executor.stream;
        List<CUBuffer<?>> buffers = new ArrayList<>();
        List<CUdeviceptr> rawBufs = new ArrayList<>();

        try {
            executor.context.attachCurrentThread();

            int blockThreads = resolveBlockThreads(req, stream);
            Kernel kernel = new Kernel(stream, "sampling");
            Kernel.Function func = kernel.makeFunction("sample_gumbel_n_children");
            func.numBlocks = numSamples;
            func.blockThreads = blockThreads;
            func.sharedMemCalc = bt -> {
                int warps = (bt + 31) >>> 5;
                return warps*(Double.BYTES + Integer.BYTES);
            };

            ModelBuffers model = executor.modelBuffers(req, buffers, rawBufs);
            CUBuffer<IntBuffer> mCounts = model.mCounts;
            CUBuffer<IntBuffer> lambdaCounts = model.lambdaCounts;
            CUdeviceptr lambdaOnlyMin = model.lambdaOnlyMin;
            CUdeviceptr lmMin = model.lmMin;
            CUBuffer<IntBuffer> lmLamSlots = model.lmLamSlots;
            CUBuffer<IntBuffer> lmMSlots = model.lmMSlots;
            CUBuffer<IntBuffer> lmMCounts = model.lmMCounts;
            CUBuffer<LongBuffer> lmOffsets = model.lmOffsets;
            CUBuffer<IntBuffer> childMSrcAll = model.childMSrcAll;
            CUBuffer<LongBuffer> childMStrideAll = model.childMStrideAll;
            CUBuffer<IntBuffer> childMTermOff = model.childMTermOff;
            CUBuffer<IntBuffer> childMTermCnt = model.childMTermCnt;
            CUBuffer<IntBuffer> childLSrcAll = model.childLSrcAll;
            CUBuffer<LongBuffer> childLStrideAll = model.childLStrideAll;
            CUBuffer<IntBuffer> childLTermOff = model.childLTermOff;
            CUBuffer<IntBuffer> childLTermCnt = model.childLTermCnt;
            CUBuffer<LongBuffer> childTableBase = model.childTableBase;
            CUdeviceptr childUpperAll = model.childUpperAll;
            CUBuffer<LongBuffer> mIdxPerSample = uploadLongs(stream,
                    mIdxSlice, buffers);
            CUBuffer<IntBuffer> outLIdx = makeInts(stream, Math.max(1, numSamples), buffers);

            int[] numSamplesArg = { numSamples };
            int[] totalLambdaStatesArg = { req.totalLambdaStates };
            int[] mCountArg = { req.mCounts.length };
            int[] lambdaCountArg = { req.lambdaCounts.length };
            int[] lmPairCountArg = { req.lmLamSlots.length };
            int[] numChildrenArg = { req.numChildren };
            double[] invRTArg = { req.invRT };
            long[] baseSeedArg = { baseSeed };

            func.setArgs(Pointer.to(
                mCounts.getDevicePointer(),
                lambdaCounts.getDevicePointer(),
                Pointer.to(lambdaOnlyMin),
                Pointer.to(lmMin),
                lmLamSlots.getDevicePointer(),
                lmMSlots.getDevicePointer(),
                lmMCounts.getDevicePointer(),
                lmOffsets.getDevicePointer(),
                childMSrcAll.getDevicePointer(),
                childMStrideAll.getDevicePointer(),
                childMTermOff.getDevicePointer(),
                childMTermCnt.getDevicePointer(),
                childLSrcAll.getDevicePointer(),
                childLStrideAll.getDevicePointer(),
                childLTermOff.getDevicePointer(),
                childLTermCnt.getDevicePointer(),
                childTableBase.getDevicePointer(),
                Pointer.to(childUpperAll),
                mIdxPerSample.getDevicePointer(),
                outLIdx.getDevicePointer(),
                Pointer.to(numSamplesArg),
                Pointer.to(totalLambdaStatesArg),
                Pointer.to(mCountArg),
                Pointer.to(lambdaCountArg),
                Pointer.to(lmPairCountArg),
                Pointer.to(numChildrenArg),
                Pointer.to(invRTArg),
                Pointer.to(baseSeedArg)
            ));

            long startNanos = System.nanoTime();
            func.runAsync();
            outLIdx.downloadAsync();
            stream.waitForGpu();

            int[] result = new int[numSamples];
            IntBuffer host = outLIdx.getHostBuffer();
            host.rewind();
            host.get(result, 0, numSamples);

            if (req.progress && reportProgress) {
                double ms = (System.nanoTime() - startNanos)/1e6;
                System.out.println(BranchDpConfig.getBackendLogPrefix() + " GPU sampling done, samples=" + numSamples
                        + ", method=" + Method.GUMBEL.propertyValue
                        + ", distinctMIdx=" + grouped.numGroups
                        + ", cdfReuse=0"
                        + ", gpus=1"
                        + ", lambdaStates=" + req.totalLambdaStates
                        + ", children=" + req.numChildren
                        + ", blockThreads=" + blockThreads
                        + ", residentChildTables=" + req.residentChildTables
                        + ", elapsedMs=" + String.format(java.util.Locale.ROOT, "%.1f", ms));
            }
            return result;
        } finally {
            for (int i = buffers.size() - 1; i >= 0; i--) {
                try {
                    buffers.get(i).cleanup();
                } catch (Throwable t) {
                    t.printStackTrace(System.err);
                }
            }
            freeRaw(rawBufs);
        }
    }

    private static int chooseGpuCountForSamples(
            Request req, int available, int numSamples) {
        if (!req.multiGpu) {
            return 1;
        }
        int cap = Math.min(available, Math.max(1, numSamples));
        if (req.maxGpus > 0) {
            cap = Math.min(cap, req.maxGpus);
        }
        return Math.max(1, cap);
    }

    private static void runMultiGpuGumbel(
            Request req, GroupedSamples grouped,
            GpuExecutor[] executors, int nGpus, int[] result)
            throws InterruptedException {
        int base = grouped.numSamples / nGpus;
        int rem = grouped.numSamples % nGpus;
        Thread[] threads = new Thread[nGpus];
        Throwable[] errors = new Throwable[nGpus];
        int sampleStart = 0;
        for (int gpuIndex = 0; gpuIndex < nGpus; gpuIndex++) {
            int sampleCount = base + (gpuIndex < rem ? 1 : 0);
            final int start = sampleStart;
            final int count = sampleCount;
            final int gi = gpuIndex;
            final GpuExecutor executor = executors[gpuIndex];
            threads[gpuIndex] = new Thread(() -> {
                try {
                    // The CUDA kernel derives its stateless random value from
                    // baseSeed + SAMPLE_SEED_STRIDE * sampleSlot.  Offset the
                    // per-slice seed so local slot s has exactly the same
                    // stream as global slot start+s in the one-GPU path.
                    long sliceSeed = req.baseSeed
                            + SAMPLE_SEED_STRIDE * (long) start;
                    int[] local = runGumbel(req, executor, grouped, start,
                            count, sliceSeed, false);
                    System.arraycopy(local, 0, result, start, count);
                } catch (Throwable t) {
                    errors[gi] = t;
                }
            }, "bms-gpu-sampling-gumbel-" + gpuIndex);
            threads[gpuIndex].start();
            sampleStart += sampleCount;
        }
        for (Thread thread : threads) {
            thread.join();
        }
        for (Throwable error : errors) {
            if (error != null) {
                throw new RuntimeException(
                        "multi-GPU Gumbel sampling failed: "
                                + error.getMessage(), error);
            }
        }
    }

    private static int chooseGpuCount(Request req, int available, int numGroups) {
        if (!req.multiGpu) {
            return 1;
        }
        int cap = Math.min(available, Math.max(1, numGroups));
        if (req.maxGpus > 0) {
            cap = Math.min(cap, req.maxGpus);
        }
        int minGroupsPerGpu = Math.max(1, req.minGroupsPerGpu);
        cap = Math.min(cap, Math.max(1, numGroups/minGroupsPerGpu));
        return Math.max(1, cap);
    }

    private static void runMultiGpu(Request req, GroupedSamples grouped,
                                    GpuExecutor[] executors, int nGpus,
                                    int[] result) throws InterruptedException {
        int base = grouped.numGroups/nGpus;
        int rem = grouped.numGroups%nGpus;
        Thread[] threads = new Thread[nGpus];
        Throwable[] errors = new Throwable[nGpus];
        SliceResult[] slices = new SliceResult[nGpus];
        int groupStart = 0;
        for (int g = 0; g < nGpus; g++) {
            int groupCount = base + (g < rem ? 1 : 0);
            WorkSlice slice = makeSlice(grouped, groupStart, groupCount);
            final int gi = g;
            threads[g] = new Thread(() -> {
                try {
                    slices[gi] = executors[gi].runSlice(req, slice);
                } catch (Throwable t) {
                    errors[gi] = t;
                }
            }, "bms-gpu-sampling-" + g);
            threads[g].start();
            groupStart += groupCount;
        }
        for (Thread thread : threads) {
            thread.join();
        }
        for (Throwable error : errors) {
            if (error != null) {
                throw new RuntimeException("multi-GPU sampling failed: " + error.getMessage(), error);
            }
        }
        for (SliceResult slice : slices) {
            copySliceResult(result, slice);
        }
    }

    private static WorkSlice makeSlice(GroupedSamples grouped, int groupStart, int groupCount) {
        int groupEnd = groupStart + groupCount;
        int sampleCount = 0;
        for (int group : grouped.groupForSample) {
            if (group >= groupStart && group < groupEnd) {
                sampleCount++;
            }
        }

        long[] mIdxByGroup = Arrays.copyOfRange(grouped.mIdxByGroup, groupStart, groupEnd);
        int[] sampleSlots = new int[sampleCount];
        int[] localGroupForSample = new int[sampleCount];
        int out = 0;
        for (int s = 0; s < grouped.numSamples; s++) {
            int group = grouped.groupForSample[s];
            if (group >= groupStart && group < groupEnd) {
                sampleSlots[out] = s;
                localGroupForSample[out] = group - groupStart;
                out++;
            }
        }
        return new WorkSlice(mIdxByGroup, sampleSlots, localGroupForSample);
    }

    private static void copySliceResult(int[] result, SliceResult slice) {
        if (slice == null) {
            return;
        }
        for (int i = 0; i < slice.sampleSlots.length; i++) {
            result[slice.sampleSlots[i]] = slice.lIdx[i];
        }
    }

    private static GpuExecutor[] makeExecutors(Request req, List<Gpu> gpus, int nGpus) {
        if (req.persistentContext || req.residentChildTables) {
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
                    Runtime.getRuntime().addShutdownHook(new Thread(SamplingGpuPhase1::cleanupPersistentPool,
                            "bms-gpu-sampling-persistent-cleanup"));
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

    private static void cleanupPersistentPool() {
        synchronized (persistentLock) {
            if (persistentPool != null) {
                persistentPool.cleanup();
                persistentPool = null;
            }
        }
    }

    private static final class GroupedSamples {
        final int numSamples;
        final int numGroups;
        final long[] mIdxByGroup;
        final int[] groupForSample;

        GroupedSamples(int numSamples, int numGroups, long[] mIdxByGroup, int[] groupForSample) {
            this.numSamples = numSamples;
            this.numGroups = numGroups;
            this.mIdxByGroup = mIdxByGroup;
            this.groupForSample = groupForSample;
        }
    }

    private static final class WorkSlice {
        final long[] mIdxByGroup;
        final int[] sampleSlots;
        final int[] groupForSample;

        WorkSlice(long[] mIdxByGroup, int[] sampleSlots, int[] groupForSample) {
            this.mIdxByGroup = mIdxByGroup;
            this.sampleSlots = sampleSlots;
            this.groupForSample = groupForSample;
        }
    }

    private static final class SliceResult {
        final int[] sampleSlots;
        final int[] lIdx;

        SliceResult(int[] sampleSlots, int[] lIdx) {
            this.sampleSlots = sampleSlots;
            this.lIdx = lIdx;
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
        private CUdeviceptr childUpperCache;
        private long childUpperCacheKey;
        private int childUpperCacheLength;
        private ModelBuffers preparedModel;
        private Object preparedIdentity;
        private Object preparedOwner;

        GpuExecutor(Gpu gpu, boolean owned) {
            this.owned = owned;
            this.context = new Context(gpu);
            this.context.attachCurrentThread();
            this.stream = new GpuStream(context);
        }

        synchronized SliceResult runSlice(Request req, WorkSlice slice) {
            context.attachCurrentThread();
            return runOnGpuSlice(req, this, slice);
        }

        ModelBuffers modelBuffers(Request req, List<CUBuffer<?>> buffers,
                                  List<CUdeviceptr> rawBufs) {
            if (req.modelIdentity != null && req.residentChildTables) {
                if (preparedModel != null && preparedIdentity == req.modelIdentity) return preparedModel;
                clearPreparedModel();
                cleanupChildUpperCache();
                List<CUBuffer<?>> retainedBuffers = new ArrayList<>();
                List<CUdeviceptr> retainedRaw = new ArrayList<>();
                try {
                    preparedModel = new ModelBuffers(req, this, retainedBuffers, retainedRaw, true);
                    preparedIdentity = req.modelIdentity;
                    preparedOwner = req.modelOwner;
                    return preparedModel;
                } catch (RuntimeException | Error ex) {
                    cleanBuffers(retainedBuffers);
                    freeRaw(retainedRaw);
                    throw ex;
                }
            }
            clearPreparedModel();
            return new ModelBuffers(req, this, buffers, rawBufs, false);
        }

        void releasePreparedModel(Object owner) {
            if (preparedModel != null && preparedOwner == owner) {
                context.attachCurrentThread();
                clearPreparedModel();
            }
        }

        private void clearPreparedModel() {
            if (preparedModel != null) {
                preparedModel.close();
                preparedModel = null;
                preparedIdentity = null;
                preparedOwner = null;
            }
        }

        CUdeviceptr childUpperBuffer(Request req, List<CUdeviceptr> rawBufs) {
            if (!req.residentChildTables || req.childUpperCacheKey == 0L) {
                // Non-resident: per-call raw buffer, freed with the rest of rawBufs.
                return uploadDoublesBig(req.childUpperAll, rawBufs);
            }
            if (childUpperCache != null
                    && childUpperCacheKey == req.childUpperCacheKey
                    && childUpperCacheLength == req.childUpperAll.length) {
                return childUpperCache;
            }
            cleanupChildUpperCache();
            // Resident: persistent raw buffer owned by this executor (not in rawBufs).
            childUpperCache = newDeviceDoubles(req.childUpperAll);
            childUpperCacheKey = req.childUpperCacheKey;
            childUpperCacheLength = req.childUpperAll.length;
            return childUpperCache;
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
            context.attachCurrentThread();
            clearPreparedModel();
            cleanupChildUpperCache();
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

        private void cleanupChildUpperCache() {
            if (childUpperCache != null) {
                try {
                    JCudaDriver.cuMemFree(childUpperCache);
                } catch (Throwable t) {
                    t.printStackTrace(System.err);
                } finally {
                    childUpperCache = null;
                    childUpperCacheKey = 0L;
                    childUpperCacheLength = 0;
                }
            }
        }
    }

    /** Device storage for one immutable prepared model; dynamic samples stay per-call. */
    private static final class ModelBuffers {
        final CUBuffer<IntBuffer> mCounts, lambdaCounts, lmLamSlots, lmMSlots, lmMCounts;
        final CUBuffer<IntBuffer> childMSrcAll, childMTermOff, childMTermCnt;
        final CUBuffer<IntBuffer> childLSrcAll, childLTermOff, childLTermCnt;
        final CUBuffer<LongBuffer> lmOffsets, childMStrideAll, childLStrideAll, childTableBase;
        final CUdeviceptr lambdaOnlyMin, lmMin, childUpperAll;
        private final List<CUBuffer<?>> buffers;
        private final List<CUdeviceptr> rawBufs;

        ModelBuffers(Request req, GpuExecutor executor, List<CUBuffer<?>> buffers,
                     List<CUdeviceptr> rawBufs, boolean retained) {
            this.buffers = buffers;
            this.rawBufs = rawBufs;
            GpuStream stream = executor.stream;
            mCounts = uploadInts(stream, req.mCounts, buffers);
            lambdaCounts = uploadInts(stream, req.lambdaCounts, buffers);
            lambdaOnlyMin = uploadDoublesBig(req.lambdaOnlyMin, rawBufs);
            lmMin = uploadDoublesBig(req.lmMin, rawBufs);
            lmLamSlots = uploadInts(stream, req.lmLamSlots, buffers);
            lmMSlots = uploadInts(stream, req.lmMSlots, buffers);
            lmMCounts = uploadInts(stream, req.lmMCounts, buffers);
            lmOffsets = uploadLongs(stream, req.lmOffsets, buffers);
            childMSrcAll = uploadInts(stream, req.childMSrcAll, buffers);
            childMStrideAll = uploadLongs(stream, req.childMStrideAll, buffers);
            childMTermOff = uploadInts(stream, req.childMTermOff, buffers);
            childMTermCnt = uploadInts(stream, req.childMTermCnt, buffers);
            childLSrcAll = uploadInts(stream, req.childLSrcAll, buffers);
            childLStrideAll = uploadLongs(stream, req.childLStrideAll, buffers);
            childLTermOff = uploadInts(stream, req.childLTermOff, buffers);
            childLTermCnt = uploadInts(stream, req.childLTermCnt, buffers);
            childTableBase = uploadLongs(stream, req.childTableBase, buffers);
            childUpperAll = retained ? uploadDoublesBig(req.childUpperAll, rawBufs)
                    : executor.childUpperBuffer(req, rawBufs);
        }

        void close() {
            cleanBuffers(buffers);
            freeRaw(rawBufs);
        }
    }

    private static void cleanBuffers(List<CUBuffer<?>> buffers) {
        for (int i = buffers.size() - 1; i >= 0; i--) {
            try { buffers.get(i).cleanup(); }
            catch (Throwable t) { t.printStackTrace(System.err); }
        }
    }

    private static SliceResult runOnGpuSlice(Request req, GpuExecutor executor, WorkSlice slice) {
        int numGroups = slice.mIdxByGroup.length;
        int numSamples = slice.sampleSlots.length;
        if (numGroups <= 0 || numSamples <= 0) {
            return new SliceResult(slice.sampleSlots, new int[numSamples]);
        }

        long cdfValuesLong = (long)numGroups*(long)req.totalLambdaStates;
        if (cdfValuesLong <= 0L || cdfValuesLong > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("GPU sampling CDF buffer too large: " + cdfValuesLong);
        }
        int cdfValues = (int)cdfValuesLong;
        GpuStream stream = executor.stream;
        List<CUBuffer<?>> buffers = new ArrayList<>();
        List<CUdeviceptr> rawBufs = new ArrayList<>();

        try {
            stream.getContext().attachCurrentThread();

            int blockThreads = resolveBlockThreads(req, stream);
            Kernel kernel = new Kernel(stream, "sampling");
            Kernel.Function cdfFunc = kernel.makeFunction("build_cdf_n_children");
            cdfFunc.numBlocks = numGroups;
            cdfFunc.blockThreads = blockThreads;
            cdfFunc.sharedMemCalc = bt -> ((bt + 31) >>> 5)*Double.BYTES;

            Kernel.Function drawFunc = kernel.makeFunction("draw_cdf_n_children");
            drawFunc.numBlocks = Math.max(1, (numSamples + blockThreads - 1)/blockThreads);
            drawFunc.blockThreads = blockThreads;
            drawFunc.sharedMemCalc = new Kernel.SharedMemCalculator.None();

            ModelBuffers model = executor.modelBuffers(req, buffers, rawBufs);
            CUBuffer<IntBuffer> mCounts = model.mCounts;
            CUBuffer<IntBuffer> lambdaCounts = model.lambdaCounts;
            CUdeviceptr lambdaOnlyMin = model.lambdaOnlyMin;
            CUdeviceptr lmMin = model.lmMin;
            CUBuffer<IntBuffer> lmLamSlots = model.lmLamSlots;
            CUBuffer<IntBuffer> lmMSlots = model.lmMSlots;
            CUBuffer<IntBuffer> lmMCounts = model.lmMCounts;
            CUBuffer<LongBuffer> lmOffsets = model.lmOffsets;
            CUBuffer<IntBuffer> childMSrcAll = model.childMSrcAll;
            CUBuffer<LongBuffer> childMStrideAll = model.childMStrideAll;
            CUBuffer<IntBuffer> childMTermOff = model.childMTermOff;
            CUBuffer<IntBuffer> childMTermCnt = model.childMTermCnt;
            CUBuffer<IntBuffer> childLSrcAll = model.childLSrcAll;
            CUBuffer<LongBuffer> childLStrideAll = model.childLStrideAll;
            CUBuffer<IntBuffer> childLTermOff = model.childLTermOff;
            CUBuffer<IntBuffer> childLTermCnt = model.childLTermCnt;
            CUBuffer<LongBuffer> childTableBase = model.childTableBase;
            CUdeviceptr childUpperAll = model.childUpperAll;
            CUBuffer<LongBuffer> mIdxByGroup = uploadLongs(stream, slice.mIdxByGroup, buffers);
            CUBuffer<DoubleBuffer> groupCdf = makeDoubles(stream, cdfValues, buffers);
            CUBuffer<DoubleBuffer> groupTotal = makeDoubles(stream, numGroups, buffers);
            CUBuffer<IntBuffer> groupForSample = uploadInts(stream, slice.groupForSample, buffers);
            CUBuffer<IntBuffer> sampleSlots = uploadInts(stream, slice.sampleSlots, buffers);
            CUBuffer<IntBuffer> outLIdx = makeInts(stream, numSamples, buffers);

            int[] numGroupsArg = { numGroups };
            int[] numSamplesArg = { numSamples };
            int[] totalLambdaStatesArg = { req.totalLambdaStates };
            int[] mCountArg = { req.mCounts.length };
            int[] lambdaCountArg = { req.lambdaCounts.length };
            int[] lmPairCountArg = { req.lmLamSlots.length };
            int[] numChildrenArg = { req.numChildren };
            double[] invRTArg = { req.invRT };
            long[] baseSeedArg = { req.baseSeed };

            cdfFunc.setArgs(Pointer.to(
                mCounts.getDevicePointer(),
                lambdaCounts.getDevicePointer(),
                Pointer.to(lambdaOnlyMin),
                Pointer.to(lmMin),
                lmLamSlots.getDevicePointer(),
                lmMSlots.getDevicePointer(),
                lmMCounts.getDevicePointer(),
                lmOffsets.getDevicePointer(),
                childMSrcAll.getDevicePointer(),
                childMStrideAll.getDevicePointer(),
                childMTermOff.getDevicePointer(),
                childMTermCnt.getDevicePointer(),
                childLSrcAll.getDevicePointer(),
                childLStrideAll.getDevicePointer(),
                childLTermOff.getDevicePointer(),
                childLTermCnt.getDevicePointer(),
                childTableBase.getDevicePointer(),
                Pointer.to(childUpperAll),
                mIdxByGroup.getDevicePointer(),
                groupCdf.getDevicePointer(),
                groupTotal.getDevicePointer(),
                Pointer.to(numGroupsArg),
                Pointer.to(totalLambdaStatesArg),
                Pointer.to(mCountArg),
                Pointer.to(lambdaCountArg),
                Pointer.to(lmPairCountArg),
                Pointer.to(numChildrenArg),
                Pointer.to(invRTArg)
            ));

            drawFunc.setArgs(Pointer.to(
                groupCdf.getDevicePointer(),
                groupTotal.getDevicePointer(),
                groupForSample.getDevicePointer(),
                sampleSlots.getDevicePointer(),
                outLIdx.getDevicePointer(),
                Pointer.to(numSamplesArg),
                Pointer.to(totalLambdaStatesArg),
                Pointer.to(baseSeedArg)
            ));

            cdfFunc.runAsync();
            drawFunc.runAsync();
            outLIdx.downloadAsync();
            stream.waitForGpu();

            int[] lIdx = new int[numSamples];
            IntBuffer host = outLIdx.getHostBuffer();
            host.rewind();
            host.get(lIdx, 0, numSamples);
            return new SliceResult(slice.sampleSlots, lIdx);
        } finally {
            for (int i = buffers.size() - 1; i >= 0; i--) {
                try {
                    buffers.get(i).cleanup();
                } catch (Throwable t) {
                    t.printStackTrace(System.err);
                }
            }
            // The resident child-upper cache (when used) is owned by the executor and
            // is NOT in rawBufs, so freeing rawBufs never frees the cache.
            freeRaw(rawBufs);
        }
    }

    private static int resolveBlockThreads(Request req, GpuStream stream) {
        int requested = Math.max(1, req.blockThreads);
        int capped = Math.min(requested, stream.getContext().getGpu().getMaxBlockThreads());
        int powerOfTwo = Integer.highestOneBit(capped);
        return Math.max(1, powerOfTwo);
    }

    private static CUBuffer<IntBuffer> uploadInts(GpuStream stream, int[] values, List<CUBuffer<?>> buffers) {
        CUBuffer<IntBuffer> buf = makeInts(stream, Math.max(1, values.length), buffers);
        IntBuffer host = buf.getHostBuffer();
        host.clear();
        if (values.length > 0) {
            host.put(values);
        }
        while (host.position() < host.capacity()) {
            host.put(0);
        }
        buf.uploadAsync();
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

    /**
     * Upload a double[] straight into raw device memory, bypassing the 2 GiB cap of a
     * single java.nio direct buffer (cuMemAlloc/cuMemcpyHtoD take long byte counts).
     * The caller must cuMemFree the returned pointer (see freeRaw).
     */
    private static CUdeviceptr uploadDoublesBig(double[] values, List<CUdeviceptr> rawBufs) {
        CUdeviceptr dptr = newDeviceDoubles(values);
        rawBufs.add(dptr);
        return dptr;
    }

    private static CUdeviceptr newDeviceDoubles(double[] values) {
        long count = Math.max(1L, values.length);
        long bytes = count*(long)Double.BYTES;
        CUdeviceptr dptr = new CUdeviceptr();
        cudaCheck(JCudaDriver.cuMemAlloc(dptr, bytes), "cuMemAlloc(" + bytes + " bytes)");
        if (values.length > 0) {
            cudaCheck(JCudaDriver.cuMemcpyHtoD(dptr, Pointer.to(values),
                    (long)values.length*(long)Double.BYTES), "cuMemcpyHtoD");
        } else {
            cudaCheck(JCudaDriver.cuMemsetD8(dptr, (byte)0, bytes), "cuMemsetD8");
        }
        return dptr;
    }

    private static void cudaCheck(int result, String what) {
        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("CUDA " + what + " failed: " + CUresult.stringFor(result));
        }
    }

    private static void freeRaw(List<CUdeviceptr> rawBufs) {
        for (int i = rawBufs.size() - 1; i >= 0; i--) {
            try {
                JCudaDriver.cuMemFree(rawBufs.get(i));
            } catch (Throwable t) {
                t.printStackTrace(System.err);
            }
        }
    }

    private static CUBuffer<IntBuffer> makeInts(GpuStream stream, int size, List<CUBuffer<?>> buffers) {
        CUBuffer<IntBuffer> buf = stream.makeIntBuffer(Math.max(1, size));
        buffers.add(buf);
        return buf;
    }

    private static CUBuffer<DoubleBuffer> makeDoubles(GpuStream stream, int size, List<CUBuffer<?>> buffers) {
        CUBuffer<DoubleBuffer> buf = stream.makeDoubleBuffer(Math.max(1, size));
        buffers.add(buf);
        return buf;
    }

    private static void markUnavailable(String reason, Throwable t) {
        unavailable = true;
        if (!unavailableLogged) {
            unavailableLogged = true;
            System.out.println(BranchDpConfig.getBackendLogPrefix() + " GPU sampling unavailable (" + reason
                    + "); falling back to Java sampling path");
            if (t != null && getConfigBoolean("branchdp.pac.sampling.gpu.trace", false)) {
                t.printStackTrace(System.err);
            }
        }
    }
}
