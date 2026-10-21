package edu.duke.cs.osprey.branchdp;

import static org.junit.jupiter.api.Assertions.*;

import edu.duke.cs.osprey.gpu.cuda.Gpus;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Direct validation for the PAC Phase-1 CUDA sampler. CPU references mirror the
 * CUDA Gumbel and CDF kernels, including their stateless splitmix64 streams, so
 * the selected lIdx values should match exactly for these synthetic requests.
 */
public class TestGpuPhase1Sampling {

    @Test
    public void preparedModelsReuseBuffersAndInvalidateWithoutChangingSeededDraws() {
        Assumptions.assumeTrue(hasCudaGpu(), "no CUDA GPU available");
        for (SamplingGpuPhase1.Method method : new SamplingGpuPhase1.Method[]{
                SamplingGpuPhase1.Method.GUMBEL, SamplingGpuPhase1.Method.CDF_DEDUP}) {
            SamplingGpuPhase1.resetForTesting();
            SamplingGpuPhase1.Request req = request(512);
            req.method = method;
            req.modelIdentity = new Object();
            req.modelOwner = new Object();
            try {
                for (int repeat = 0; repeat < 3; repeat++) {
                    req = req.forSamples(req.mIdxPerSample, req.baseSeed + 31, false);
                    int[] expected = method == SamplingGpuPhase1.Method.GUMBEL
                            ? cpuGumbelReference(req) : cpuCdfReference(req);
                    assertArrayEquals(expected, SamplingGpuPhase1.sample(req));
                }
                req.lambdaOnlyMin = req.lambdaOnlyMin.clone();
                req.lambdaOnlyMin[0] -= 40.0;
                req.modelIdentity = new Object();
                int[] expected = method == SamplingGpuPhase1.Method.GUMBEL
                        ? cpuGumbelReference(req) : cpuCdfReference(req);
                assertArrayEquals(expected, SamplingGpuPhase1.sample(req));
                SamplingGpuPhase1.releasePreparedModel(req.modelOwner);
                assertArrayEquals(expected, SamplingGpuPhase1.sample(req));
            } finally {
                SamplingGpuPhase1.resetForTesting();
            }
        }
    }

    @Test
    public void gpuMatchesCpuReference_gumbel_multiChild() {
        Assumptions.assumeTrue(hasCudaGpu(), "no CUDA GPU available");
        SamplingGpuPhase1.resetForTesting();

        SamplingGpuPhase1.Request req = request(64);
        req.method = SamplingGpuPhase1.Method.GUMBEL;
        int[] expected = cpuGumbelReference(req);
        int[] actual = SamplingGpuPhase1.sample(req);

        assertNotNull(actual, "GPU sampling returned fallback on a CUDA node");
        assertArrayEquals(expected, actual);
        for (int lIdx : actual) {
            assertTrue(lIdx >= 0 && lIdx < req.totalLambdaStates);
        }
    }

    @Test
    public void gpuMatchesCpuReference_dedupCdf_multiChild() {
        Assumptions.assumeTrue(hasCudaGpu(), "no CUDA GPU available");
        SamplingGpuPhase1.resetForTesting();

        SamplingGpuPhase1.Request req = request(512);
        req.method = SamplingGpuPhase1.Method.CDF_DEDUP;
        int[] expected = cpuCdfReference(req);
        int[] actual = SamplingGpuPhase1.sample(req);

        assertNotNull(actual, "GPU sampling returned fallback on a CUDA node");
        assertArrayEquals(expected, actual);
        for (int lIdx : actual) {
            assertTrue(lIdx >= 0 && lIdx < req.totalLambdaStates);
        }
    }

    @Test
    public void gpuDistributionMatchesAnalyticSoftmax_largeN() {
        Assumptions.assumeTrue(hasCudaGpu(), "no CUDA GPU available");
        SamplingGpuPhase1.resetForTesting();

        int n = 50_000;
        SamplingGpuPhase1.Request req = request(n);
        req.method = SamplingGpuPhase1.Method.CDF_DEDUP;
        Arrays.fill(req.mIdxPerSample, 0L);
        req.baseSeed = 0x51a7157b0bd1e995L;

        int[] actual = SamplingGpuPhase1.sample(req);
        assertNotNull(actual, "GPU sampling returned fallback on a CUDA node");

        int[] counts = new int[req.totalLambdaStates];
        for (int lIdx : actual) {
            assertTrue(lIdx >= 0 && lIdx < req.totalLambdaStates);
            counts[lIdx]++;
        }

        double[] expected = analyticProbabilities(req, 0L);
        for (int lIdx = 0; lIdx < expected.length; lIdx++) {
            double observed = (double)counts[lIdx]/(double)n;
            double p = expected[lIdx];
            double sixSigma = 6.0*Math.sqrt(Math.max(0.0, p*(1.0 - p)/(double)n));
            double tolerance = Math.max(0.003, sixSigma);
            assertEquals(p, observed, tolerance,
                    "lambda state " + lIdx + " frequency outside Monte Carlo tolerance");
        }
    }

    private static boolean hasCudaGpu() {
        try {
            return !Gpus.get().getGpus().isEmpty();
        } catch (Throwable t) {
            return false;
        }
    }

    private static SamplingGpuPhase1.Request request(int samples) {
        SamplingGpuPhase1.Request req = new SamplingGpuPhase1.Request();
        req.totalLambdaStates = 12;
        req.blockThreads = 128;
        req.numChildren = 2;
        req.invRT = 1.25;
        req.mCounts = new int[] { 2, 3 };
        req.lambdaCounts = new int[] { 3, 4 };
        req.lambdaOnlyMin = new double[req.totalLambdaStates];
        for (int i = 0; i < req.lambdaOnlyMin.length; i++) {
            req.lambdaOnlyMin[i] = -0.17*i + 0.03*(i % 5);
        }

        req.lmLamSlots = new int[] { 0, 1 };
        req.lmMSlots = new int[] { 1, 0 };
        req.lmMCounts = new int[] { 3, 2 };
        req.lmOffsets = new long[] { 0, 9 };
        req.lmMin = new double[17];
        for (int i = 0; i < req.lmMin.length; i++) {
            req.lmMin[i] = 0.11*Math.sin(i + 0.25) - 0.07*Math.cos(0.5*i);
        }

        req.childMSrcAll = new int[] { 0 };
        req.childMStrideAll = new long[] { 4 };
        req.childMTermOff = new int[] { 0, 1 };
        req.childMTermCnt = new int[] { 1, 0 };
        req.childLSrcAll = new int[] { 1, 0 };
        req.childLStrideAll = new long[] { 1, 1 };
        req.childLTermOff = new int[] { 0, 1 };
        req.childLTermCnt = new int[] { 1, 1 };
        req.childTableBase = new long[] { 0, 8 };
        req.childUpperAll = new double[11];
        for (int i = 0; i < req.childUpperAll.length; i++) {
            req.childUpperAll[i] = 0.19*i - 0.04*(i % 3);
        }
        req.childUpperCacheKey = 0x6142d97f2a5b6c31L;

        req.mIdxPerSample = new long[samples];
        for (int s = 0; s < req.mIdxPerSample.length; s++) {
            req.mIdxPerSample[s] = Math.floorMod(splitmix64(0x5eed1234L + s), 6L);
        }
        req.baseSeed = 0x1234abcd55aa7711L;
        req.progress = true;
        req.multiGpu = true;
        req.maxGpus = 4;
        req.minGroupsPerGpu = 1;
        req.persistentContext = true;
        req.residentChildTables = true;
        req.method = SamplingGpuPhase1.Method.GUMBEL;
        return req;
    }

    private static int[] cpuGumbelReference(SamplingGpuPhase1.Request req) {
        int[] out = new int[req.mIdxPerSample.length];
        int[] mLocal = new int[req.mCounts.length];
        int[] lambdaLocal = new int[req.lambdaCounts.length];

        for (int s = 0; s < out.length; s++) {
            decodeState(req.mIdxPerSample[s], req.mCounts, mLocal);
            double bestScore = Double.NEGATIVE_INFINITY;
            int bestIdx = req.totalLambdaStates - 1;
            for (int lIdx = 0; lIdx < req.totalLambdaStates; lIdx++) {
                double eMin = localEnergy(req, lIdx, mLocal, lambdaLocal);
                double fUpper = childSum(req, mLocal, lambdaLocal);
                double logWeight = -eMin*req.invRT + fUpper;
                if (Double.isNaN(logWeight)) {
                    continue;
                }
                double score = logWeight + gumbelNoise(req.baseSeed, s, lIdx);
                if (score > bestScore) {
                    bestScore = score;
                    bestIdx = lIdx;
                }
            }
            out[s] = bestIdx;
        }
        return out;
    }

    private static int[] cpuCdfReference(SamplingGpuPhase1.Request req) {
        int[] out = new int[req.mIdxPerSample.length];
        Map<Long, double[]> cdfs = new HashMap<>();
        for (int s = 0; s < out.length; s++) {
            long mIdx = req.mIdxPerSample[s];
            double[] cdf = cdfs.computeIfAbsent(mIdx, key -> buildCdf(req, key));
            out[s] = sampleFromCdf(cdf, uniform01(req.baseSeed, s));
        }
        return out;
    }

    private static double[] analyticProbabilities(SamplingGpuPhase1.Request req, long mIdx) {
        double[] cdf = buildCdf(req, mIdx);
        double total = cdf[cdf.length - 1];
        assertTrue(total > 0.0 && Double.isFinite(total));
        double[] p = new double[cdf.length];
        double prev = 0.0;
        for (int i = 0; i < cdf.length; i++) {
            p[i] = (cdf[i] - prev)/total;
            prev = cdf[i];
        }
        return p;
    }

    private static double[] buildCdf(SamplingGpuPhase1.Request req, long mIdx) {
        double[] cdf = new double[req.totalLambdaStates];
        int[] mLocal = new int[req.mCounts.length];
        int[] lambdaLocal = new int[req.lambdaCounts.length];
        decodeState(mIdx, req.mCounts, mLocal);

        double maxLog = Double.NEGATIVE_INFINITY;
        for (int lIdx = 0; lIdx < req.totalLambdaStates; lIdx++) {
            double eMin = localEnergy(req, lIdx, mLocal, lambdaLocal);
            double fUpper = childSum(req, mLocal, lambdaLocal);
            double logWeight = -eMin*req.invRT + fUpper;
            cdf[lIdx] = logWeight;
            if (!Double.isNaN(logWeight) && logWeight > maxLog) {
                maxLog = logWeight;
            }
        }

        if (!Double.isFinite(maxLog)) {
            Arrays.fill(cdf, 0.0);
            return cdf;
        }

        double running = 0.0;
        for (int lIdx = 0; lIdx < req.totalLambdaStates; lIdx++) {
            double weight = Math.exp(cdf[lIdx] - maxLog);
            if (!Double.isFinite(weight)) {
                weight = 0.0;
            }
            running += weight;
            cdf[lIdx] = running;
        }
        return cdf;
    }

    private static int sampleFromCdf(double[] cdf, double u) {
        double total = cdf[cdf.length - 1];
        if (!(total > 0.0) || !Double.isFinite(total)) {
            return cdf.length - 1;
        }
        double target = u*total;
        int lo = 0;
        int hi = cdf.length - 1;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (target < cdf[mid]) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    private static double localEnergy(SamplingGpuPhase1.Request req, int lIdx,
                                      int[] mLocal, int[] lambdaLocal) {
        decodeState(lIdx, req.lambdaCounts, lambdaLocal);
        double e = req.lambdaOnlyMin[lIdx];
        for (int p = 0; p < req.lmLamSlots.length; p++) {
            int lamSlot = req.lmLamSlots[p];
            int mSlot = req.lmMSlots[p];
            long off = req.lmOffsets[p]
                    + (long)lambdaLocal[lamSlot]*(long)req.lmMCounts[p]
                    + (long)mLocal[mSlot];
            e += req.lmMin[(int)off];
        }
        return e;
    }

    private static double childSum(SamplingGpuPhase1.Request req, int[] mLocal,
                                   int[] lambdaLocal) {
        double sum = 0.0;
        for (int c = 0; c < req.numChildren; c++) {
            long idx = 0L;
            for (int t = 0; t < req.childMTermCnt[c]; t++) {
                int off = req.childMTermOff[c] + t;
                idx += (long)mLocal[req.childMSrcAll[off]]*req.childMStrideAll[off];
            }
            for (int t = 0; t < req.childLTermCnt[c]; t++) {
                int off = req.childLTermOff[c] + t;
                idx += (long)lambdaLocal[req.childLSrcAll[off]]*req.childLStrideAll[off];
            }
            sum += req.childUpperAll[(int)(req.childTableBase[c] + idx)];
        }
        return sum;
    }

    private static void decodeState(long idx, int[] counts, int[] out) {
        for (int i = counts.length - 1; i >= 0; i--) {
            out[i] = (int)(idx % counts[i]);
            idx /= counts[i];
        }
    }

    private static double uniform01(long baseSeed, int sampleSlot) {
        long h = splitmix64(baseSeed + 0x9E3779B97F4A7C15L*(long)sampleSlot);
        return ((double)(h >>> 11) + 0.5)*(1.0/9007199254740992.0);
    }

    private static double gumbelNoise(long baseSeed, int sampleSlot, int lIdx) {
        long h = splitmix64(baseSeed + 0x9E3779B97F4A7C15L*(long)sampleSlot);
        h = splitmix64(h + (long)lIdx);
        double u = ((double)(h >>> 11) + 0.5)*(1.0/9007199254740992.0);
        return -Math.log(-Math.log(u));
    }

    private static long splitmix64(long x) {
        x += 0x9E3779B97F4A7C15L;
        x = (x ^ (x >>> 30))*0xBF58476D1CE4E5B9L;
        x = (x ^ (x >>> 27))*0x94D049BB133111EBL;
        return x ^ (x >>> 31);
    }
}
