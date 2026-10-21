/*
** This file is part of OSPREY 3.0
**
** OSPREY Protein Redesign Software Version 3.0
** Copyright (C) 2001-2018 Bruce Donald Lab, Duke University
**
** OSPREY is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License version 2
** as published by the Free Software Foundation.
*/

package edu.duke.cs.osprey.branchdp;

import java.math.BigInteger;
import java.util.Arrays;

/**
 * Pure planning and host-side packing for the fully bounded branch-DP GPU
 * fallback.
 *
 * Unlike the legacy flattened-union slicer, this planner gives every parent
 * union-M dimension and every parent-lambda dimension its own rectangular tile
 * extent.  For a child whose fold plan projects onto a subset of those
 * dimensions, the exact maximum packed tile size is therefore the Cartesian
 * product of just that subset.  This is the important property that makes both
 * multi-child row tiling and row-internal lambda tiling predictable before any
 * CUDA allocation is attempted.
 */
final class DPGpuOutOfCore {

    private static final long DIRECT_DOUBLE_STATES = Integer.MAX_VALUE / (long)Double.BYTES;
    private static final long DIRECT_LONG_STATES = Integer.MAX_VALUE / (long)Long.BYTES;
    private static final long OUTPUT_BYTES_PER_STATE = Long.BYTES + 6L * Double.BYTES;
    private static final int GATHER_CHUNK_STATES = 16 * 1024;

    private DPGpuOutOfCore() {
    }

    /** Host reference for the exact partial format used by the CUDA lambda
     * tiler. Kept here so partition/extreme-value semantics can be tested
     * without a GPU and compared directly with device results. */
    static final class LogSumExpPartial {
        double max = Double.NEGATIVE_INFINITY;
        double sumExp = 0.0;

        void add(double value) {
            if (Double.isNaN(value)) {
                max = Double.NaN;
                sumExp = Double.NaN;
                return;
            }
            if (value == Double.NEGATIVE_INFINITY || Double.isNaN(max)) {
                return;
            }
            if (value > max) {
                sumExp = max == Double.NEGATIVE_INFINITY
                        ? 1.0
                        : sumExp * Math.exp(max - value) + 1.0;
                max = value;
            } else {
                sumExp += Math.exp(value - max);
            }
        }

        void merge(LogSumExpPartial other) {
            if (Double.isNaN(max) || Double.isNaN(sumExp)
                    || Double.isNaN(other.max)
                    || Double.isNaN(other.sumExp)) {
                max = Double.NaN;
                sumExp = Double.NaN;
                return;
            }
            if (other.max == Double.NEGATIVE_INFINITY) {
                return;
            }
            if (max == Double.NEGATIVE_INFINITY) {
                max = other.max;
                sumExp = other.sumExp;
                return;
            }
            double mergedMax = Math.max(max, other.max);
            sumExp = sumExp * Math.exp(max - mergedMax)
                    + other.sumExp * Math.exp(other.max - mergedMax);
            max = mergedMax;
        }

        double finish() {
            return max == Double.NEGATIVE_INFINITY || Double.isNaN(max)
                    ? max
                    : max + Math.log(sumExp);
        }
    }

    static final class Plan {
        final int[] mTileExtents;
        final int[] lambdaTileExtents;
        final long mBoxCount;
        final long lambdaBoxCount;
        final long maxMBoxStates;
        final long maxLambdaTileStates;
        final long maxChildPackedStates;
        final long maxChildRowKeys;
        final long maxChildLambdaKeys;
        final int outputStatesPerTile;
        final long estimatedDeviceBytes;
        final long budgetBytes;
        final boolean multiChildRowTiling;
        final boolean lambdaTiling;
        /** Parent lambda mixed-radix strides reused for every lambda box. */
        final long[] lambdaParentStrides;
        /** Child M projection dimensions reused for every M box. */
        final int[][] childMUnionDims;

        Plan(int[] mTileExtents, int[] lambdaTileExtents, Shape shape,
             long mBoxCount, long lambdaBoxCount, long budgetBytes,
             boolean multiChildRowTiling, boolean lambdaTiling,
             int[] lambdaCounts, int[][] childMUnionDims) {
            this.mTileExtents = mTileExtents;
            this.lambdaTileExtents = lambdaTileExtents;
            this.mBoxCount = mBoxCount;
            this.lambdaBoxCount = lambdaBoxCount;
            this.maxMBoxStates = shape.mVolume;
            this.maxLambdaTileStates = shape.lambdaVolume;
            this.maxChildPackedStates = shape.childPackedStates;
            this.maxChildRowKeys = shape.childRowKeys;
            this.maxChildLambdaKeys = shape.childLambdaKeys;
            this.outputStatesPerTile = shape.outputStates;
            this.estimatedDeviceBytes = shape.estimatedBytes;
            this.budgetBytes = budgetBytes;
            this.multiChildRowTiling = multiChildRowTiling;
            this.lambdaTiling = lambdaTiling;
            this.lambdaParentStrides = mixedRadixStrides(lambdaCounts);
            this.childMUnionDims = clone2d(childMUnionDims);
        }
    }

    /**
     * Allocation-free description of the dimensions and fixed metadata used by
     * the bounded planner.  Runtime requests and root-scoring shapes both map to
     * this representation, so admission and cost modeling cannot drift apart.
     */
    static final class PlanningInput {
        final int mPositionCount;
        final int[] lambdaCounts;
        final int[] unionMCounts;
        final int[][] childMUnionDims;
        final int[][] childLambdaDims;
        final int numChildren;
        final long parentFreeStateCount;
        final long childSliceMaxBytes;
        final long outputWorkspaceMaxBytes;
        final long lmTermCount;
        final long lmPairCount;
        final long childMTermTotal;
        final long childLTermTotal;
        final long tripleFactorCount;
        final long tripleTermCount;

        PlanningInput(int mPositionCount, int[] lambdaCounts,
                      int[] unionMCounts, int[][] childMUnionDims,
                      int[][] childLambdaDims, long parentFreeStateCount,
                      long childSliceMaxBytes, long outputWorkspaceMaxBytes,
                      long lmTermCount, long lmPairCount,
                      long childMTermTotal, long childLTermTotal) {
            this(mPositionCount, lambdaCounts, unionMCounts,
                    childMUnionDims, childLambdaDims, parentFreeStateCount,
                    childSliceMaxBytes, outputWorkspaceMaxBytes,
                    lmTermCount, lmPairCount, childMTermTotal,
                    childLTermTotal, 0L, 0L);
        }

        PlanningInput(int mPositionCount, int[] lambdaCounts,
                      int[] unionMCounts, int[][] childMUnionDims,
                      int[][] childLambdaDims, long parentFreeStateCount,
                      long childSliceMaxBytes, long outputWorkspaceMaxBytes,
                      long lmTermCount, long lmPairCount,
                      long childMTermTotal, long childLTermTotal,
                      long tripleFactorCount, long tripleTermCount) {
            this.mPositionCount = mPositionCount;
            this.lambdaCounts = lambdaCounts.clone();
            this.unionMCounts = unionMCounts.clone();
            this.childMUnionDims = clone2d(childMUnionDims);
            this.childLambdaDims = clone2d(childLambdaDims);
            this.numChildren = childMUnionDims.length;
            this.parentFreeStateCount = parentFreeStateCount;
            this.childSliceMaxBytes = childSliceMaxBytes;
            this.outputWorkspaceMaxBytes = outputWorkspaceMaxBytes;
            this.lmTermCount = lmTermCount;
            this.lmPairCount = lmPairCount;
            this.childMTermTotal = childMTermTotal;
            this.childLTermTotal = childLTermTotal;
            this.tripleFactorCount = tripleFactorCount;
            this.tripleTermCount = tripleTermCount;
        }

        static PlanningInput fromRequest(DPGpuFullDP.Request req) {
            if (!supports(req)) {
                return null;
            }
            int[][] childMUnionDims = new int[req.numChildren][];
            int[][] childLambdaDims = new int[req.numChildren][];
            for (int c = 0; c < req.numChildren; c++) {
                int mOff = req.childMTermOff[c];
                int mCnt = req.childMTermCnt[c];
                if (!validRange(mOff, mCnt, req.childMSrcAll.length)) {
                    return null;
                }
                childMUnionDims[c] = new int[mCnt];
                for (int t = 0; t < mCnt; t++) {
                    int parentSlot = req.childMSrcAll[mOff + t];
                    int unionDim = findSlot(req.unionMSlots, parentSlot);
                    if (unionDim < 0) {
                        return null;
                    }
                    childMUnionDims[c][t] = unionDim;
                }

                int lOff = req.childLTermOff[c];
                int lCnt = req.childLTermCnt[c];
                if (!validRange(lOff, lCnt, req.childLSrcAll.length)) {
                    return null;
                }
                childLambdaDims[c] = new int[lCnt];
                for (int t = 0; t < lCnt; t++) {
                    childLambdaDims[c][t] = req.childLSrcAll[lOff + t];
                }
            }
            return new PlanningInput(req.mCounts.length, req.lambdaCounts,
                    req.unionMCounts, childMUnionDims, childLambdaDims,
                    req.parentFreeStateCount, req.childSliceMaxBytes,
                    req.outOfCoreOutputWorkspaceMaxBytes, req.lmRigid.length,
                    req.lmLamSlots.length, req.childMSrcAll.length,
                    req.childLSrcAll.length, req.tripleOffsets.length,
                    req.tripleRigid.length);
        }
    }

    private static final class Shape {
        final boolean feasible;
        final long mVolume;
        final long lambdaVolume;
        final long childPackedStates;
        final long childRowKeys;
        final long childLambdaKeys;
        final int outputStates;
        final long estimatedBytes;

        Shape(boolean feasible, long mVolume, long lambdaVolume,
              long childPackedStates, long childRowKeys, long childLambdaKeys,
              int outputStates, long estimatedBytes) {
            this.feasible = feasible;
            this.mVolume = mVolume;
            this.lambdaVolume = lambdaVolume;
            this.childPackedStates = childPackedStates;
            this.childRowKeys = childRowKeys;
            this.childLambdaKeys = childLambdaKeys;
            this.outputStates = outputStates;
            this.estimatedBytes = estimatedBytes;
        }

        static Shape infeasible() {
            return new Shape(false, Long.MAX_VALUE, Long.MAX_VALUE,
                    Long.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE,
                    0, Long.MAX_VALUE);
        }
    }

    /**
     * Choose a deterministic rectangular M/lambda tile shape under one device's
     * usable-byte budget.  This method deliberately reads no system properties;
     * execution policy and any configured budget cap are applied by the caller.
     */
    static Plan choosePlan(DPGpuFullDP.Request req, long budgetBytes) {
        return choosePlan(PlanningInput.fromRequest(req), budgetBytes);
    }

    static Plan choosePlan(PlanningInput input, long budgetBytes) {
        if (!supports(input) || budgetBytes <= 0L) {
            return null;
        }

        int[] mExtents = new int[input.unionMCounts.length];
        int[] lambdaExtents = new int[input.lambdaCounts.length];
        Arrays.fill(mExtents, 1);
        Arrays.fill(lambdaExtents, 1);

        Shape current = evaluate(input, mExtents, lambdaExtents, budgetBytes);
        if (!current.feasible) {
            return null;
        }
        BigInteger currentTraffic = estimateTrafficBytes(
                input, mExtents, lambdaExtents, current);

        // Coordinate doubling makes planning logarithmic in every cardinality.
        // At each step choose the feasible growth that reduces modeled transfer
        // traffic per added byte, with tile-box reduction as a secondary term.
        // The traffic model includes repeated child H2D gathers, lambda-index /
        // energy uploads, output D2H, and fixed metadata.  This prevents an
        // early high-cardinality dimension from consuming the whole budget while
        // creating a plan that looks small by box count but rereads the same
        // child rows many times.
        int maxSteps = 4 * (mExtents.length + lambdaExtents.length + 1) * 32;
        for (int step = 0; step < maxSteps; step++) {
            boolean bestIsM = false;
            int bestDim = -1;
            int bestExtent = -1;
            Shape bestShape = null;
            BigInteger bestTraffic = null;
            double bestScore = Double.NEGATIVE_INFINITY;

            for (int dim = 0; dim < mExtents.length; dim++) {
                int next = growExtent(mExtents[dim], input.unionMCounts[dim]);
                if (next == mExtents[dim]) {
                    continue;
                }
                int old = mExtents[dim];
                mExtents[dim] = next;
                Shape candidate = evaluate(input, mExtents, lambdaExtents, budgetBytes);
                mExtents[dim] = old;
                if (!candidate.feasible) {
                    continue;
                }
                BigInteger candidateTraffic = estimateTrafficBytes(
                        input, mExtents, lambdaExtents, candidate);
                double score = growthScore(input.unionMCounts[dim], old, next,
                        current.estimatedBytes, candidate.estimatedBytes,
                        currentTraffic, candidateTraffic);
                if (score > bestScore) {
                    bestScore = score;
                    bestShape = candidate;
                    bestTraffic = candidateTraffic;
                    bestIsM = true;
                    bestDim = dim;
                    bestExtent = next;
                }
            }

            for (int dim = 0; dim < lambdaExtents.length; dim++) {
                int next = growExtent(lambdaExtents[dim], input.lambdaCounts[dim]);
                if (next == lambdaExtents[dim]) {
                    continue;
                }
                int old = lambdaExtents[dim];
                lambdaExtents[dim] = next;
                Shape candidate = evaluate(input, mExtents, lambdaExtents, budgetBytes);
                lambdaExtents[dim] = old;
                if (!candidate.feasible) {
                    continue;
                }
                BigInteger candidateTraffic = estimateTrafficBytes(
                        input, mExtents, lambdaExtents, candidate);
                double score = growthScore(input.lambdaCounts[dim], old, next,
                        current.estimatedBytes, candidate.estimatedBytes,
                        currentTraffic, candidateTraffic);
                if (score > bestScore) {
                    bestScore = score;
                    bestShape = candidate;
                    bestTraffic = candidateTraffic;
                    bestIsM = false;
                    bestDim = dim;
                    bestExtent = next;
                }
            }

            if (bestShape == null) {
                break;
            }
            if (bestIsM) {
                mExtents[bestDim] = bestExtent;
            } else {
                lambdaExtents[bestDim] = bestExtent;
            }
            current = bestShape;
            currentTraffic = bestTraffic;
        }

        long mBoxCount = tileBoxCount(input.unionMCounts, mExtents);
        long lambdaBoxCount = tileBoxCount(input.lambdaCounts, lambdaExtents);
        boolean mTiled = false;
        for (int i = 0; i < mExtents.length; i++) {
            if (mExtents[i] < input.unionMCounts[i]) {
                mTiled = true;
                break;
            }
        }
        boolean lambdaTiled = false;
        for (int i = 0; i < lambdaExtents.length; i++) {
            if (lambdaExtents[i] < input.lambdaCounts[i]) {
                lambdaTiled = true;
                break;
            }
        }

        return new Plan(mExtents.clone(), lambdaExtents.clone(), current,
                mBoxCount, lambdaBoxCount, budgetBytes,
                input.numChildren >= 2 && mTiled, lambdaTiled,
                input.lambdaCounts, input.childMUnionDims);
    }

    static long estimateMinimumDeviceBytes(DPGpuFullDP.Request req) {
        return estimateMinimumDeviceBytes(PlanningInput.fromRequest(req));
    }

    static long estimateMinimumDeviceBytes(PlanningInput input) {
        if (!supports(input)) {
            return Long.MAX_VALUE;
        }
        int[] m = new int[input.unionMCounts.length];
        int[] l = new int[input.lambdaCounts.length];
        Arrays.fill(m, 1);
        Arrays.fill(l, 1);
        Shape shape = evaluate(input, m, l, Long.MAX_VALUE);
        if (!shape.feasible) {
            return Long.MAX_VALUE;
        }
        // evaluate(Long.MAX_VALUE) maximizes the output workspace.  The true
        // minimum path needs only one M output state.
        long withoutOutput = shape.estimatedBytes
                - multiply(shape.outputStates, OUTPUT_BYTES_PER_STATE);
        return add(withoutOutput, OUTPUT_BYTES_PER_STATE);
    }

    /**
     * Conservative single-GPU traffic upper bound for the selected OOC plan.
     * It includes packed lower/upper child states, row/lambda key metadata,
     * streamed lambda indices/energies, and the fixed metadata upload. Boundary
     * M boxes may fit more free-M states per output block; using the largest box
     * repeat count deliberately overestimates those cheaper boundary boxes.
     */
    static BigInteger estimateTrafficBytes(PlanningInput input, Plan plan) {
        if (!supports(input) || plan == null) {
            return null;
        }
        long freePerBlock = Math.max(1L,
                plan.outputStatesPerTile / Math.max(1L, plan.maxMBoxStates));
        BigInteger freeRepeats = BigInteger.valueOf(
                ceilDiv(input.parentFreeStateCount, freePerBlock));
        BigInteger mBoxes = BigInteger.valueOf(plan.mBoxCount);
        BigInteger lambdaBoxes = BigInteger.valueOf(plan.lambdaBoxCount);
        BigInteger blockCount = mBoxes.multiply(lambdaBoxes)
                .multiply(freeRepeats);

        BigInteger traffic = BigInteger.valueOf(fixedDeviceBytes(input));
        for (int c = 0; c < input.numChildren; c++) {
            BigInteger rowStates = projectedTileStateSum(input.unionMCounts,
                    plan.mTileExtents, input.childMUnionDims[c]);
            BigInteger lambdaStates = projectedTileStateSum(input.lambdaCounts,
                    plan.lambdaTileExtents, input.childLambdaDims[c]);
            BigInteger packedStates = rowStates.multiply(lambdaStates)
                    .multiply(freeRepeats);
            traffic = traffic.add(packedStates.multiply(
                    BigInteger.valueOf(2L * Double.BYTES)));
            traffic = traffic.add(rowStates.multiply(lambdaBoxes)
                    .multiply(freeRepeats)
                    .multiply(BigInteger.valueOf(Long.BYTES)));
            traffic = traffic.add(lambdaStates.multiply(mBoxes)
                    .multiply(freeRepeats)
                    .multiply(BigInteger.valueOf(Long.BYTES)));
        }

        // Per packed block: packed base, row base/count, lambda base/count.
        traffic = traffic.add(blockCount
                .multiply(BigInteger.valueOf(input.numChildren))
                .multiply(BigInteger.valueOf(3L * Long.BYTES
                        + 2L * Integer.BYTES)));

        BigInteger totalLambdaStates = productBig(input.lambdaCounts);
        traffic = traffic.add(totalLambdaStates.multiply(mBoxes)
                .multiply(freeRepeats)
                .multiply(BigInteger.valueOf(Integer.BYTES
                        + 2L * Double.BYTES)));
        // Every output state is uploaded once as an M index and downloaded
        // once as a lower/upper pair.  These terms are constant across tile
        // shapes, but keeping them in the estimate makes the planner's reported
        // H2D/D2H accounting complete.
        BigInteger outputStates = productBig(input.unionMCounts)
                .multiply(BigInteger.valueOf(input.parentFreeStateCount));
        traffic = traffic.add(outputStates.multiply(
                BigInteger.valueOf(Long.BYTES
                        + 2L * Double.BYTES)));
        return traffic;
    }

    private static BigInteger estimateTrafficBytes(
            PlanningInput input, int[] mExtents, int[] lambdaExtents,
            Shape shape) {
        if (shape == null || !shape.feasible) {
            return null;
        }
        Plan plan = new Plan(
                mExtents.clone(), lambdaExtents.clone(), shape,
                tileBoxCount(input.unionMCounts, mExtents),
                tileBoxCount(input.lambdaCounts, lambdaExtents),
                Long.MAX_VALUE,
                input.numChildren >= 2,
                false, input.lambdaCounts, input.childMUnionDims);
        return estimateTrafficBytes(input, plan);
    }

    private static boolean supports(DPGpuFullDP.Request req) {
        return req != null
                && req.numChildren >= 0
                && req.numChildren <= 64
                && req.mCounts != null
                && req.mCounts.length <= DPGpuFullDP.MAX_EDGE_POSITIONS
                && req.lambdaCounts != null
                && req.lambdaCounts.length <= DPGpuFullDP.MAX_EDGE_POSITIONS
                && req.lambdaOnlyRigid != null
                && req.lambdaOnlyMin != null
                && req.lambdaOnlyRigid.length == req.totalLambdaStates
                && req.lambdaOnlyMin.length == req.totalLambdaStates
                && req.lmRigid != null
                && req.lmMin != null
                && req.lmLamSlots != null
                && req.lmMSlots != null
                && req.lmMCounts != null
                && req.lmOffsets != null
                && req.lmRigid.length == req.lmMin.length
                && req.lmLamSlots.length == req.lmMSlots.length
                && req.lmLamSlots.length == req.lmMCounts.length
                && req.lmLamSlots.length == req.lmOffsets.length
                && req.tripleSlots != null
                && req.tripleStrides != null
                && req.tripleOffsets != null
                && req.tripleRigid != null
                && req.tripleMin != null
                && req.tripleSlots.length == req.tripleOffsets.length * 3L
                && req.tripleStrides.length == req.tripleSlots.length
                && req.tripleRigid.length == req.tripleMin.length
                && req.childMSrcAll != null
                && req.childMStrideAll != null
                && req.childMCountsAll != null
                && req.childMPackedStrideAll != null
                && req.childMTermOff != null
                && req.childMTermCnt != null
                && req.childLSrcAll != null
                && req.childLStrideAll != null
                && req.childLCountsAll != null
                && req.childLPackedStrideAll != null
                && req.childLTermOff != null
                && req.childLTermCnt != null
                && req.childMSrcAll.length == req.childMStrideAll.length
                && req.childMSrcAll.length == req.childMCountsAll.length
                && req.childMSrcAll.length == req.childMPackedStrideAll.length
                && req.childLSrcAll.length == req.childLStrideAll.length
                && req.childLSrcAll.length == req.childLCountsAll.length
                && req.childLSrcAll.length == req.childLPackedStrideAll.length
                && req.childMTermOff.length == req.numChildren
                && req.childMTermCnt.length == req.numChildren
                && req.childLTermOff.length == req.numChildren
                && req.childLTermCnt.length == req.numChildren
                && req.childTables != null
                && req.childTables.length == req.numChildren
                && req.unionMSlots != null
                && req.unionMCounts != null
                && req.unionMParentStrides != null
                && req.unionMSlots.length == req.unionMCounts.length
                && req.unionMSlots.length == req.unionMParentStrides.length
                && req.freeMCounts != null
                && req.freeMParentStrides != null
                && req.freeMCounts.length == req.freeMParentStrides.length
                && req.parentFreeStateCount > 0L
                && req.totalLambdaStates > 0;
    }

    private static boolean supports(PlanningInput input) {
        if (input == null
                || input.mPositionCount < 0
                || input.mPositionCount > DPGpuFullDP.MAX_EDGE_POSITIONS
                || input.lambdaCounts.length > DPGpuFullDP.MAX_EDGE_POSITIONS
                || input.unionMCounts.length > input.mPositionCount
                || input.numChildren < 0 || input.numChildren > 64
                || input.childLambdaDims.length != input.numChildren
                || input.parentFreeStateCount <= 0L
                || input.outputWorkspaceMaxBytes <= 0L
                || input.lmTermCount < 0L
                || input.lmTermCount > Integer.MAX_VALUE
                || input.lmPairCount < 0L
                || input.lmPairCount > Integer.MAX_VALUE
                || input.childMTermTotal < 0L
                || input.childMTermTotal > Integer.MAX_VALUE
                || input.childLTermTotal < 0L
                || input.childLTermTotal > Integer.MAX_VALUE
                || !positive(input.unionMCounts)
                || !positive(input.lambdaCounts)
                || product(input.lambdaCounts) > Integer.MAX_VALUE) {
            return false;
        }
        for (int c = 0; c < input.numChildren; c++) {
            if (!validDimensions(input.childMUnionDims[c],
                    input.unionMCounts.length)
                    || !validDimensions(input.childLambdaDims[c],
                    input.lambdaCounts.length)) {
                return false;
            }
        }
        return true;
    }

    private static Shape evaluate(PlanningInput input, int[] mExtents,
                                  int[] lambdaExtents, long budgetBytes) {
        long mVolume = product(mExtents);
        long lambdaVolume = product(lambdaExtents);
        if (mVolume <= 0L || lambdaVolume <= 0L
                || lambdaVolume > DIRECT_DOUBLE_STATES) {
            return Shape.infeasible();
        }

        long childPacked = 0L;
        long childRows = 0L;
        long childLambdas = 0L;
        for (int c = 0; c < input.numChildren; c++) {
            long rows = projectedStates(mExtents, input.childMUnionDims[c]);
            long lambdas = projectedStates(lambdaExtents,
                    input.childLambdaDims[c]);
            childRows = add(childRows, rows);
            childLambdas = add(childLambdas, lambdas);
            childPacked = add(childPacked, multiply(rows, lambdas));
        }
        if (childPacked > DIRECT_DOUBLE_STATES
                || childRows > DIRECT_LONG_STATES
                || childLambdas > DIRECT_LONG_STATES) {
            return Shape.infeasible();
        }

        long childBytes = multiply(childPacked, 2L * Double.BYTES);
        long childCap = input.childSliceMaxBytes > 0L
                ? input.childSliceMaxBytes
                : Long.MAX_VALUE;
        if (childBytes > childCap) {
            return Shape.infeasible();
        }

        long bytes = fixedDeviceBytes(input);
        bytes = addBytes(bytes, input.numChildren, Long.BYTES); // childPackedBase
        bytes = addBytes(bytes, input.numChildren, Long.BYTES); // childRowKeyBase
        bytes = addBytes(bytes, input.numChildren, Integer.BYTES); // childRowKeyCount
        bytes = addBytes(bytes, childRows, Long.BYTES); // childRowKeysAll
        bytes = addBytes(bytes, input.numChildren, Long.BYTES); // childLambdaKeyBase
        bytes = addBytes(bytes, input.numChildren, Integer.BYTES); // childLambdaKeyCount
        bytes = addBytes(bytes, childLambdas, Long.BYTES); // childLambdaKeysAll
        bytes = addBytes(bytes, childPacked, Double.BYTES); // child lower
        bytes = addBytes(bytes, childPacked, Double.BYTES); // child upper
        bytes = addBytes(bytes, lambdaVolume, Integer.BYTES); // global lambda indices
        bytes = addBytes(bytes, lambdaVolume, Double.BYTES); // lambda-only rigid tile
        bytes = addBytes(bytes, lambdaVolume, Double.BYTES); // lambda-only min tile
        if (bytes >= budgetBytes) {
            return Shape.infeasible();
        }

        // The legacy full-DP mStateChunk is deliberately not a limit here.
        // OOC gathers depend only on the M/lambda box, not on the free-M output
        // block.  Capping this workspace at the old 1M-state chunk caused the
        // same 1--2 GiB child tile to be gathered/uploaded roughly 20--25 times
        // on the production shapes.  Use the explicit combined workspace cap
        // instead, still bounded by actual usable VRAM and every direct-buffer
        // element limit below.
        long outputWorkspaceBytes = Math.min(budgetBytes - bytes,
                input.outputWorkspaceMaxBytes);
        long maxOutput = outputWorkspaceBytes / OUTPUT_BYTES_PER_STATE;
        maxOutput = Math.min(maxOutput, DIRECT_DOUBLE_STATES);
        long allFreeOutputs = multiply(mVolume, input.parentFreeStateCount);
        maxOutput = Math.min(maxOutput, allFreeOutputs);
        if (maxOutput < mVolume || maxOutput <= 0L || maxOutput > Integer.MAX_VALUE) {
            return Shape.infeasible();
        }
        int outputStates = (int)maxOutput;
        long estimated = add(bytes, multiply(outputStates, OUTPUT_BYTES_PER_STATE));
        if (estimated > budgetBytes) {
            return Shape.infeasible();
        }
        return new Shape(true, mVolume, lambdaVolume, childPacked,
                childRows, childLambdas, outputStates, estimated);
    }

    /** Fixed buffers that remain resident while all M/lambda tiles execute. */
    private static long fixedDeviceBytes(PlanningInput input) {
        long bytes = 0L;
        bytes = addBytes(bytes, input.mPositionCount, Integer.BYTES);
        bytes = addBytes(bytes, input.lambdaCounts.length, Integer.BYTES);
        bytes = addBytes(bytes, input.lmTermCount, Double.BYTES);
        bytes = addBytes(bytes, input.lmTermCount, Double.BYTES);
        bytes = addBytes(bytes, input.lmPairCount, Integer.BYTES);
        bytes = addBytes(bytes, input.lmPairCount, Integer.BYTES);
        bytes = addBytes(bytes, input.lmPairCount, Integer.BYTES);
        bytes = addBytes(bytes, input.lmPairCount, Long.BYTES);
        bytes = addBytes(bytes, 3L * input.tripleFactorCount, Integer.BYTES);
        bytes = addBytes(bytes, 3L * input.tripleFactorCount, Long.BYTES);
        bytes = addBytes(bytes, input.tripleFactorCount, Long.BYTES);
        bytes = addBytes(bytes, input.tripleTermCount, Double.BYTES);
        bytes = addBytes(bytes, input.tripleTermCount, Double.BYTES);
        bytes = addBytes(bytes, input.childMTermTotal, Integer.BYTES);
        bytes = addBytes(bytes, input.childMTermTotal, Long.BYTES);
        bytes = addBytes(bytes, input.numChildren, Integer.BYTES);
        bytes = addBytes(bytes, input.numChildren, Integer.BYTES);
        bytes = addBytes(bytes, input.childLTermTotal, Integer.BYTES);
        bytes = addBytes(bytes, input.childLTermTotal, Long.BYTES);
        bytes = addBytes(bytes, input.numChildren, Integer.BYTES);
        bytes = addBytes(bytes, input.numChildren, Integer.BYTES);
        return bytes;
    }

    private static double growthScore(int count, int oldExtent, int newExtent,
                                      long oldBytes, long newBytes,
                                      BigInteger oldTraffic,
                                      BigInteger newTraffic) {
        long oldTiles = ceilDiv(count, oldExtent);
        long newTiles = ceilDiv(count, newExtent);
        double boxBenefit = Math.log((double)oldTiles / (double)newTiles);
        long added = Math.max(1L, newBytes - oldBytes);
        double benefit = boxBenefit * 0.1;
        if (oldTraffic != null && newTraffic != null
                && oldTraffic.signum() > 0 && newTraffic.signum() > 0) {
            benefit += logBigInteger(oldTraffic) - logBigInteger(newTraffic);
        } else {
            benefit += boxBenefit;
        }
        return benefit / Math.log1p((double)added);
    }

    private static double logBigInteger(BigInteger value) {
        int bitLength = value.bitLength();
        if (bitLength <= 53) {
            return Math.log(value.doubleValue());
        }
        int shift = bitLength - 53;
        long leading = value.shiftRight(shift).longValue();
        return Math.log((double)leading) + shift * Math.log(2.0);
    }

    private static int growExtent(int current, int count) {
        if (current >= count) {
            return count;
        }
        long doubled = Math.max((long)current + 1L, 2L * current);
        return (int)Math.min((long)count, doubled);
    }

    private static long projectedStates(int[] extents, int[] dimensions) {
        long states = 1L;
        for (int dimension : dimensions) {
            states = multiply(states, extents[dimension]);
        }
        return states;
    }

    static final class TensorTile {
        final int[] starts;
        final int[] sizes;
        final long volume;

        TensorTile(int[] starts, int[] sizes, long volume) {
            this.starts = starts;
            this.sizes = sizes;
            this.volume = volume;
        }
    }

    static TensorTile tileForOrdinal(int[] counts, int[] extents, long ordinal) {
        long boxes = tileBoxCount(counts, extents);
        if (ordinal < 0L || ordinal >= boxes) {
            throw new IndexOutOfBoundsException("tile ordinal " + ordinal
                    + " outside [0," + boxes + ")");
        }
        int[] starts = new int[counts.length];
        int[] sizes = new int[counts.length];
        long remaining = ordinal;
        long volume = 1L;
        for (int i = counts.length - 1; i >= 0; i--) {
            long nBoxes = ceilDiv(counts[i], extents[i]);
            long box = remaining % nBoxes;
            remaining /= nBoxes;
            long start = box * (long)extents[i];
            starts[i] = (int)start;
            sizes[i] = (int)Math.min((long)extents[i], counts[i] - start);
            volume = multiply(volume, sizes[i]);
        }
        return new TensorTile(starts, sizes, volume);
    }

    static final class MBox {
        final TensorTile tile;
        final long[] unionParentBases;
        final long[][] childRowKeys;

        MBox(TensorTile tile, long[] unionParentBases, long[][] childRowKeys) {
            this.tile = tile;
            this.unionParentBases = unionParentBases;
            this.childRowKeys = childRowKeys;
        }
    }

    static MBox buildMBox(DPGpuFullDP.Request req, Plan plan, long boxOrdinal) {
        TensorTile tile = tileForOrdinal(req.unionMCounts, plan.mTileExtents, boxOrdinal);
        if (tile.volume > Integer.MAX_VALUE) {
            throw new DPGpuFullDP.GpuMemoryExceededException(
                    "Out-of-core M box exceeds Java index limits: " + tile.volume);
        }
        int states = (int)tile.volume;
        long[] unionParentBases = new long[states];
        long[][] candidates = new long[req.numChildren][states];
        int[] local = new int[tile.sizes.length];
        int[] unionValues = new int[tile.sizes.length];

        for (int state = 0; state < states; state++) {
            decodeState(state, tile.sizes, local);
            long parent = 0L;
            for (int u = 0; u < unionValues.length; u++) {
                int value = tile.starts[u] + local[u];
                unionValues[u] = value;
                parent += (long)value * req.unionMParentStrides[u];
            }
            unionParentBases[state] = parent;
            for (int c = 0; c < req.numChildren; c++) {
                long rowKey = 0L;
                int off = req.childMTermOff[c];
                for (int t = 0; t < req.childMTermCnt[c]; t++) {
                    rowKey += (long)unionValues[plan.childMUnionDims[c][t]]
                            * req.childMPackedStrideAll[off + t];
                }
                candidates[c][state] = rowKey;
            }
        }

        long[][] keys = new long[req.numChildren][];
        for (int c = 0; c < req.numChildren; c++) {
            keys[c] = sortedUnique(candidates[c]);
        }
        return new MBox(tile, unionParentBases, keys);
    }

    static final class OutputBlock {
        final long[] mIndices;

        OutputBlock(long[] mIndices) {
            this.mIndices = mIndices;
        }
    }

    static OutputBlock buildOutputBlock(DPGpuFullDP.Request req, MBox box,
                                        long freeStart, int freeCount) {
        if (freeStart < 0L || freeCount < 0
                || freeStart + (long)freeCount > req.parentFreeStateCount) {
            throw new IndexOutOfBoundsException("free-M range [" + freeStart + ","
                    + (freeStart + freeCount) + ") outside [0,"
                    + req.parentFreeStateCount + ")");
        }
        long outputCount = multiply(box.tile.volume, freeCount);
        if (outputCount > Integer.MAX_VALUE) {
            throw new DPGpuFullDP.GpuMemoryExceededException(
                    "Out-of-core output block exceeds Java index limits: " + outputCount);
        }
        long[] freeOffsets = new long[freeCount];
        int[] freeLocal = new int[req.freeMCounts.length];
        for (int i = 0; i < freeCount; i++) {
            long key = freeStart + i;
            decodeState(key, req.freeMCounts, freeLocal);
            long offset = 0L;
            for (int f = 0; f < freeLocal.length; f++) {
                offset += (long)freeLocal[f] * req.freeMParentStrides[f];
            }
            freeOffsets[i] = offset;
        }

        long[] mIndices = new long[(int)outputCount];
        int out = 0;
        for (long unionBase : box.unionParentBases) {
            for (long freeOffset : freeOffsets) {
                mIndices[out++] = unionBase + freeOffset;
            }
        }
        return new OutputBlock(mIndices);
    }

    static final class LambdaBox {
        final TensorTile tile;
        final int[] lambdaIndices;
        final long[][] childLambdaKeys;

        LambdaBox(TensorTile tile, int[] lambdaIndices, long[][] childLambdaKeys) {
            this.tile = tile;
            this.lambdaIndices = lambdaIndices;
            this.childLambdaKeys = childLambdaKeys;
        }
    }

    static LambdaBox buildLambdaBox(DPGpuFullDP.Request req, Plan plan, long boxOrdinal) {
        TensorTile tile = tileForOrdinal(req.lambdaCounts,
                plan.lambdaTileExtents, boxOrdinal);
        if (tile.volume > Integer.MAX_VALUE) {
            throw new DPGpuFullDP.GpuMemoryExceededException(
                    "Out-of-core lambda box exceeds Java index limits: " + tile.volume);
        }
        int states = (int)tile.volume;
        int[] lambdaIndices = new int[states];
        long[][] candidates = new long[req.numChildren][states];
        long[] parentStrides = plan.lambdaParentStrides;
        int[] local = new int[tile.sizes.length];
        int[] lambdaValues = new int[tile.sizes.length];

        for (int state = 0; state < states; state++) {
            decodeState(state, tile.sizes, local);
            long parent = 0L;
            for (int l = 0; l < lambdaValues.length; l++) {
                int value = tile.starts[l] + local[l];
                lambdaValues[l] = value;
                parent += (long)value * parentStrides[l];
            }
            if (parent < 0L || parent >= req.totalLambdaStates) {
                throw new IllegalStateException("Lambda tile produced invalid parent index "
                        + parent + " of " + req.totalLambdaStates);
            }
            lambdaIndices[state] = (int)parent;
            for (int c = 0; c < req.numChildren; c++) {
                long key = 0L;
                int off = req.childLTermOff[c];
                for (int t = 0; t < req.childLTermCnt[c]; t++) {
                    key += (long)lambdaValues[req.childLSrcAll[off + t]]
                            * req.childLPackedStrideAll[off + t];
                }
                candidates[c][state] = key;
            }
        }

        long[][] keys = new long[req.numChildren][];
        for (int c = 0; c < req.numChildren; c++) {
            keys[c] = sortedUnique(candidates[c]);
        }
        return new LambdaBox(tile, lambdaIndices, keys);
    }

    static final class PackedChildBlock {
        final long[] childPackedBase;
        final long[] childRowKeyBase;
        final int[] childRowKeyCount;
        final long[] childRowKeysAll;
        final long[] childLambdaKeyBase;
        final int[] childLambdaKeyCount;
        final long[] childLambdaKeysAll;
        final long packedStates;
        final double[] lowerPacked;
        final double[] upperPacked;

        PackedChildBlock(long[] childPackedBase,
                         long[] childRowKeyBase, int[] childRowKeyCount,
                         long[] childRowKeysAll,
                         long[] childLambdaKeyBase, int[] childLambdaKeyCount,
                         long[] childLambdaKeysAll,
                         long packedStates, double[] lowerPacked,
                         double[] upperPacked) {
            this.childPackedBase = childPackedBase;
            this.childRowKeyBase = childRowKeyBase;
            this.childRowKeyCount = childRowKeyCount;
            this.childRowKeysAll = childRowKeysAll;
            this.childLambdaKeyBase = childLambdaKeyBase;
            this.childLambdaKeyCount = childLambdaKeyCount;
            this.childLambdaKeysAll = childLambdaKeysAll;
            this.packedStates = packedStates;
            this.lowerPacked = lowerPacked;
            this.upperPacked = upperPacked;
        }
    }

    /** CPU-testable heap packing of one rectangular M x lambda child tile. */
    static PackedChildBlock buildPackedChildBlock(DPGpuFullDP.Request req,
                                                   MBox mBox,
                                                   LambdaBox lambdaBox) {
        long rowKeyTotal = 0L;
        long lambdaKeyTotal = 0L;
        long packedStates = 0L;
        for (int c = 0; c < req.numChildren; c++) {
            long rows = mBox.childRowKeys[c].length;
            long lambdas = lambdaBox.childLambdaKeys[c].length;
            rowKeyTotal = add(rowKeyTotal, rows);
            lambdaKeyTotal = add(lambdaKeyTotal, lambdas);
            packedStates = add(packedStates, multiply(rows, lambdas));
        }
        if (rowKeyTotal > Integer.MAX_VALUE || lambdaKeyTotal > Integer.MAX_VALUE
                || packedStates > Integer.MAX_VALUE) {
            throw new DPGpuFullDP.GpuMemoryExceededException(
                    "Out-of-core packed child block exceeds Java index limits: rows="
                            + rowKeyTotal + ", lambdas=" + lambdaKeyTotal
                            + ", states=" + packedStates);
        }

        long[] packedBase = new long[req.numChildren];
        long[] rowBase = new long[req.numChildren];
        int[] rowCount = new int[req.numChildren];
        long[] rowKeys = new long[(int)rowKeyTotal];
        long[] lambdaBase = new long[req.numChildren];
        int[] lambdaCount = new int[req.numChildren];
        long[] lambdaKeys = new long[(int)lambdaKeyTotal];
        double[] lower = new double[(int)packedStates];
        double[] upper = new double[(int)packedStates];

        long packedOff = 0L;
        int rowOff = 0;
        int lambdaOff = 0;
        for (int c = 0; c < req.numChildren; c++) {
            long[] childRows = mBox.childRowKeys[c];
            long[] childLambdas = lambdaBox.childLambdaKeys[c];
            packedBase[c] = packedOff;
            rowBase[c] = rowOff;
            rowCount[c] = childRows.length;
            System.arraycopy(childRows, 0, rowKeys, rowOff, childRows.length);
            rowOff += childRows.length;
            lambdaBase[c] = lambdaOff;
            lambdaCount[c] = childLambdas.length;
            System.arraycopy(childLambdas, 0, lambdaKeys, lambdaOff,
                    childLambdas.length);
            lambdaOff += childLambdas.length;

            fillChild(req, c, childRows, childLambdas, packedOff, lower, upper);
            packedOff += multiply(childRows.length, childLambdas.length);
        }
        return new PackedChildBlock(packedBase, rowBase, rowCount, rowKeys,
                lambdaBase, lambdaCount, lambdaKeys, packedStates, lower, upper);
    }

    private static void fillChild(DPGpuFullDP.Request req, int child,
                                  long[] rowKeys, long[] lambdaKeys,
                                  long destinationBase,
                                  double[] lower, double[] upper) {
        long[] rowOriginalBases = new long[rowKeys.length];
        for (int i = 0; i < rowKeys.length; i++) {
            rowOriginalBases[i] = childOriginalRowBase(req, child, rowKeys[i]);
        }
        long[] lambdaOriginalOffsets = new long[lambdaKeys.length];
        for (int i = 0; i < lambdaKeys.length; i++) {
            lambdaOriginalOffsets[i] = childOriginalLambdaOffset(req, child,
                    lambdaKeys[i]);
        }
        long childStates = multiply(rowKeys.length, lambdaKeys.length);
        long chunks = ceilDiv(childStates, GATHER_CHUNK_STATES);
        DPGpuFullDP.parallelGatherChunks(chunks, chunk -> {
            long start = chunk * GATHER_CHUNK_STATES;
            long end = Math.min(childStates, start + GATHER_CHUNK_STATES);
            int lambdaCount = lambdaKeys.length;
            long row = start / lambdaCount;
            int lambda = (int)(start - row * lambdaCount);
            for (long linear = start; linear < end; linear++) {
                long original = rowOriginalBases[(int)row]
                        + lambdaOriginalOffsets[lambda];
                int destination = (int)(destinationBase + linear);
                req.childTables[child].readPair(original, lower, upper, destination);
                lambda++;
                if (lambda == lambdaCount) {
                    lambda = 0;
                    row++;
                }
            }
        });
    }

    private static long childOriginalRowBase(DPGpuFullDP.Request req, int child,
                                             long rowKey) {
        long index = 0L;
        int off = req.childMTermOff[child];
        for (int t = 0; t < req.childMTermCnt[child]; t++) {
            long packedStride = req.childMPackedStrideAll[off + t];
            int parentSlot = req.childMSrcAll[off + t];
            int digit = (int)((rowKey / packedStride)
                    % (long)req.mCounts[parentSlot]);
            index += (long)digit * req.childMStrideAll[off + t];
        }
        return index;
    }

    private static long childOriginalLambdaOffset(DPGpuFullDP.Request req,
                                                   int child, long lambdaKey) {
        long index = 0L;
        int off = req.childLTermOff[child];
        for (int t = 0; t < req.childLTermCnt[child]; t++) {
            long packedStride = req.childLPackedStrideAll[off + t];
            int parentSlot = req.childLSrcAll[off + t];
            int digit = (int)((lambdaKey / packedStride)
                    % (long)req.lambdaCounts[parentSlot]);
            index += (long)digit * req.childLStrideAll[off + t];
        }
        return index;
    }

    private static long[] sortedUnique(long[] values) {
        if (values.length == 0) {
            return values;
        }
        Arrays.sort(values);
        int unique = 1;
        for (int i = 1; i < values.length; i++) {
            if (values[i] != values[unique - 1]) {
                values[unique++] = values[i];
            }
        }
        return Arrays.copyOf(values, unique);
    }

    private static long[] mixedRadixStrides(int[] counts) {
        long[] strides = new long[counts.length];
        long stride = 1L;
        for (int i = counts.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride = multiply(stride, counts[i]);
        }
        return strides;
    }

    private static void decodeState(long state, int[] counts, int[] out) {
        for (int i = counts.length - 1; i >= 0; i--) {
            out[i] = (int)(state % counts[i]);
            state /= counts[i];
        }
    }

    private static int[][] clone2d(int[][] values) {
        int[][] copy = new int[values.length][];
        for (int i = 0; i < values.length; i++) {
            copy[i] = values[i].clone();
        }
        return copy;
    }

    private static boolean validRange(int offset, int count, int length) {
        return offset >= 0 && count >= 0 && offset <= length - count;
    }

    private static boolean positive(int[] values) {
        for (int value : values) {
            if (value <= 0) {
                return false;
            }
        }
        return true;
    }

    private static boolean validDimensions(int[] dimensions, int count) {
        if (dimensions == null) {
            return false;
        }
        for (int dimension : dimensions) {
            if (dimension < 0 || dimension >= count) {
                return false;
            }
        }
        return true;
    }

    private static BigInteger projectedTileStateSum(int[] counts,
                                                     int[] extents,
                                                     int[] projectedDimensions) {
        boolean[] projected = new boolean[counts.length];
        for (int dimension : projectedDimensions) {
            projected[dimension] = true;
        }
        BigInteger states = BigInteger.ONE;
        for (int i = 0; i < counts.length; i++) {
            long factor = projected[i]
                    ? counts[i]
                    : ceilDiv(counts[i], extents[i]);
            states = states.multiply(BigInteger.valueOf(factor));
        }
        return states;
    }

    private static BigInteger productBig(int[] values) {
        BigInteger product = BigInteger.ONE;
        for (int value : values) {
            product = product.multiply(BigInteger.valueOf(value));
        }
        return product;
    }

    private static int findSlot(int[] slots, int target) {
        for (int i = 0; i < slots.length; i++) {
            if (slots[i] == target) {
                return i;
            }
        }
        return -1;
    }

    private static long tileBoxCount(int[] counts, int[] extents) {
        long boxes = 1L;
        for (int i = 0; i < counts.length; i++) {
            boxes = multiply(boxes, ceilDiv(counts[i], extents[i]));
        }
        return boxes;
    }

    private static long product(int[] values) {
        long out = 1L;
        for (int value : values) {
            out = multiply(out, value);
        }
        return out;
    }

    private static long ceilDiv(long value, long divisor) {
        return value / divisor + (value % divisor == 0L ? 0L : 1L);
    }

    private static long addBytes(long current, long count, int elementBytes) {
        return add(current, multiply(Math.max(1L, count), elementBytes));
    }

    private static long add(long a, long b) {
        if (a < 0L || b < 0L || a > Long.MAX_VALUE - b) {
            return Long.MAX_VALUE;
        }
        return a + b;
    }

    private static long multiply(long a, long b) {
        if (a < 0L || b < 0L) {
            return Long.MAX_VALUE;
        }
        if (a == 0L || b == 0L) {
            return 0L;
        }
        if (a > Long.MAX_VALUE / b) {
            return Long.MAX_VALUE;
        }
        return a * b;
    }
}
