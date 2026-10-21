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
import edu.duke.cs.osprey.confspace.SimpleConfSpace;
import edu.duke.cs.osprey.ematrix.EnergyMatrix;
import edu.duke.cs.osprey.energy.ConfEnergyCalculator;
import edu.duke.cs.osprey.kstar.pfunc.BoltzmannCalculator;
import edu.duke.cs.osprey.kstar.pfunc.PartitionFunction;
import edu.duke.cs.osprey.parallelism.Parallelism;
import edu.duke.cs.osprey.tools.MathTools;

import java.math.BigDecimal;
import java.util.*;

/** Branch decomposition, resource planning, and DP bounds for PACK*. */
public abstract class BranchDpBackend implements PartitionFunction.WithConfDB {
    private static final double TESS_FALLBACK_THRESHOLD = 0.5;
    private static final String ROOT_SPLIT_MAX_FSET_PROPERTY = "branchdp.rootSplit.maxFset";
    private static final String ROOT_SPLIT_HOST_BUDGET_PROPERTY = "branchdp.rootSplit.hostBudgetBytes";
    private static final double DEFAULT_ROOT_SPLIT_HOST_HEAP_FRACTION = 0.70;
    protected final SimpleConfSpace confSpace;
    protected final ConfEnergyCalculator minimizingEcalc;
    protected final Parallelism parallelism;
    protected final BoltzmannCalculator bc = new BoltzmannCalculator(PartitionFunction.decimalPrecision);
    protected final String stateName;
    protected EnergyMatrix branchRigidEmat;
    protected EnergyMatrix branchMinimizingEmat;
    protected InteractionGraph interactionGraph;
    protected RootedTreeNode rootedRoot;
    protected RootedTreeEdge rootedRootEdge;
    protected RCs searchRCs;
    protected boolean dpTablesReady;
    protected BigDecimal flatSumZLower = BigDecimal.ZERO;
    protected BigDecimal flatSumZUpper = MathTools.BigPositiveInfinity;
    protected PartitionFunction.Values values = new Values();
    protected double targetEpsilon;
    protected double epsilonBound = 1.0;
    protected int totalMinimizations;
    protected int numConfsEnergied;
    private Status status = Status.Estimating;
    private boolean closed;

    public static class Values extends PartitionFunction.Values {
        public Values() { pstar = MathTools.BigPositiveInfinity; }
        @Override public BigDecimal calcUpperBound() { return pstar; }
        @Override public BigDecimal calcLowerBound() { return qstar; }
        @Override public double getEffectiveEpsilon() {
            if (pstar == MathTools.BigPositiveInfinity) return 1.0;
            if (pstar.signum() == 0 && qstar.signum() == 0) return 0.0;
            return MathTools.bigDivide(pstar.subtract(qstar), pstar, decimalPrecision).doubleValue();
        }
    }

    protected BranchDpBackend(SimpleConfSpace confSpace, EnergyMatrix rigidEmat,
            EnergyMatrix minimizingEmat, ConfEnergyCalculator minimizingEcalc,
            RCs rcs, Parallelism parallelism, String stateName) {
        this.confSpace = confSpace;
        this.minimizingEcalc = minimizingEcalc;
        this.parallelism = parallelism == null ? Parallelism.makeCpu(1) : parallelism;
        this.stateName = stateName == null || stateName.isBlank() ? "standalone" : stateName.trim();
        this.searchRCs = rcs;
        double residualBudget = Math.max(0.0,
                getConfigDouble("branchdp.cutoff.residualBudget", 0.5));
        interactionGraph = InteractionGraph.buildWithResidualBudget(
                confSpace, rigidEmat, minimizingEmat, rcs, residualBudget, true);
        branchRigidEmat = sparseMatrix(rigidEmat);
        branchMinimizingEmat = sparseMatrix(minimizingEmat);
        BranchDecomposition decomposition = new BranchDecomposition(interactionGraph);
        decomposition.compute();
        RootingCandidate selected = selectConfiguredRoot(
                decomposition, rcs, true);
        if (selected == null) return;
        rootedRoot = selected.root;
        rootedRootEdge = rootedRoot.getLeftChild().getChildOfEdge();
        double rt = BoltzmannCalculator.RClassic * BoltzmannCalculator.TClassic;
        RootedTreeEdge.postOrderInitIncremental(rootedRoot,
                branchRigidEmat, branchMinimizingEmat, interactionGraph, rt, false);
        RootedTreeEdge.postOrderComputeFullDP(rootedRoot);
        dpTablesReady = true;
        flatSumZLower = fromLog(rootedRootEdge.getLogZLower(0), rt);
        flatSumZUpper = fromLog(rootedRootEdge.getLogZUpper(0), rt);
        updateBound();
    }

    private EnergyMatrix sparseMatrix(EnergyMatrix source) {
        EnergyMatrix matrix = new EnergyMatrix(source);
        for (int i=0;i<matrix.getNumPos();i++) for (int j=0;j<i;j++) {
            if (interactionGraph.hasEdge(i,j)) continue;
            for (int a=0;a<matrix.getNumConfAtPos(i);a++)
                for (int b=0;b<matrix.getNumConfAtPos(j);b++) matrix.setPairwise(i,a,j,b,0.0);
        }
        return matrix;
    }

    private BigDecimal fromLog(double log, double rt) {
        if (log == Double.NEGATIVE_INFINITY) return BigDecimal.ZERO;
        if (log == Double.POSITIVE_INFINITY) return MathTools.BigPositiveInfinity;
        if (Double.isNaN(log)) throw new IllegalStateException("DP returned a NaN partition function");
        return bc.calcPrecise(-rt * log);
    }

    protected ConfEnergyCalculator getMinimizingEcalc() { return minimizingEcalc; }
    protected void updateBound() {
        epsilonBound = computeEpsilon(flatSumZLower, flatSumZUpper);
    }
    @Override public void init(double epsilon) {
        targetEpsilon = epsilon;
        values = new Values();
        status = Status.Estimating;
        numConfsEnergied = 0;
    }
    @Override public PartitionFunction.Values getValues() { return values; }
    protected void setStatus(Status value) { status = value; }
    @Override public Status getStatus() { return status; }
    @Override public int getNumConfsEvaluated() { return numConfsEnergied; }
    @Override public int getParallelism() { return parallelism.getParallelism(); }
    @Override public void setReportProgress(boolean value) {}
    @Override public void setConfListener(ConfListener listener) {}
    @Override public void setStabilityThreshold(BigDecimal threshold) {}
    @Override public void setConfDB(edu.duke.cs.osprey.confspace.ConfDB db,
            edu.duke.cs.osprey.confspace.ConfDB.Key key) {}
    @Override public void compute(int maxNumConfs) {
        if (closed) throw new IllegalStateException("partition function is closed");
        if (!dpTablesReady) { computeWithoutBranchDecomposition(maxNumConfs); return; }
        beforeBranchDecompositionSearch(maxNumConfs);
    }
    protected abstract void computeWithoutBranchDecomposition(int maxNumConfs);
    protected abstract boolean beforeBranchDecompositionSearch(int maxNumConfs);
    @Override public PartitionFunction.Result makeResult() {
        PartitionFunction.Result result = new PartitionFunction.Result(getStatus(),getValues(),getNumConfsEvaluated());
        attachExtraStats(result);
        return result;
    }
    protected void attachExtraStats(PartitionFunction.Result result) {}
    protected void finishWithCurrentDPBounds(String message) {
        System.out.println(message);
        values.qstar = flatSumZLower;
        values.pstar = flatSumZUpper;
        values.qprime = flatSumZUpper.subtract(flatSumZLower);
        if (epsilonBound <= targetEpsilon) setStatus(Status.Estimated);
    }
    public synchronized void releaseLargeMemory() {
        if (closed) return;
        closed = true;
        RootedTreeEdge.postOrderReleaseLargeMemory(rootedRoot);
        rootedRoot = null; rootedRootEdge = null;
        branchRigidEmat = null; branchMinimizingEmat = null; searchRCs = null;
        dpTablesReady = false;
    }

    private double computeEpsilon(BigDecimal zLower, BigDecimal zUpper) {
        if (zUpper.signum() <= 0) return Double.POSITIVE_INFINITY;
        if (zLower.signum() <= 0) return 1.0;

        double eps = MathTools.bigDivide(zUpper.subtract(zLower), zUpper,
                PartitionFunction.decimalPrecision).doubleValue();
        return Math.max(0.0, eps);
    }

    protected double getConfigDouble(String key, double defaultValue) {
        return BranchDpConfig.getBackendDouble(key, defaultValue, BranchDpConfig.getBackendLogPrefix());
    }

    public static final class RootingCandidate {
        public final RootedTreeNode root;
        public final int splitEdgeIndex;
        public final double logTESS;
        public final int maxFsetSize;
        public final int branchingEdges;
        public final int totalFsetEdges;
        public final int rootFsetSize;
        public final long maxDPTableBytes;
        public final long totalDPTableBytes;
        // Estimated live Java-heap storage for DP tables plus per-edge
        // enumeration/lambda arrays. This is the hard root-feasibility metric;
        // DP-table bytes alone omit up to four bytes per int-addressable M state.
        public final long totalHostBytes;
        public final double logDPWork;

        RootingCandidate(RootedTreeNode root, int splitEdgeIndex, double logTESS,
                         int maxFsetSize, int branchingEdges,
                         int totalFsetEdges, int rootFsetSize,
                         long maxDPTableBytes, long totalDPTableBytes,
                         long totalHostBytes,
                         double logDPWork) {
            this.root = root;
            this.splitEdgeIndex = splitEdgeIndex;
            this.logTESS = logTESS;
            this.maxFsetSize = maxFsetSize;
            this.branchingEdges = branchingEdges;
            this.totalFsetEdges = totalFsetEdges;
            this.rootFsetSize = rootFsetSize;
            this.maxDPTableBytes = maxDPTableBytes;
            this.totalDPTableBytes = totalDPTableBytes;
            this.totalHostBytes = totalHostBytes;
            this.logDPWork = logDPWork;
        }

    }

    public static final class RootSelectionException extends IllegalStateException {
        RootSelectionException(String message) { super(message); }
    }

    /**
     * Select a root for an already computed decomposition using the same policy
     * as initial branch-DP. A preview allocates no enumeration arrays or tables.
     * Call within the caller's backend configuration scope (including PACK* aliases).
     */
    public static RootingCandidate selectConfiguredRoot(BranchDecomposition decomposition,
            RCs rcs, boolean initEnumerationArrays) {
        RootSelector selector = new RootSelector(decomposition,
                Math.max(1, BranchDpConfig.getBackendInteger(ROOT_SPLIT_MAX_FSET_PROPERTY,
                        2, BranchDpConfig.getBackendLogPrefix())));
        RootingCandidate preview = selector.selectRooting(rcs);
        if (preview == null) {
            if (decomposition.getTree().getNumEdges() == 0) return null;
            throw new RootSelectionException("No root satisfies the DP state-count limits");
        }
        if (preview.totalHostBytes > selector.selectedRootHostBudgetBytes) {
            RootedTreeEdge.postOrderReleaseLargeMemory(preview.root);
            throw new RootSelectionException("Requested root exceeds the configured Java-heap budget");
        }
        if (!initEnumerationArrays) return preview;
        RootedTreeEdge.postOrderReleaseLargeMemory(preview.root);
        return selector.evaluateRootSplit(rcs, preview.splitEdgeIndex, true);
    }

    /** Shared initial/proposal root scoring; never constructs a backend or runs DP. */
    private static final class RootSelector {
        private final BranchDecomposition branchDecomposition;
        private final int rootSplitMaxFset;
        private long selectedRootHostBudgetBytes = Long.MAX_VALUE;

        RootSelector(BranchDecomposition decomposition, int maxFset) {
            this.branchDecomposition = decomposition;
            this.rootSplitMaxFset = maxFset;
        }

        private long getConfigBytes(String key, long fallback) {
            return BranchDpConfig.getBackendBytes(key, fallback, BranchDpConfig.getBackendLogPrefix());
        }

        private RootingCandidate selectRooting(RCs rcs) {
            int numEdges = branchDecomposition.getTree().getNumEdges();
            if (numEdges == 0) return null;

            selectedRootHostBudgetBytes = resolveRootHostBudgetBytes();

            double logNaive = computeLogNaive(rcs);
            RootingCandidate best = null;
            RootingCandidate leastHost = null;
            for (int splitIdx = 0; splitIdx < numEdges; splitIdx++) {
                RootingCandidate candidate = evaluateRootSplit(rcs, splitIdx, false);
                if (candidate == null) continue;
                if (leastHost == null || candidate.totalHostBytes < leastHost.totalHostBytes) {
                    leastHost = candidate;
                }
                boolean fitsHostBudget = candidate.totalHostBytes <= selectedRootHostBudgetBytes;
                if (!fitsHostBudget) {
                    RootedTreeEdge.postOrderReleaseLargeMemory(candidate.root);
                    continue;
                }
                boolean better = isBetterWorkRooting(candidate, best, logNaive);
                if (better) {
                    if (best != null) RootedTreeEdge.postOrderReleaseLargeMemory(best.root);
                    best = candidate;
                } else {
                    RootedTreeEdge.postOrderReleaseLargeMemory(candidate.root);
                }
            }
            if (best == null && leastHost != null) {
                throw new RootSelectionException(BranchDpConfig.getBackendLogPrefix()
                        + " no root split fits the Java-heap budget "
                        + formatBytes(selectedRootHostBudgetBytes)
                        + "; minimum estimated live host storage is "
                        + formatBytes(leastHost.totalHostBytes)
                        + " at split edge " + leastHost.splitEdgeIndex
                        + ". Raise -D" + ROOT_SPLIT_HOST_BUDGET_PROPERTY
                        + " only if the node and -Xmx have sufficient headroom.");
            }
            if (best == null) {
                return evaluateRootSplit(rcs, 0, false);
            }
            return best;
        }

        private long resolveRootHostBudgetBytes() {
            long configured = getConfigBytes(ROOT_SPLIT_HOST_BUDGET_PROPERTY, 0L);
            long maxHeap = Runtime.getRuntime().maxMemory();
            long budget = configured > 0L
                    ? configured
                    : Math.max(1L, (long) Math.floor(maxHeap * DEFAULT_ROOT_SPLIT_HOST_HEAP_FRACTION));
            return budget;
        }

        private RootingCandidate evaluateRootSplit(RCs rcs, int splitEdgeIndex,
                                                   boolean initEnumerationArrays) {
            RootedTreeNode root = branchDecomposition.rootBranchTree(rcs, splitEdgeIndex);
            if (root == null) return null;

            try {
                RootedTreeEdge.postOrderCompLlambda(root, initEnumerationArrays);
            } catch (DPTableTooLargeException e) {
                if (!initEnumerationArrays) {
                    RootedTreeEdge.postOrderReleaseLargeMemory(root);
                    return null;
                }
                throw e;
            }
            RootedTreeEdge rootEdge = root.getLeftChild().getChildOfEdge();
            rootEdge.compactTree();

            List<RootedTreeEdge> lambdaEdges = new ArrayList<>();
            RootedTreeEdge.collectLambdaEdges(root, lambdaEdges);

            int maxFsetSize = 0;
            int branchingEdges = 0;
            int totalFsetEdges = 0;
            long maxDPTableBytes = 0L;
            long totalDPTableBytes = 0L;
            long totalHostBytes = 0L;
            double logDPWork = Double.NEGATIVE_INFINITY;
            for (RootedTreeEdge edge : lambdaEdges) {
                int fsetSize = edge.getFset() == null ? 0 : edge.getFset().size();
                maxFsetSize = Math.max(maxFsetSize, fsetSize);
                if (fsetSize > 1) {
                    branchingEdges++;
                }
                totalFsetEdges += fsetSize;

                long mStates = Math.max(1L, edge.getMStateCount());
                long lambdaStates = Math.max(1, edge.getTotalLambdaStates());
                long tableBytes = estimateDPTableBytes(edge.getMStateCount());
                boolean fileBacked = RootedTreeEdge.shouldUseFileBackedDPTable(edge.getMStateCount());
                long hostBytes = fileBacked
                        ? estimateDPAuxHostBytes(edge.getMStateCount(), edge.getTotalLambdaStates())
                        : estimateDPHostBytes(edge.getMStateCount(), edge.getTotalLambdaStates());
                maxDPTableBytes = Math.max(maxDPTableBytes, tableBytes);
                totalDPTableBytes = saturatingAdd(totalDPTableBytes, tableBytes);
                totalHostBytes = saturatingAdd(totalHostBytes, hostBytes);
                logDPWork = RootedTreeEdge.logSumExp(logDPWork,
                        Math.log((double) mStates) + Math.log((double) lambdaStates));
            }

            int rootFsetSize = rootEdge.getFset() == null ? 0 : rootEdge.getFset().size();
            return new RootingCandidate(root, splitEdgeIndex, rootEdge.computeLogTESS(),
                    maxFsetSize, branchingEdges, totalFsetEdges, rootFsetSize,
                    maxDPTableBytes, totalDPTableBytes,
                    totalHostBytes, logDPWork);
        }

        private boolean isBetterRooting(RootingCandidate candidate, RootingCandidate best,
                                        double logNaive) {
            if (best == null) return true;

            boolean candidateValid = Math.exp(candidate.logTESS - logNaive) <= TESS_FALLBACK_THRESHOLD;
            boolean bestValid = Math.exp(best.logTESS - logNaive) <= TESS_FALLBACK_THRESHOLD;
            if (candidateValid != bestValid) return candidateValid;

            boolean candidateCapped = candidate.maxFsetSize <= rootSplitMaxFset;
            boolean bestCapped = best.maxFsetSize <= rootSplitMaxFset;
            if (candidateCapped != bestCapped) return candidateCapped;

            if (candidate.maxFsetSize != best.maxFsetSize) {
                return candidate.maxFsetSize > best.maxFsetSize;
            }
            if (candidate.branchingEdges != best.branchingEdges) {
                return candidate.branchingEdges > best.branchingEdges;
            }
            if (candidate.rootFsetSize != best.rootFsetSize) {
                return candidate.rootFsetSize > best.rootFsetSize;
            }
            if (candidate.totalFsetEdges != best.totalFsetEdges) {
                return candidate.totalFsetEdges > best.totalFsetEdges;
            }
            int tessCmp = Double.compare(candidate.logTESS, best.logTESS);
            if (tessCmp != 0) return tessCmp < 0;
            return candidate.splitEdgeIndex < best.splitEdgeIndex;
        }

        private boolean isBetterWorkRooting(RootingCandidate candidate, RootingCandidate best,
                                            double logNaive) {
            if (best == null) return true;

            boolean candidateValid = Math.exp(candidate.logTESS - logNaive) <= TESS_FALLBACK_THRESHOLD;
            boolean bestValid = Math.exp(best.logTESS - logNaive) <= TESS_FALLBACK_THRESHOLD;
            if (candidateValid != bestValid) return candidateValid;

            int workCmp = Double.compare(candidate.logDPWork, best.logDPWork);
            if (workCmp != 0) return workCmp < 0;

            int maxTableCmp = Long.compare(candidate.maxDPTableBytes, best.maxDPTableBytes);
            if (maxTableCmp != 0) return maxTableCmp < 0;

            int totalTableCmp = Long.compare(candidate.totalDPTableBytes, best.totalDPTableBytes);
            if (totalTableCmp != 0) return totalTableCmp < 0;

            boolean candidateCapped = candidate.maxFsetSize <= rootSplitMaxFset;
            boolean bestCapped = best.maxFsetSize <= rootSplitMaxFset;
            if (candidateCapped != bestCapped) return candidateCapped;

            return isBetterRooting(candidate, best, logNaive);
        }

    }

    private static double computeLogNaive(RCs rcs) {
        double logNaive = 0.0;
        for (int pos = 0; pos < rcs.getNumPos(); pos++) {
            if (rcs.getNum(pos) > 0) {
                logNaive += Math.log(rcs.getNum(pos));
            }
        }
        return logNaive;
    }

    private static long saturatingAdd(long a, long b) {
        if (Long.MAX_VALUE - a < b) {
            return Long.MAX_VALUE;
        }
        return a + b;
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

    public static long estimateDPTableBytes(long mStates) {
        if (mStates > Long.MAX_VALUE / (2L * (long) Double.BYTES)) {
            return Long.MAX_VALUE;
        }
        return 2L * (long) mStates * (long) Double.BYTES;
    }

    public static long estimateDPHostBytes(long mStates, long lambdaStates) {
        long tableBytes = estimateDPTableBytes(mStates);
        return saturatingAdd(tableBytes, estimateDPAuxHostBytes(mStates, lambdaStates));
    }

    public static long estimateDPAuxHostBytes(long mStates, long lambdaStates) {
        long enumerationBytes = mStates >= 0L && mStates <= Integer.MAX_VALUE
                ? saturatingMultiply(mStates, Integer.BYTES)
                : 0L;
        long lambdaBytes = lambdaStates <= 0L
                ? 0L
                : saturatingMultiply(lambdaStates, 2L * Double.BYTES);
        return saturatingAdd(enumerationBytes, lambdaBytes);
    }

    private static String formatBytes(long bytes) {
        if (bytes < 1024L) {
            return bytes + " B";
        }
        double value = bytes / 1024.0;
        String[] units = {"KiB", "MiB", "GiB", "TiB"};
        int unit = 0;
        while (value >= 1024.0 && unit < units.length - 1) {
            value /= 1024.0;
            unit++;
        }
        return String.format(Locale.ROOT, "%.1f %s", value, units[unit]);
    }

}
