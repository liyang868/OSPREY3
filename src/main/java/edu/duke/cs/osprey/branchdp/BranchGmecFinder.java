package edu.duke.cs.osprey.branchdp;

import edu.duke.cs.osprey.astar.conf.RCs;
import edu.duke.cs.osprey.confspace.ConfSearch;
import edu.duke.cs.osprey.ematrix.EnergyMatrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/** BWM-style min-sum DP and traceback on an existing rooted decomposition. */
public final class BranchGmecFinder {

    private BranchGmecFinder() {}

    /**
     * Find the GMEC of the supplied sparse pairwise proposal matrix. The score includes the
     * matrix constant. Returns null when no finite conformation exists.
     * Partition-function tables are untouched; temporary minimum/argmin tables
     * use the same dense, sharded, or mapped storage policy as ordinary DP.
     */
    public static ConfSearch.ScoredConf find(RootedTreeNode root, EnergyMatrix matrix,
                                             InteractionGraph graph, RCs rcs) {
        if (matrix.hasHigherOrderTerms()) {
            throw new IllegalArgumentException("GMEC traceback requires a pairwise energy matrix");
        }
        for (int pos = 0; pos < rcs.getNumPos(); pos++) {
            if (rcs.getNum(pos) == 0) return null;
        }

        List<RootedTreeEdge> edges = new ArrayList<>();
        RootedTreeEdge.collectLambdaEdges(root, edges);
        Map<RootedTreeEdge, DPTable> minima = new IdentityHashMap<>();
        try {
            for (RootedTreeEdge edge : edges) {
                DPTable table = edge.makeDPTable();
                minima.put(edge, table);
                for (long m = 0; m < edge.getMStateCount(); m++) {
                    int[] mState = edge.decodeMStatePublic(m);
                    double bestEnergy = Double.POSITIVE_INFINITY;
                    int bestLambda = -1;
                    for (int lambda = 0; lambda < edge.getTotalLambdaStates(); lambda++) {
                        double energy = edge.computeLocalEnergyForLambdaIndex(
                                mState, lambda, matrix, graph);
                        if (!Double.isFinite(energy)) continue;
                        int[] lambdaState = edge.decodeLambdaState(lambda);
                        for (RootedTreeEdge child : edge.getFset()) {
                            int[] childM = edge.getMstateForFullState(mState, lambdaState, child);
                            energy += minima.get(child).lower(child.computeIndexInA(childM));
                        }
                        if (Double.isFinite(energy) && energy < bestEnergy) {
                            bestEnergy = energy;
                            bestLambda = lambda;
                        }
                    }
                    // An int lambda index is exactly representable as a double.
                    table.set(m, bestEnergy, bestLambda);
                }
            }

            int[] assignments = new int[rcs.getNumPos()];
            Arrays.fill(assignments, -1);
            double energy = matrix.getConstTerm();
            if (!edges.isEmpty()) {
                RootedTreeEdge rootEdge = edges.get(edges.size() - 1);
                if (!Double.isFinite(minima.get(rootEdge).lower(0))) return null;
                energy += minima.get(rootEdge).lower(0);
                trace(rootEdge, new int[0], minima, assignments);
            }
            // Zero/one-position spaces and isolated vertices have no tree edge.
            for (int pos = 0; pos < assignments.length; pos++) {
                if (assignments[pos] >= 0) continue;
                for (int other = 0; other < assignments.length; other++) {
                    if (graph.hasEdge(pos, other)) {
                        throw new IllegalStateException("GMEC decomposition omits position " + pos);
                    }
                }
                double best = Double.POSITIVE_INFINITY;
                for (int rc : rcs.get(pos)) {
                    double value = matrix.getOneBody(pos, rc);
                    if (Double.isFinite(value) && value < best) {
                        best = value;
                        assignments[pos] = rc;
                    }
                }
                if (assignments[pos] < 0) return null;
                energy += best;
            }
            return Double.isFinite(energy) ? new ConfSearch.ScoredConf(assignments, energy) : null;
        } finally {
            minima.values().forEach(DPTable::close);
        }
    }

    private static void trace(RootedTreeEdge edge, int[] mState,
                              Map<RootedTreeEdge, DPTable> minima, int[] assignments) {
        int lambda = (int) minima.get(edge).upper(edge.computeIndexInA(mState));
        if (lambda < 0) throw new IllegalStateException("No finite GMEC traceback state");
        edge.writeLambdaStateGlobal(lambda, assignments);
        int[] lambdaState = edge.decodeLambdaState(lambda);
        for (RootedTreeEdge child : edge.getFset()) {
            trace(child, edge.getMstateForFullState(mState, lambdaState, child), minima, assignments);
        }
    }
}
