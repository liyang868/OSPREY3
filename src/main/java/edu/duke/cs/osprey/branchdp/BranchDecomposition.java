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
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;

import java.math.BigInteger;
import java.util.*;

/**
 * Computes a branch decomposition of a residue interaction graph.
 * Uses the heuristic algorithm from Hicks (2005): star graph → push → 2-sep → 3-sep → eigenvector.
 *
 * Adapted from BWM's BranchDecomposition.BranchDecompositionH.
 * Key changes: int position indices instead of String vertex names,
 * InteractionGraph input instead of file I/O,
 * Apache Commons Math 3 instead of JAMA for eigenvalue decomposition.
 */
public class BranchDecomposition {
    public enum Strategy { HICKS, WEIGHTED_HICKS }

    private BranchTree bt;
    private Set<Integer> graphVertices;
    private int numPositions;
    private InteractionGraph interactionGraph;
    private final Strategy strategy;
    private final int[] positionStateCounts;
    private final boolean logWeightedRefinement;
    private boolean computed = false;
    private static final int DEFAULT_WEIGHTED_HICKS_RESTARTS = 24;
    private static final int DEFAULT_WEIGHTED_HICKS_RANDOM_MOVES = 32;

    /**
     * Build a branch decomposition from an interaction graph.
     * Call compute() to run the algorithm.
     */
    public BranchDecomposition(InteractionGraph graph) {
        this(graph, Strategy.HICKS, null, true);
    }
    public BranchDecomposition(InteractionGraph graph, Strategy strategy) {
        this(graph, strategy, null, true);
    }
    public BranchDecomposition(InteractionGraph graph, Strategy strategy, int[] counts) {
        this(graph, strategy, counts, true);
    }
    public BranchDecomposition(InteractionGraph graph, Strategy strategy, int[] counts, boolean logRefinement) {
        this.interactionGraph = graph;
        this.strategy = strategy == null ? Strategy.HICKS : strategy;
        this.numPositions = graph.getNumPositions();
        this.positionStateCounts = normalizePositionStateCounts(counts, numPositions);
        this.logWeightedRefinement = logRefinement;
        resetTree();
        initializeHicksTree();
    }

    private void resetTree() {
        this.bt = new BranchTree();
        this.graphVertices = new LinkedHashSet<>();
    }

    private void initializeHicksTree() {
        resetTree();
        // Build initial leaf nodes from interaction graph edges (analogous to readGraphFile)
        Set<Integer> gvDup = new LinkedHashSet<>();
        Set<Integer> seen = new LinkedHashSet<>();

        for (int[] edge : interactionGraph.getEdgeList()) {
            int pos1 = edge[0];
            int pos2 = edge[1];
            bt.addNode(new BranchNode(true, pos1, pos2));

            if (!seen.add(pos1)) gvDup.add(pos1);
            if (!seen.add(pos2)) gvDup.add(pos2);
        }
        graphVertices.addAll(seen);

        // Build star graph
        constructStarGraph(gvDup);
    }

    /** Run the branch decomposition algorithm. */

    public void compute() {
        if (computed) return;
        computed = true;
        computeHicks(strategy == Strategy.WEIGHTED_HICKS);
    }

    private void computeHicks(boolean weightedPush) {
        boolean done = false;
        boolean done2sep = false;
        boolean done3sep = false;

        while (!done) {
            LinkedHashSet<BranchNode> bigNodes = getNodesToSplit();

            if (!bigNodes.isEmpty()) {
                LinkedHashSet<BranchEdge> sx = new LinkedHashSet<>();
                LinkedHashSet<BranchEdge> sy = new LinkedHashSet<>();

                BranchNode pn;

                if (weightedPush) {
                    pn = pushBestWeighted(bigNodes, sx, sy);
                } else {
                    pn = findPushNode(bigNodes);
                }

                if (pn != null) {
                    if (!weightedPush) {
                    pushNode(pn, sx, sy);
                    }
                } else if (!done2sep) {
                    boolean found = false;
                    for (BranchNode bn : bigNodes) {
                        if (find2sep(bn, sx, sy)) {
                            pn = bn;
                            found = true;
                            break;
                        }
                    }
                    if (!found) done2sep = true;
                } else if (!done3sep) {
                    boolean found = false;
                    for (BranchNode bn : bigNodes) {
                        if (find3sep(bn, sx, sy)) {
                            pn = bn;
                            found = true;
                            break;
                        }
                    }
                    if (!found) done3sep = true;
                } else {
                    pn = bigNodes.iterator().next();
                    doEigen(pn, sx, sy);
                }

                if (!sx.isEmpty()) {
                    performSplit(pn, sx, sy);
                }
            } else {
                done = true;
            }
        }

        if (weightedPush) {
            refineCompletedTreeByStateCount("weighted_hicks");
        }
    }

    private void refineCompletedTreeByStateCount(String label) {
        int swaps = refineWeightedNearestNeighborInterchanges();
        int improvements = refineWeightedNniRestarts();
        if (logWeightedRefinement) {
            System.out.println("BranchDecomposition " + label + ": weighted NNI swaps=" + swaps
                    + ", restartImprovements=" + improvements);
            logWeightedCost(label);
        }
    }

    public BranchTree getTree() { return bt; }
    public int getBranchwidth() { return bt.getBranchwidth(); }

    private static int[] normalizePositionStateCounts(int[] counts, int numPositions) {
        int[] normalized = new int[numPositions];
        Arrays.fill(normalized, 1);
        if (counts != null) {
            for (int i = 0; i < Math.min(counts.length, numPositions); i++) {
                normalized[i] = Math.max(1, counts[i]);
            }
        }
        return normalized;
    }

    // ========== Star graph construction ==========

    private void constructStarGraph(Set<Integer> gvDup) {
        BranchNode starNode = new BranchNode(false, -1, -1);
        bt.addNode(starNode);
        int starNodeIndex = starNode.getIndex();

        for (int i = 0; i < bt.getNumNodes(); i++) {
            if (i != starNodeIndex) {
                int eInd = bt.addEdge(i, starNodeIndex);
                BranchEdge be = bt.getEdge(eInd);
                BranchNode bn = bt.getNode(i);
                int pos1 = bn.getPos1();
                int pos2 = bn.getPos2();

                if (gvDup.contains(pos1)) be.addM(pos1);
                if (gvDup.contains(pos2)) be.addM(pos2);
            }
        }
    }

    private BigInteger stateCount(Collection<Integer> positions) {
        // Most preview boundaries fit in a long. Avoid allocating one
        // BigInteger per position, promoting before overflow when necessary.
        long small = 1;
        BigInteger large = null;
        for (int pos : positions) {
            int factor = positionStateCounts[pos];
            if (large != null) {
                large = large.multiply(BigInteger.valueOf(factor));
            } else if (factor > 0 && small <= Long.MAX_VALUE / factor) {
                small *= factor;
            } else {
                large = BigInteger.valueOf(small).multiply(BigInteger.valueOf(factor));
            }
        }
        return large == null ? BigInteger.valueOf(small) : large;
    }

    private void logWeightedCost(String label) {
        BigInteger maxStates = BigInteger.ZERO;
        BigInteger totalStates = BigInteger.ZERO;
        for (int edgeIndex = 0; edgeIndex < bt.getNumEdges(); edgeIndex++) {
            BigInteger states = stateCount(bt.getEdge(edgeIndex).getM());
            maxStates = maxStates.max(states);
            totalStates = totalStates.add(states);
        }
        System.out.println("BranchDecomposition " + label + ": bw=" + bt.getBranchwidth()
                + ", maxBoundaryStates=" + maxStates
                + ", totalBoundaryStates=" + totalStates);
    }

    private static final class WeightedNniChoice {
        final BranchEdge central;
        final BranchNode leftNode;
        final BranchNode rightNode;
        final BranchEdge leftSwap;
        final BranchEdge rightSwap;
        final LinkedHashSet<Integer> newMiddle;
        final BigInteger stateReduction;
        final int sizeReduction;

        WeightedNniChoice(BranchEdge central,
                          BranchNode leftNode, BranchNode rightNode,
                          BranchEdge leftSwap, BranchEdge rightSwap,
                          LinkedHashSet<Integer> newMiddle,
                          BigInteger stateReduction, int sizeReduction) {
            this.central = central;
            this.leftNode = leftNode;
            this.rightNode = rightNode;
            this.leftSwap = leftSwap;
            this.rightSwap = rightSwap;
            this.newMiddle = newMiddle;
            this.stateReduction = stateReduction;
            this.sizeReduction = sizeReduction;
        }

        boolean isBetterThan(WeightedNniChoice other) {
            if (other == null) return true;
            int cmp = stateReduction.compareTo(other.stateReduction);
            if (cmp != 0) return cmp > 0;
            if (sizeReduction != other.sizeReduction) {
                return sizeReduction > other.sizeReduction;
            }
            return newMiddle.size() < other.newMiddle.size();
        }
    }

    /**
     * Refine a completed cubic branch tree with nearest-neighbor interchanges.
     * An NNI changes only one internal edge's leaf bipartition; the four outer
     * cuts (and their M sets) are preserved. Accepting only a strictly smaller
     * RC-state product therefore monotonically lowers total table pressure and
     * never increases the original branchwidth.
     */
    private int refineWeightedNearestNeighborInterchanges() {
        int branchwidthCap = bt.getBranchwidth();
        int swaps = 0;
        int maxSwaps = Math.max(1, 20 * bt.getNumEdges());

        while (swaps < maxSwaps) {
            WeightedNniChoice best = null;
            for (int edgeIndex = 0; edgeIndex < bt.getNumEdges(); edgeIndex++) {
                BranchEdge central = bt.getEdge(edgeIndex);
                BranchNode leftNode = central.getn1();
                BranchNode rightNode = central.getn2();
                if (leftNode.getIsLeaf() || rightNode.getIsLeaf()
                        || leftNode.getNumEdges() != 3 || rightNode.getNumEdges() != 3) {
                    continue;
                }

                BranchEdge[] leftOuter = outerEdges(leftNode, central);
                BranchEdge[] rightOuter = outerEdges(rightNode, central);
                if (leftOuter.length != 2 || rightOuter.length != 2) continue;

                // AB|CD -> AC|BD and AD|BC. Swapping leftOuter[1] with either
                // right edge enumerates both non-equivalent NNI alternatives.
                best = betterNni(best, central, leftNode, rightNode,
                        leftOuter[0], leftOuter[1], rightOuter[0], rightOuter[1],
                        branchwidthCap);
                best = betterNni(best, central, leftNode, rightNode,
                        leftOuter[0], leftOuter[1], rightOuter[1], rightOuter[0],
                        branchwidthCap);
            }

            if (best == null) break;
            applyWeightedNni(best);
            swaps++;
        }
        return swaps;
    }

    private static final class WeightedTreeCost {
        final BigInteger maxStates;
        final BigInteger totalStates;
        final int branchwidth;

        WeightedTreeCost(BigInteger maxStates, BigInteger totalStates, int branchwidth) {
            this.maxStates = maxStates;
            this.totalStates = totalStates;
            this.branchwidth = branchwidth;
        }

        boolean isBetterThan(WeightedTreeCost other) {
            int cmp = maxStates.compareTo(other.maxStates);
            if (cmp != 0) return cmp < 0;
            cmp = totalStates.compareTo(other.totalStates);
            if (cmp != 0) return cmp < 0;
            return branchwidth < other.branchwidth;
        }
    }

    /** Deterministic random-restart NNI search to escape one-swap local minima. */
    private int refineWeightedNniRestarts() {
        int restarts = getWeightedHicksInteger(
                "restarts", DEFAULT_WEIGHTED_HICKS_RESTARTS);
        int randomMoves = getWeightedHicksInteger(
                "randomMoves", DEFAULT_WEIGHTED_HICKS_RANDOM_MOVES);
        if (restarts <= 0 || randomMoves <= 0 || bt.getNumEdges() < 5) return 0;

        int branchwidthCap = bt.getBranchwidth();
        BranchTree bestTree = bt.deepCopy();
        WeightedTreeCost bestCost = weightedTreeCost();
        Random random = new Random(weightedHicksSeed());
        int improvements = 0;

        for (int restart = 0; restart < restarts; restart++) {
            bt = bestTree.deepCopy();
            for (int move = 0; move < randomMoves; move++) {
                List<WeightedNniChoice> choices = legalNniChoices(branchwidthCap, false);
                if (choices.isEmpty()) break;
                applyWeightedNni(choices.get(random.nextInt(choices.size())));
            }
            refineWeightedNearestNeighborInterchanges();
            WeightedTreeCost cost = weightedTreeCost();
            if (cost.isBetterThan(bestCost)) {
                bestCost = cost;
                bestTree = bt.deepCopy();
                improvements++;
            }
        }
        bt = bestTree;
        return improvements;
    }

    private List<WeightedNniChoice> legalNniChoices(int branchwidthCap,
                                                     boolean requireImprovement) {
        List<WeightedNniChoice> choices = new ArrayList<>();
        for (int edgeIndex = 0; edgeIndex < bt.getNumEdges(); edgeIndex++) {
            BranchEdge central = bt.getEdge(edgeIndex);
            BranchNode leftNode = central.getn1();
            BranchNode rightNode = central.getn2();
            if (leftNode.getIsLeaf() || rightNode.getIsLeaf()
                    || leftNode.getNumEdges() != 3 || rightNode.getNumEdges() != 3) {
                continue;
            }
            BranchEdge[] leftOuter = outerEdges(leftNode, central);
            BranchEdge[] rightOuter = outerEdges(rightNode, central);
            if (leftOuter.length != 2 || rightOuter.length != 2) continue;

            WeightedNniChoice first = makeNniChoice(
                    central, leftNode, rightNode,
                    leftOuter[0], leftOuter[1], rightOuter[0], rightOuter[1],
                    branchwidthCap, requireImprovement);
            WeightedNniChoice second = makeNniChoice(
                    central, leftNode, rightNode,
                    leftOuter[0], leftOuter[1], rightOuter[1], rightOuter[0],
                    branchwidthCap, requireImprovement);
            if (first != null) choices.add(first);
            if (second != null) choices.add(second);
        }
        return choices;
    }

    private WeightedTreeCost weightedTreeCost() {
        BigInteger max = BigInteger.ZERO;
        BigInteger total = BigInteger.ZERO;
        for (int edgeIndex = 0; edgeIndex < bt.getNumEdges(); edgeIndex++) {
            BigInteger states = stateCount(bt.getEdge(edgeIndex).getM());
            max = max.max(states);
            total = total.add(states);
        }
        return new WeightedTreeCost(max, total, bt.getBranchwidth());
    }

    private long weightedHicksSeed() {
        long seed = 0x9E3779B97F4A7C15L;
        for (int count : positionStateCounts) {
            seed ^= count + 0x9E3779B9L + (seed << 6) + (seed >>> 2);
        }
        for (int[] edge : interactionGraph.getEdgeList()) {
            seed ^= (long) edge[0] * 0xBF58476D1CE4E5B9L;
            seed = Long.rotateLeft(seed, 17) ^ edge[1];
        }
        return seed;
    }

    private static int getWeightedHicksInteger(String suffix, int defaultValue) {
        String neutral = "branchdp.decomp.weightedHicks." + suffix;
        String value = System.getProperty(neutral);
        if (value == null || value.trim().isEmpty()) {
            value = System.getProperty("packstar.decomp.weightedHicks." + suffix);
        }
        if (value == null || value.trim().isEmpty()) return defaultValue;
        try {
            return Math.max(0, Integer.parseInt(value.trim()));
        } catch (NumberFormatException ex) {
            System.err.println("BranchDecomposition weighted_hicks: invalid "
                    + neutral + "='" + value + "', using " + defaultValue + ".");
            return defaultValue;
        }
    }

    private WeightedNniChoice betterNni(WeightedNniChoice best,
                                         BranchEdge central,
                                         BranchNode leftNode, BranchNode rightNode,
                                         BranchEdge leftKeep, BranchEdge leftSwap,
                                         BranchEdge rightSwap, BranchEdge rightKeep,
                                         int branchwidthCap) {
        WeightedNniChoice candidate = makeNniChoice(
                central, leftNode, rightNode,
                leftKeep, leftSwap, rightSwap, rightKeep,
                branchwidthCap, true);
        return candidate != null && candidate.isBetterThan(best) ? candidate : best;
    }

    private WeightedNniChoice makeNniChoice(BranchEdge central,
                                             BranchNode leftNode, BranchNode rightNode,
                                             BranchEdge leftKeep, BranchEdge leftSwap,
                                             BranchEdge rightSwap, BranchEdge rightKeep,
                                             int branchwidthCap,
                                             boolean requireImprovement) {
        // (leftKeep U rightSwap) intersect (leftSwap U rightKeep), preserving
        // the original insertion order without constructing the second union
        // or allocating entries that the intersection immediately discards.
        LinkedHashSet<Integer> newMiddle = new LinkedHashSet<>();
        for (int pos : leftKeep.getM()) {
            if (leftSwap.getM().contains(pos) || rightKeep.getM().contains(pos)) newMiddle.add(pos);
        }
        for (int pos : rightSwap.getM()) {
            if (leftSwap.getM().contains(pos) || rightKeep.getM().contains(pos)) newMiddle.add(pos);
        }
        if (newMiddle.size() > branchwidthCap) return null;

        // Random restart moves sample uniformly from legal choices and never
        // compare their scores. Only the improving NNI search needs products.
        BigInteger reduction = requireImprovement
                ? stateCount(central.getM()).subtract(stateCount(newMiddle))
                : BigInteger.ZERO;
        int sizeReduction = central.getM().size() - newMiddle.size();
        if (requireImprovement
                && (reduction.signum() < 0
                || (reduction.signum() == 0 && sizeReduction <= 0))) {
            return null;
        }

        return new WeightedNniChoice(
                central, leftNode, rightNode, leftSwap, rightSwap,
                newMiddle, reduction, sizeReduction);
    }

    private BranchEdge[] outerEdges(BranchNode node, BranchEdge central) {
        BranchEdge[] incident = bt.getEdgesForNode(node.getIndex());
        BranchEdge[] outer = new BranchEdge[Math.max(0, incident.length - 1)];
        int offset = 0;
        for (BranchEdge edge : incident) {
            if (edge != central) outer[offset++] = edge;
        }
        return offset == outer.length ? outer : Arrays.copyOf(outer, offset);
    }

    private void applyWeightedNni(WeightedNniChoice choice) {
        BranchNode leftOther = otherEndpoint(choice.leftSwap, choice.leftNode);
        BranchNode rightOther = otherEndpoint(choice.rightSwap, choice.rightNode);
        LinkedHashSet<Integer> leftMiddle = new LinkedHashSet<>(choice.leftSwap.getM());
        LinkedHashSet<Integer> rightMiddle = new LinkedHashSet<>(choice.rightSwap.getM());

        int leftIndex = edgeIndex(choice.leftSwap);
        int rightIndex = edgeIndex(choice.rightSwap);
        if (leftIndex < 0 || rightIndex < 0) {
            throw new IllegalStateException("weighted NNI edge disappeared during refinement");
        }
        if (leftIndex > rightIndex) {
            bt.deleteEdge(leftIndex);
            bt.deleteEdge(rightIndex);
        } else {
            bt.deleteEdge(rightIndex);
            bt.deleteEdge(leftIndex);
        }

        int newLeft = bt.addEdge(choice.leftNode.getIndex(), rightOther.getIndex());
        int newRight = bt.addEdge(choice.rightNode.getIndex(), leftOther.getIndex());
        if (newLeft < 0 || newRight < 0) {
            throw new IllegalStateException("weighted NNI produced a duplicate tree edge");
        }
        bt.getEdge(newLeft).setM(rightMiddle);
        bt.getEdge(newRight).setM(leftMiddle);
        choice.central.setM(choice.newMiddle);
    }

    private int edgeIndex(BranchEdge target) {
        for (int edgeIndex = 0; edgeIndex < bt.getNumEdges(); edgeIndex++) {
            if (bt.getEdge(edgeIndex) == target) return edgeIndex;
        }
        return -1;
    }

    private static BranchNode otherEndpoint(BranchEdge edge, BranchNode node) {
        return edge.getn1() == node ? edge.getn2() : edge.getn1();
    }

    // ========== Node splitting logic ==========

    private LinkedHashSet<BranchNode> getNodesToSplit() {
        LinkedHashSet<BranchNode> pn = new LinkedHashSet<>();
        for (int i = 0; i < bt.getNumNodes(); i++) {
            BranchNode bn = bt.getNode(i);
            if (!bn.getIsLeaf() && bn.getNumEdges() > 3)
                pn.add(bn);
        }
        return pn;
    }

    private BranchNode findPushNode(LinkedHashSet<BranchNode> bigNodes) {
        for (BranchNode cn : bigNodes) {
            BranchEdge[] ce = bt.getEdgesForNode(cn.getIndex());
            for (int j = 0; j < cn.getNumEdges(); j++) {
                if (!ce[j].isChecked(cn))
                    return cn;
            }
        }
        return null;
    }

    private static final class WeightedPushChoice {
        final BranchNode node;
        final BranchEdge[] edges;
        final int left;
        final int right;
        final LinkedHashSet<Integer> boundary;
        final BigInteger boundaryStates;

        WeightedPushChoice(BranchNode node, BranchEdge[] edges, int left, int right,
                           LinkedHashSet<Integer> boundary,
                           BigInteger boundaryStates) {
            this.node = node;
            this.edges = edges;
            this.left = left;
            this.right = right;
            this.boundary = boundary;
            this.boundaryStates = boundaryStates;
        }

        boolean isBetterThan(WeightedPushChoice other) {
            if (other == null) return true;
            int cmp = boundaryStates.compareTo(other.boundaryStates);
            if (cmp != 0) return cmp < 0;
            if (boundary.size() != other.boundary.size()) {
                return boundary.size() < other.boundary.size();
            }
            if (node.getIndex() != other.node.getIndex()) {
                return node.getIndex() < other.node.getIndex();
            }
            if (left != other.left) return left < other.left;
            return right < other.right;
        }
    }

    /**
     * Hicks' push inequality controls separator cardinality but the historical
     * implementation accepts the first valid pair.  Evaluate every valid push
     * and choose the new separator with the smallest RC Cartesian product.
     * Every accepted choice still satisfies the original inequality, so this
     * changes tie-breaking/shape rather than relaxing branchwidth safety.
     */
    private BranchNode pushBestWeighted(LinkedHashSet<BranchNode> bigNodes,
                                        LinkedHashSet<BranchEdge> sx,
                                        LinkedHashSet<BranchEdge> sy) {
        WeightedPushChoice best = null;
        for (BranchNode node : bigNodes) {
            BranchEdge[] edges = bt.getEdgesForNode(node.getIndex());
            boolean[][] positionInEdge = new boolean[numPositions][edges.length];
            for (int edgeIndex = 0; edgeIndex < edges.length; edgeIndex++) {
                for (int pos : edges[edgeIndex].getM()) {
                    if (pos >= 0 && pos < numPositions) {
                        positionInEdge[pos][edgeIndex] = true;
                    }
                }
            }

            for (int i = 0; i < edges.length; i++) {
                if (edges[i].isChecked(node)) continue;
                for (int j = 0; j < edges.length; j++) {
                    if (i == j) continue;
                    LinkedHashSet<Integer> boundary = pushBoundary(
                            edges, positionInEdge, i, j);
                    if (boundary.size() > Math.max(
                            edges[i].getM().size(), edges[j].getM().size())) {
                        continue;
                    }
                    WeightedPushChoice candidate = new WeightedPushChoice(
                            node, edges, i, j, boundary, stateCount(boundary));
                    if (candidate.isBetterThan(best)) {
                        best = candidate;
                    }
                }
            }
        }

        if (best == null) {
            // No unchecked edge participates in a legal push. Mark the exhausted
            // search space exactly as repeated legacy pushNode() calls would.
            for (BranchNode node : bigNodes) {
                for (BranchEdge edge : bt.getEdgesForNode(node.getIndex())) {
                    edge.setChecked(node, true);
                }
            }
            return null;
        }

        best.edges[best.left].setChecked(best.node, true);
        sx.add(best.edges[best.left]);
        sx.add(best.edges[best.right]);
        for (int edgeIndex = 0; edgeIndex < best.edges.length; edgeIndex++) {
            if (edgeIndex != best.left && edgeIndex != best.right) {
                sy.add(best.edges[edgeIndex]);
            }
        }
        return best.node;
    }

    private LinkedHashSet<Integer> pushBoundary(BranchEdge[] edges,
                                                 boolean[][] positionInEdge,
                                                 int left, int right) {
        LinkedHashSet<Integer> boundary = new LinkedHashSet<>(edges[left].getM());
        boundary.addAll(edges[right].getM());
        boundary.removeIf(pos -> {
            if (pos < 0 || pos >= numPositions) return false;
            for (int edgeIndex = 0; edgeIndex < edges.length; edgeIndex++) {
                if (edgeIndex != left && edgeIndex != right
                        && positionInEdge[pos][edgeIndex]) {
                    return false;
                }
            }
            return true;
        });
        return boundary;
    }

    // ========== Push operation ==========

    private void pushNode(BranchNode pn, LinkedHashSet<BranchEdge> sx, LinkedHashSet<BranchEdge> sy) {
        BranchEdge[] pe = bt.getEdgesForNode(pn.getIndex());

        // Track which positions appear in which edge M-sets
        boolean[][] vInMe = new boolean[numPositions][pe.length];
        for (int k = 0; k < pe.length; k++) {
            for (int pos : pe[k].getM()) {
                if (pos >= 0 && pos < numPositions)
                    vInMe[pos][k] = true;
            }
        }

        for (int i = 0; i < pe.length; i++) {
            if (!pe[i].isChecked(pn)) {
                pe[i].setChecked(pn, true);

                for (int j = 0; j < pe.length; j++) {
                    if (i != j) {
                        LinkedHashSet<Integer> uMij = new LinkedHashSet<>(pe[i].getM());
                        uMij.addAll(pe[j].getM());

                        // Remove vertices only in M(i) ∪ M(j) but no other edges
                        LinkedHashSet<Integer> toRemove = new LinkedHashSet<>();
                        for (int pos : uMij) {
                            if (pos < 0 || pos >= numPositions) continue;
                            boolean found = false;
                            for (int l = 0; l < pe.length; l++) {
                                if (l != i && l != j && vInMe[pos][l]) {
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) toRemove.add(pos);
                        }
                        uMij.removeAll(toRemove);

                        // Check pushing inequality
                        if (uMij.size() <= Math.max(pe[i].getM().size(), pe[j].getM().size())) {
                            sx.add(pe[i]);
                            sx.add(pe[j]);
                            for (int k = 0; k < pe.length; k++) {
                                if (k != i && k != j)
                                    sy.add(pe[k]);
                            }
                            return;
                        }
                    }
                }
            }
        }
    }

    // ========== 2-separation ==========

    private boolean find2sep(BranchNode pn, LinkedHashSet<BranchEdge> sx, LinkedHashSet<BranchEdge> sy) {
        BranchEdge[] D = bt.getEdgesForNode(pn.getIndex());
        LinkedHashSet<Integer> S = new LinkedHashSet<>();
        for (BranchEdge e : D) S.addAll(e.getM());
        Object[] Sa = S.toArray();
        BranchTree H = constructGraphH(Sa, D);
        return find23sepHelper(H, sx, sy, D, true);
    }

    // ========== 3-separation ==========

    private boolean find3sep(BranchNode pn, LinkedHashSet<BranchEdge> sx, LinkedHashSet<BranchEdge> sy) {
        BranchEdge[] D = bt.getEdgesForNode(pn.getIndex());
        LinkedHashSet<Integer> S = new LinkedHashSet<>();
        for (BranchEdge e : D) S.addAll(e.getM());
        Object[] Sa = S.toArray();
        BranchTree H = constructGraphH(Sa, D);

        // Check for 3-separation via triangle
        for (int i = 0; i < H.getNumNodes(); i++) {
            BranchNode x1 = H.getNode(i);
            if (x1.getNumEdges() == 3) {
                for (int j = 0; j < H.getNumNodes(); j++) {
                    if (j != i) {
                        BranchNode x2 = H.getNode(j);
                        if (H.edgeExists(x1.getIndex(), x2.getIndex())) {
                            for (int k = j + 1; k < H.getNumNodes(); k++) {
                                if (k != i) {
                                    BranchNode x3 = H.getNode(k);
                                    if (H.edgeExists(x1.getIndex(), x3.getIndex())
                                            && H.edgeExists(x2.getIndex(), x3.getIndex())) {
                                        // Found a triangle
                                        LinkedHashSet<BranchEdge> sxtmp = new LinkedHashSet<>();
                                        LinkedHashSet<BranchEdge> sytmp = new LinkedHashSet<>();
                                        for (BranchEdge d : D) {
                                            sytmp.add(d);
                                            if (d.getM().contains(x1.getPos1()))
                                                sxtmp.add(d);
                                        }
                                        sytmp.removeAll(sxtmp);
                                        if (sytmp.size() >= 2) {
                                            sx.addAll(sxtmp);
                                            sy.addAll(sytmp);
                                            return true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Check for 3-separations with L and R sets of size >= 2
        return find23sepHelper(H, sx, sy, D, false);
    }

    // ========== 2/3-separation helper ==========

    private boolean find23sepHelper(BranchTree H, LinkedHashSet<BranchEdge> sx, LinkedHashSet<BranchEdge> sy,
                                     BranchEdge[] D, boolean do2sep) {
        // Find node with minimum degree
        BranchNode nn = null;
        int minDegree = Integer.MAX_VALUE;
        for (int i = 0; i < H.getNumNodes(); i++) {
            if (H.getNode(i).getNumEdges() < minDegree) {
                nn = H.getNode(i);
                minDegree = H.getNode(i).getNumEdges();
            }
        }
        if (nn == null) return false;

        // Find neighbors of nn
        BranchEdge[] cDedges = H.getEdgesForNode(nn.getIndex());
        BranchNode[] nnNeighborsN = new BranchNode[cDedges.length];
        LinkedHashSet<Integer> nnNeighbors = new LinkedHashSet<>();
        for (int i = 0; i < cDedges.length; i++) {
            if (cDedges[i].getn1().getIndex() == nn.getIndex()) {
                nnNeighbors.add(cDedges[i].getn2().getIndex());
                nnNeighborsN[i] = cDedges[i].getn2();
            } else {
                nnNeighbors.add(cDedges[i].getn1().getIndex());
                nnNeighborsN[i] = cDedges[i].getn1();
            }
        }

        // Check all non-neighbors of nn
        for (int i = 0; i < H.getNumNodes(); i++) {
            if (!nnNeighbors.contains(H.getNode(i).getIndex()) && H.getNode(i).getIndex() != nn.getIndex()) {
                BranchEdge e = new BranchEdge(new BranchNode(false, -1, -1), new BranchNode(false, -1, -1));

                if (do2sep && find2sepHelper(H, nn, H.getNode(i), sx, sy, D, e))
                    return true;
                if (!do2sep && findKsepPairV(H, nn, H.getNode(i), 3, sx, sy, D, e))
                    return true;
            }
        }

        // Check pairs of neighbors of nn
        for (int i = 0; i < nnNeighborsN.length; i++) {
            for (int j = i + 1; j < nnNeighborsN.length; j++) {
                if (!H.edgeExists(nnNeighborsN[i].getIndex(), nnNeighborsN[j].getIndex())) {
                    BranchEdge e = new BranchEdge(new BranchNode(false, -1, -1), new BranchNode(false, -1, -1));

                    if (do2sep && find2sepHelper(H, nnNeighborsN[i], nnNeighborsN[j], sx, sy, D, e))
                        return true;
                    if (!do2sep && findKsepPairV(H, nnNeighborsN[i], nnNeighborsN[j], 3, sx, sy, D, e))
                        return true;
                }
            }
        }

        return false;
    }

    private boolean find2sepHelper(BranchTree H, BranchNode v, BranchNode w,
                                    LinkedHashSet<BranchEdge> sx, LinkedHashSet<BranchEdge> sy,
                                    BranchEdge[] D, BranchEdge e) {
        if (!H.edgeExists(v.getIndex(), w.getIndex())) {
            MinVertexCut mvc = new MinVertexCut();
            mvc.findMinVertexCut(H, v.getIndex(), w.getIndex());
            e.setM(mvc.getCutSet());

            if (mvc.getCutSet().size() == 2 && mvc.getLset().size() > 0 && mvc.getRset().size() > 0) {
                completeXY(mvc.getLset(), mvc.getRset(), D, sx, sy, false);
                return true;
            }
        }
        return false;
    }

    // ========== k-separation with vertex contraction ==========

    private boolean findKsepPairV(BranchTree H, BranchNode v, BranchNode w, int k,
                                   LinkedHashSet<BranchEdge> sx, LinkedHashSet<BranchEdge> sy,
                                   BranchEdge[] D, BranchEdge e) {
        if (!H.edgeExists(v.getIndex(), w.getIndex())) {
            MinVertexCut mvc = new MinVertexCut();
            mvc.findMinVertexCut(H, v.getIndex(), w.getIndex());
            e.setM(mvc.getCutSet());

            if (mvc.getCutSet().size() == k) {
                // Check if both L and R have size >= 2, or M is not independent in original graph
                if ((mvc.getLset().size() >= 2 && mvc.getRset().size() >= 2)
                        || !isIndependentSetGraph(mvc.getCutSet())) {
                    completeXY(mvc.getLset(), mvc.getRset(), D, sx, sy, false);
                    return true;
                }

                if (v.getNumEdges() <= k && w.getNumEdges() <= k) {
                    // Contract v and w to their neighbors
                    LinkedHashSet<BranchNode> vN = H.getNeighborsForNode(v, false);
                    LinkedHashSet<BranchNode> wN = H.getNeighborsForNode(w, false);
                    Object[] vNa = vN.toArray();
                    Object[] wNa = wN.toArray();

                    for (Object vNeighObj : vNa) {
                        BranchNode vNeigh = (BranchNode) vNeighObj;
                        if (!wN.contains(vNeigh)) {
                            for (Object wNeighObj : wNa) {
                                BranchNode wNeigh = (BranchNode) wNeighObj;
                                if (!vN.contains(wNeigh)) {
                                    if (!H.edgeExists(vNeigh.getIndex(), wNeigh.getIndex())) {
                                        int[] xyInd = new int[2];
                                        BranchTree modH = contractV(H, v, vNeigh, vNa, w, wNeigh, wNa, xyInd);
                                        mvc = new MinVertexCut();
                                        mvc.findMinVertexCut(modH, xyInd[0], xyInd[1]);
                                        if (mvc.getCutSet().size() <= k) {
                                            mvc.addToLset(v.getPos1());
                                            mvc.addToRset(w.getPos1());
                                            completeXY(mvc.getLset(), mvc.getRset(), D, sx, sy, false);
                                            e.setM(mvc.getCutSet());
                                            return true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else if (v.getNumEdges() <= k || w.getNumEdges() <= k) {
                    // One has degree <= k, the other > k
                    BranchNode sn = v.getNumEdges() <= k ? v : w;
                    BranchNode ln = v.getNumEdges() <= k ? w : v;
                    LinkedHashSet<BranchNode> sN = H.getNeighborsForNode(sn, false);
                    Object[] sNa = sN.toArray();

                    for (Object sNeighObj : sNa) {
                        BranchNode sNeigh = (BranchNode) sNeighObj;
                        if (!H.edgeExists(sNeigh.getIndex(), ln.getIndex())) {
                            int[] xyInd = new int[2];
                            BranchTree modH = contractV(H, sn, sNeigh, sNa, xyInd, ln);
                            mvc = new MinVertexCut();
                            mvc.findMinVertexCut(modH, xyInd[0], xyInd[1]);
                            if (mvc.getCutSet().size() <= k) {
                                mvc.addToLset(sn.getPos1());
                                completeXY(mvc.getLset(), mvc.getRset(), D, sx, sy, false);
                                e.setM(mvc.getCutSet());
                                return true;
                            }
                        }
                    }
                } else {
                    // Both have degree > k
                    completeXY(mvc.getLset(), mvc.getRset(), D, sx, sy, false);
                    e.setM(mvc.getCutSet());
                    return true;
                }
            }
        }
        return false;
    }

    /** Check if positions in set s are independent in the original interaction graph. */
    private boolean isIndependentSetGraph(LinkedHashSet<Integer> s) {
        Integer[] sA = s.toArray(new Integer[0]);
        for (int i = 0; i < sA.length; i++) {
            for (int j = i + 1; j < sA.length; j++) {
                if (sA[i] >= 0 && sA[j] >= 0 && interactionGraph.hasEdge(sA[i], sA[j]))
                    return false;
            }
        }
        return true;
    }

    // ========== Vertex contraction ==========

    /** Contract nodes v→x and w→y in graph H. */
    private BranchTree contractV(BranchTree h, BranchNode v, BranchNode x, Object[] vNa,
                                  BranchNode w, BranchNode y, Object[] wNa, int[] xyInd) {
        BranchTree modH = h.deepCopy();
        BranchNode vMod = modH.getNode(v.getIndex());
        BranchNode xMod = modH.getNode(x.getIndex());
        BranchNode wMod = modH.getNode(w.getIndex());
        BranchNode yMod = modH.getNode(y.getIndex());
        unifyEdges(modH, vNa, xMod);
        unifyEdges(modH, wNa, yMod);
        modH.deleteNode(vMod.getIndex());
        modH.deleteNode(wMod.getIndex());
        xyInd[0] = xMod.getIndex();
        xyInd[1] = yMod.getIndex();
        return modH;
    }

    /** Contract node v→x in graph H (single contraction). */
    private BranchTree contractV(BranchTree h, BranchNode v, BranchNode x, Object[] vNa,
                                  int[] xyInd, BranchNode ln) {
        BranchTree modH = h.deepCopy();
        BranchNode vMod = modH.getNode(v.getIndex());
        BranchNode xMod = modH.getNode(x.getIndex());
        BranchNode lnMod = modH.getNode(ln.getIndex());
        unifyEdges(modH, vNa, xMod);
        modH.deleteNode(vMod.getIndex());
        xyInd[0] = xMod.getIndex();
        xyInd[1] = lnMod.getIndex();
        return modH;
    }

    private void unifyEdges(BranchTree h, Object[] vNa, BranchNode x) {
        for (Object nodeObj : vNa) {
            BranchNode node = (BranchNode) nodeObj;
            if (x.getIndex() != node.getIndex())
                h.addEdge(x.getIndex(), node.getIndex());
        }
    }

    // ========== Eigenvector heuristic ==========

    private void doEigen(BranchNode pn, LinkedHashSet<BranchEdge> sx, LinkedHashSet<BranchEdge> sy) {
        BranchEdge[] D = bt.getEdgesForNode(pn.getIndex());

        // Union of M-sets
        LinkedHashSet<Integer> S = new LinkedHashSet<>();
        for (BranchEdge e : D) S.addAll(e.getM());
        Object[] Sa = S.toArray();

        // Nv[i] = edges in D that contain vertex Sa[i] in their M-set
        BranchEdge[][] Nv = new BranchEdge[Sa.length][];
        for (int i = 0; i < Sa.length; i++) {
            List<BranchEdge> edges = new ArrayList<>();
            for (BranchEdge d : D) {
                if (d.getM().contains(Sa[i]))
                    edges.add(d);
            }
            Nv[i] = edges.toArray(new BranchEdge[0]);
        }

        // Build F matrix (Laplacian-like)
        double[][] F = new double[D.length][D.length];
        for (int i = 0; i < D.length; i++) {
            for (int j = 0; j < D.length; j++) {
                F[i][j] = 0.0;
                if (i != j) {
                    for (int k = 0; k < Nv.length; k++) {
                        if (isElement(Nv[k], D[i]) && isElement(Nv[k], D[j]))
                            F[i][j] += -1.0 / (Nv[k].length - 1);
                    }
                } else {
                    for (int k = 0; k < Nv.length; k++) {
                        if (isElement(Nv[k], D[i]))
                            F[i][j]++;
                    }
                }
            }
        }

        // Eigenvalue decomposition using Apache Commons Math 3
        RealMatrix Fm = new Array2DRowRealMatrix(F);
        EigenDecomposition ed = new EigenDecomposition(Fm);
        double[] eigenvalues = ed.getRealEigenvalues();

        // Find second smallest eigenvalue
        double[] sorted = Arrays.copyOf(eigenvalues, eigenvalues.length);
        Arrays.sort(sorted);
        double secondSmallest = sorted.length > 1 ? sorted[1] : sorted[0];
        int evlind = -1;
        for (int i = 0; i < eigenvalues.length; i++) {
            if (eigenvalues[i] == secondSmallest) {
                evlind = i;
                break;
            }
        }

        // Get eigenvector for second smallest eigenvalue
        double[] secondEvc = ed.getEigenvector(evlind).toArray();

        // Sort D by eigenvector component
        double[] secondEvcSorted = Arrays.copyOf(secondEvc, secondEvc.length);
        Arrays.sort(secondEvcSorted);
        BranchEdge[] Dsorted = new BranchEdge[D.length];
        boolean[] used = new boolean[D.length];
        for (int i = 0; i < secondEvcSorted.length; i++) {
            for (int j = 0; j < secondEvc.length; j++) {
                if (!used[j] && secondEvc[j] == secondEvcSorted[i]) {
                    Dsorted[i] = D[j];
                    used[j] = true;
                    break;
                }
            }
        }

        // Partition: first |D|/3 to sx, last |D|/3 to sy
        int numE = (int) Math.ceil(D.length / 3.0);
        for (int i = 0; i < numE; i++) {
            sx.add(Dsorted[i]);
            sy.add(Dsorted[Dsorted.length - 1 - i]);
        }

        // Build helper graph H and find min vertex cut
        BranchTree H = constructGraphH(Sa, D);
        int[] vi = new int[2];
        H = modifyGraphH(H, sx, sy, vi);

        MinVertexCut mvc = new MinVertexCut();
        mvc.findMinVertexCut(H.deepCopy(), vi[0], vi[1]);
        mvc.getCutSet().remove(-1);
        mvc.getLset().remove(-1);
        mvc.getRset().remove(-1);

        // Complete the partition
        completeXY(mvc.getLset(), mvc.getRset(), D, sx, sy, false);
    }

    // ========== Helper graph construction ==========

    /**
     * Construct graph H with nodes = vertices in Sa (M-set positions),
     * edges between positions that co-occur in some M-set.
     */
    private BranchTree constructGraphH(Object[] Sa, BranchEdge[] D) {
        BranchTree H = new BranchTree();
        for (Object posObj : Sa) {
            int pos = (Integer) posObj;
            H.addNode(new BranchNode(true, pos, -1)); // vertex identity in pos1
        }
        for (int i = 0; i < H.getNumNodes(); i++) {
            int v1 = H.getNode(i).getPos1();
            for (int j = i + 1; j < H.getNumNodes(); j++) {
                int v2 = H.getNode(j).getPos1();
                for (BranchEdge d : D) {
                    if (d.getM().contains(v1) && d.getM().contains(v2)) {
                        H.addEdge(i, j);
                        break;
                    }
                }
            }
        }
        return H;
    }

    /**
     * Modify graph H by adding source and sink nodes connected to vertices in sx/sy M-sets.
     */
    private BranchTree modifyGraphH(BranchTree H, LinkedHashSet<BranchEdge> sx,
                                     LinkedHashSet<BranchEdge> sy, int[] vi) {
        BranchNode va = new BranchNode(true, -1, -1); // source sentinel
        H.addNode(va);
        addEdgesToArtificialNode(H, va, sx);

        BranchNode vb = new BranchNode(true, -1, -1); // sink sentinel
        H.addNode(vb);
        addEdgesToArtificialNode(H, vb, sy);

        vi[0] = va.getIndex();
        vi[1] = vb.getIndex();
        return H;
    }

    /** Add edges from artificial node vn to all positions in M-sets of the given edges. */
    private void addEdgesToArtificialNode(BranchTree H, BranchNode vn, LinkedHashSet<BranchEdge> edgeSet) {
        LinkedHashSet<Integer> positions = new LinkedHashSet<>();
        for (BranchEdge be : edgeSet) {
            positions.addAll(be.getM());
        }
        for (int pos : positions) {
            for (int j = 0; j < H.getNumNodes(); j++) {
                if (H.getNode(j).getPos1() == pos) {
                    H.addEdge(vn.getIndex(), H.getNode(j).getIndex());
                    break;
                }
            }
        }
    }

    // ========== Edge partition completion ==========

    /**
     * Complete the partition of edges D into sx and sy based on L-set and R-set.
     * For each unassigned edge in D:
     *   - if M intersects only L: add to sx
     *   - if M intersects only R: add to sy
     *   - if M intersects neither: add to sy (default)
     */
    private void completeXY(LinkedHashSet<Integer> Lset, LinkedHashSet<Integer> Rset,
                             BranchEdge[] D, LinkedHashSet<BranchEdge> sx,
                             LinkedHashSet<BranchEdge> sy, boolean inX) {
        for (BranchEdge d : D) {
            if (!sx.contains(d) && !sy.contains(d)) {
                LinkedHashSet<Integer> eM = d.getM();
                LinkedHashSet<Integer> intersectL = new LinkedHashSet<>(Lset);
                intersectL.retainAll(eM);
                LinkedHashSet<Integer> intersectR = new LinkedHashSet<>(Rset);
                intersectR.retainAll(eM);

                if (intersectR.isEmpty() && intersectL.isEmpty()) {
                    if (inX) sx.add(d);
                    else sy.add(d);
                } else if (intersectR.isEmpty()) {
                    sx.add(d);
                } else if (intersectL.isEmpty()) {
                    sy.add(d);
                } else {
                    // Edge connects both sides -- add to the larger set
                    if (sx.size() <= sy.size()) sx.add(d);
                    else sy.add(d);
                }
            }
        }
    }

    // ========== Split execution ==========

    private void performSplit(BranchNode bn, LinkedHashSet<BranchEdge> sx, LinkedHashSet<BranchEdge> sy) {
        BranchNode bx = new BranchNode(false, -1, -1);
        bt.addNode(bx);
        BranchNode by = new BranchNode(false, -1, -1);
        bt.addNode(by);

        BranchEdge[] bnEdges = bt.getEdgesForNode(bn.getIndex());

        for (BranchEdge bnEdge : bnEdges) {
            BranchNode otherNode;
            if (bnEdge.getn1().getIndex() == bn.getIndex())
                otherNode = bnEdge.getn2();
            else
                otherNode = bnEdge.getn1();

            int eInd;
            if (sx.contains(bnEdge))
                eInd = bt.addEdge(bx.getIndex(), otherNode.getIndex());
            else
                eInd = bt.addEdge(by.getIndex(), otherNode.getIndex());

            BranchEdge ce = bt.getEdge(eInd);
            ce.setM(bnEdge.getM());
            ce.setChecked(otherNode, bnEdge.isChecked(otherNode));
            if (sx.contains(bnEdge))
                ce.setChecked(bx, bnEdge.isChecked(bn));
            else
                ce.setChecked(by, bnEdge.isChecked(bn));
        }

        bt.deleteNode(bn.getIndex());

        int bxbyInd = bt.addEdge(bx.getIndex(), by.getIndex());
        BranchEdge bxby = bt.getEdge(bxbyInd);
        compMLRsets(bxby, sx, sy);
    }

    /** Compute M-set for new edge = intersection of union(M-sets in sx) and union(M-sets in sy). */
    private void compMLRsets(BranchEdge e, LinkedHashSet<BranchEdge> sx, LinkedHashSet<BranchEdge> sy) {
        LinkedHashSet<Integer> uMsx = new LinkedHashSet<>();
        for (BranchEdge edge : sx) uMsx.addAll(edge.getM());

        LinkedHashSet<Integer> uMsy = new LinkedHashSet<>();
        for (BranchEdge edge : sy) uMsy.addAll(edge.getM());

        LinkedHashSet<Integer> xyM = new LinkedHashSet<>(uMsx);
        xyM.retainAll(uMsy);
        e.setM(xyM);
    }

    // ========== Utility ==========

    private boolean isElement(Object[] a, Object e) {
        for (Object obj : a) {
            if (obj == e) return true;
        }
        return false;
    }

    // ========== Tree Rooting (BWM*-style) ==========

    /**
     * Convert the unrooted BranchTree into a rooted binary tree.
     * Following BWM*'s transformRootedTree():
     * 1. Create virtual root (name=-3) and s-node (name=-2)
     * 2. Root edge (root→s) has empty M-set, isRootEdge=true
     * 3. Use edge[0] as the split edge: its two endpoints become s's children
     * 4. Recursively assign children for internal nodes
     *
     * @return the virtual root RootedTreeNode
     */
    public RootedTreeNode rootBranchTree(RCs rcs) {
        return rootBranchTree(rcs, 0);
    }

    public RootedTreeNode rootBranchTree(RCs rcs, int splitEdgeIndex) {
        if (bt.getNumEdges() == 0) {
            return null;
        }
        if (bt.getNumEdges() == 1) {
            return rootSingleEdgeTree(rcs);
        }
        if (splitEdgeIndex < 0 || splitEdgeIndex >= bt.getNumEdges()) {
            throw new IllegalArgumentException("Invalid branch decomposition split edge index "
                    + splitEdgeIndex + " for " + bt.getNumEdges() + " edges");
        }

        // Map from BranchNode index → RootedTreeNode
        Map<Integer, RootedTreeNode> nodeMap = new HashMap<>();
        for (int i = 0; i < bt.getNumNodes(); i++) {
            BranchNode bn = bt.getNode(i);
            nodeMap.put(i, new RootedTreeNode(i, bn.getIsLeaf(), bn.getPos1(), bn.getPos2()));
        }

        // Create virtual root and s-node
        RootedTreeNode root = new RootedTreeNode(-3, false, -1, -1);
        RootedTreeNode s = new RootedTreeNode(-2, false, -1, -1);

        // Root edge: root → s, M = empty, isRootEdge = true
        RootedTreeEdge rootEdge = new RootedTreeEdge(root, s, new LinkedHashSet<>(), true, rcs);
        s.setChildOfEdge(rootEdge);
        s.setParent(root);
        root.setLeftChild(s);

        // Use the requested split edge. The legacy behavior is splitEdgeIndex=0.
        BranchEdge splitEdge = bt.getEdge(splitEdgeIndex);
        LinkedHashSet<Integer> splitM = splitEdge.getM();
        int n1Idx = splitEdge.getn1().getIndex();
        int n2Idx = splitEdge.getn2().getIndex();

        RootedTreeNode leftNode = nodeMap.get(n1Idx);
        RootedTreeNode rightNode = nodeMap.get(n2Idx);

        // s → leftNode (with split M-set)
        RootedTreeEdge leftEdge = new RootedTreeEdge(s, leftNode, splitM, false, rcs);
        leftNode.setChildOfEdge(leftEdge);
        leftNode.setParent(s);
        s.setLeftChild(leftNode);

        // s → rightNode (with split M-set)
        RootedTreeEdge rightEdge = new RootedTreeEdge(s, rightNode, splitM, false, rcs);
        rightNode.setChildOfEdge(rightEdge);
        rightNode.setParent(s);
        s.setRightChild(rightNode);

        // Track which edges have been used (the split edge is used)
        Set<Integer> usedEdges = new HashSet<>();
        usedEdges.add(splitEdgeIndex);

        // Recursively build subtrees
        if (!leftNode.getIsLeaf()) {
            buildRootedHelper(leftNode, nodeMap, usedEdges, rcs);
        }
        if (!rightNode.getIsLeaf()) {
            buildRootedHelper(rightNode, nodeMap, usedEdges, rcs);
        }

        return root;
    }

    private RootedTreeNode rootSingleEdgeTree(RCs rcs) {
        BranchEdge onlyEdge = bt.getEdge(0);
        BranchNode n1 = onlyEdge.getn1();
        BranchNode n2 = onlyEdge.getn2();
        BranchNode leaf = n1.getIsLeaf() ? n1 : (n2.getIsLeaf() ? n2 : null);
        if (leaf == null) {
            throw new RuntimeException("Single-edge branch tree has no leaf endpoint");
        }

        RootedTreeNode root = new RootedTreeNode(-3, false, -1, -1);
        RootedTreeNode leafNode = new RootedTreeNode(
                leaf.getIndex(), true, leaf.getPos1(), leaf.getPos2());
        RootedTreeEdge rootEdge = new RootedTreeEdge(
                root, leafNode, new LinkedHashSet<>(), true, rcs);

        leafNode.setChildOfEdge(rootEdge);
        leafNode.setParent(root);
        root.setLeftChild(leafNode);
        return root;
    }

    /**
     * For an internal node that already has its parent set,
     * find the 2 remaining incident edges in the unrooted tree
     * and assign them as left/right children.
     */
    private void buildRootedHelper(RootedTreeNode node, Map<Integer, RootedTreeNode> nodeMap,
                                    Set<Integer> usedEdges, RCs rcs) {
        int nodeIdx = node.getName();

        // Find the 2 unused edges incident to this node
        List<Integer> childEdgeIndices = new ArrayList<>();
        for (int i = 0; i < bt.getNumEdges(); i++) {
            if (usedEdges.contains(i)) continue;
            BranchEdge be = bt.getEdge(i);
            if (be.getn1().getIndex() == nodeIdx || be.getn2().getIndex() == nodeIdx) {
                childEdgeIndices.add(i);
            }
        }

        // Degenerate decompositions for tiny sparse graphs can produce unary
        // internal nodes. Keep those rooted instead of failing before DP starts.
        if (childEdgeIndices.size() > 2) {
            throw new RuntimeException("Internal node " + nodeIdx + " has " + childEdgeIndices.size()
                    + " remaining edges (expected <= 2)");
        }

        for (int side = 0; side < childEdgeIndices.size(); side++) {
            int edgeIdx = childEdgeIndices.get(side);
            usedEdges.add(edgeIdx);
            BranchEdge be = bt.getEdge(edgeIdx);

            // The child is the node on the other end
            int childIdx = (be.getn1().getIndex() == nodeIdx) ?
                    be.getn2().getIndex() : be.getn1().getIndex();
            RootedTreeNode childNode = nodeMap.get(childIdx);

            RootedTreeEdge edge = new RootedTreeEdge(node, childNode, be.getM(), false, rcs);
            childNode.setChildOfEdge(edge);
            childNode.setParent(node);

            if (side == 0) node.setLeftChild(childNode);
            else node.setRightChild(childNode);

            if (!childNode.getIsLeaf()) {
                buildRootedHelper(childNode, nodeMap, usedEdges, rcs);
            }
        }
    }

}
