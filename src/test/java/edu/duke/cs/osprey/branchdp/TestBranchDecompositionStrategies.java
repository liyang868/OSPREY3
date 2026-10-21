package edu.duke.cs.osprey.branchdp;

import org.junit.jupiter.api.Test;

import java.math.BigInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TestBranchDecompositionStrategies {

    @Test
    public void boundaryProductsRemainExactAcrossLongOverflow() throws Exception {
        int[] counts = {Integer.MAX_VALUE, Integer.MAX_VALUE, 3, 7, 11, 13};
        BranchDecomposition bd = new BranchDecomposition(InteractionGraph.buildComplete(6),
                BranchDecomposition.Strategy.WEIGHTED_HICKS, counts);
        var method = BranchDecomposition.class.getDeclaredMethod("stateCount", java.util.Collection.class);
        method.setAccessible(true);
        for (int mask = 0; mask < 64; mask++) {
            java.util.List<Integer> positions = new java.util.ArrayList<>();
            BigInteger expected = BigInteger.ONE;
            for (int pos = 0; pos < counts.length; pos++) if ((mask & (1 << pos)) != 0) {
                positions.add(pos);
                expected = expected.multiply(BigInteger.valueOf(counts[pos]));
            }
            assertEquals(expected, method.invoke(bd, positions));
            java.util.Collections.reverse(positions);
            assertEquals(expected, method.invoke(bd, positions));
        }
    }

    @Test
    public void defaultHicksBuildsRootableBranchTree() {
        InteractionGraph graph = InteractionGraph.buildComplete(5);
        BranchDecomposition bd = new BranchDecomposition(
                graph, BranchDecomposition.Strategy.HICKS);

        bd.compute();

        assertEquals(graph.getNumEdges(), countLeaves(bd.getTree()));
        assertTrue(bd.getBranchwidth() <= graph.getNumPositions());
        assertNotNull(bd.rootBranchTree(null));
    }

    @Test
    public void weightedHicksBuildsRootableTreeWithoutIncreasingWeightedMaximum() {
        InteractionGraph graph = InteractionGraph.buildComplete(6);
        int[] rcCounts = {97, 2, 3, 5, 7, 11};

        BranchDecomposition hicks = new BranchDecomposition(
                graph, BranchDecomposition.Strategy.HICKS, rcCounts);
        hicks.compute();
        BranchDecomposition weighted = new BranchDecomposition(
                graph, BranchDecomposition.Strategy.WEIGHTED_HICKS, rcCounts);
        weighted.compute();

        assertEquals(graph.getNumEdges(), countLeaves(weighted.getTree()));
        assertNotNull(weighted.rootBranchTree(null));
        assertTrue(maxBoundaryStates(weighted.getTree(), rcCounts)
                        .compareTo(maxBoundaryStates(hicks.getTree(), rcCounts)) <= 0,
                "weighted Hicks/NNI must not increase the maximum separator state product");
    }

    private static int countLeaves(BranchTree tree) {
        int count = 0;
        for (int i = 0; i < tree.getNumNodes(); i++) {
            if (tree.getNode(i).getIsLeaf()) {
                count++;
            }
        }
        return count;
    }

    private static BigInteger maxBoundaryStates(BranchTree tree, int[] rcCounts) {
        BigInteger max = BigInteger.ZERO;
        for (int edgeIndex = 0; edgeIndex < tree.getNumEdges(); edgeIndex++) {
            BigInteger states = BigInteger.ONE;
            for (int pos : tree.getEdge(edgeIndex).getM()) {
                states = states.multiply(BigInteger.valueOf(rcCounts[pos]));
            }
            max = max.max(states);
        }
        return max;
    }

}
