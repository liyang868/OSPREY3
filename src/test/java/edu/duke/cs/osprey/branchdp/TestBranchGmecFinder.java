package edu.duke.cs.osprey.branchdp;

import edu.duke.cs.osprey.astar.conf.RCs;
import edu.duke.cs.osprey.confspace.ConfSearch;
import edu.duke.cs.osprey.ematrix.EnergyMatrix;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class TestBranchGmecFinder {

    @TempDir Path temporary;

    @Test public void matchesEnumerationWithoutChangingSamplingTablesInEveryStorageMode() throws Exception {
        Properties saved = (Properties) System.getProperties().clone();
        try {
            System.setProperty("branchdp.dp.gpu", "false");
            System.setProperty("branchdp.dp.shardSize", "3");
            System.setProperty("branchdp.dp.mmap.chunkEntries", "3");
            System.setProperty("branchdp.dp.mmap.dir", temporary.toString());
            for (String mode : List.of("dense", "sharded", "mmap")) {
                System.setProperty("branchdp.dp.tableMode", mode);
                for (int seed = 0; seed < 3; seed++) {
                    int[] counts = {5, 4, 3, 5, 2};
                    RCs allowed = new RCs(new int[][]{{0, 2, 4}, {1, 3}, {0, 2}, {2, 4}, {0, 1}});
                    InteractionGraph graph = InteractionGraph.buildFromEdges(5, List.of(
                            new int[]{0, 1}, new int[]{1, 2}, new int[]{2, 3},
                            new int[]{3, 4}, new int[]{0, 4}));
                    EnergyMatrix proposal = new EnergyMatrix(5, counts, 0);
                    proposal.setConstTerm(17.25);
                    Random random = new Random(seed);
                    for (int i = 0; i < counts.length; i++) {
                        for (int a = 0; a < counts[i]; a++) {
                            proposal.setOneBody(i, a, random.nextDouble() * 4 - 2);
                            for (int j = 0; j < i; j++) for (int b = 0; b < counts[j]; b++) {
                                proposal.setPairwise(i, a, j, b,
                                        graph.hasEdge(i, j) ? random.nextDouble() * 8 - 4 : -1000);
                            }
                        }
                    }
                    RootedTreeNode root = root(graph, allowed);
                    try {
                        RootedTreeEdge.postOrderInitIncremental(root, proposal, proposal, graph, 0.6);
                        RootedTreeEdge.postOrderComputeFullDP(root);
                        List<RootedTreeEdge> edges = new ArrayList<>();
                        RootedTreeEdge.collectLambdaEdges(root, edges);
                        List<Double> before = tableValues(edges);
                        long filesBefore;
                        try (var files = Files.list(temporary)) { filesBefore = files.count(); }

                        ConfSearch.ScoredConf gmec = BranchGmecFinder.find(root, proposal, graph, allowed);
                        assertNotNull(gmec);
                        assertEquals(enumerate(proposal, graph, allowed, new int[5], 0), gmec.getScore(), 1e-10);
                        assertEquals(gmec.getScore(), energy(proposal, graph, gmec.getAssignments()), 1e-10);
                        for (int i = 0; i < 5; i++) {
                            int chosen = gmec.getAssignments()[i];
                            assertTrue(Arrays.stream(allowed.get(i)).anyMatch(rc -> rc == chosen));
                        }
                        assertEquals(before, tableValues(edges));
                        assertArrayEquals(gmec.getAssignments(),
                                BranchGmecFinder.find(root, proposal, graph, allowed).getAssignments());
                        try (var files = Files.list(temporary)) { assertEquals(filesBefore, files.count()); }
                    } finally {
                        RootedTreeEdge.postOrderReleaseLargeMemory(root);
                    }
                    try (var files = Files.list(temporary)) { assertEquals(0, files.count()); }
                }
            }
        } finally {
            System.setProperties(saved);
        }
    }

    @Test public void learnedProposalCorrectionsCanChangeTheGmec() {
        RCs allowed = new RCs(new int[][]{{0, 2}, {0, 2}});
        InteractionGraph graph = InteractionGraph.buildComplete(2);
        EnergyMatrix original = new EnergyMatrix(2, new int[]{3, 3}, 0);
        original.setOneBody(0, 2, 3.0);
        original.setOneBody(1, 2, 3.0);
        EnergyMatrix proposal = new EnergyMatrix(original);
        proposal.setConstTerm(9);
        proposal.setPairwise(0, 2, 1, 2, -20.0);
        RootedTreeNode root = root(graph, allowed);
        try {
            assertArrayEquals(new int[]{0, 0}, BranchGmecFinder.find(root, original, graph, allowed).getAssignments());
            var mode = BranchGmecFinder.find(root, proposal, graph, allowed);
            assertArrayEquals(new int[]{2, 2}, mode.getAssignments());
            assertEquals(-5, mode.getScore(), 1e-12);
        } finally {
            RootedTreeEdge.postOrderReleaseLargeMemory(root);
        }
    }

    @Test public void handlesEmptyAndSinglePositionSpacesAndNoFiniteConformation() {
        EnergyMatrix empty = new EnergyMatrix(0, new int[0], 0);
        empty.setConstTerm(7);
        var zero = BranchGmecFinder.find(null, empty, InteractionGraph.buildComplete(0), new RCs(new int[0][]));
        assertArrayEquals(new int[0], zero.getAssignments());
        assertEquals(7, zero.getScore());

        EnergyMatrix single = new EnergyMatrix(1, new int[]{5}, 0);
        single.setConstTerm(7);
        single.setOneBody(0, 1, 2.0);
        single.setOneBody(0, 4, -3.0);
        var mode = BranchGmecFinder.find(null, single, InteractionGraph.buildComplete(1), new RCs(new int[][]{{1, 4}}));
        assertArrayEquals(new int[]{4}, mode.getAssignments());
        assertEquals(4, mode.getScore());
        assertNull(BranchGmecFinder.find(null, single, InteractionGraph.buildComplete(1), new RCs(new int[][]{{}})));
        single.setOneBody(0, 1, Double.POSITIVE_INFINITY);
        single.setOneBody(0, 4, Double.NaN);
        assertNull(BranchGmecFinder.find(null, single, InteractionGraph.buildComplete(1), new RCs(new int[][]{{1, 4}})));
    }

    private static RootedTreeNode root(InteractionGraph graph, RCs allowed) {
        BranchDecomposition decomposition = new BranchDecomposition(graph);
        decomposition.compute();
        RootedTreeNode root = decomposition.rootBranchTree(allowed);
        RootedTreeEdge.postOrderCompLlambda(root);
        return root;
    }

    private static List<Double> tableValues(List<RootedTreeEdge> edges) {
        List<Double> values = new ArrayList<>();
        for (RootedTreeEdge edge : edges) for (long m = 0; m < edge.getMStateCount(); m++) {
            values.add(edge.getLogZLower(m));
            values.add(edge.getLogZUpper(m));
        }
        return values;
    }

    private static double enumerate(EnergyMatrix matrix, InteractionGraph graph, RCs rcs, int[] conf, int pos) {
        if (pos == conf.length) return energy(matrix, graph, conf);
        double best = Double.POSITIVE_INFINITY;
        for (int rc : rcs.get(pos)) {
            conf[pos] = rc;
            best = Math.min(best, enumerate(matrix, graph, rcs, conf, pos + 1));
        }
        return best;
    }

    private static double energy(EnergyMatrix matrix, InteractionGraph graph, int[] conf) {
        double energy = matrix.getConstTerm();
        for (int i = 0; i < conf.length; i++) {
            energy += matrix.getOneBody(i, conf[i]);
            for (int j = 0; j < i; j++) if (graph.hasEdge(i, j))
                energy += matrix.getPairwise(i, conf[i], j, conf[j]);
        }
        return energy;
    }
}
