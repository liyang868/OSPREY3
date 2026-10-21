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
import edu.duke.cs.osprey.structure.Residue;

import java.util.*;

/**
 * Represents a residue interaction graph built from OSPREY's energy matrices.
 * Positions are vertices; edges connect positions with significant pairwise energy interactions.
 */
public class InteractionGraph {

    private final int numPositions;
    private final boolean[][] adjacency;
    private final List<int[]> edgeList;

    private InteractionGraph(int numPositions, boolean[][] adjacency) {
        this.numPositions = numPositions;
        this.adjacency = adjacency;
        this.edgeList = new ArrayList<>();
        for (int i = 0; i < numPositions; i++) {
            for (int j = i + 1; j < numPositions; j++) {
                if (adjacency[i][j]) {
                    edgeList.add(new int[]{i, j});
                }
            }
        }
    }

    /**
     * One omitted interaction edge and its worst-case absolute pairwise energy.
     * The residual value is the quantity used in sparse-to-full perturbation bounds:
     * |E_full(conf) - E_sparse(conf)| <= sum residuals over cut edges.
     */
    public static class CutEdge {
        public final int pos1;
        public final int pos2;
        public final double minDistance;
        public final double maxAbsEnergy;

        public CutEdge(int pos1, int pos2, double minDistance, double maxAbsEnergy) {
            this.pos1 = pos1;
            this.pos2 = pos2;
            this.minDistance = minDistance;
            this.maxAbsEnergy = maxAbsEnergy;
        }
    }

    /**
     * Build an interaction graph by cutting low-risk pairwise interactions until
     * a global omitted-energy budget is exhausted.
     *
     * Unlike distance cutoffs, this rule gives a direct perturbation bound:
     * |E_full(conf) - E_sparse(conf)| is bounded by the sum of omitted-edge residuals.
     */
    public static InteractionGraph buildWithResidualBudget(
            SimpleConfSpace confSpace, EnergyMatrix rigidEmat, EnergyMatrix minimizingEmat,
            RCs rcs, double residualBudget, boolean keepConnected) {

        int numPos = rcs.getNumPos();
        boolean[][] adj = new boolean[numPos][numPos];
        int[] degree = new int[numPos];
        List<CutEdge> candidates = new ArrayList<>();

        for (int i = 0; i < numPos; i++) {
            for (int j = i + 1; j < numPos; j++) {
                adj[i][j] = true;
                adj[j][i] = true;
                degree[i]++;
                degree[j]++;

                double minDist = calcMinTemplateDistance(confSpace, i, j);
                double maxAbsEnergy = calcMaxAbsPairEnergy(rigidEmat, minimizingEmat, rcs, i, j);
                if (Double.isFinite(maxAbsEnergy)) {
                    candidates.add(new CutEdge(i, j, minDist, maxAbsEnergy));
                }
            }
        }

        candidates.sort((a, b) -> {
            int energyCmp = Double.compare(a.maxAbsEnergy, b.maxAbsEnergy);
            if (energyCmp != 0) return energyCmp;
            return -Double.compare(a.minDistance, b.minDistance);
        });

        int edgesCut = 0;
        double residual = 0.0;
        double budget = Math.max(0.0, residualBudget);

        for (CutEdge edge : candidates) {
            if (residual + edge.maxAbsEnergy > budget + 1e-12) {
                continue;
            }
            if (keepConnected && wouldDisconnect(adj, degree, edge.pos1, edge.pos2)) {
                continue;
            }

            adj[edge.pos1][edge.pos2] = false;
            adj[edge.pos2][edge.pos1] = false;
            degree[edge.pos1]--;
            degree[edge.pos2]--;
            residual += edge.maxAbsEnergy;
            edgesCut++;

            System.out.println("Cutting (" + edge.pos1 + "," + edge.pos2
                    + "): residual " + String.format("%.4f", edge.maxAbsEnergy)
                    + ", distance " + String.format("%.2f", edge.minDistance)
                    + ", budgetUsed " + String.format("%.4f", residual)
                    + "/" + String.format("%.4f", budget));
        }

        int totalPossible = numPos * (numPos - 1) / 2;
        int edgesKept = totalPossible - edgesCut;
        System.out.println("InteractionGraph residual-budget cutoff: kept " + edgesKept + "/" + totalPossible
                + " edges (cut " + edgesCut + "), residualUpperBound="
                + String.format("%.6f", residual)
                + ", residualBudget=" + String.format("%.6f", budget)
                + ", keepConnected=" + keepConnected);

        return new InteractionGraph(numPos, adj);
    }

    private static boolean wouldDisconnect(boolean[][] adj, int[] degree, int cutI, int cutJ) {
        if (degree[cutI] <= 1 || degree[cutJ] <= 1) {
            return true;
        }

        adj[cutI][cutJ] = false;
        adj[cutJ][cutI] = false;
        boolean connected = isConnected(adj);
        adj[cutI][cutJ] = true;
        adj[cutJ][cutI] = true;

        return !connected;
    }

    private static boolean isConnected(boolean[][] adj) {
        int n = adj.length;
        if (n <= 1) return true;

        boolean[] seen = new boolean[n];
        ArrayDeque<Integer> queue = new ArrayDeque<>();
        seen[0] = true;
        queue.add(0);

        while (!queue.isEmpty()) {
            int i = queue.remove();
            for (int j = 0; j < n; j++) {
                if (adj[i][j] && !seen[j]) {
                    seen[j] = true;
                    queue.add(j);
                }
            }
        }

        for (boolean visited : seen) {
            if (!visited) return false;
        }
        return true;
    }

    private static double calcMinTemplateDistance(SimpleConfSpace confSpace, int i, int j) {
        SimpleConfSpace.Position posI = confSpace.positions.get(i);
        SimpleConfSpace.Position posJ = confSpace.positions.get(j);
        Residue resI = posI.strand.mol.getResByPDBResNumber(posI.resNum);
        Residue resJ = posJ.strand.mol.getResByPDBResNumber(posJ.resNum);
        return resI.distanceTo(resJ);
    }

    private static double calcMaxAbsPairEnergy(EnergyMatrix rigidEmat, EnergyMatrix minimizingEmat,
                                               RCs rcs, int i, int j) {
        double maxAbsEnergy = 0.0;
        for (int rci = 0; rci < rcs.getNum(i); rci++) {
            int rc1 = rcs.get(i, rci);
            for (int rcj = 0; rcj < rcs.getNum(j); rcj++) {
                int rc2 = rcs.get(j, rcj);
                double rigidE = rigidEmat.getPairwise(i, rc1, j, rc2);
                double minE = minimizingEmat.getPairwise(i, rc1, j, rc2);
                maxAbsEnergy = maxAbs(maxAbsEnergy, rigidE);
                maxAbsEnergy = maxAbs(maxAbsEnergy, minE);
            }
        }
        return maxAbsEnergy;
    }

    private static double maxAbs(double current, double value) {
        if (Double.isNaN(value)) return current;
        if (Double.isInfinite(value)) return Double.POSITIVE_INFINITY;
        return Math.max(current, Math.abs(value));
    }

    public boolean hasEdge(int pos1, int pos2) {
        return adjacency[pos1][pos2];
    }

    public int getNumPositions() {
        return numPositions;
    }

    public int getNumEdges() {
        return edgeList.size();
    }

    public List<int[]> getEdgeList() {
        return edgeList;
    }

    /**
     * Build a complete interaction graph where all position pairs are edges.
     * Matches the BWM* approach of using the full interaction graph without filtering.
     */
    public static InteractionGraph buildComplete(int numPositions) {
        boolean[][] adj = new boolean[numPositions][numPositions];
        for (int i = 0; i < numPositions; i++) {
            for (int j = i + 1; j < numPositions; j++) {
                adj[i][j] = true;
                adj[j][i] = true;
            }
        }
        return new InteractionGraph(numPositions, adj);
    }

    public static InteractionGraph buildFromEdges(int numPositions, Collection<int[]> edges) {
        boolean[][] adj = new boolean[numPositions][numPositions];
        for (int[] edge : edges) {
            if (edge == null || edge.length < 2) {
                throw new IllegalArgumentException("Edge must contain two positions.");
            }
            int i = edge[0];
            int j = edge[1];
            if (i < 0 || i >= numPositions || j < 0 || j >= numPositions) {
                throw new IllegalArgumentException("Edge position outside graph: " + i + "," + j);
            }
            if (i == j) {
                throw new IllegalArgumentException("Self edges are not supported: " + i);
            }
            adj[i][j] = true;
            adj[j][i] = true;
        }
        return new InteractionGraph(numPositions, adj);
    }
}
