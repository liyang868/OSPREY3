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

import java.util.LinkedHashSet;

/**
 * Computes minimum vertex cuts using Ford-Fulkerson max-flow.
 * Used in branch decomposition for finding 2- and 3-separations.
 *
 * Adapted from BWM's BranchDecomposition.MinVertexCut, using Integer position indices.
 */
public class MinVertexCut {

    private LinkedHashSet<Integer> Lset;
    private LinkedHashSet<Integer> cutSet;
    private LinkedHashSet<Integer> Rset;

    public MinVertexCut() {
    }

    /**
     * Find the min vertex cut for undirected graph H, from source node si to sink node ti.
     * The cut set is stored in cutSet, and the resulting two partitions are in Lset and Rset.
     * Graph H is not modified.
     */
    public void findMinVertexCut(BranchTree H, int si, int ti) {
        Lset = new LinkedHashSet<>();
        cutSet = new LinkedHashSet<>();
        Rset = new LinkedHashSet<>();

        // Transform H into a directed graph suitable for max flow
        int[][] vMap = new int[H.getNumNodes()][2];
        int[] splitV = new int[2 * H.getNumNodes()];
        BranchTree dH = transformGraph(H.deepCopy(), vMap, splitV);

        int siNew = vMap[si][1];
        int tiNew = vMap[ti][0];

        // Compute min cut using max flow
        LinkedHashSet<Integer> S1 = new LinkedHashSet<>();
        LinkedHashSet<Integer> S2 = new LinkedHashSet<>();
        compMinCut(dH, siNew, tiNew, S1, S2, splitV);

        // Extract minimum vertex cut and partition
        for (int i = 0; i < H.getNumNodes(); i++) {
            int pos = H.getNode(i).getPos1(); // vertex identity stored in pos1
            if (i == si)
                Lset.add(pos);
            else if (i == ti)
                Rset.add(pos);
            else {
                if (S1.contains(vMap[i][0]) && S1.contains(vMap[i][1]))
                    Lset.add(pos);
                else if (S2.contains(vMap[i][0]) && S2.contains(vMap[i][1]))
                    Rset.add(pos);
                else
                    cutSet.add(pos);
            }
        }

        // Remove sentinel values (-1 for internal nodes)
        Lset.remove(-1);
        cutSet.remove(-1);
        Rset.remove(-1);
    }

    private BranchTree transformGraph(BranchTree H, int[][] vMap, int[] splitV) {
        BranchTree dH = new BranchTree();
        for (int i = 0; i < H.getNumNodes(); i++) {
            BranchNode nodeH = H.getNode(i);
            BranchNode bn1 = new BranchNode(nodeH.getIsLeaf(), nodeH.getPos1(), nodeH.getPos2());
            dH.addNode(bn1);
            vMap[i][0] = bn1.getIndex();
            BranchNode bn2 = new BranchNode(nodeH.getIsLeaf(), nodeH.getPos1(), nodeH.getPos2());
            dH.addNode(bn2);
            vMap[i][1] = bn2.getIndex();
            splitV[vMap[i][0]] = vMap[i][1];
            splitV[vMap[i][1]] = vMap[i][0];
            dH.addEdge(vMap[i][0], vMap[i][1]);
        }
        for (int j = 0; j < H.getNumEdges(); j++) {
            BranchEdge be = H.getEdge(j);
            int i1 = be.getn1().getIndex();
            int i2 = be.getn2().getIndex();
            dH.addEdge(vMap[i1][1], vMap[i2][0]);
            dH.addEdge(vMap[i2][1], vMap[i1][0]);
        }
        return dH;
    }

    private void compMinCut(BranchTree dH, int s, int t, LinkedHashSet<Integer> S1,
                            LinkedHashSet<Integer> S2, int[] splitV) {
        int n = dH.getNumNodes();
        int[][] c = new int[n][n];
        int[][] f = new int[n][n];
        int[][] rc = new int[n][n];

        for (int i = 0; i < dH.getNumEdges(); i++) {
            BranchEdge curE = dH.getEdge(i);
            int n1 = curE.getn1().getIndex();
            int n2 = curE.getn2().getIndex();
            c[n1][n2] = 1;
            rc[n1][n2] = 1;
        }

        BreadthFirstSearch bfs = new BreadthFirstSearch(dH, s, t);
        int[] p;
        while ((p = bfs.getNextPath(rc)) != null) {
            int rcp = Integer.MAX_VALUE;
            for (int i = 0; i < p.length - 1; i++)
                rcp = Math.min(rcp, rc[p[i]][p[i + 1]]);
            for (int i = 0; i < p.length - 1; i++) {
                f[p[i]][p[i + 1]] += rcp;
                f[p[i + 1]][p[i]] = -f[p[i]][p[i + 1]];
            }
            for (int i = 0; i < p.length - 1; i++) {
                rc[p[i]][p[i + 1]] = c[p[i]][p[i + 1]] - f[p[i]][p[i + 1]];
                rc[p[i + 1]][p[i]] = c[p[i + 1]][p[i]] - f[p[i + 1]][p[i]];
            }
        }

        int[] lnNodes = getLayeredNetworkNodes(dH, s, t, c, f, splitV);
        for (int node : lnNodes)
            S1.add(node);
        for (int i = 0; i < n; i++)
            S2.add(i);
        S2.removeAll(S1);
    }

    private int[] getLayeredNetworkNodes(BranchTree dH, int s, int t, int[][] c, int[][] f, int[] splitV) {
        int n = dH.getNumNodes();

        // Set external edge capacities to infinity
        for (int i = 0; i < dH.getNumEdges(); i++) {
            BranchEdge curE = dH.getEdge(i);
            int n1 = curE.getn1().getIndex();
            int n2 = curE.getn2().getIndex();
            if (splitV[n1] != n2)
                c[n1][n2] = Integer.MAX_VALUE;
        }

        int[][] Vi = new int[n][];
        Vi[0] = new int[]{s};
        int[] Vall = new int[n];
        int curAll = 0;
        Vall[curAll++] = s;

        int ind = 0;
        boolean done = false;
        while (!done) {
            int[] T = new int[n];
            int curT = 0;
            for (int i = 0; i < Vi[ind].length; i++) {
                for (int j = 0; j < n; j++) {
                    if (!isElement(j, Vall, curAll)) {
                        if ((f[Vi[ind][i]][j] < c[Vi[ind][i]][j]) || (f[j][Vi[ind][i]] > 0)) {
                            T[curT++] = j;
                            Vall[curAll++] = j;
                        }
                    }
                }
            }
            if (curT == 0)
                done = true;
            else if (isElement(t, T, curT)) {
                Vi[ind + 1] = new int[]{t};
                done = true;
            } else {
                int[] tmp = new int[curT];
                System.arraycopy(T, 0, tmp, 0, curT);
                Vi[ind + 1] = tmp;
                ind++;
            }
        }

        int[] result = new int[curAll];
        System.arraycopy(Vall, 0, result, 0, curAll);
        return result;
    }

    private boolean isElement(int a, int[] A, int sizeA) {
        for (int i = 0; i < sizeA; i++) {
            if (A[i] == a)
                return true;
        }
        return false;
    }

    public LinkedHashSet<Integer> getCutSet() {
        return cutSet;
    }

    public LinkedHashSet<Integer> getLset() {
        return Lset;
    }

    public LinkedHashSet<Integer> getRset() {
        return Rset;
    }

    public void addToLset(int pos) {
        Lset.add(pos);
    }

    public void addToRset(int pos) {
        Rset.add(pos);
    }
}
