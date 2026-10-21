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
import java.util.concurrent.ArrayBlockingQueue;

/**
 * BFS for finding augmenting paths in the max-flow algorithm.
 * Adapted from BWM's BranchDecomposition.BreadthFirstSearch.
 */
public class BreadthFirstSearch {

    private static final int WHITE = 0;
    private static final int GRAY = 1;
    private static final int BLACK = 2;

    private int[] color;
    private int[] d;
    private int[] pi;

    private BranchTree G;
    private int s;
    private int t;
    private int[][] adj;

    private ArrayBlockingQueue<Integer> q;

    public BreadthFirstSearch(BranchTree graph, int sInd, int tInd) {
        G = graph;
        s = sInd;
        t = tInd;
        int numV = G.getNumNodes();

        adj = new int[numV][];
        int[] curInd = new int[numV];
        for (int i = 0; i < G.getNumEdges(); i++) {
            BranchEdge curE = G.getEdge(i);
            int n1 = curE.getn1().getIndex();
            int n2 = curE.getn2().getIndex();

            if (adj[n1] == null) {
                adj[n1] = new int[curE.getn1().getNumEdges()];
                curInd[n1] = 0;
            }
            adj[n1][curInd[n1]] = n2;
            curInd[n1]++;

            if (adj[n2] == null) {
                adj[n2] = new int[curE.getn2().getNumEdges()];
                curInd[n2] = 0;
            }
            adj[n2][curInd[n2]] = n1;
            curInd[n2]++;
        }
    }

    public int[] getNextPath(int[][] rc) {
        init();
        while (!q.isEmpty()) {
            int u = q.remove();
            if (adj[u] == null) {
                color[u] = BLACK;
                continue;
            }
            for (int vc = 0; vc < adj[u].length; vc++) {
                int v = adj[u][vc];
                if (v == t) {
                    pi[v] = u;
                    return tracePath();
                } else if (color[v] == WHITE) {
                    if (rc[u][v] > 0) {
                        color[v] = GRAY;
                        d[v] = d[u] + 1;
                        pi[v] = u;
                        q.add(v);
                    }
                }
            }
            color[u] = BLACK;
        }
        return null;
    }

    private int[] tracePath() {
        LinkedHashSet<Integer> p = new LinkedHashSet<>();
        int curNode = t;
        while (curNode != s) {
            p.add(curNode);
            curNode = pi[curNode];
        }
        p.add(s);
        Object[] pA = p.toArray();
        int[] pAr = new int[pA.length];
        for (int i = 0; i < pAr.length; i++) {
            pAr[i] = (Integer) pA[pA.length - 1 - i];
        }
        return pAr;
    }

    public void init() {
        int numV = G.getNumNodes();
        color = new int[numV];
        d = new int[numV];
        pi = new int[numV];
        for (int i = 0; i < numV; i++) {
            color[i] = WHITE;
            d[i] = Integer.MAX_VALUE;
            pi[i] = -1;
        }
        color[s] = GRAY;
        q = new ArrayBlockingQueue<>(numV);
        q.add(s);
    }
}
