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

import java.io.*;
import java.util.LinkedHashSet;

/**
 * A tree data structure for branch decomposition.
 * Stores arrays of BranchNode and BranchEdge objects.
 *
 * Adapted from BWM's BranchDecomposition.BranchTree.
 */
public class BranchTree implements Serializable {

    private BranchNode[] bn;
    private int numNodes;
    private BranchEdge[] be;
    private int numEdges;

    public BranchTree() {
        bn = new BranchNode[50];
        numNodes = 0;
        be = new BranchEdge[50];
        numEdges = 0;
    }

    public void addNode(BranchNode tn) {
        if (numNodes >= bn.length)
            bn = doubleSize(bn);
        bn[numNodes] = tn;
        bn[numNodes].setIndex(numNodes);
        numNodes++;
    }

    public void deleteNode(int index) {
        deleteEdges(index);
        for (int i = index; i < numNodes - 1; i++) {
            bn[i] = bn[i + 1];
            bn[i].setIndex(i);
        }
        bn[numNodes - 1] = null;
        numNodes--;
    }

    public int addEdge(int i1, int i2) {
        if (!edgeExists(i1, i2)) {
            if (numEdges >= be.length)
                be = doubleSize(be);
            be[numEdges] = new BranchEdge(bn[i1], bn[i2]);
            bn[i1].addEdge();
            bn[i2].addEdge();
            numEdges++;
            return numEdges - 1;
        }
        return -1;
    }

    public boolean edgeExists(int i1, int i2) {
        for (int i = 0; i < numEdges; i++) {
            int n1 = be[i].getn1().getIndex();
            int n2 = be[i].getn2().getIndex();
            if ((n1 == i1 && n2 == i2) || (n1 == i2 && n2 == i1))
                return true;
        }
        return false;
    }

    public void deleteEdge(int index) {
        bn[be[index].getn1().getIndex()].deleteEdge();
        bn[be[index].getn2().getIndex()].deleteEdge();
        for (int i = index; i < numEdges - 1; i++) {
            be[i] = be[i + 1];
        }
        be[numEdges - 1] = null;
        numEdges--;
    }

    private void deleteEdges(int index) {
        int i = 0;
        while (i < numEdges) {
            if (be[i].getn1().getIndex() == index || be[i].getn2().getIndex() == index)
                deleteEdge(i);
            else
                i++;
        }
    }

    public BranchNode getNode(int index) {
        return bn[index];
    }

    public int getNumNodes() {
        return numNodes;
    }

    public BranchEdge getEdge(int index) {
        return be[index];
    }

    public int getNumEdges() {
        return numEdges;
    }

    public BranchEdge[] getEdgesForNode(int index) {
        BranchEdge[] bnEdges = new BranchEdge[bn[index].getNumEdges()];
        int c = 0;
        for (int i = 0; i < numEdges; i++) {
            if (be[i].getn1().getIndex() == index || be[i].getn2().getIndex() == index) {
                bnEdges[c] = be[i];
                c++;
            }
            if (c >= bnEdges.length)
                break;
        }
        return bnEdges;
    }

    public LinkedHashSet<BranchNode> getNeighborsForNode(BranchNode node, boolean directed) {
        LinkedHashSet<BranchNode> vN = new LinkedHashSet<>();
        for (int i = 0; i < numEdges; i++) {
            if (be[i].getn1().getIndex() == node.getIndex())
                vN.add(be[i].getn2());
            else if (!directed && be[i].getn2().getIndex() == node.getIndex())
                vN.add(be[i].getn1());
        }
        return vN;
    }

    private BranchEdge[] doubleSize(BranchEdge[] b) {
        BranchEdge[] tmp = new BranchEdge[b.length * 2];
        System.arraycopy(b, 0, tmp, 0, b.length);
        return tmp;
    }

    private BranchNode[] doubleSize(BranchNode[] b) {
        BranchNode[] tmp = new BranchNode[b.length * 2];
        System.arraycopy(b, 0, tmp, 0, b.length);
        return tmp;
    }

    public BranchTree deepCopy() {
        try {
            ByteArrayOutputStream b = new ByteArrayOutputStream();
            ObjectOutputStream fout = new ObjectOutputStream(b);
            fout.writeObject(this);
            fout.flush();
            fout.close();
            ObjectInputStream fin = new ObjectInputStream(new ByteArrayInputStream(b.toByteArray()));
            return (BranchTree) fin.readObject();
        } catch (Exception e) {
            throw new RuntimeException("BranchTree deep copy failed", e);
        }
    }

    /** Returns the maximum M-set size across all edges (the branchwidth). */
    public int getBranchwidth() {
        int bw = 0;
        for (int i = 0; i < numEdges; i++) {
            bw = Math.max(bw, be[i].getM().size());
        }
        return bw;
    }
}
