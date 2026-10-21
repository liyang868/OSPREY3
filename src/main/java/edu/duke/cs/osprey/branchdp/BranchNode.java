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

import java.io.Serializable;

/**
 * A node in a branch decomposition tree.
 * Leaf nodes represent edges in the original interaction graph (storing the two position endpoints).
 * Internal nodes are split points in the decomposition.
 *
 * Adapted from BWM's BranchDecomposition.BranchNode, using int position indices instead of String vertex names.
 */
public class BranchNode implements Serializable {

    private boolean isLeaf;
    private int graphPos1;  // first position index (-1 for internal nodes)
    private int graphPos2;  // second position index (-1 for internal nodes)
    private int numEdges;
    private int index;

    public BranchNode(boolean leafNode, int pos1, int pos2) {
        this.isLeaf = leafNode;
        this.graphPos1 = pos1;
        this.graphPos2 = pos2;
        this.numEdges = 0;
        this.index = -1;
    }

    public int getPos1() {
        return graphPos1;
    }

    public int getPos2() {
        return graphPos2;
    }

    public int getNumEdges() {
        return numEdges;
    }

    public boolean getIsLeaf() {
        return isLeaf;
    }

    public int getIndex() {
        return index;
    }

    public void setIndex(int i) {
        index = i;
    }

    public void addEdge() {
        numEdges++;
    }

    public void deleteEdge() {
        numEdges--;
    }
}
