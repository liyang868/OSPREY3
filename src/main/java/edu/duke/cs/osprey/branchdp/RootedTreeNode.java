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

/**
 * A node in the rooted branch decomposition tree.
 *
 * Leaf nodes represent edges in the original interaction graph (storing two position endpoints).
 * Internal nodes are split points. The virtual root (name=-3) and s-node (name=-2) are created
 * during tree rooting.
 *
 * Adapted from BWM*'s TreeNode.
 */
public class RootedTreeNode {

    private final int name;
    private final boolean isLeaf;
    private final int graphV1;  // interaction graph position 1 (-1 for internal)
    private final int graphV2;  // interaction graph position 2 (-1 for internal)

    private RootedTreeNode parent;
    private RootedTreeNode leftChild;
    private RootedTreeNode rightChild;
    private RootedTreeEdge childOfEdge;  // edge connecting this node to its parent

    public RootedTreeNode(int name, boolean isLeaf, int v1, int v2) {
        this.name = name;
        this.isLeaf = isLeaf;
        this.graphV1 = v1;
        this.graphV2 = v2;
    }

    public int getName() { return name; }
    public boolean getIsLeaf() { return isLeaf; }
    public int getV1() { return graphV1; }
    public int getV2() { return graphV2; }

    public RootedTreeNode getParent() { return parent; }
    public void setParent(RootedTreeNode p) { this.parent = p; }

    public RootedTreeNode getLeftChild() { return leftChild; }
    public void setLeftChild(RootedTreeNode lc) { this.leftChild = lc; }

    public RootedTreeNode getRightChild() { return rightChild; }
    public void setRightChild(RootedTreeNode rc) { this.rightChild = rc; }

    public RootedTreeEdge getChildOfEdge() { return childOfEdge; }
    public void setChildOfEdge(RootedTreeEdge e) { this.childOfEdge = e; }

    @Override
    public String toString() {
        if (isLeaf) return "Leaf(" + name + ": " + graphV1 + "-" + graphV2 + ")";
        return "Internal(" + name + ")";
    }
}
