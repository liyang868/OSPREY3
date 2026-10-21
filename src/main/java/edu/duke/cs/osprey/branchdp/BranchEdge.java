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
import java.util.LinkedHashSet;

/**
 * An edge in a branch decomposition tree.
 * Each edge stores the M-set (middle set): the set of positions that appear on both sides of the cut.
 *
 * Adapted from BWM's BranchDecomposition.BranchEdge, using Integer position indices instead of String vertex names.
 */
public class BranchEdge implements Serializable {

    private BranchNode n1;
    private BranchNode n2;
    private LinkedHashSet<Integer> M;
    private boolean checkedn1;
    private boolean checkedn2;

    public BranchEdge(BranchNode bn1, BranchNode bn2) {
        n1 = bn1;
        n2 = bn2;
        M = new LinkedHashSet<>();
        checkedn1 = false;
        checkedn2 = false;
    }

    public LinkedHashSet<Integer> getM() {
        return M;
    }

    public BranchNode getn1() {
        return n1;
    }

    public BranchNode getn2() {
        return n2;
    }

    public void setM(LinkedHashSet<Integer> nM) {
        M = new LinkedHashSet<>(nM);
    }

    public void addM(int pos) {
        M.add(pos);
    }

    public boolean isChecked(BranchNode bn) {
        if (bn.getIndex() == n1.getIndex())
            return checkedn1;
        else if (bn.getIndex() == n2.getIndex())
            return checkedn2;
        else {
            throw new IllegalArgumentException("Edge not incident to node " + bn.getIndex());
        }
    }

    public void setChecked(BranchNode bn, boolean c) {
        if (bn.getIndex() == n1.getIndex())
            checkedn1 = c;
        else if (bn.getIndex() == n2.getIndex())
            checkedn2 = c;
        else {
            throw new IllegalArgumentException("Edge not incident to node " + bn.getIndex());
        }
    }
}
