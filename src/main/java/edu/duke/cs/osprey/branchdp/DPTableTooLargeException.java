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

import java.util.Arrays;

/**
 * Dense Java arrays cannot index DP tables above Integer.MAX_VALUE states.
 */
public class DPTableTooLargeException extends IllegalStateException {

    public final long mStates;
    public final int[] mPositions;
    public final String stateName;

    public DPTableTooLargeException(String stateName, long mStates,
                                    int[] mPositions, String detail) {
        super(BranchDpConfig.getBackendLogPrefix() + " status=TooLargeForDenseDP, state=" + stateName
                + ", mStates=" + mStates
                + ", positions=" + Arrays.toString(mPositions)
                + ". " + detail);
        this.stateName = stateName;
        this.mStates = mStates;
        this.mPositions = Arrays.copyOf(mPositions, mPositions.length);
    }
}
