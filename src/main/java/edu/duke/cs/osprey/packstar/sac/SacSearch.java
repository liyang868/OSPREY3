/* SPDX-License-Identifier: GPL-2.0-only */
package edu.duke.cs.osprey.packstar.sac;

import edu.duke.cs.osprey.confspace.SimpleConfSpace;
import edu.duke.cs.osprey.ematrix.EnergyMatrix;

import java.io.IOException;
import java.nio.file.Path;

/** Entry point for default SAC sequence search. */
public final class SacSearch {
    private final edu.duke.cs.osprey.packstar.search.SacSearch search;

    public SacSearch(Path outputDirectory, String designId) throws IOException {
        search = new edu.duke.cs.osprey.packstar.search.SacSearch(outputDirectory, designId);
    }

    public void run(SimpleConfSpace complex, EnergyMatrix complexEm, SimpleConfSpace protein, EnergyMatrix proteinEm,
                    SimpleConfSpace ligand, EnergyMatrix ligandEm) throws IOException {
        search.run(complex, complexEm, protein, proteinEm, ligand, ligandEm);
    }
}
