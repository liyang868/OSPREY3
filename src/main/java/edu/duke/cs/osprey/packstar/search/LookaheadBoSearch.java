/* SPDX-License-Identifier: GPL-2.0-only */
package edu.duke.cs.osprey.packstar.search;

import edu.duke.cs.osprey.branchdp.BranchDpConfig;
import edu.duke.cs.osprey.confspace.SimpleConfSpace;
import edu.duke.cs.osprey.ematrix.EnergyMatrix;

import java.io.IOException;
import java.nio.file.Path;

/** Online two-step Bayesian optimization of the three-state Em log binding score. */
public final class LookaheadBoSearch {
    public static final String ROUTE = "lookahead_bo";

    private final Path resultFile;
    private final long seed;
    private final int queries;
    private final double seconds;

    public LookaheadBoSearch(Path outputDirectory, String designId) {
        seed = Long.parseLong(System.getProperty("packstar.bo.seed", "42"));
        queries = Integer.parseInt(System.getProperty("packstar.bo.queries", "1000"));
        seconds = Double.parseDouble(System.getProperty("packstar.bo.seconds", "600"));
        if (queries < 1 || !Double.isFinite(seconds) || seconds <= 0) {
            throw new IllegalArgumentException("Lookahead BO needs positive query and time budgets");
        }
        resultFile = outputDirectory.resolve(designId + "_" + ROUTE + ".json");
    }

    public void run(SimpleConfSpace complex, EnergyMatrix complexEm, SimpleConfSpace protein, EnergyMatrix proteinEm,
                    SimpleConfSpace ligand, EnergyMatrix ligandEm) throws IOException {
        var model = new EmEnergyModel(new SimpleConfSpace[]{complex, protein, ligand},
            new EnergyMatrix[]{complexEm, proteinEm, ligandEm});
        try (var scope = BranchDpConfig.enterPackStarAliasScope()) {
            var warm = new EmSequenceSearch(model, seed, queries, seconds);
            warm.prepare();
            var search = new EmSequenceSearch(warm, seed);
            var planner = new LookaheadPlanner(search, seed);
            try {
                var pool = search.pool();
                while (!pool.actions().isEmpty()) {
                    var action = planner.choose(pool.actions());
                    var executed = search.execute(action, planner::observe);
                    planner.recordCost(action.kind, executed);
                    pool = search.pool();
                }
            } catch (EmSequenceSearch.Horizon end) {
                search.stop = "time_budget";
            }
            var result = search.result(ROUTE);
            result.put("bayes", planner.gp.report());
            result.put("lookahead", planner.report());
            EmSequenceSearch.writeJson(resultFile, result);
        }
    }
}
