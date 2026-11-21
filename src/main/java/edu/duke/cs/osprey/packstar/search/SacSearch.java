/* SPDX-License-Identifier: GPL-2.0-only */
package edu.duke.cs.osprey.packstar.search;

import edu.duke.cs.osprey.branchdp.BranchDpConfig;
import edu.duke.cs.osprey.branchdp.BranchDpBackend.RootSelectionException;
import edu.duke.cs.osprey.branchdp.DPTableTooLargeException;
import edu.duke.cs.osprey.confspace.SimpleConfSpace;
import edu.duke.cs.osprey.ematrix.EnergyMatrix;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;

import static edu.duke.cs.osprey.packstar.search.EmSequenceSearch.*;
import static edu.duke.cs.osprey.packstar.search.EmEnergyModel.*;

/** Budgeted sequence search on the three-state Em log binding score. */
public final class SacSearch {
    private final Path policyFile, resultFile;
    private final SacPolicy policy;
    private final long seed;
    private final int queries, episodes;
    private final double seconds;
    private final boolean training;

    public SacSearch(Path outputDirectory, String designId) throws IOException {
        String mode = System.getProperty("packstar.sac.mode", "search");
        if (!mode.equals("train") && !mode.equals("search")) throw new IllegalArgumentException("SAC mode must be train or search");
        training = mode.equals("train");
        seed = Long.parseLong(System.getProperty("packstar.sac.seed", "42"));
        queries = Integer.parseInt(System.getProperty("packstar.sac.queries", "1000"));
        seconds = Double.parseDouble(System.getProperty("packstar.sac.seconds", "600"));
        episodes = training ? Integer.parseInt(System.getProperty("packstar.sac.episodes", "1")) : 1;
        if (queries < 1 || episodes < 1 || !Double.isFinite(seconds) || seconds <= 0) {
            throw new IllegalArgumentException("SAC needs positive query, time and episode budgets");
        }
        policyFile = Path.of(System.getProperty("packstar.sac.policy", outputDirectory.resolve(designId + "_sac_policy.json").toString()));
        resultFile = outputDirectory.resolve(designId + "_sac.json");
        policy = training ? new SacPolicy(seed) : SacPolicy.load(policyFile);
    }

    public void run(SimpleConfSpace complex, EnergyMatrix complexEm, SimpleConfSpace protein, EnergyMatrix proteinEm,
                    SimpleConfSpace ligand, EnergyMatrix ligandEm) throws IOException {
        var model = new EmEnergyModel(new SimpleConfSpace[]{complex, protein, ligand}, new EnergyMatrix[]{complexEm, proteinEm, ligandEm});
        try (var scope = BranchDpConfig.enterPackStarAliasScope()) {
            EmSequenceSearch warm = new EmSequenceSearch(model, seed, queries, seconds);
            warm.prepare();
            for (int epoch = 0; epoch < episodes; epoch++) {
                EmSequenceSearch search = new EmSequenceSearch(warm, seed + epoch);
                List<SacPolicy.Step> steps = runEpisode(search, seed + epoch);
                if (training) {
                    if (steps.isEmpty()) throw new IllegalStateException("No SAC transitions after warm initialization: " + search.stop);
                    policy.train(steps, (epoch + 1.0) / episodes);
                } else {
                    writeJson(resultFile, search.result("sac"));
                }
            }
            if (training) {
                policy.save(policyFile);
            }
        }
    }

    private static double scale(double value) { return Math.max(-5, Math.min(5, value / 10)); }
    private static double[] features(EmSequenceSearch search, Action a, Pool pool) {
        search.check();
        double n = Math.max(1, search.model.names.length), work = 0;
        for (int[] seq : a.candidates) {
            try {
                for (Cache cache : search.caches) work += cache.estimate(seq);
            } catch (RootSelectionException | DPTableTooLargeException unavailable) {
                work = Long.MAX_VALUE;
            }
        }
        work = Math.max(1, work);
        double cost = .00002 + work * search.secondsPerWork;
        long reuse = reuse(a, pool.actions());
        double novelty = search.novelty(a);
        Mutation m = a.match, evidence = m == null ? a.evidence : m;
        double gain = Math.max(0, a.hint), info = information(a, reuse, novelty);
        if (a.evidence != null && a.evidence.bounded()) {
            gain /= 1 + a.evidence.width() / 10;
        }
        boolean bound = m != null && m.bounded();
        if (bound) gain = Math.max(0, m.lower());
        double priority = (gain + info) / Math.sqrt(Math.max(.00002, cost));
        double boundary = 0, region = 0;
        if (evidence != null) for (Region r : evidence.regions()) if (r != null) {
            boundary += r.boundary().cardinality();
            region += r.variables().cardinality();
        }
        double compatible = m == null ? 0 : pool.elites().stream().filter(e -> m.compatible(e.seq())).count() / 3.0;
        boolean outside = m != null;
        if (m != null) for (int s = 0; s < search.caches.length; s++) {
            if (m.regions()[s] != null && !m.regions()[s].equals(search.caches[s].state.whole)) outside = false;
        }
        double now = search.elapsed(), remaining = Math.max(0, search.seconds - now);
        return SacPolicy.embed(new double[]{
            remaining / search.seconds, (search.budget - search.attempts) / (double)search.budget, scale(search.best.score() - search.wt.score()), (now - search.lastImprovement) / search.seconds,
            a.kind == Kind.LOCAL ? 1 : 0, a.kind == Kind.EXPAND ? 1 : 0, 0, a.kind == Kind.BATCH ? 1 : 0, a.kind == Kind.REMOTE ? 1 : 0, 0,
            a.positions.length / n, a.candidates.size() / 4.0, scale(a.background.score() - search.best.score()), scale(a.hint), scale(a.synergy),
            bound ? 1 : 0, bound ? scale(m.lower()) : 0, bound ? scale(m.upper()) : 0,
            evidence != null && evidence.bounded() ? scale(evidence.width()) : 0, search.available(m, a.candidates.get(0)) ? 1 : 0, outside ? 1 : 0,
            Math.log1p(work) / 24, Math.min(5, cost / Math.max(.000001, remaining)), Math.log1p(reuse) / 4,
            novelty, boundary / (3 * n), region / (3 * n), compatible, Math.log1p(search.mutations.size()) / 8,
            scale(a.background.score() + a.hint - search.best.score()), 0, search.warmCount / (double)Math.max(1, search.attempts),
            Math.log1p(priority) / 10, Math.log1p(search.seconds) / 10
        });
    }

    private record Decision(List<Action> actions, double[][] state) {}
    private static final Decision END = new Decision(List.of(), new double[0][]);

    private static Decision decision(EmSequenceSearch search) {
        Pool pool = search.pool();
        if (pool.actions().isEmpty()) return END;
        return new Decision(pool.actions(), pool.actions().stream().map(a -> features(search, a, pool)).toArray(double[][]::new));
    }

    private List<SacPolicy.Step> runEpisode(EmSequenceSearch search, long episodeSeed) {
        var behavior = new Random(episodeSeed ^ 0x706f6c696379L);
        List<SacPolicy.Step> steps = new ArrayList<>();
        try {
            Decision current = decision(search);
            while (current != END) {
                search.check();
                int choice = policy.choose(current.state, behavior, training);
                double before = search.best.score();
                int observations = search.rows.size();
                Decision next;
                try {
                    search.execute(current.actions.get(choice), null);
                    next = decision(search);
                } catch (Horizon end) { search.stop = "time_budget"; next = END; }
                if (training && search.rows.size() > observations) {
                    steps.add(new SacPolicy.Step(current.state, choice, (search.best.score() - before) / 10, next.state));
                }
                current = next;
            }
        } catch (Horizon end) { search.stop = "time_budget"; }
        // The clock can also expire between two actions. Close the last real transition.
        if (training && !steps.isEmpty() && search.stop.equals("time_budget")) {
            int i = steps.size() - 1;
            var last = steps.get(i);
            steps.set(i, new SacPolicy.Step(last.state(), last.action(), last.reward(), END.state));
        }
        return steps;
    }
}
