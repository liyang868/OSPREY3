/* SPDX-License-Identifier: GPL-2.0-only */
package edu.duke.cs.osprey.packstar.search;

import edu.duke.cs.osprey.branchdp.BranchDpBackend.RootSelectionException;
import edu.duke.cs.osprey.branchdp.DPTableTooLargeException;

import java.util.*;

import static edu.duke.cs.osprey.packstar.search.EmCategoricalGP.*;
import static edu.duke.cs.osprey.packstar.search.EmEnergyModel.*;
import static edu.duke.cs.osprey.packstar.search.EmSequenceSearch.*;

/** Compares first actions by fantasy-conditioned improvement after a second action. */
final class LookaheadPlanner {
    private static final int FANTASIES = 8;
    private static final long PLAN_NANOS = 1_000_000_000L;
    private static final double COST_FLOOR = .00002;
    private static final double COVARIANCE_JITTER = 1e-6;
    private static final double OBSERVATION_VARIANCE = 1;

    private record Rank(Action action, double priority, double std, double seconds) {}
    private record Continuation(double seconds, double[] means, double[][] cross, double[][] covariance,
                                double[][] lower) {}
    private static final class PlanTimeout extends RuntimeException {}

    final EmCategoricalGP gp;
    private final EmSequenceSearch search;
    private final Random rng;
    private final Map<Key, Prediction> posterior = new HashMap<>();
    private final Map<Kind, Double> rates = new EnumMap<>(Kind.class);
    private int calls, fallbacks, timeouts;
    private double planningSeconds;

    LookaheadPlanner(EmSequenceSearch search, long seed) {
        this.search = search;
        gp = new EmCategoricalGP(search.model.reference, search::check);
        rng = new Random(seed ^ 0x706c616eL);
    }

    void observe(Row row) {
        gp.update(row.seq(), baseline(row.seq()), row.score());
        posterior.clear();
    }

    double baseline(int[] sequence) {
        int[] reference = search.model.reference;
        double wt = search.wt.score(), value = wt;
        double[] singles = new double[sequence.length];
        int[] slice = reference.clone();
        for (int i = 0; i < sequence.length; i++) if (sequence[i] != reference[i]) {
            slice[i] = sequence[i];
            Row row = search.rows.get(new Key(slice));
            if (row != null && Double.isFinite(row.score())) singles[i] = row.score() - wt;
            value += singles[i];
            slice[i] = reference[i];
        }
        for (int i = 0; i < sequence.length; i++) if (sequence[i] != reference[i]) {
            slice[i] = sequence[i];
            for (int j = i + 1; j < sequence.length; j++) if (sequence[j] != reference[j]) {
                slice[j] = sequence[j];
                Row row = search.rows.get(new Key(slice));
                if (row != null && Double.isFinite(row.score())) value += row.score() - wt - singles[i] - singles[j];
                slice[j] = reference[j];
            }
            slice[i] = reference[i];
        }
        return value;
    }

    private Prediction prediction(int[] sequence) {
        return posterior.computeIfAbsent(new Key(sequence), key -> gp.predict(sequence, baseline(sequence)));
    }

    void recordCost(Kind kind, Execution execution) {
        if (execution.work() > 0) {
            rates.merge(kind, execution.seconds() / execution.work(), (before, observed) -> .8 * before + .2 * observed);
        }
    }

    private double cost(Action action, Cache.Forecast[] caches, Runnable check) {
        double work = 0;
        try {
            for (int[] sequence : action.candidates) {
                for (var cache : caches) work += cache.scoreWork(sequence, check);
            }
        } catch (RootSelectionException | DPTableTooLargeException unavailable) {
            return Double.POSITIVE_INFINITY;
        }
        return COST_FLOOR + Math.max(1, work) * rates.getOrDefault(action.kind, search.secondsPerWork);
    }

    private Cache.Forecast[] forecast() {
        return Arrays.stream(search.caches).map(Cache::forecast).toArray(Cache.Forecast[]::new);
    }

    private List<Rank> rank(List<Action> pool) {
        List<Rank> ranked = new ArrayList<>();
        for (Action action : pool) {
            double std = 0, optimistic = Double.NEGATIVE_INFINITY;
            for (int[] sequence : action.candidates) {
                Prediction p = prediction(sequence);
                double spread = Math.sqrt(p.variance()), upper = p.mean() + 1.96 * spread;
                if (upper > optimistic) {
                    optimistic = upper;
                    std = spread;
                }
            }
            double seconds = cost(action, forecast(), search::check);
            double info = information(action, reuse(action, pool), search.novelty(action));
            double gain = Math.max(0, optimistic - search.best.score());
            ranked.add(new Rank(action, (gain + info) / Math.sqrt(seconds), std, seconds));
        }
        return ranked;
    }

    Action choose(List<Action> pool) {
        posterior.clear();
        List<Rank> ratings = rank(pool);
        List<Rank> ranked = ratings.stream().sorted(Comparator.comparingDouble(Rank::priority).reversed()).toList();
        Action fallback = ranked.get(0).action;
        if (search.step % 5 == 4) {
            fallback = ranked.stream().map(Rank::action).filter(a -> a.kind == Kind.REMOTE || a.kind == Kind.EXPAND)
                .findFirst().orElse(pool.get(0));
        }

        long begun = System.nanoTime(), deadline = begun + PLAN_NANOS;
        Runnable check = () -> {
            search.check();
            if (System.nanoTime() - deadline >= 0) throw new PlanTimeout();
        };
        calls++;
        try {
            var firsts = new LinkedHashSet<>(ranked.subList(0, Math.min(8, ranked.size())));
            ratings.stream().sorted(Comparator.comparingDouble(Rank::std).reversed()).limit(4).forEach(firsts::add);
            ratings.stream().sorted(Comparator.comparingDouble(Rank::seconds)).limit(4).forEach(firsts::add);
            for (Rank rank : ranked) {
                if (firsts.size() >= 16) break;
                firsts.add(rank);
            }
            Action chosen = null;
            double best = Double.NEGATIVE_INFINITY;
            for (Rank rank : firsts) {
                check.run();
                Action first = rank.action;
                if (rank.seconds > search.remaining() || first.candidates.size() > search.remainingQueries()) continue;
                double utility = utility(first, rank.seconds, ranked, check);
                if (utility > best) {
                    best = utility;
                    chosen = first;
                }
            }
            if (chosen != null) return chosen;
            return fallback(fallback);
        } catch (PlanTimeout timeout) {
            timeouts++;
            return fallback(fallback);
        } catch (IllegalArgumentException numerical) {
            return fallback(fallback);
        } finally {
            planningSeconds += (System.nanoTime() - begun) / 1e9;
        }
    }

    private Action fallback(Action action) {
        fallbacks++;
        return action;
    }

    private double utility(Action first, double firstCost, List<Rank> ranked, Runnable check) {
        List<int[]> sequences = first.candidates;
        List<Prediction> priors = sequences.stream().map(this::prediction).toList();
        double[][] lower = cholesky(joint(sequences, priors, OBSERVATION_VARIANCE), check);
        List<Continuation> next = new ArrayList<>();
        for (Action second : successors(first, ranked, check)) {
            check.run();
            if (sequences.size() + second.candidates.size() > search.remainingQueries()) continue;
            var caches = forecast();
            cost(first, caches, check);
            double totalCost = firstCost + cost(second, caches, check);
            if (totalCost <= search.remaining()) {
                next.add(condition(second.candidates, sequences, priors, lower, totalCost, check));
            }
        }

        double best = search.best.score(), utility = 0;
        for (int fantasy = 0; fantasy < FANTASIES; fantasy++) {
            check.run();
            double[] delta = draw(lower);
            double incumbent = best;
            for (int i = 0; i < delta.length; i++) incumbent = Math.max(incumbent, priors.get(i).mean() + delta[i]);
            double[] solved = backward(lower, forward(lower, delta));
            double secondBest = 0;
            for (Continuation second : next) {
                check.run();
                if (second.seconds <= search.remaining()) {
                    secondBest = Math.max(secondBest, improvement(second, solved, incumbent));
                }
            }
            utility += (incumbent - best + secondBest) / FANTASIES;
        }
        return utility;
    }

    private List<Action> successors(Action first, List<Rank> ranked, Runnable check) {
        List<Action> next = new ArrayList<>();
        Set<Key> observed = new HashSet<>();
        for (int[] sequence : first.candidates) observed.add(new Key(sequence));
        for (Rank rank : ranked) {
            if (next.size() == 8) break;
            if (rank.action.candidates.stream().noneMatch(s -> observed.contains(new Key(s)))) next.add(rank.action);
        }
        for (int[] background : first.candidates) {
            for (int i = 0; i < background.length && next.size() < 16; i++) {
                int count = search.model.alphabet[i].length;
                if (count <= 1) continue;
                check.run();
                int[] sequence = background.clone();
                sequence[i] = (sequence[i] + 1 + rng.nextInt(count - 1)) % count;
                Key key = new Key(sequence);
                if (!search.rows.containsKey(key) && !search.failed.contains(key) && !observed.contains(key)) {
                    next.add(new Action(Kind.LOCAL, first.background, new int[]{i}, List.of(sequence), 0, 0));
                }
            }
        }
        return next;
    }

    private Continuation condition(List<int[]> sequences, List<int[]> first, List<Prediction> priors,
                                   double[][] lower, double seconds, Runnable check) {
        List<Prediction> predictions = sequences.stream().map(this::prediction).toList();
        int n = sequences.size();
        double[] means = new double[n];
        double[][] cross = new double[n][], projection = new double[n][];
        for (int i = 0; i < n; i++) {
            Prediction p = predictions.get(i);
            means[i] = p.mean();
            cross[i] = new double[first.size()];
            for (int j = 0; j < first.size(); j++) {
                cross[i][j] = gp.covariance(sequences.get(i), p, first.get(j), priors.get(j));
            }
            projection[i] = forward(lower, cross[i]);
        }
        double[][] covariance = joint(sequences, predictions, 0);
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
            for (int k = 0; k < first.size(); k++) covariance[i][j] -= projection[i][k] * projection[j][k];
        }
        double[][] conditional = n == 1 ? null : cholesky(covariance, check);
        return new Continuation(seconds, means, cross, covariance, conditional);
    }

    private double improvement(Continuation second, double[] solved, double incumbent) {
        double[] means = second.means.clone();
        for (int i = 0; i < means.length; i++) {
            for (int j = 0; j < solved.length; j++) means[i] += second.cross[i][j] * solved[j];
        }
        if (means.length == 1) {
            return expectedImprovement(means[0], Math.max(0, second.covariance[0][0]), incumbent);
        }
        double expected = 0;
        for (int sample = 0; sample < FANTASIES; sample++) {
            double max = incumbent;
            double[] delta = draw(second.lower);
            for (int i = 0; i < means.length; i++) max = Math.max(max, means[i] + delta[i]);
            expected += (max - incumbent) / FANTASIES;
        }
        return expected;
    }

    private double[] draw(double[][] lower) {
        double[] z = new double[lower.length], delta = new double[lower.length];
        for (int i = 0; i < z.length; i++) z[i] = rng.nextGaussian();
        for (int i = 0; i < delta.length; i++) {
            for (int j = 0; j <= i; j++) delta[i] += lower[i][j] * z[j];
        }
        return delta;
    }

    private double[][] joint(List<int[]> sequences, List<Prediction> predictions, double noise) {
        double[][] covariance = new double[sequences.size()][sequences.size()];
        for (int i = 0; i < covariance.length; i++) for (int j = 0; j <= i; j++) {
            double value = gp.covariance(sequences.get(i), predictions.get(i), sequences.get(j), predictions.get(j));
            covariance[i][j] = covariance[j][i] = value + (i == j ? noise + COVARIANCE_JITTER : 0);
        }
        return covariance;
    }

    static double expectedImprovement(double mean, double variance, double incumbent) {
        if (variance <= 1e-12) return Math.max(0, mean - incumbent);
        double sd = Math.sqrt(variance), z = (mean - incumbent) / sd;
        double t = 1 / (1 + .2316419 * Math.abs(z));
        double pdf = Math.exp(-.5 * z * z) / Math.sqrt(2 * Math.PI);
        double tail = pdf * t * (.319381530 + t * (-.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
        double cdf = z >= 0 ? 1 - tail : tail;
        return Math.max(0, (mean - incumbent) * cdf + sd * pdf);
    }

    Map<String, Object> report() {
        return Map.of("depth", 2, "fantasies", FANTASIES, "calls", calls, "fallbacks", fallbacks, "timeouts", timeouts,
            "seconds", planningSeconds, "cost_rates", rates);
    }
}
