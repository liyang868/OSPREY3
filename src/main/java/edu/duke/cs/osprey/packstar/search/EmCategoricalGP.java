/* SPDX-License-Identifier: GPL-2.0-only */
package edu.duke.cs.osprey.packstar.search;

import java.util.*;

import static edu.duke.cs.osprey.packstar.search.EmEnergyModel.Key;
import static edu.duke.cs.osprey.packstar.search.EmSequenceSearch.distance;

/** Gaussian process on paid higher-order residuals of the wild-type pairwise expansion. */
final class EmCategoricalGP {
    private static final int CAPACITY = 512;
    private static final int REFIT_INTERVAL = 32;
    private static final double NOISE = .01, JITTER = 1e-8, SCALE = 10;

    private record Observation(Key sequence, double residual, double score) {}
    record Prediction(double mean, double variance, double[] solved) {}

    private final int[] reference;
    private final Runnable check;
    private final Map<Key, Observation> ledger = new LinkedHashMap<>();
    private List<Observation> active = new ArrayList<>();
    private double[] rho;
    private double[][] l = new double[0][];
    private double[] alpha = new double[0];
    private int refits, hyperProposals;
    private double fitSeconds;

    EmCategoricalGP(int[] reference, Runnable check) {
        this.reference = reference.clone();
        this.check = check;
        rho = new double[reference.length];
        Arrays.fill(rho, .5);
    }

    private double kernel(int[] a, int[] b, double[] parameters) {
        // Measured WT, single and double substitutions have zero residual.
        if (distance(reference, a) <= 2 || distance(reference, b) <= 2) return 0;
        double value = 4;
        for (int i = 0; i < a.length; i++) if (a[i] != b[i]) value *= parameters[i];
        return value;
    }

    void update(int[] sequence, double baseline, double score) {
        if (!Double.isFinite(baseline) || !Double.isFinite(score) || distance(reference, sequence) <= 2) return;
        Key key = new Key(sequence);
        if (ledger.containsKey(key)) return;
        check.run();
        long started = System.nanoTime();
        var observation = new Observation(key, (score - baseline) / SCALE, score);
        ledger.put(key, observation);
        int count = ledger.size();
        try {
            if (count <= CAPACITY && count % REFIT_INTERVAL != 0) {
                append(observation);
            } else {
                List<Observation> selected = selectActive();
                double[] parameters = count % REFIT_INTERVAL == 0 ? optimize(selected, count) : rho;
                double[][] factor = cholesky(matrix(selected, parameters), check);
                double[] weights = solve(factor, selected);
                active = selected;
                rho = parameters;
                l = factor;
                alpha = weights;
                refits++;
            }
        } catch (RuntimeException error) {
            ledger.remove(key);
            throw error;
        } finally {
            fitSeconds += (System.nanoTime() - started) / 1e9;
        }
    }

    private List<Observation> selectActive() {
        var all = new ArrayList<>(ledger.values());
        if (all.size() <= CAPACITY) return all;
        var chosen = new LinkedHashSet<Observation>();
        all.stream().sorted(Comparator.comparingDouble(Observation::score).reversed()).limit(128).forEach(chosen::add);
        int recent = 0;
        for (int i = all.size() - 1; i >= 0 && recent < 192; i--) {
            if (chosen.add(all.get(i))) recent++;
        }
        int[] distances = new int[all.size()];
        Arrays.fill(distances, Integer.MAX_VALUE);
        for (Observation center : chosen) updateDistances(all, distances, center);
        while (chosen.size() < CAPACITY) {
            check.run();
            int index = -1, farthest = -1;
            for (int i = 0; i < all.size(); i++) {
                if (distances[i] > farthest && !chosen.contains(all.get(i))) {
                    index = i;
                    farthest = distances[i];
                }
            }
            Observation next = all.get(index);
            chosen.add(next);
            updateDistances(all, distances, next);
        }
        return new ArrayList<>(chosen);
    }

    private void updateDistances(List<Observation> all, int[] distances, Observation center) {
        check.run();
        for (int i = 0; i < all.size(); i++) {
            distances[i] = Math.min(distances[i], distance(center.sequence.values, all.get(i).sequence.values));
        }
    }

    private double[][] matrix(List<Observation> data, double[] parameters) {
        double[][] matrix = new double[data.size()][data.size()];
        for (int i = 0; i < matrix.length; i++) {
            if ((i & 15) == 0) check.run();
            for (int j = 0; j <= i; j++) {
                double value = kernel(data.get(i).sequence.values, data.get(j).sequence.values, parameters);
                matrix[i][j] = matrix[j][i] = value + (i == j ? NOISE + JITTER : 0);
            }
        }
        return matrix;
    }

    static double[][] cholesky(double[][] matrix, Runnable check) {
        double[][] lower = new double[matrix.length][matrix.length];
        for (int i = 0; i < lower.length; i++) {
            if ((i & 15) == 0) check.run();
            for (int j = 0; j <= i; j++) {
                double value = matrix[i][j];
                for (int k = 0; k < j; k++) value -= lower[i][k] * lower[j][k];
                if (i == j) {
                    if (!(value > 0) || !Double.isFinite(value)) throw new IllegalArgumentException("Non-positive GP covariance");
                    lower[i][j] = Math.sqrt(value);
                } else {
                    lower[i][j] = value / lower[j][j];
                }
            }
        }
        return lower;
    }

    static double[] forward(double[][] lower, double[] b) {
        double[] x = new double[b.length];
        for (int i = 0; i < b.length; i++) {
            double value = b[i];
            for (int j = 0; j < i; j++) value -= lower[i][j] * x[j];
            x[i] = value / lower[i][i];
        }
        return x;
    }

    static double[] backward(double[][] lower, double[] b) {
        double[] x = new double[b.length];
        for (int i = b.length - 1; i >= 0; i--) {
            double value = b[i];
            for (int j = i + 1; j < b.length; j++) value -= lower[j][i] * x[j];
            x[i] = value / lower[i][i];
        }
        return x;
    }

    private static double[] residuals(List<Observation> data) {
        return data.stream().mapToDouble(Observation::residual).toArray();
    }

    private static double[] solve(double[][] lower, List<Observation> data) {
        return backward(lower, forward(lower, residuals(data)));
    }

    private void append(Observation observation) {
        int n = active.size();
        double[] cross = new double[n];
        for (int i = 0; i < n; i++) cross[i] = kernel(active.get(i).sequence.values, observation.sequence.values, rho);
        double[] solved = forward(l, cross);
        double diagonal = 4 + NOISE + JITTER;
        for (double value : solved) diagonal -= value * value;
        double[][] expanded = Arrays.copyOf(l, n + 1);
        expanded[n] = Arrays.copyOf(solved, n + 1);
        expanded[n][n] = Math.sqrt(Math.max(JITTER, diagonal));
        active.add(observation);
        alpha = solve(expanded, active);
        l = expanded;
    }

    private double nll(List<Observation> data, double[] parameters) {
        double[][] lower = cholesky(matrix(data, parameters), check);
        double[] solved = forward(lower, residuals(data));
        double loss = 0;
        for (int i = 0; i < solved.length; i++) loss += .5 * solved[i] * solved[i] + Math.log(lower[i][i]);
        return loss;
    }

    private double[] optimize(List<Observation> data, int count) {
        double[] parameters = rho.clone();
        double loss = nll(data, parameters);
        for (int k = 0; k < 25; k++) {
            check.run();
            int site = (k + count / REFIT_INTERVAL) % parameters.length;
            double old = parameters[site];
            double increment = (k / parameters.length) % 2 == 0 ? .08 : -.08;
            parameters[site] = Math.max(.02, Math.min(.98, old + increment));
            double candidate = nll(data, parameters);
            hyperProposals++;
            if (candidate < loss) loss = candidate;
            else parameters[site] = old;
        }
        return parameters;
    }

    Prediction predict(int[] sequence, double baseline) {
        check.run();
        double[] cross = new double[active.size()];
        double mean = baseline;
        for (int i = 0; i < cross.length; i++) {
            cross[i] = kernel(active.get(i).sequence.values, sequence, rho);
            mean += SCALE * alpha[i] * cross[i];
        }
        double[] solved = forward(l, cross);
        double variance = kernel(sequence, sequence, rho);
        for (double value : solved) variance -= value * value;
        return new Prediction(mean, SCALE * SCALE * Math.max(0, variance), solved);
    }

    double covariance(int[] a, Prediction pa, int[] b, Prediction pb) {
        double value = kernel(a, b, rho);
        for (int i = 0; i < pa.solved.length; i++) value -= pa.solved[i] * pb.solved[i];
        return SCALE * SCALE * value;
    }

    Map<String, Object> report() {
        return Map.of("model", "WT F2 + categorical product diffusion-kernel residual GP",
            "active_points", active.size(), "paid_high_order_observations", ledger.size(),
            "full_refits", refits, "hyperparameter_proposals", hyperProposals, "rho_by_site", rho,
            "fit_seconds", fitSeconds, "observation_variance_scaled", NOISE);
    }
}
