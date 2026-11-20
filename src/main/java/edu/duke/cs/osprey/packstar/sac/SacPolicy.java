/* SPDX-License-Identifier: GPL-2.0-only */
package edu.duke.cs.osprey.packstar.sac;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static edu.duke.cs.osprey.packstar.sac.SacSearch.JSON;

/** Default discrete SAC from MS-PACK*. One categorical distribution over the current legal actions. */
final class SacPolicy {
    static final String[] FEATURES = {
        "remaining_time", "remaining_queries", "best_gain", "stagnation", "is_local", "is_expand",
        "is_transfer", "is_batch", "is_remote", "is_prepare", "block_fraction", "batch_fraction",
        "background_gap", "observed_motif_gain", "pair_synergy", "has_interval", "interval_lower",
        "interval_upper", "interval_width", "response_available", "outside_available", "log_estimated_work",
        "estimated_cost_fraction", "future_reuse", "novelty", "boundary_fraction", "region_fraction",
        "compatible_background_fraction", "response_count", "posterior_mean_gap", "posterior_std",
        "warm_fraction", "information_priority", "budget_log_seconds"
    };
    private static final String SCHEMA = "mspackstar.competitive-policy.v1";
    private static final int WIDTH = 1 + FEATURES.length + 96;
    private static final int BATCH = 64, CAPACITY = 12_000;
    private static final int[][] SIGNS = projectionSigns();

    private final double[] actor = new double[WIDTH];
    private final double[][] critics = new double[2][WIDTH], targets = new double[2][WIDTH];
    private final Adam actorAdam = new Adam();
    private final Adam[] criticAdam = {new Adam(), new Adam()};
    private final List<Step> replay = new ArrayList<>();
    private final Random random;
    private double alpha = .001;
    private int version;

    record Step(double[][] state, int action, double reward, double[][] next) {}

    SacPolicy(long seed) {
        random = new Random(seed ^ 0x127abcdeL);
        for (int h = 0; h < critics.length; h++) {
            Random init = new Random(seed + 104729L * (h + 1));
            for (int j = 0; j < WIDTH; j++) critics[h][j] = .01 * init.nextGaussian() / Math.sqrt(WIDTH);
            targets[h] = critics[h].clone();
        }
    }

    private static int[][] projectionSigns() {
        int[][] signs = new int[96][FEATURES.length];
        for (int k = 0; k < signs.length; k++) {
            for (int i = 0; i < FEATURES.length; i++) {
                long hash = 1009L * (k + 1) + 7919L * (i + 1);
                hash = (hash ^ (hash >>> 30)) * 0xbf58476d1ce4e5b9L;
                hash = (hash ^ (hash >>> 27)) * 0x94d049bb133111ebL;
                signs[k][i] = (hash ^ (hash >>> 31)) < 0 ? -1 : 1;
            }
        }
        return signs;
    }

    static double[] embed(double[] features) {
        if (features.length != FEATURES.length) throw new IllegalArgumentException("SAC feature count");
        double[] x = new double[WIDTH];
        x[0] = 1;
        for (int i = 0; i < features.length; i++) x[i + 1] = Math.max(-5, Math.min(5, features[i]));
        for (int k = 0; k < SIGNS.length; k++) {
            double sum = 0;
            for (int i = 0; i < features.length; i++) sum += SIGNS[k][i] * x[i + 1] / Math.sqrt(features.length);
            x[1 + features.length + k] = Math.tanh(sum);
        }
        return x;
    }

    private static double dot(double[] w, double[] x) {
        double sum = 0;
        for (int j = 0; j < w.length; j++) sum += w[j] * x[j];
        return sum;
    }

    private double[] probabilities(double[][] state) {
        double[] p = new double[state.length];
        double max = Double.NEGATIVE_INFINITY, sum = 0;
        for (int i = 0; i < p.length; i++) max = Math.max(max, p[i] = dot(actor, state[i]));
        for (int i = 0; i < p.length; i++) sum += p[i] = Math.exp(p[i] - max);
        for (int i = 0; i < p.length; i++) p[i] /= sum;
        return p;
    }

    int choose(double[][] state, Random behavior, boolean training) {
        if (state.length == 0) throw new IllegalArgumentException("empty SAC action pool");
        if (training) {
            double[] p = probabilities(state);
            double draw = behavior.nextDouble(), sum = 0;
            for (int i = 0; i < p.length - 1; i++) if ((sum += p[i]) > draw) return i;
            return p.length - 1;
        }
        int best = 0;
        double top = dot(actor, state[0]);
        for (int i = 1; i < state.length; i++) {
            double value = dot(actor, state[i]);
            if (value > top) { best = i; top = value; }
        }
        return best;
    }

    private static double minQ(double[][] weights, double[] x) {
        return Math.min(dot(weights[0], x), dot(weights[1], x));
    }

    // An empty successor represents a completed horizon. Interrupted transitions never enter replay.
    private double softValue(double[][] state) {
        double sum = 0;
        double[] p = probabilities(state);
        for (int i = 0; i < p.length; i++) {
            sum += p[i] * (minQ(targets, state[i]) - alpha * Math.log(Math.max(1e-300, p[i])));
        }
        return sum;
    }

    void train(List<Step> episode, double progress) {
        replay.addAll(episode);
        if (replay.size() > CAPACITY) replay.subList(0, replay.size() - CAPACITY).clear();
        alpha = .001 * Math.max(0, 1 - progress / .8);
        for (int k = 0; !replay.isEmpty() && k < Math.max(2, 2 * episode.size() / BATCH); k++) update();
        version++;
    }

    private void update() {
        double[][] criticGrad = new double[2][WIDTH];
        double[] actorGrad = new double[WIDTH];
        for (int b = 0; b < BATCH; b++) {
            Step step = replay.get(random.nextInt(replay.size()));
            double[] x = step.state[step.action];
            double y = step.reward + softValue(step.next);
            for (int h = 0; h < 2; h++) {
                double huberGradient = Math.max(-1, Math.min(1, dot(critics[h], x) - y));
                for (int j = 0; j < WIDTH; j++) criticGrad[h][j] += huberGradient * x[j] / BATCH;
            }
            double[] p = probabilities(step.state), loss = new double[p.length];
            double mean = 0;
            for (int a = 0; a < p.length; a++) {
                loss[a] = alpha * Math.log(Math.max(1e-300, p[a])) - minQ(critics, step.state[a]);
                mean += p[a] * loss[a];
            }
            for (int a = 0; a < p.length; a++) {
                for (int j = 0; j < WIDTH; j++) actorGrad[j] += p[a] * (loss[a] - mean) * step.state[a][j] / BATCH;
            }
        }
        actorAdam.apply(actor, actorGrad, 1e-4);
        for (int h = 0; h < 2; h++) {
            criticAdam[h].apply(critics[h], criticGrad[h], 3e-4);
            for (int j = 0; j < WIDTH; j++) targets[h][j] = .995 * targets[h][j] + .005 * critics[h][j];
        }
    }

    private static final class Adam {
        final double[] mean = new double[WIDTH], variance = new double[WIDTH];
        int steps;

        void apply(double[] w, double[] gradient, double rate) {
            double squared = dot(gradient, gradient);
            double scale = Math.min(1, .5 / Math.max(1e-30, Math.sqrt(squared)));
            steps++;
            double b1 = 1 - Math.pow(.9, steps), b2 = 1 - Math.pow(.999, steps);
            for (int j = 0; j < WIDTH; j++) {
                double g = gradient[j] * scale;
                mean[j] = .9 * mean[j] + .1 * g;
                variance[j] = .999 * variance[j] + .001 * g * g;
                w[j] -= rate * (mean[j] / b1) / (Math.sqrt(variance[j] / b2) + 1e-8);
            }
        }
    }

    void save(Path path) throws IOException {
        ObjectNode doc = JSON.createObjectNode();
        doc.put("schema", SCHEMA).put("route", "sac").put("rl_lookahead", false);
        doc.set("feature_names", JSON.valueToTree(FEATURES));
        doc.put("embedding_width", WIDTH).put("version", version).put("alpha", alpha);
        doc.set("actor", JSON.valueToTree(actor));
        doc.set("q", JSON.valueToTree(critics));
        doc.set("target", JSON.valueToTree(targets));
        doc.set("value", JSON.valueToTree(new double[WIDTH])); // reserved by the original v1 reader
        SacSearch.writeJson(path, doc);
    }

    static SacPolicy load(Path path) throws IOException {
        JsonNode doc = JSON.readTree(path.toFile());
        if (doc == null || !SCHEMA.equals(doc.path("schema").asText()) || !"sac".equals(doc.path("route").asText())
            || (doc.has("rl_lookahead") && !doc.path("rl_lookahead").equals(JSON.valueToTree(false)))
            || !doc.path("feature_names").equals(JSON.valueToTree(FEATURES))) {
            throw new IllegalArgumentException("Expected a default MS-PACK* SAC policy with matching action features");
        }
        SacPolicy policy = new SacPolicy(0);
        readWeights(doc.path("actor"), policy.actor);
        if (!doc.path("q").isArray() || doc.path("q").size() != 2) throw new IllegalArgumentException("Expected two SAC critics");
        for (int h = 0; h < 2; h++) readWeights(doc.path("q").get(h), policy.critics[h]);
        policy.version = doc.path("version").asInt();
        return policy;
    }

    private static void readWeights(JsonNode values, double[] dest) {
        if (!values.isArray() || values.size() != WIDTH) throw new IllegalArgumentException("Wrong SAC weight dimensions");
        for (int i = 0; i < WIDTH; i++) {
            if (!values.get(i).isNumber() || !Double.isFinite(dest[i] = values.get(i).asDouble())) {
                throw new IllegalArgumentException("Expected finite SAC weights");
            }
        }
    }
}
