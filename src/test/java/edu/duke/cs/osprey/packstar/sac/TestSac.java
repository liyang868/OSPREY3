/* SPDX-License-Identifier: GPL-2.0-only */
package edu.duke.cs.osprey.packstar.sac;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import edu.duke.cs.osprey.ematrix.EnergyMatrix;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class TestSac {
    @Test
    public void defaultPolicyRoundTrip(@TempDir Path dir) throws Exception {
        SacPolicy policy = new SacPolicy(42);
        double[] f = new double[SacPolicy.FEATURES.length];
        f[4] = 1;
        double[][] state = {SacPolicy.embed(f), SacPolicy.embed(new double[f.length])};
        var json = new ObjectMapper();
        Path file = dir.resolve("policy.json");
        policy.save(file);
        var before = json.readTree(file.toFile());
        assertNotEquals(before.path("q").get(0), before.path("q").get(1));
        policy.train(List.of(new SacPolicy.Step(state, 0, .25, state),
                             new SacPolicy.Step(state, 1, .1, new double[0][])), .25);
        policy.save(file);
        var doc = (ObjectNode)json.readTree(file.toFile());
        assertNotEquals(before.path("actor"), doc.path("actor"));
        assertNotEquals(before.path("target"), doc.path("target"));
        var loaded = SacPolicy.load(file);
        for (boolean train : List.of(false, true)) {
            Random a = new Random(5), b = new Random(5);
            for (int i = 0; i < 20; i++) assertEquals(policy.choose(state, a, train), loaded.choose(state, b, train));
        }
        doc.remove("rl_lookahead"); // default v1 checkpoints predate the mode flag
        json.writeValue(file.toFile(), doc);
        assertDoesNotThrow(() -> SacPolicy.load(file));
        doc.put("rl_lookahead", true);
        json.writeValue(file.toFile(), doc);
        assertThrows(IllegalArgumentException.class, () -> SacPolicy.load(file));
    }

    @Test
    public void cachedStateScoresAndResponseBoundsAgreeWithEnumeration() {
        int[] sizes = {3, 3, 3, 3, 3};
        var matrix = new EnergyMatrix(sizes.length, sizes, 0);
        matrix.setConstTerm(-2.5);
        int[][][] domains = new int[sizes.length][][];
        for (int p = 0; p < sizes.length; p++) {
            domains[p] = new int[][]{{0, 2}, {1}};
            for (int a = 0; a < sizes[p]; a++) {
                matrix.setOneBody(p, a, .3 * (a - p));
                for (int q = 0; q < p; q++) for (int b = 0; b < sizes[q]; b++) {
                    matrix.setPairwise(p, a, q, b, .02 * (1 + p + a) * (q - b));
                }
            }
        }
        var state = new SacEnergyModel.State(matrix, new int[]{2, 4, 1, 3, 0}, domains);
        var cache = new SacEnergyModel.Cache(state);
        int[] wt = new int[sizes.length], target = wt.clone();
        target[2] = 1;
        var r = cache.response(wt, target, () -> {});
        double[] before = cache.message(r, wt), after = cache.message(r, target);
        double lo = Double.POSITIVE_INFINITY, hi = Double.NEGATIVE_INFINITY;
        for (int j = 0; j < before.length; j++) {
            lo = Math.min(lo, after[j] - before[j]);
            hi = Math.max(hi, after[j] - before[j]);
        }
        double gain = cache.score(target, () -> {}) - cache.score(wt, () -> {});
        assertTrue(lo <= gain + 1e-12 && gain <= hi + 1e-12);
        for (int mask = 0; mask < 32; mask++) {
            int[] seq = new int[sizes.length];
            for (int p = 0; p < seq.length; p++) seq[p] = (mask >>> p) & 1;
            int[][] rows = state.rows(seq);
            double exact = Math.log(sum(matrix, rows, new int[rows.length], 0, matrix.getConstTerm()));
            assertEquals(exact, cache.score(seq, () -> {}), 1e-11, Arrays.toString(seq));
        }
    }

    private static double sum(EnergyMatrix matrix, int[][] rows, int[] conf, int p, double energy) {
        if (p == rows.length) return Math.exp(-energy / SacEnergyModel.RT);
        double z = 0;
        for (int r : rows[p]) {
            conf[p] = r;
            double e = energy + matrix.getOneBody(p, r);
            for (int q = 0; q < p; q++) e += matrix.getPairwise(p, r, q, conf[q]);
            z += sum(matrix, rows, conf, p + 1, e);
        }
        return z;
    }
}
