/* SPDX-License-Identifier: GPL-2.0-only */
package edu.duke.cs.osprey.packstar.search;

import edu.duke.cs.osprey.ematrix.EnergyMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class TestLookaheadBo {
    @Test
    public void categoricalPosteriorUsesPaidHigherOrderResiduals() {
        var gp = new EmCategoricalGP(new int[4], () -> {});
        int[] warm = {1, 1, 0, 0}, observed = {1, 1, 1, 0}, query = {1, 1, 1, 1};
        gp.update(warm, 3, 8);
        assertEquals(0, gp.predict(warm, 3).variance());
        assertEquals(17, gp.predict(query, 17).mean());
        assertEquals(400, gp.predict(query, 17).variance());

        gp.update(observed, 11, 17);
        gp.update(observed, 11, 100);
        double observationCovariance = 4 + .01 + 1e-8;
        var posterior = gp.predict(query, 17);
        assertEquals(17 + 12 / observationCovariance, posterior.mean(), 1e-12);
        assertEquals(100 * (4 - 4 / observationCovariance), posterior.variance(), 1e-12);
        assertEquals(posterior.variance(), gp.covariance(query, posterior, query, posterior), 1e-12);
        assertEquals(1, gp.report().get("paid_high_order_observations"));
    }

    @Test
    public void forecastMatchesScoringWorkWithoutPopulatingTheLiveCache() {
        int[] sizes = {4, 4, 4, 4};
        var matrix = new EnergyMatrix(sizes.length, sizes, 0);
        int[][][] domains = new int[sizes.length][][];
        for (int p = 0; p < sizes.length; p++) {
            domains[p] = new int[][]{{0, 1}, {2, 3}};
            for (int a = 0; a < sizes[p]; a++) {
                matrix.setOneBody(p, a, .25 * (a - p));
                for (int q = 0; q < p; q++) for (int b = 0; b < sizes[q]; b++) {
                    matrix.setPairwise(p, a, q, b, .02 * (p + a) * (q - b));
                }
            }
        }
        var state = new EmEnergyModel.State(matrix, new int[]{0, 1, 2, 3}, domains);
        var cache = new EmEnergyModel.Cache(state);
        int[] first = {0, 0, 0, 0, 0}, sameState = {0, 0, 0, 0, 1}, second = {1, 0, 0, 0, 1};
        Runnable check = () -> {};
        var branch = cache.forecast();
        double firstWork = branch.scoreWork(first, check);
        assertTrue(firstWork > 0);
        assertEquals(0, branch.scoreWork(sameState, check));
        double secondWork = branch.scoreWork(second, check);
        assertEquals(0, cache.work);
        assertEquals(firstWork, cache.forecast().scoreWork(sameState, check));

        cache.score(first, check);
        assertEquals(firstWork, cache.work);
        cache.score(sameState, check);
        assertEquals(firstWork, cache.work);
        cache.score(second, check);
        assertEquals(firstWork + secondWork, cache.work);
    }
}
