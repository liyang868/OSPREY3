package edu.duke.cs.osprey.packstar;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/** Tests exact-key CCD work reduction and ordered result scatter metadata. */
public class TestPackStarConformationDedup {

    @Test
    public void exactAssignmentsAreUniqueAndMapBackToEverySample() {
        List<int[]> samples = new ArrayList<>(List.of(
                new int[]{1, 2, 3},
                new int[]{4, 5, 6},
                new int[]{1, 2, 3},
                new int[]{4, 5, 6},
                new int[]{1, 2, 4}));

        PackStarEstimator.UniqueConformationBatch batch =
                PackStarEstimator.deduplicateConformations(samples);

        assertEquals(3, batch.uniqueConformations.size());
        assertArrayEquals(new int[]{1, 2, 3},
                batch.uniqueConformations.get(0));
        assertArrayEquals(new int[]{4, 5, 6},
                batch.uniqueConformations.get(1));
        assertArrayEquals(new int[]{1, 2, 4},
                batch.uniqueConformations.get(2));
        assertArrayEquals(new int[]{0, 1, 0, 1, 2},
                batch.uniqueIndexBySample);
    }

    @Test
    public void equalValuesUseValueEqualityEvenWithDifferentArrays() {
        List<int[]> samples = List.of(
                new int[]{-1, 0, 9},
                Arrays.copyOf(new int[]{-1, 0, 9}, 3));

        PackStarEstimator.UniqueConformationBatch batch =
                PackStarEstimator.deduplicateConformations(samples);

        assertEquals(1, batch.uniqueConformations.size());
        assertArrayEquals(new int[]{0, 0}, batch.uniqueIndexBySample);
    }
}
