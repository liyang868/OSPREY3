package edu.duke.cs.osprey.packstar;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TestPackStarRandomStreams {

    @Test
    public void logicalIdentityDeterminesSeed() {
        long first = PackStarEstimator.deriveRandomSeed(
                42L, "packstar|kstar-state-2|pac-v1|rcs-A");
        PackStarEstimator.deriveRandomSeed(
                42L, "packstar|unrelated|pac-v1|rcs-B");
        long repeated = PackStarEstimator.deriveRandomSeed(
                42L, "packstar|kstar-state-2|pac-v1|rcs-A");

        assertEquals(first, repeated);
    }

    @Test
    public void stateIdentityAndBaseSeedCreateDistinctStreams() {
        long base = PackStarEstimator.deriveRandomSeed(42L, "state-A");
        assertNotEquals(base,
                PackStarEstimator.deriveRandomSeed(42L, "state-B"));
        assertNotEquals(base,
                PackStarEstimator.deriveRandomSeed(43L, "state-A"));
    }

    @Test
    public void blankIdentityIsRejected() {
        assertThrows(IllegalArgumentException.class, () ->
                PackStarEstimator.deriveRandomSeed(42L, "  "));
    }
}
