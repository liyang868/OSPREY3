package edu.duke.cs.osprey.packstar;

import edu.duke.cs.osprey.tools.*;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestPackStarByteBoundedCache {
    @Test public void retainsRecentValuesWithinBudgetAndRejectsOversizedEntries() {
        var cache = new ByteBoundedCache<String, String>(10);
        cache.put("a", "A", 4);
        cache.put("b", "B", 4);
        assertEquals("A", cache.get("a"));
        cache.put("c", "C", 4);
        assertNull(cache.get("b"));
        cache.put("huge", "H", 11);
        assertEquals("A", cache.get("a"));
        cache.put("a", "replacement", 7);
        assertNull(cache.get("c"));
        assertEquals(7, cache.bytes());
        cache.clear();
        assertEquals(0, cache.bytes());
        assertNull(cache.get("a"));
    }

    @Test public void zeroDisablesRetentionAndLargeLimitsDoNotOverflow() {
        var disabled = new ByteBoundedCache<String, String>(0);
        disabled.put("x", "X", 1);
        assertNull(disabled.get("x"));
        var large = new ByteBoundedCache<String, String>(Long.MAX_VALUE);
        large.put("a", "A", Long.MAX_VALUE - 1);
        large.put("b", "B", 2);
        assertNull(large.get("a"));
        assertEquals(2, large.bytes());
    }
}
