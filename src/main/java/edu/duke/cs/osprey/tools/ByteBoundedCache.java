package edu.duke.cs.osprey.tools;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/** A small synchronized LRU whose caller supplies conservative retained-byte estimates. */
public final class ByteBoundedCache<K, V> {

    private static final class Entry<V> {
        final V value;
        final long bytes;
        Entry(V value, long bytes) { this.value = value; this.bytes = bytes; }
    }

    private final long maximumBytes;
    private final Map<K, Entry<V>> entries = new LinkedHashMap<>(16, 0.75f, true);
    private long bytes;
    private long hits;
    private long misses;

    public ByteBoundedCache(long maximumBytes) {
        if (maximumBytes < 0) throw new IllegalArgumentException("negative cache byte limit");
        this.maximumBytes = maximumBytes;
    }

    public boolean canStore(long entryBytes) {
        return entryBytes > 0 && entryBytes <= maximumBytes;
    }

    public synchronized V get(K key) {
        Entry<V> entry = entries.get(key);
        if (entry == null) { misses++; return null; }
        hits++;
        return entry.value;
    }

    /** Values must be immutable while retained; oversized entries do not evict useful ones. */
    public synchronized void put(K key, V value, long entryBytes) {
        Objects.requireNonNull(value);
        if (!canStore(entryBytes)) return;
        Entry<V> previous = entries.remove(key);
        if (previous != null) bytes -= previous.bytes;
        while (bytes > maximumBytes - entryBytes) {
            var oldest = entries.entrySet().iterator();
            bytes -= oldest.next().getValue().bytes;
            oldest.remove();
        }
        entries.put(key, new Entry<>(value, entryBytes));
        bytes += entryBytes;
    }

    public synchronized void clear() { entries.clear(); bytes = 0; }
    public synchronized long bytes() { return bytes; }
    public synchronized long hits() { return hits; }
    public synchronized long misses() { return misses; }
}
