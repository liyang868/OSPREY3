/*
** This file is part of OSPREY 3.0
**
** OSPREY Protein Redesign Software Version 3.0
** Copyright (C) 2001-2018 Bruce Donald Lab, Duke University
**
** OSPREY is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License version 2
** as published by the Free Software Foundation.
**
** You should have received a copy of the GNU General Public License
** along with OSPREY.  If not, see <http://www.gnu.org/licenses/>.
*/

package edu.duke.cs.osprey.branchdp;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;

public interface DPTable extends AutoCloseable {

    long size();

    double lower(long idx);

    double upper(long idx);

    void readPair(long idx, double[] lowerDest, double[] upperDest, int destIndex);

    /** Paired read into pinned/direct buffers. Absolute puts leave buffer positions
     *  untouched, so independent gather chunks can safely fill disjoint indices. */
    void readPair(long idx, DoubleBuffer lowerDest, DoubleBuffer upperDest, int destIndex);

    void set(long idx, double lower, double upper);

    void fill(double lower, double upper);

    void copyFrom(long start, DoubleBuffer lowerSrc, DoubleBuffer upperSrc, int count);

    void copyFromIndexed(long[] indices, DoubleBuffer lowerSrc, DoubleBuffer upperSrc, int count);

    boolean isDenseArrayBacked();

    double[] lowerArrayUnsafe();

    double[] upperArrayUnsafe();

    /**
     * The table's backing storage as one or more contiguous chunks, in index
     * order (chunk 0 covers [0, chunk0.length), chunk 1 continues immediately
     * after, etc.). For a dense table this is a single-element array wrapping
     * the whole backing array; for a sharded table this is the shard array
     * directly. Unlike {@link #lowerArrayUnsafe()}/{@link #upperArrayUnsafe()},
     * array-backed tables expose these chunks to callers such as the resident
     * GPU uploader. File-backed tables deliberately reject this accessor and
     * are streamed through {@link #readPair(long, DoubleBuffer, DoubleBuffer,
     * int)} instead; callers must check {@link #supportsArrayChunks()} first.
     */
    double[][] lowerChunks();

    double[][] upperChunks();

    /** Whether lowerChunks()/upperChunks() can expose Java double[] storage. */
    default boolean supportsArrayChunks() {
        return true;
    }

    /** True for page-cache-backed tables whose logical bytes are not Java heap. */
    default boolean isFileBacked() {
        return false;
    }

    long estimatedBytes();

    @Override
    default void close() {
    }
}

final class DenseDPTable implements DPTable {

    private final double[] lower;
    private final double[] upper;

    DenseDPTable(long size) {
        if (size > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Dense DP table size exceeds Java array limits: " + size);
        }
        this.lower = new double[(int) size];
        this.upper = new double[(int) size];
    }

    @Override
    public long size() {
        return lower.length;
    }

    @Override
    public double lower(long idx) {
        return lower[(int) idx];
    }

    @Override
    public double upper(long idx) {
        return upper[(int) idx];
    }

    @Override
    public void readPair(long idx, double[] lowerDest, double[] upperDest, int destIndex) {
        if (idx < 0L || idx >= lower.length) {
            throw new IndexOutOfBoundsException("DP table index " + idx
                    + " outside [0," + lower.length + ")");
        }
        int i = (int)idx;
        lowerDest[destIndex] = lower[i];
        upperDest[destIndex] = upper[i];
    }

    @Override
    public void readPair(long idx, DoubleBuffer lowerDest, DoubleBuffer upperDest, int destIndex) {
        if (idx < 0L || idx >= lower.length) {
            throw new IndexOutOfBoundsException("DP table index " + idx
                    + " outside [0," + lower.length + ")");
        }
        int i = (int)idx;
        lowerDest.put(destIndex, lower[i]);
        upperDest.put(destIndex, upper[i]);
    }

    @Override
    public void set(long idx, double lower, double upper) {
        int i = (int) idx;
        this.lower[i] = lower;
        this.upper[i] = upper;
    }

    @Override
    public void fill(double lower, double upper) {
        Arrays.fill(this.lower, lower);
        Arrays.fill(this.upper, upper);
    }

    @Override
    public void copyFrom(long start, DoubleBuffer lowerSrc, DoubleBuffer upperSrc, int count) {
        if (count < 0 || start < 0 || start + (long)count > lower.length) {
            throw new IndexOutOfBoundsException("DP table copy [" + start + ","
                    + (start + (long)count) + ") outside [0," + lower.length + ")");
        }
        int offset = (int)start;
        lowerSrc.rewind();
        upperSrc.rewind();
        lowerSrc.get(lower, offset, count);
        upperSrc.get(upper, offset, count);
    }

    @Override
    public void copyFromIndexed(long[] indices, DoubleBuffer lowerSrc, DoubleBuffer upperSrc, int count) {
        if (count < 0 || count > indices.length) {
            throw new IndexOutOfBoundsException("DP table indexed copy count " + count
                    + " outside [0," + indices.length + ")");
        }
        lowerSrc.rewind();
        upperSrc.rewind();
        for (int i = 0; i < count; i++) {
            long idx = indices[i];
            if (idx < 0 || idx >= lower.length) {
                throw new IndexOutOfBoundsException("DP table indexed copy target " + idx
                        + " outside [0," + lower.length + ")");
            }
            int offset = (int)idx;
            lower[offset] = lowerSrc.get();
            upper[offset] = upperSrc.get();
        }
    }

    @Override
    public boolean isDenseArrayBacked() {
        return true;
    }

    @Override
    public double[] lowerArrayUnsafe() {
        return lower;
    }

    @Override
    public double[] upperArrayUnsafe() {
        return upper;
    }

    @Override
    public double[][] lowerChunks() {
        return new double[][] { lower };
    }

    @Override
    public double[][] upperChunks() {
        return new double[][] { upper };
    }

    @Override
    public long estimatedBytes() {
        return 2L * (long) lower.length * (long) Double.BYTES;
    }
}

final class ShardedDPTable implements DPTable {

    private final long size;
    private final int shardSize;
    private final int shardShift;
    private final int shardMask;
    private final double[][] lowerShards;
    private final double[][] upperShards;

    ShardedDPTable(long size, int shardSize) {
        if (size < 0) {
            throw new IllegalArgumentException("Negative DP table size: " + size);
        }
        if (shardSize <= 0) {
            throw new IllegalArgumentException("Shard size must be positive: " + shardSize);
        }
        long shardCountLong = (size + shardSize - 1L) / shardSize;
        if (shardCountLong > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Too many DP shards: " + shardCountLong);
        }
        this.size = size;
        this.shardSize = shardSize;
        boolean powerOfTwo = (shardSize & (shardSize - 1)) == 0;
        this.shardShift = powerOfTwo ? Integer.numberOfTrailingZeros(shardSize) : -1;
        this.shardMask = powerOfTwo ? shardSize - 1 : -1;
        this.lowerShards = new double[(int) shardCountLong][];
        this.upperShards = new double[(int) shardCountLong][];

        for (int s = 0; s < lowerShards.length; s++) {
            long shardStart = (long) s * (long) shardSize;
            int len = (int) Math.min((long) shardSize, size - shardStart);
            lowerShards[s] = new double[len];
            upperShards[s] = new double[len];
        }
    }

    @Override
    public long size() {
        return size;
    }

    @Override
    public double lower(long idx) {
        int shard = shardIndex(idx);
        int offset = shardOffset(idx);
        return lowerShards[shard][offset];
    }

    @Override
    public double upper(long idx) {
        int shard = shardIndex(idx);
        int offset = shardOffset(idx);
        return upperShards[shard][offset];
    }

    @Override
    public void readPair(long idx, double[] lowerDest, double[] upperDest, int destIndex) {
        checkIndex(idx);
        int shard = rawShardIndex(idx);
        int offset = rawShardOffset(idx);
        lowerDest[destIndex] = lowerShards[shard][offset];
        upperDest[destIndex] = upperShards[shard][offset];
    }

    @Override
    public void readPair(long idx, DoubleBuffer lowerDest, DoubleBuffer upperDest, int destIndex) {
        checkIndex(idx);
        int shard = rawShardIndex(idx);
        int offset = rawShardOffset(idx);
        lowerDest.put(destIndex, lowerShards[shard][offset]);
        upperDest.put(destIndex, upperShards[shard][offset]);
    }

    @Override
    public void set(long idx, double lower, double upper) {
        int shard = shardIndex(idx);
        int offset = shardOffset(idx);
        lowerShards[shard][offset] = lower;
        upperShards[shard][offset] = upper;
    }

    @Override
    public void fill(double lower, double upper) {
        for (double[] shard : lowerShards) {
            Arrays.fill(shard, lower);
        }
        for (double[] shard : upperShards) {
            Arrays.fill(shard, upper);
        }
    }

    @Override
    public void copyFrom(long start, DoubleBuffer lowerSrc, DoubleBuffer upperSrc, int count) {
        if (count < 0 || start < 0 || start + (long)count > size) {
            throw new IndexOutOfBoundsException("DP table copy [" + start + ","
                    + (start + (long)count) + ") outside [0," + size + ")");
        }
        lowerSrc.rewind();
        upperSrc.rewind();
        long idx = start;
        int remaining = count;
        while (remaining > 0) {
            int shard = shardIndex(idx);
            int offset = shardOffset(idx);
            int len = Math.min(remaining, lowerShards[shard].length - offset);
            lowerSrc.get(lowerShards[shard], offset, len);
            upperSrc.get(upperShards[shard], offset, len);
            idx += len;
            remaining -= len;
        }
    }

    @Override
    public void copyFromIndexed(long[] indices, DoubleBuffer lowerSrc, DoubleBuffer upperSrc, int count) {
        if (count < 0 || count > indices.length) {
            throw new IndexOutOfBoundsException("DP table indexed copy count " + count
                    + " outside [0," + indices.length + ")");
        }
        lowerSrc.rewind();
        upperSrc.rewind();
        for (int i = 0; i < count; i++) {
            long idx = indices[i];
            int shard = shardIndex(idx);
            int offset = shardOffset(idx);
            lowerShards[shard][offset] = lowerSrc.get();
            upperShards[shard][offset] = upperSrc.get();
        }
    }

    @Override
    public boolean isDenseArrayBacked() {
        return false;
    }

    @Override
    public double[] lowerArrayUnsafe() {
        throw new IllegalStateException("Sharded DP table is not backed by one lower[] array");
    }

    @Override
    public double[] upperArrayUnsafe() {
        throw new IllegalStateException("Sharded DP table is not backed by one upper[] array");
    }

    @Override
    public double[][] lowerChunks() {
        return lowerShards;
    }

    @Override
    public double[][] upperChunks() {
        return upperShards;
    }

    @Override
    public long estimatedBytes() {
        if (size > Long.MAX_VALUE / (2L * (long) Double.BYTES)) {
            return Long.MAX_VALUE;
        }
        return 2L * size * (long) Double.BYTES;
    }

    private int shardIndex(long idx) {
        checkIndex(idx);
        return rawShardIndex(idx);
    }

    private int shardOffset(long idx) {
        return rawShardOffset(idx);
    }

    private int rawShardIndex(long idx) {
        return shardShift >= 0
                ? (int)(idx >>> shardShift)
                : (int)(idx / (long)shardSize);
    }

    private int rawShardOffset(long idx) {
        return shardShift >= 0
                ? (int)(idx & (long)shardMask)
                : (int)(idx % (long)shardSize);
    }

    private void checkIndex(long idx) {
        if (idx < 0 || idx >= size) {
            throw new IndexOutOfBoundsException("DP table index " + idx + " outside [0," + size + ")");
        }
    }
}

/**
 * File-backed, chunk-mapped DP storage for tables larger than node RAM/JVM heap.
 * Lower and upper arrays occupy two contiguous regions of one sparse temporary
 * file. Absolute DoubleBuffer operations are safe for disjoint parallel tiles.
 */
final class MappedDPTable implements DPTable {

    private final long size;
    private final int chunkEntries;
    private final int chunkShift;
    private final int chunkMask;
    private final Path path;
    private final FileChannel channel;
    private final MappedByteBuffer[] lowerMaps;
    private final MappedByteBuffer[] upperMaps;
    private final DoubleBuffer[] lowerViews;
    private final DoubleBuffer[] upperViews;
    private volatile boolean closed = false;

    MappedDPTable(long size, int chunkEntries, Path directory) {
        if (size < 0L) {
            throw new IllegalArgumentException("Negative mapped DP table size: " + size);
        }
        if (chunkEntries <= 0 || (long) chunkEntries * Double.BYTES > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Mapped DP chunk entries must fit one ByteBuffer: "
                    + chunkEntries);
        }
        if (size > Long.MAX_VALUE / (2L * Double.BYTES)) {
            throw new IllegalArgumentException("Mapped DP table byte size overflows long: " + size);
        }

        this.size = size;
        this.chunkEntries = chunkEntries;
        boolean powerOfTwo = (chunkEntries & (chunkEntries - 1)) == 0;
        this.chunkShift = powerOfTwo ? Integer.numberOfTrailingZeros(chunkEntries) : -1;
        this.chunkMask = powerOfTwo ? chunkEntries - 1 : -1;

        long chunkCountLong = (size + chunkEntries - 1L) / chunkEntries;
        if (chunkCountLong > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Too many mapped DP chunks: " + chunkCountLong);
        }
        int chunkCount = (int) chunkCountLong;
        this.lowerMaps = new MappedByteBuffer[chunkCount];
        this.upperMaps = new MappedByteBuffer[chunkCount];
        this.lowerViews = new DoubleBuffer[chunkCount];
        this.upperViews = new DoubleBuffer[chunkCount];

        Path createdPath = null;
        FileChannel openedChannel = null;
        try {
            Files.createDirectories(directory);
            createdPath = Files.createTempFile(directory, "packstar-dp-", ".bin");
            openedChannel = FileChannel.open(createdPath,
                    StandardOpenOption.READ, StandardOpenOption.WRITE,
                    StandardOpenOption.TRUNCATE_EXISTING);

            long lowerBytes = size * (long) Double.BYTES;
            long totalBytes = lowerBytes * 2L;
            if (totalBytes > 0L) {
                openedChannel.position(totalBytes - 1L);
                openedChannel.write(ByteBuffer.wrap(new byte[]{0}));
            }

            for (int chunk = 0; chunk < chunkCount; chunk++) {
                long start = (long) chunk * chunkEntries;
                int entries = (int) Math.min((long) chunkEntries, size - start);
                long bytes = (long) entries * Double.BYTES;
                lowerMaps[chunk] = openedChannel.map(FileChannel.MapMode.READ_WRITE,
                        start * (long) Double.BYTES, bytes);
                upperMaps[chunk] = openedChannel.map(FileChannel.MapMode.READ_WRITE,
                        lowerBytes + start * (long) Double.BYTES, bytes);
                lowerMaps[chunk].order(ByteOrder.nativeOrder());
                upperMaps[chunk].order(ByteOrder.nativeOrder());
                lowerViews[chunk] = lowerMaps[chunk].asDoubleBuffer();
                upperViews[chunk] = upperMaps[chunk].asDoubleBuffer();
            }
        } catch (IOException ex) {
            cleanupFailedConstruction(createdPath, openedChannel, lowerMaps, upperMaps);
            throw new UncheckedIOException("Unable to create mapped DP table in " + directory, ex);
        } catch (RuntimeException | Error ex) {
            cleanupFailedConstruction(createdPath, openedChannel, lowerMaps, upperMaps);
            throw ex;
        }
        this.path = createdPath;
        this.channel = openedChannel;
    }

    @Override
    public long size() {
        return size;
    }

    @Override
    public double lower(long idx) {
        checkOpenAndIndex(idx);
        return lowerViews[rawChunkIndex(idx)].get(rawChunkOffset(idx));
    }

    @Override
    public double upper(long idx) {
        checkOpenAndIndex(idx);
        return upperViews[rawChunkIndex(idx)].get(rawChunkOffset(idx));
    }

    @Override
    public void readPair(long idx, double[] lowerDest, double[] upperDest, int destIndex) {
        checkOpenAndIndex(idx);
        int chunk = rawChunkIndex(idx);
        int offset = rawChunkOffset(idx);
        lowerDest[destIndex] = lowerViews[chunk].get(offset);
        upperDest[destIndex] = upperViews[chunk].get(offset);
    }

    @Override
    public void readPair(long idx, DoubleBuffer lowerDest, DoubleBuffer upperDest, int destIndex) {
        checkOpenAndIndex(idx);
        int chunk = rawChunkIndex(idx);
        int offset = rawChunkOffset(idx);
        lowerDest.put(destIndex, lowerViews[chunk].get(offset));
        upperDest.put(destIndex, upperViews[chunk].get(offset));
    }

    @Override
    public void set(long idx, double lower, double upper) {
        checkOpenAndIndex(idx);
        int chunk = rawChunkIndex(idx);
        int offset = rawChunkOffset(idx);
        lowerViews[chunk].put(offset, lower);
        upperViews[chunk].put(offset, upper);
    }

    @Override
    public void fill(double lower, double upper) {
        ensureOpen();
        for (int chunk = 0; chunk < lowerViews.length; chunk++) {
            DoubleBuffer lo = lowerViews[chunk];
            DoubleBuffer hi = upperViews[chunk];
            for (int offset = 0; offset < lo.capacity(); offset++) {
                lo.put(offset, lower);
                hi.put(offset, upper);
            }
        }
    }

    @Override
    public void copyFrom(long start, DoubleBuffer lowerSrc, DoubleBuffer upperSrc, int count) {
        ensureOpen();
        if (count < 0 || start < 0L || start + (long) count > size) {
            throw new IndexOutOfBoundsException("Mapped DP copy [" + start + ","
                    + (start + (long) count) + ") outside [0," + size + ")");
        }
        lowerSrc.rewind();
        upperSrc.rewind();
        long index = start;
        int sourceOffset = 0;
        int remaining = count;
        while (remaining > 0) {
            int chunk = rawChunkIndex(index);
            int offset = rawChunkOffset(index);
            int length = Math.min(remaining, lowerViews[chunk].capacity() - offset);
            putRange(lowerViews[chunk], offset, lowerSrc, sourceOffset, length);
            putRange(upperViews[chunk], offset, upperSrc, sourceOffset, length);
            index += length;
            sourceOffset += length;
            remaining -= length;
        }
    }

    private static void putRange(DoubleBuffer destination, int destinationOffset,
                                 DoubleBuffer source, int sourceOffset, int length) {
        DoubleBuffer src = source.duplicate();
        src.position(sourceOffset);
        src.limit(sourceOffset + length);
        DoubleBuffer dst = destination.duplicate();
        dst.position(destinationOffset);
        dst.put(src);
    }

    @Override
    public void copyFromIndexed(long[] indices, DoubleBuffer lowerSrc,
                                DoubleBuffer upperSrc, int count) {
        ensureOpen();
        if (count < 0 || count > indices.length) {
            throw new IndexOutOfBoundsException("Mapped DP indexed copy count " + count
                    + " outside [0," + indices.length + ")");
        }
        lowerSrc.rewind();
        upperSrc.rewind();
        for (int i = 0; i < count; i++) {
            long idx = indices[i];
            checkIndex(idx);
            int chunk = rawChunkIndex(idx);
            int offset = rawChunkOffset(idx);
            lowerViews[chunk].put(offset, lowerSrc.get());
            upperViews[chunk].put(offset, upperSrc.get());
        }
    }

    @Override
    public boolean isDenseArrayBacked() {
        return false;
    }

    @Override
    public double[] lowerArrayUnsafe() {
        throw new IllegalStateException("Mapped DP table is not backed by a Java lower[] array");
    }

    @Override
    public double[] upperArrayUnsafe() {
        throw new IllegalStateException("Mapped DP table is not backed by a Java upper[] array");
    }

    @Override
    public double[][] lowerChunks() {
        throw new IllegalStateException("Mapped DP table has no Java-array chunks");
    }

    @Override
    public double[][] upperChunks() {
        throw new IllegalStateException("Mapped DP table has no Java-array chunks");
    }

    @Override
    public boolean supportsArrayChunks() {
        return false;
    }

    @Override
    public boolean isFileBacked() {
        return true;
    }

    @Override
    public long estimatedBytes() {
        return 2L * size * (long) Double.BYTES;
    }

    Path pathForTesting() {
        return path;
    }

    @Override
    public synchronized void close() {
        if (closed) return;
        closed = true;
        try {
            channel.close();
        } catch (IOException ex) {
            System.err.println("Unable to close mapped DP channel " + path + ": " + ex.getMessage());
        }
        for (MappedByteBuffer map : lowerMaps) cleanMapping(map);
        for (MappedByteBuffer map : upperMaps) cleanMapping(map);
        try {
            Files.deleteIfExists(path);
        } catch (IOException ex) {
            // Unix permits unlinking mapped files; if a platform does not, leave
            // one diagnostic and let the job-local cleanup remove it later.
            System.err.println("Unable to delete mapped DP file " + path + ": " + ex.getMessage());
        }
    }

    private static void cleanMapping(MappedByteBuffer map) {
        if (map == null) return;
        try {
            Class<?> unsafeClass = Class.forName("sun.misc.Unsafe");
            Field field = unsafeClass.getDeclaredField("theUnsafe");
            field.setAccessible(true);
            Object unsafe = field.get(null);
            Method cleaner = unsafeClass.getMethod("invokeCleaner", ByteBuffer.class);
            cleaner.invoke(unsafe, map);
        } catch (Throwable ignored) {
            // Closing/unlinking remains correct on Linux; GC will unmap later.
        }
    }

    private static void cleanupFailedConstruction(Path path, FileChannel channel,
                                                  MappedByteBuffer[] lowerMaps,
                                                  MappedByteBuffer[] upperMaps) {
        if (channel != null) {
            try {
                channel.close();
            } catch (IOException ignored) {
            }
        }
        for (MappedByteBuffer map : lowerMaps) cleanMapping(map);
        for (MappedByteBuffer map : upperMaps) cleanMapping(map);
        if (path != null) {
            try {
                Files.deleteIfExists(path);
            } catch (IOException ignored) {
            }
        }
    }

    private int rawChunkIndex(long idx) {
        return chunkShift >= 0
                ? (int) (idx >>> chunkShift)
                : (int) (idx / (long) chunkEntries);
    }

    private int rawChunkOffset(long idx) {
        return chunkShift >= 0
                ? (int) (idx & (long) chunkMask)
                : (int) (idx % (long) chunkEntries);
    }

    private void checkOpenAndIndex(long idx) {
        ensureOpen();
        checkIndex(idx);
    }

    private void checkIndex(long idx) {
        if (idx < 0L || idx >= size) {
            throw new IndexOutOfBoundsException("Mapped DP index " + idx
                    + " outside [0," + size + ")");
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Mapped DP table is closed: " + path);
        }
    }
}
