package edu.duke.cs.osprey.packstar;

import edu.duke.cs.osprey.parallelism.*;

import org.junit.jupiter.api.Test;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.*;

public class TestPackStarTaskCapacity {
    @Test public void freesOneSlotWithoutWaitingForTheSlowTask() throws Exception {
        var tasks = new ThreadPoolTaskExecutor();
        tasks.start(2);
        var slowRelease = new CountDownLatch(1);
        var finished = new AtomicInteger();
        var producer = Executors.newSingleThreadExecutor();
        try {
            tasks.submit(() -> {
                try { slowRelease.await(); }
                catch (InterruptedException ex) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException(ex);
                }
                return 1;
            }, n -> finished.incrementAndGet());
            tasks.submit(() -> 2, n -> finished.incrementAndGet());
            Future<?> next = producer.submit(() -> {
                tasks.waitForCapacity(2);
                tasks.submit(() -> 3, n -> finished.incrementAndGet());
            });
            next.get(5, TimeUnit.SECONDS);
            assertEquals(1L, slowRelease.getCount());
            assertTrue(tasks.getNumRunningTasks() <= 2);
            slowRelease.countDown();
            tasks.waitForFinish();
            assertEquals(3, finished.get());
        } finally {
            slowRelease.countDown();
            producer.shutdownNow();
            tasks.stopAndWait(5000);
        }
    }

    @Test public void failuresArePropagatedWithoutWaitingForeverForACallback() throws Exception {
        var tasks = new ThreadPoolTaskExecutor();
        tasks.start(1);
        var producer = Executors.newSingleThreadExecutor();
        try {
            tasks.submit(() -> { throw new IllegalStateException("CCD failed"); }, n -> fail());
            Future<?> pending = producer.submit(() -> tasks.waitForCapacity(1));
            ExecutionException error = assertThrows(ExecutionException.class,
                    () -> pending.get(5, TimeUnit.SECONDS));
            assertInstanceOf(TaskExecutor.TaskException.class, error.getCause());
        } finally {
            producer.shutdownNow();
            tasks.stopAndWait(5000);
        }
    }
}
