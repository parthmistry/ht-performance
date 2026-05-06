package post.parthmistry.htperformance;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.GZIPInputStream;

public class RegexLogAnalyzer {

    private static final Pattern MAIN_PATTERN = Pattern.compile(
            "^(?<timestamp>\\S+)\\s+" +
            "(?<level>INFO|WARN|ERROR|DEBUG|TRACE)\\s+" +
            "\\[(?<service>[a-zA-Z0-9\\-]+)]\\s+" +
            "\\[traceId=(?<traceId>[a-fA-F0-9]+)]\\s+" +
            "userId=(?<userId>\\d+)\\s+" +
            "ip=(?<ip>(?:\\d{1,3}\\.){3}\\d{1,3})\\s+" +
            "method=(?<method>GET|POST|PUT|PATCH|DELETE)\\s+" +
            "path=(?<path>\\S+)\\s+" +
            "status=(?<status>\\d{3})\\s+" +
            "latencyMs=(?<latency>\\d+)\\s+" +
            "msg=\"(?<message>[^\"]*)\".*$"
    );

    private static final Pattern ERROR_PATTERN = Pattern.compile(
            "ERROR.*exception=\"(?<exception>[^\"]+)\".*stack=\"(?<stack>[^\"]+)\""
    );

    private static final Pattern SLOW_API_PATTERN = Pattern.compile(
            "method=(GET|POST|PUT|PATCH|DELETE)\\s+path=(/api/\\S+)\\s+status=\\d{3}\\s+latencyMs=([1-9]\\d{3,})"
    );

    private static final List<String> POISON_PILL = Collections.emptyList();

    private static final int BATCH_SIZE = 10_000;

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.out.println("Usage:");
            System.out.println("  java RegexLogAnalyzer <threads> <log-file-1> [log-file-2] ...");
            System.out.println();
            System.out.println("Example:");
            System.out.println("  java RegexLogAnalyzer 8 app.log");
            return;
        }

        int threads = Integer.parseInt(args[0]);

        List<Path> files = new ArrayList<>();
        for (int i = 1; i < args.length; i++) {
            files.add(Paths.get(args[i]));
        }

        long start = System.nanoTime();

        BlockingQueue<List<String>> queue = new ArrayBlockingQueue<>(threads * 4);

        List<WorkerStats> workerStats = new ArrayList<>();
        List<Thread> workers = new ArrayList<>();

        for (int i = 0; i < threads; i++) {
            WorkerStats stats = new WorkerStats();
            workerStats.add(stats);

            Thread worker = new Thread(new LogWorker(queue, stats), "log-worker-" + i);
            worker.start();
            workers.add(worker);
        }

        long totalBytes = 0;
        for (Path file : files) {
            totalBytes += Files.size(file);
        }

        // Read all files in parallel — up to `threads` reader threads, one per file
        int readerCount = Math.min(threads, files.size());
        ExecutorService readerPool = Executors.newFixedThreadPool(readerCount);
        AtomicLong submittedLinesAtomic = new AtomicLong(0);
        List<Future<?>> readerFutures = new ArrayList<>();

        for (Path file : files) {
            readerFutures.add(readerPool.submit(() -> {
                try {
                    long lines = readFileIntoQueue(file, queue);
                    submittedLinesAtomic.addAndGet(lines);
                } catch (IOException | InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException(e);
                }
            }));
        }

        readerPool.shutdown();

        // Wait for all readers to finish, then send poison pills
        for (Future<?> f : readerFutures) {
            f.get();
        }

        for (int i = 0; i < threads; i++) {
            queue.put(POISON_PILL);
        }

        for (Thread worker : workers) {
            worker.join();
        }

        long submittedLines = submittedLinesAtomic.get();

        FinalStats finalStats = merge(workerStats);

        long end = System.nanoTime();
        double elapsedSec = (end - start) / 1_000_000_000.0;
        double gbProcessed = totalBytes / 1024.0 / 1024.0 / 1024.0;

        printStats(finalStats, submittedLines, totalBytes, elapsedSec, gbProcessed);
    }

    private static long readFileIntoQueue(Path file, BlockingQueue<List<String>> queue)
            throws IOException, InterruptedException {

        long lineCount = 0;
        boolean isGzip = file.getFileName().toString().endsWith(".gz");

        InputStream raw = Files.newInputStream(file);
        InputStream source = isGzip ? new GZIPInputStream(raw, 256 * 1024) : raw;

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(source, StandardCharsets.UTF_8), 8 * 1024 * 1024)) {

            List<String> batch = new ArrayList<>(BATCH_SIZE);

            String line;
            while ((line = reader.readLine()) != null) {
                batch.add(line);
                lineCount++;

                if (batch.size() >= BATCH_SIZE) {
                    queue.put(batch);
                    batch = new ArrayList<>(BATCH_SIZE);
                }
            }

            if (!batch.isEmpty()) {
                queue.put(batch);
            }
        }

        return lineCount;
    }

    private static class LogWorker implements Runnable {

        private final BlockingQueue<List<String>> queue;

        private final WorkerStats stats;

        LogWorker(BlockingQueue<List<String>> queue, WorkerStats stats) {
            this.queue = queue;
            this.stats = stats;
        }

        @Override
        public void run() {
            try {
                while (true) {
                    List<String> batch = queue.take();

                    if (batch == POISON_PILL) {
                        break;
                    }

                    for (String line : batch) {
                        processLine(line, stats);
                    }
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }

        private void processLine(String line, WorkerStats stats) {
            stats.totalLines++;

            Matcher mainMatcher = MAIN_PATTERN.matcher(line);

            if (!mainMatcher.matches()) {
                stats.unmatchedLines++;
                return;
            }

            String level = mainMatcher.group("level");
            String service = mainMatcher.group("service");
            String method = mainMatcher.group("method");
            String path = mainMatcher.group("path");
            int status = Integer.parseInt(mainMatcher.group("status"));
            int latency = Integer.parseInt(mainMatcher.group("latency"));

            increment(stats.levelCounts, level);
            increment(stats.serviceCounts, service);
            increment(stats.methodCounts, method);
            increment(stats.pathCounts, path);
            increment(stats.statusCounts, String.valueOf(status));

            stats.totalLatency += latency;
            stats.maxLatency = Math.max(stats.maxLatency, latency);

            if (latency >= 1000) {
                stats.slowRequests++;
            }

            if (status >= 500) {
                stats.serverErrors++;
            } else if (status >= 400) {
                stats.clientErrors++;
            }

            if ("ERROR".equals(level)) {
                stats.errorLines++;

                Matcher errorMatcher = ERROR_PATTERN.matcher(line);
                if (errorMatcher.find()) {
                    String exception = errorMatcher.group("exception");
                    increment(stats.exceptionCounts, exception);
                }
            }

            if ("WARN".equals(level)) {
                stats.warnLines++;
            }

            Matcher slowMatcher = SLOW_API_PATTERN.matcher(line);
            if (slowMatcher.find()) {
                String slowKey = slowMatcher.group(1) + " " + slowMatcher.group(2);
                increment(stats.slowEndpointCounts, slowKey);
            }
        }

        private void increment(Map<String, Long> map, String key) {
            map.merge(key, 1L, Long::sum);
        }
    }

    private static class WorkerStats {
        long totalLines;
        long unmatchedLines;

        long errorLines;
        long warnLines;
        long slowRequests;
        long serverErrors;
        long clientErrors;

        long totalLatency;
        long maxLatency;

        Map<String, Long> levelCounts = new HashMap<>();
        Map<String, Long> serviceCounts = new HashMap<>();
        Map<String, Long> methodCounts = new HashMap<>();
        Map<String, Long> pathCounts = new HashMap<>();
        Map<String, Long> statusCounts = new HashMap<>();
        Map<String, Long> exceptionCounts = new HashMap<>();
        Map<String, Long> slowEndpointCounts = new HashMap<>();
    }

    private static class FinalStats {
        long totalLines;
        long unmatchedLines;

        long errorLines;
        long warnLines;
        long slowRequests;
        long serverErrors;
        long clientErrors;

        long totalLatency;
        long maxLatency;

        Map<String, Long> levelCounts = new HashMap<>();
        Map<String, Long> serviceCounts = new HashMap<>();
        Map<String, Long> methodCounts = new HashMap<>();
        Map<String, Long> pathCounts = new HashMap<>();
        Map<String, Long> statusCounts = new HashMap<>();
        Map<String, Long> exceptionCounts = new HashMap<>();
        Map<String, Long> slowEndpointCounts = new HashMap<>();
    }

    private static FinalStats merge(List<WorkerStats> allStats) {
        FinalStats finalStats = new FinalStats();

        for (WorkerStats stats : allStats) {
            finalStats.totalLines += stats.totalLines;
            finalStats.unmatchedLines += stats.unmatchedLines;
            finalStats.errorLines += stats.errorLines;
            finalStats.warnLines += stats.warnLines;
            finalStats.slowRequests += stats.slowRequests;
            finalStats.serverErrors += stats.serverErrors;
            finalStats.clientErrors += stats.clientErrors;
            finalStats.totalLatency += stats.totalLatency;
            finalStats.maxLatency = Math.max(finalStats.maxLatency, stats.maxLatency);

            mergeMap(finalStats.levelCounts, stats.levelCounts);
            mergeMap(finalStats.serviceCounts, stats.serviceCounts);
            mergeMap(finalStats.methodCounts, stats.methodCounts);
            mergeMap(finalStats.pathCounts, stats.pathCounts);
            mergeMap(finalStats.statusCounts, stats.statusCounts);
            mergeMap(finalStats.exceptionCounts, stats.exceptionCounts);
            mergeMap(finalStats.slowEndpointCounts, stats.slowEndpointCounts);
        }

        return finalStats;
    }

    private static void mergeMap(Map<String, Long> target, Map<String, Long> source) {
        for (Map.Entry<String, Long> entry : source.entrySet()) {
            target.merge(entry.getKey(), entry.getValue(), Long::sum);
        }
    }

    private static void printStats(
            FinalStats stats,
            long submittedLines,
            long totalBytes,
            double elapsedSec,
            double gbProcessed
    ) {
        System.out.println();
        System.out.println("========== LOG ANALYSIS STATS ==========");
        System.out.println();

        System.out.printf("Files size              : %.3f GB%n", gbProcessed);
        System.out.printf("Submitted lines         : %,d%n", submittedLines);
        System.out.printf("Processed lines         : %,d%n", stats.totalLines);
        System.out.printf("Unmatched lines         : %,d%n", stats.unmatchedLines);
        System.out.printf("Elapsed time            : %.3f sec%n", elapsedSec);
        System.out.printf("Throughput              : %.3f GB/sec%n", gbProcessed / elapsedSec);
        System.out.printf("Lines/sec               : %,.0f%n", stats.totalLines / elapsedSec);

        System.out.println();

        double avgLatency = stats.totalLines == 0
                ? 0.0
                : stats.totalLatency / (double) stats.totalLines;

        System.out.printf("Average latency         : %.2f ms%n", avgLatency);
        System.out.printf("Max latency             : %,d ms%n", stats.maxLatency);
        System.out.printf("Slow requests >=1000 ms : %,d%n", stats.slowRequests);
        System.out.printf("Client errors 4xx       : %,d%n", stats.clientErrors);
        System.out.printf("Server errors 5xx       : %,d%n", stats.serverErrors);
        System.out.printf("ERROR lines             : %,d%n", stats.errorLines);
        System.out.printf("WARN lines              : %,d%n", stats.warnLines);

        printMap("Log levels", stats.levelCounts, 20);
        printMap("Services", stats.serviceCounts, 20);
        printMap("HTTP methods", stats.methodCounts, 20);
        printMap("HTTP statuses", stats.statusCounts, 30);
        printMap("Top paths", stats.pathCounts, 10);
        printMap("Exceptions", stats.exceptionCounts, 10);
        printMap("Slow endpoints", stats.slowEndpointCounts, 10);
    }

    private static void printMap(String title, Map<String, Long> map, int limit) {
        System.out.println();
        System.out.println("---- " + title + " ----");

        map.entrySet()
                .stream()
                .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
                .limit(limit)
                .forEach(e -> System.out.printf("%-40s %,d%n", e.getKey(), e.getValue()));
    }

}
