package post.parthmistry.htperformance;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.zip.GZIPOutputStream;

public class LogGenerator {

    private static final String[] LEVELS = {
            "INFO", "WARN", "ERROR", "DEBUG", "TRACE"
    };

    private static final String[] SERVICES = {
            "auth-service",
            "payment-service",
            "order-service",
            "inventory-service",
            "notification-service",
            "search-service",
            "billing-service"
    };

    private static final String[] METHODS = {
            "GET", "POST", "PUT", "PATCH", "DELETE"
    };

    private static final String[] PATHS = {
            "/api/login",
            "/api/logout",
            "/api/orders",
            "/api/orders/{id}",
            "/api/payment",
            "/api/refund",
            "/api/products",
            "/api/search",
            "/api/cart",
            "/api/inventory",
            "/api/notifications"
    };

    private static final int[] STATUSES = {
            200, 201, 204, 301, 400, 401, 403, 404, 409, 429, 500, 502, 503
    };

    private static final String[] MESSAGES = {
            "Request completed",
            "User authenticated",
            "Payment processed",
            "Inventory updated",
            "Order created",
            "Cache miss",
            "Cache hit",
            "Database timeout",
            "External API failed",
            "Retrying request",
            "Validation failed",
            "Rate limit exceeded",
            "Unexpected exception occurred"
    };

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.out.println("Usage:");
            System.out.println("  java LogGenerator <output-prefix> <size-gb> <num-threads>");
            System.out.println();
            System.out.println("Example:");
            System.out.println("  java LogGenerator app.log 8 4");
            System.out.println("  (produces app.log.part0.gz ... app.log.part3.gz, each 2 GB of uncompressed logs)");
            return;
        }

        String outputPrefix = args[0];
        double sizeGb = Double.parseDouble(args[1]);
        int numThreads = Integer.parseInt(args[2]);

        long totalTargetBytes = (long) (sizeGb * 1024L * 1024L * 1024L);
        long bytesPerFile = totalTargetBytes / numThreads;

        Path firstFile = Paths.get(outputPrefix + ".part0.gz");
        Path parent = firstFile.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }

        System.out.printf("Generating %.2f GB across %d file(s), ~%.2f GB each (uncompressed)%n",
                sizeGb, numThreads, bytesPerFile / 1024.0 / 1024.0 / 1024.0);

        long overallStart = System.nanoTime();

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<Future<FileResult>> futures = new ArrayList<>();

        for (int i = 0; i < numThreads; i++) {
            final int partIndex = i;
            final long seed = 42L + i;
            Path outputFile = Paths.get(outputPrefix + ".part" + partIndex + ".gz");

            futures.add(executor.submit(new Callable<FileResult>() {
                @Override
                public FileResult call() throws Exception {
                    return generateFile(outputFile, bytesPerFile, seed, partIndex, numThreads);
                }
            }));
        }

        executor.shutdown();

        long totalLines = 0;
        long totalUncompressedBytes = 0;

        for (int i = 0; i < futures.size(); i++) {
            FileResult result = futures.get(i).get();
            System.out.printf("Part %d: %,d lines, %.2f GB uncompressed -> %s%n",
                    i, result.lineCount,
                    result.writtenBytes / 1024.0 / 1024.0 / 1024.0,
                    result.outputFile.getFileName());
            totalLines += result.lineCount;
            totalUncompressedBytes += result.writtenBytes;
        }

        long overallEnd = System.nanoTime();

        System.out.println();
        System.out.println("Done.");
        System.out.printf("Total uncompressed size : %.2f GB%n", totalUncompressedBytes / 1024.0 / 1024.0 / 1024.0);
        System.out.printf("Total lines             : %,d%n", totalLines);
        System.out.printf("Total time              : %.2f sec%n", (overallEnd - overallStart) / 1_000_000_000.0);
    }

    private static FileResult generateFile(Path outputFile, long targetBytes, long seed, int partIndex, int totalParts)
            throws IOException {

        Random random = new Random(seed);
        long writtenBytes = 0;
        long lineCount = 0;
        long start = System.nanoTime();

        try (OutputStream fos = Files.newOutputStream(outputFile,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
             GZIPOutputStream gzip = new GZIPOutputStream(fos, 256 * 1024);
             BufferedWriter writer = new BufferedWriter(
                     new OutputStreamWriter(gzip, StandardCharsets.UTF_8), 8 * 1024 * 1024)) {

            StringBuilder batch = new StringBuilder(8 * 1024 * 1024);

            while (writtenBytes < targetBytes) {
                String line = generateLogLine(random);
                batch.append(line).append('\n');
                lineCount++;

                if (batch.length() >= 8 * 1024 * 1024) {
                    byte[] bytes = batch.toString().getBytes(StandardCharsets.UTF_8);
                    writer.write(batch.toString());
                    writtenBytes += bytes.length;
                    batch.setLength(0);

                    if (lineCount % 500_000 == 0) {
                        double percent = writtenBytes * 100.0 / targetBytes;
                        double gb = writtenBytes / 1024.0 / 1024.0 / 1024.0;
                        double elapsed = (System.nanoTime() - start) / 1_000_000_000.0;
                        System.out.printf("\r[part%d/%d] %.2f%% | %.2f GB | %,d lines | %.2f sec",
                                partIndex, totalParts - 1, percent, gb, lineCount, elapsed);
                    }
                }
            }

            if (!batch.isEmpty()) {
                byte[] bytes = batch.toString().getBytes(StandardCharsets.UTF_8);
                writer.write(batch.toString());
                writtenBytes += bytes.length;
            }
        }

        return new FileResult(outputFile, writtenBytes, lineCount);
    }

    private static class FileResult {
        final Path outputFile;
        final long writtenBytes;
        final long lineCount;

        FileResult(Path outputFile, long writtenBytes, long lineCount) {
            this.outputFile = outputFile;
            this.writtenBytes = writtenBytes;
            this.lineCount = lineCount;
        }
    }

    private static String generateLogLine(Random random) {
        String level = LEVELS[random.nextInt(LEVELS.length)];
        String service = SERVICES[random.nextInt(SERVICES.length)];
        String method = METHODS[random.nextInt(METHODS.length)];
        String path = PATHS[random.nextInt(PATHS.length)];
        int status = STATUSES[random.nextInt(STATUSES.length)];
        int latency = random.nextInt(5000);
        int userId = 1 + random.nextInt(1_000_000);

        String ip = random.nextInt(256) + "." +
                random.nextInt(256) + "." +
                random.nextInt(256) + "." +
                random.nextInt(256);

        String traceId = UUID.randomUUID().toString().replace("-", "");
        String message = MESSAGES[random.nextInt(MESSAGES.length)];

        String extra = "";

        if (level.equals("ERROR")) {
            extra = " exception=\"java.lang.RuntimeException\" stack=\"com.example.Service.call(Service.java:"
                    + (20 + random.nextInt(500)) + ")\"";
        } else if (level.equals("WARN")) {
            extra = " warningCode=W" + (1000 + random.nextInt(9000));
        }

        return Instant.now()
                + " "
                + level
                + "  ["
                + service
                + "] [traceId="
                + traceId
                + "] userId="
                + userId
                + " ip="
                + ip
                + " method="
                + method
                + " path="
                + path
                + " status="
                + status
                + " latencyMs="
                + latency
                + " msg=\""
                + message
                + "\""
                + extra;
    }
}
