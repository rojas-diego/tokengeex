use std::{
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
    thread::ThreadId,
    time::Instant,
};

use rayon::current_num_threads;

pub struct Task {
    pub start: Instant,
    pub desc: String,
    pub num_samples: usize,
    pub num_samples_processed: AtomicUsize,
    pub num_bytes_processed: AtomicUsize,
}

pub struct LocalTask<'a> {
    task: &'a Task,
    start: Instant,
    num_chunk_samples: usize,
    num_bytes_processed: usize,
    tid: u64,
}

impl Task {
    pub fn new(desc: &str, num_samples: usize, chunk_size: usize) -> Self {
        log::info!(
            "{} | {} samples | {} threads | {} chunks | {} chunk size",
            desc,
            num_samples,
            current_num_threads(),
            (num_samples + chunk_size) / chunk_size,
            chunk_size
        );

        Self {
            start: Instant::now(),
            desc: desc.to_string(),
            num_samples,
            num_samples_processed: AtomicUsize::new(0),
            num_bytes_processed: AtomicUsize::new(0),
        }
    }

    pub fn finish(&self) {
        let total_bytes_processed = self.num_bytes_processed.load(Relaxed);
        let total_samples_processed = self.num_samples_processed.load(Relaxed);

        log::info!(
            "{} | FINISHED {:.2}MB in {:.2}m | {} samples | {:.2}MB/s",
            self.desc,
            (total_bytes_processed as f64) / 1024.0 / 1024.0,
            (self.start.elapsed().as_secs() as f64) / 60.0,
            total_samples_processed,
            mb_per_sec(total_bytes_processed, self.start),
        );
    }

    pub fn local(&self, num_chunk_samples: usize) -> LocalTask<'_> {
        LocalTask {
            task: self,
            start: Instant::now(),
            num_chunk_samples,
            num_bytes_processed: 0,
            tid: unsafe { std::mem::transmute::<ThreadId, u64>(std::thread::current().id()) },
        }
    }
}

impl<'a> LocalTask<'a> {
    pub fn record(&mut self, num_bytes: usize) {
        self.num_bytes_processed += num_bytes;
    }

    pub fn finish(&self) {
        self.task
            .num_samples_processed
            .fetch_add(self.num_chunk_samples, Relaxed);
        self.task
            .num_bytes_processed
            .fetch_add(self.num_bytes_processed, Relaxed);

        let total_bytes_processed = self.task.num_bytes_processed.load(Relaxed);
        let total_samples_processed = self.task.num_samples_processed.load(Relaxed);

        let percent_done = (total_samples_processed as f64 / self.task.num_samples as f64) * 100.0;
        let eta = (self.task.start.elapsed().as_secs_f64() / percent_done) * (100.0 - percent_done);

        log::debug!(
            "Worker {:>3} | ETA {:>5}s | {:>6.2}% | Task {:>5.2}MB/s | Thread {:>5.2}MB/s ({:05.2}MB in {}s)",
            self.tid,
            eta.round(),
            percent_done,
            mb_per_sec(total_bytes_processed, self.task.start),
            mb_per_sec(self.num_bytes_processed, self.start),
            (self.num_bytes_processed as f64) / 1024.0 / 1024.0,
            self.start.elapsed().as_secs(),
        );
    }
}

// We chunk up samples into chunks that are at most 1/10th of the
// per-thread workload because too large chunks can cause some threads
// to be idle while others are still working. We also prevent
// chunks from being too small to avoid too much overhead.
pub fn par_chunk_size(num_samples: usize, min_chunk_size: usize, f: usize) -> usize {
    let chunk_size = num_samples / current_num_threads() / f;
    std::cmp::max(1, std::cmp::max(chunk_size, min_chunk_size))
}

pub fn mb_per_sec(n: usize, since: std::time::Instant) -> f64 {
    (n as f64 / 1024.0 / 1024.0) / since.elapsed().as_secs_f64()
}
