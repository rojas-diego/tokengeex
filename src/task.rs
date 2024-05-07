use rayon::current_num_threads;
use std::{
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed},
        Arc,
    },
    time::Instant,
};

#[derive(Clone)]
pub struct Task {
    inner: Arc<TaskInner>,
}

struct TaskInner {
    pub start: Instant,
    pub desc: String,
    pub num_samples: usize,
    pub num_samples_processed: AtomicUsize,
    pub num_bytes_processed: AtomicUsize,
    pub finished: AtomicBool,
}

pub struct LocalTask {
    task: Arc<TaskInner>,
    num_chunk_samples: usize,
    num_bytes_processed: usize,
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
            inner: Arc::new(TaskInner {
                start: Instant::now(),
                desc: desc.to_string(),
                num_samples,
                num_samples_processed: AtomicUsize::new(0),
                num_bytes_processed: AtomicUsize::new(0),
                finished: AtomicBool::new(false),
            }),
        }
    }

    pub fn start(&self) {
        let task = self.inner.clone();

        std::thread::spawn(move || {
            loop {
                std::thread::sleep(std::time::Duration::from_secs(1));

                if task.finished.load(Relaxed) {
                    break;
                }

                let num_samples_processed = task.num_samples_processed.load(Relaxed);
                let num_bytes_processed = task.num_bytes_processed.load(Relaxed);

                if num_samples_processed >= task.num_samples {
                    break;
                }

                let percent_done = (num_samples_processed as f64 / task.num_samples as f64) * 100.0;

                if percent_done == 0.0 {
                    continue;
                }

                let eta =
                    (task.start.elapsed().as_secs_f64() / percent_done) * (100.0 - percent_done);

                log::debug!(
                    "{} | {:>6.2}% | ETA {:>5}s | {:>5.2}MB/s | {:>5.2}MB/s per thread",
                    task.desc,
                    percent_done,
                    eta.round(),
                    mb_per_sec(num_bytes_processed, task.start),
                    mb_per_sec(num_bytes_processed, task.start) / current_num_threads() as f64,
                );
            }

            let num_bytes_processed = task.num_bytes_processed.load(Relaxed);

            log::info!(
                "FINISHED {} | {} samples | {:.2}MB/s | {:.2}s",
                task.desc,
                task.num_samples,
                mb_per_sec(num_bytes_processed, task.start),
                task.start.elapsed().as_secs_f64(),
            );
        });
    }

    pub fn local(&self, num_samples: usize) -> LocalTask {
        LocalTask {
            task: self.inner.clone(),
            num_chunk_samples: num_samples,
            num_bytes_processed: 0,
        }
    }

    pub fn finish(&self) {
        self.inner.finished.store(true, Relaxed);
    }
}

impl LocalTask {
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
    }
}

// We chunk up samples into chunks that are at most f/10th of the
// per-thread workload because too large chunks can cause some threads
// to be idle while others are still working. We also prevent
// chunks from being too small to avoid too much overhead.
pub fn par_chunk_size(num_samples: usize, f: usize) -> usize {
    let chunk_size = num_samples / current_num_threads() / f;
    std::cmp::max(1, chunk_size)
}

pub fn mb_per_sec(n: usize, since: std::time::Instant) -> f64 {
    (n as f64 / 1024.0 / 1024.0) / since.elapsed().as_secs_f64()
}
