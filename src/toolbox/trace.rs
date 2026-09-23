// Copyright 2026 The Sashiko Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! An optional record of every tool call a review stage makes.
//!
//! Set `SASHIKO_TOOL_TRACE` to a file path and each tool call, and the end of
//! each stage, is appended to it as one JSON line. Unset, nothing is recorded.

use serde_json::{Value, json};
use std::io::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Stage runs are numbered per process, not per trace, because each patch
/// gets its own ToolBox and all of them append to the same file.
static STAGE_RUNS: AtomicU64 = AtomicU64::new(0);

enum Sink {
    File(std::fs::File),
    Memory(Vec<Value>),
}

pub struct ToolTrace {
    sink: Mutex<Sink>,
}

impl ToolTrace {
    /// Opens the file named by `SASHIKO_TOOL_TRACE` for appending, or returns
    /// None when the variable is unset or the file cannot be opened.
    pub fn from_env() -> Option<Self> {
        let path = std::env::var_os("SASHIKO_TOOL_TRACE")?;
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            Ok(file) => Some(Self {
                sink: Mutex::new(Sink::File(file)),
            }),
            Err(e) => {
                tracing::warn!(
                    "SASHIKO_TOOL_TRACE: cannot open {}: {}",
                    path.to_string_lossy(),
                    e
                );
                None
            }
        }
    }

    /// Keeps records in memory, for tests to read back with `records`.
    pub fn in_memory() -> Self {
        Self {
            sink: Mutex::new(Sink::Memory(Vec::new())),
        }
    }

    /// A fresh id for one execution of one stage, so that two runs of the
    /// same stage, concurrent or retried, are told apart.
    pub fn next_stage_run() -> u64 {
        STAGE_RUNS.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Appends one record, stamped with the wall clock in milliseconds.
    pub fn record(&self, mut record: Value) {
        if let Some(obj) = record.as_object_mut() {
            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            obj.insert("ts_ms".to_string(), json!(ts));
        }
        let mut sink = self.sink.lock().unwrap();
        match &mut *sink {
            Sink::File(file) => {
                // One write per line: with O_APPEND that keeps lines from
                // several ToolBoxes in one process from interleaving.
                let mut line = record.to_string();
                line.push('\n');
                if let Err(e) = file.write_all(line.as_bytes()) {
                    tracing::warn!("SASHIKO_TOOL_TRACE: write failed: {}", e);
                }
            }
            Sink::Memory(records) => records.push(record),
        }
    }

    /// The records kept by an in-memory trace. Empty for a file trace.
    pub fn records(&self) -> Vec<Value> {
        match &*self.sink.lock().unwrap() {
            Sink::Memory(records) => records.clone(),
            Sink::File(_) => Vec::new(),
        }
    }
}
