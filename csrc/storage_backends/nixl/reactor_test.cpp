// SPDX-License-Identifier: Apache-2.0
// Deterministic behavior tests for the internal CrossCallReactor admission
// and load-fallback state machine (no NIXL agent, no Ceph).
// Build:
//   g++ -std=c++20 -O1 -pthread -I csrc/storage_backends -I
//   csrc/storage_backends/nixl -I "$NIXL_INCLUDE_DIR"
//   csrc/storage_backends/nixl/reactor_test.cpp -L "$NIXL_LIB_DIR" -lnixl
//   -Wl,-rpath,"$NIXL_LIB_DIR" -o reactor_test
// Then run ./reactor_test. libnixl is linked only for
// nixlEnumStrings::statusStr; no live NIXL behavior is exercised or claimed.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "reactor.h"

namespace {

#define CHECK(cond)                                                  \
  do {                                                               \
    if (!(cond)) {                                                   \
      std::fprintf(stderr, "FAILED: %s at %s:%d\n", #cond, __FILE__, \
                   __LINE__);                                        \
      std::exit(1);                                                  \
    }                                                                \
  } while (0)

using namespace lmcache::connector;

template <typename Pred>
bool wait_until(Pred pred, int timeout_ms = 5000) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (!pred()) {
    if (std::chrono::steady_clock::now() > deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

struct FakeState {
  std::atomic<int> live{0};
  std::atomic<int> polls{0};
  std::atomic<int> post_calls{0};
  std::atomic<bool> release{false};
  std::atomic<bool> posted{false};
  std::atomic<bool> completed{false};
};

// Fallback outcomes for a failed first load transfer.
constexpr int kFallbackHardThrow = 0;  // query error: raw throw, tile fails
constexpr int kFallbackSoft = 1;       // all keys present: false bits, ok
constexpr int kFallbackEmpty = 2;      // none present: empty retry
constexpr int kFallbackRetry = 3;      // present subset: real retry

// Ordered event log shared by the driver and its fake operations.
struct TraceLog {
  std::mutex mu;
  std::vector<std::string> events;
  void add(const std::string& event) {
    std::lock_guard<std::mutex> lock(mu);
    events.push_back(event);
  }
};

struct Script {
  bool posts = true;
  int prepare_mode = 0;  // 0 ok, 1 throw hard, 2 throw load-soft
  bool throw_on_post = false;
  bool stop_on_post = false;
  bool stop_during_prepare = false;
  bool stop_in_complete = false;
  int hold_polls = 0;  // >0: stay IN_PROG for this many polls even released
  nixl_status_t post_status = NIXL_IN_PROG;
  nixl_status_t terminal_status = NIXL_SUCCESS;
  std::vector<uint8_t> soft_results;
  std::vector<uint8_t> finish_results;
  std::string name = "op";
  std::shared_ptr<FakeState> state = std::make_shared<FakeState>();
  int fallback_mode = kFallbackRetry;
  std::vector<uint8_t> fallback_soft_results;
  std::shared_ptr<Script> retry_script;
  // Stamped by FakeDriver::set.
  TraceLog* trace = nullptr;
  std::atomic<bool>* stop_flag = nullptr;
};

class FakeOperation final : public NixlPendingOperation {
 public:
  explicit FakeOperation(const Script& script)
      : name_(script.name),
        state_(script.state),
        trace_(script.trace),
        stop_flag_(script.stop_flag),
        post_status_(script.post_status),
        terminal_status_(script.terminal_status),
        finish_results_(script.finish_results),
        throw_on_post_(script.throw_on_post),
        script_stop_on_post_(script.stop_on_post),
        stop_in_complete_(script.stop_in_complete),
        hold_polls_(script.hold_polls) {
    state_->live.fetch_add(1);
  }
  ~FakeOperation() override { state_->live.fetch_add(-1); }

  nixl_status_t post() override {
    state_->post_calls.fetch_add(1);
    if (throw_on_post_ && stop_flag_ != nullptr && script_stop_on_post_) {
      stop_flag_->store(true);
    }
    if (throw_on_post_) throw std::runtime_error("fake submission failure");
    state_->posted.store(true);
    if (post_status_ != NIXL_IN_PROG) latched_ = true;
    return post_status_;
  }

  nixl_status_t poll() override {
    const int count = state_->polls.fetch_add(1) + 1;
    if (post_status_ == NIXL_IN_PROG && !state_->release.load() &&
        (hold_polls_ == 0 || count <= hold_polls_)) {
      return NIXL_IN_PROG;
    }
    latched_ = true;
    return terminal_status_;
  }

  std::vector<uint8_t> complete() override {
    release();
    if (latched_ && terminal_status_ != NIXL_SUCCESS) {
      // Mirrors the real operation: failed transfers keep false bits.
      return std::vector<uint8_t>(finish_results_.size(), 0);
    }
    return std::move(finish_results_);
  }

  void discard() noexcept override { release(); }

 private:
  void release() {
    state_->completed.store(true);
    if (stop_flag_ != nullptr && stop_in_complete_) {
      stop_flag_->store(true);
    }
    if (trace_ != nullptr) trace_->add(name_ + ":complete");
  }

  std::string name_;
  std::shared_ptr<FakeState> state_;
  TraceLog* trace_;
  std::atomic<bool>* stop_flag_;
  nixl_status_t post_status_;
  nixl_status_t terminal_status_;
  std::vector<uint8_t> finish_results_;
  bool throw_on_post_ = false;
  bool script_stop_on_post_ = false;
  bool stop_in_complete_ = false;
  int hold_polls_ = 0;
  bool latched_ = false;
};

class FakeDriver final : public NixlReactorDriver {
 public:
  explicit FakeDriver(std::atomic<bool>* stop_flag) : stop_flag_(stop_flag) {}

  void set(uint64_t future_id, Script script) {
    script.trace = &trace_log_;
    script.stop_flag = stop_flag_;
    if (script.retry_script) {
      script.retry_script->trace = &trace_log_;
      script.retry_script->stop_flag = stop_flag_;
    }
    scripts_[future_id] = std::move(script);
  }

  std::shared_ptr<FakeState> state(uint64_t future_id) {
    return scripts_.at(future_id).state;
  }

  std::shared_ptr<FakeState> retry_state(uint64_t future_id) {
    return scripts_.at(future_id).retry_script->state;
  }

  bool posts_tile(const Request& request) const override {
    return scripts_.at(request.future_id).posts;
  }

  void execute_sync(const Request& request) override {
    std::lock_guard<std::mutex> lock(trace_mu_);
    sync_order_.push_back(request.future_id);
  }

  std::unique_ptr<NixlPendingOperation> prepare(
      const Request& request) override {
    const Script& script = scripts_.at(request.future_id);
    {
      std::lock_guard<std::mutex> lock(trace_mu_);
      prepare_order_.push_back(request.future_id);
    }
    if (script.stop_during_prepare) stop_flag_->store(true);
    if (script.prepare_mode == 1) {
      throw std::runtime_error("fake hard prepare failure");
    }
    if (script.prepare_mode == 2) {
      throw NixlLoadPrepareError("fake load prepare failure",
                                 script.soft_results);
    }
    return std::make_unique<FakeOperation>(script);
  }

  std::unique_ptr<NixlPendingOperation> prepare_load_fallback(
      const Request& request) override {
    const Script& script = scripts_.at(request.future_id);
    {
      std::lock_guard<std::mutex> lock(trace_mu_);
      fallback_order_.push_back(request.future_id);
    }
    trace_log_.add("fallback:" + script.name);
    switch (script.fallback_mode) {
      case kFallbackHardThrow:
        throw std::runtime_error("fake fallback query failure");
      case kFallbackSoft:
        throw NixlLoadPrepareError("fake load retry skipped",
                                   script.fallback_soft_results);
      case kFallbackEmpty: {
        Script empty = script.retry_script ? *script.retry_script : Script{};
        empty.post_status = NIXL_SUCCESS;
        empty.terminal_status = NIXL_SUCCESS;
        empty.finish_results.assign(request.keys.size(), 0);
        return std::make_unique<FakeOperation>(empty);
      }
      default:
        return std::make_unique<FakeOperation>(*script.retry_script);
    }
  }

  void log_load_error(const std::string& message) override {
    std::lock_guard<std::mutex> lock(trace_mu_);
    logs_.push_back(message);
  }

  size_t trace_index(const std::string& event) {
    std::lock_guard<std::mutex> lock(trace_log_.mu);
    for (size_t index = 0; index < trace_log_.events.size(); ++index) {
      if (trace_log_.events[index] == event) return index;
    }
    return 1000;
  }

  std::vector<uint64_t> prepare_order() {
    std::lock_guard<std::mutex> lock(trace_mu_);
    return prepare_order_;
  }

  size_t fallback_calls() {
    std::lock_guard<std::mutex> lock(trace_mu_);
    return fallback_order_.size();
  }

  size_t sync_calls() {
    std::lock_guard<std::mutex> lock(trace_mu_);
    return sync_order_.size();
  }

  size_t log_count(const std::string& needle) {
    std::lock_guard<std::mutex> lock(trace_mu_);
    size_t count = 0;
    for (const std::string& log : logs_) {
      if (log.find(needle) != std::string::npos) ++count;
    }
    return count;
  }

 private:
  std::atomic<bool>* stop_flag_;
  std::unordered_map<uint64_t, Script> scripts_;  // frozen before run
  TraceLog trace_log_;
  std::mutex trace_mu_;
  std::vector<uint64_t> prepare_order_;
  std::vector<uint64_t> fallback_order_;
  std::vector<uint64_t> sync_order_;
  std::vector<std::string> logs_;
};

std::shared_ptr<BatchState> make_batch(Op op, size_t num_items) {
  auto batch = std::make_shared<BatchState>();
  batch->batch_op = op;
  batch->remaining_tiles.store(1);
  batch->per_key_results.assign(num_items, 0);
  return batch;
}

Request make_request(uint64_t future_id, Op op, size_t num_keys,
                     std::shared_ptr<BatchState> batch, size_t start_idx) {
  Request request;
  request.future_id = future_id;
  request.op = op;
  for (size_t index = 0; index < num_keys; ++index) {
    request.keys.push_back("k" + std::to_string(index));
  }
  request.batch = std::move(batch);
  request.start_idx = start_idx;
  return request;
}

class Harness {
 public:
  void start() {
    worker_ = std::thread([this] {
      CrossCallReactor reactor(driver);
      try {
        reactor.run(
            queue_mu_, queue_cv, queue_, [this] { return stop.load(); },
            [this](const Request& request, const Completion& completion) {
              std::lock_guard<std::mutex> lock(emit_mu_);
              ++emit_attempts_;
              if (throw_next_emit_) {
                throw_next_emit_ = false;
                throw std::runtime_error("fake emit failure");
              }
              if (emits_.size() >= emit_fail_after) {
                throw std::runtime_error("fake emit failure");
              }
              emits_.push_back(
                  {request.future_id, completion.ok, completion.error});
            });
      } catch (...) {
        run_threw.store(true);
      }
    });
  }

  void push(Request request) {
    {
      std::lock_guard<std::mutex> lock(queue_mu_);
      queue_.push(std::move(request));
    }
    queue_cv.notify_one();
  }

  void stop_and_join() {
    stop.store(true);
    queue_cv.notify_all();
    worker_.join();
  }

  void join() { worker_.join(); }

  size_t emit_count() {
    std::lock_guard<std::mutex> lock(emit_mu_);
    return emits_.size();
  }

  // The next completion callback throws once; later callbacks record.
  void fail_next_emit() {
    std::lock_guard<std::mutex> lock(emit_mu_);
    throw_next_emit_ = true;
  }

  size_t emit_attempts() {
    std::lock_guard<std::mutex> lock(emit_mu_);
    return emit_attempts_;
  }

  // The future_id completed exactly once, optionally with the given status
  // and an error substring.
  bool emitted(uint64_t future_id, std::optional<bool> ok = {},
               const std::string& error_needle = "") {
    std::lock_guard<std::mutex> lock(emit_mu_);
    size_t count = 0;
    for (const Emit& emit : emits_) {
      if (emit.future_id != future_id) continue;
      ++count;
      if (ok && emit.ok != *ok) return false;
      if (!error_needle.empty() &&
          emit.error.find(error_needle) == std::string::npos) {
        return false;
      }
    }
    return count == 1;
  }

  FakeDriver driver{&stop};
  std::atomic<bool> stop{false};
  std::atomic<bool> run_threw{false};
  size_t emit_fail_after = std::numeric_limits<size_t>::max();
  std::condition_variable queue_cv;

 private:
  struct Emit {
    uint64_t future_id;
    bool ok;
    std::string error;
  };

  std::mutex queue_mu_;
  std::queue<Request> queue_;
  std::mutex emit_mu_;
  std::vector<Emit> emits_;
  size_t emit_attempts_ = 0;
  bool throw_next_emit_ = false;
  std::thread worker_;
};

Script held_script(std::string name = "op") {
  Script script;
  script.name = std::move(name);
  return script;  // state->release stays false: pending until released
}

Script done_script(std::vector<uint8_t> finish_results = {},
                   std::string name = "op") {
  Script script;
  script.name = std::move(name);
  script.finish_results = std::move(finish_results);
  script.state->release.store(true);
  return script;
}

Script retry_script(nixl_status_t terminal, std::vector<uint8_t> bits,
                    std::string name = "retry") {
  Script script;
  script.name = std::move(name);
  script.terminal_status = terminal;
  script.finish_results = std::move(bits);
  script.state->release.store(true);
  return script;
}

// Posted tiles are admitted as they arrive: three tiles overlap a held
// one, complete, and each tile gets exactly one completion and one
// operation release.
void test_overlap_admission_and_exact_once() {
  Harness harness;
  harness.driver.set(1, held_script("held"));
  for (uint64_t id = 2; id <= 4; ++id) harness.driver.set(id, done_script());
  for (uint64_t id = 1; id <= 4; ++id) {
    harness.push(make_request(id, Op::BATCH_TILE_SET, 1,
                              make_batch(Op::BATCH_TILE_SET, 1), 0));
  }
  harness.start();
  CHECK(wait_until([&] { return harness.emit_count() == 3; }));
  // Tiles 2..4 posted and completed while tile 1 still holds its handle.
  CHECK(harness.driver.prepare_order() == std::vector<uint64_t>({1, 2, 3, 4}));
  CHECK(!harness.driver.state(1)->completed.load());
  harness.driver.state(1)->release.store(true);
  CHECK(wait_until([&] { return harness.emit_count() == 4; }));
  harness.stop_and_join();
  for (uint64_t id = 1; id <= 4; ++id) {
    CHECK(harness.emitted(id, true));
    CHECK(harness.driver.state(id)->completed.load());
    CHECK(harness.driver.state(id)->live.load() == 0);
  }
  std::printf("PASS overlap_admission_and_exact_once\n");
}

// A successful first transfer never queries, logs, or falls back.
void test_fast_load_success() {
  Harness harness;
  harness.driver.set(1, done_script({1, 1}, "first"));
  auto batch = make_batch(Op::BATCH_TILE_GET, 2);
  batch->per_key_results = {9, 9};
  harness.push(make_request(1, Op::BATCH_TILE_GET, 2, batch, 0));
  harness.start();
  CHECK(wait_until([&] { return harness.emit_count() == 1; }));
  harness.stop_and_join();
  CHECK(harness.driver.fallback_calls() == 0);
  CHECK(harness.driver.log_count("") == 0);
  CHECK(batch->per_key_results == std::vector<uint8_t>({1, 1}));
  CHECK(harness.emitted(1, true));
  std::printf("PASS fast_load_success\n");
}

// Taxonomy of a terminally failed first load transfer.
struct FallbackCase {
  const char* name;
  int fallback_mode;
  std::shared_ptr<Script> retry;
  std::vector<uint8_t> soft_bits;
  bool expect_ok;
  std::vector<uint8_t> expect_bits;  // compared only when expect_ok
  const char* expect_log;            // nullptr = nothing logged
};

void test_fallback_taxonomy() {
  const FallbackCase cases[] = {
      {"all-present-soft",
       kFallbackSoft,
       nullptr,
       {0, 0},
       true,
       {0, 0},
       "fake load retry skipped"},
      {"none-present-empty",
       kFallbackEmpty,
       nullptr,
       {},
       true,
       {0, 0},
       nullptr},
      {"query-error-hard", kFallbackHardThrow, nullptr, {}, false, {}, nullptr},
      {"retry-failed-soft",
       kFallbackRetry,
       std::make_shared<Script>(
           retry_script(NIXL_ERR_BACKEND, {0, 0}, "retry")),
       {},
       true,
       {0, 0},
       "NIXL transfer progress failed"},
  };
  for (const FallbackCase& test : cases) {
    Harness harness;
    Script first = done_script({}, "first");
    first.terminal_status = NIXL_ERR_NOT_FOUND;
    first.fallback_mode = test.fallback_mode;
    first.retry_script = test.retry;
    first.fallback_soft_results = test.soft_bits;
    harness.driver.set(1, first);
    auto batch = make_batch(Op::BATCH_TILE_GET, 2);
    batch->per_key_results = {9, 9};
    harness.push(make_request(1, Op::BATCH_TILE_GET, 2, batch, 0));
    harness.start();
    CHECK(wait_until([&] { return harness.emit_count() == 1; }));
    harness.stop_and_join();
    CHECK(harness.driver.fallback_calls() == 1);
    CHECK(harness.driver.state(1)->live.load() == 0);  // failed op released
    CHECK(harness.emitted(1, test.expect_ok));
    if (test.expect_ok) {
      CHECK(batch->per_key_results == test.expect_bits);
    } else {
      CHECK(batch->per_key_results == std::vector<uint8_t>({9, 9}));
    }
    CHECK(test.expect_log ? harness.driver.log_count(test.expect_log) == 1
                          : harness.driver.log_count("") == 0);
    std::printf("PASS fallback_taxonomy[%s]\n", test.name);
  }
}

// Strict-subset retry: one active slot, release-before-query order, one
// completion, and position-preserved bits from start_idx.
void test_retry_slot_and_release_order() {
  Harness harness;
  Script first = held_script("first");
  first.terminal_status = NIXL_ERR_NOT_FOUND;
  first.retry_script =
      std::make_shared<Script>(retry_script(NIXL_SUCCESS, {0, 1, 0}, "retry"));
  first.retry_script->state->release.store(false);  // pending until released
  harness.driver.set(1, first);
  auto batch = make_batch(Op::BATCH_TILE_GET, 4);
  batch->per_key_results = {9, 9, 9, 9};
  harness.push(make_request(1, Op::BATCH_TILE_GET, 3, batch, 1));
  harness.start();
  CHECK(wait_until([&] { return harness.driver.state(1)->posted.load(); }));
  harness.driver.state(1)->release.store(true);  // first transfer fails
  CHECK(
      wait_until([&] { return harness.driver.retry_state(1)->posted.load(); }));
  CHECK(harness.driver.state(1)->live.load() == 0);
  CHECK(harness.driver.trace_index("first:complete") <
        harness.driver.trace_index("fallback:first"));
  CHECK(harness.emit_count() == 0);  // the retry replaces the same slot
  harness.driver.retry_state(1)->release.store(true);
  CHECK(wait_until([&] { return harness.emit_count() == 1; }));
  harness.stop_and_join();
  CHECK(harness.emitted(1, true));
  CHECK(batch->per_key_results == std::vector<uint8_t>({9, 0, 1, 0}));
  CHECK(harness.driver.fallback_calls() == 1);  // no fallback loop
  std::printf("PASS retry_slot_and_release_order\n");
}

// Immediate first-attempt failures take the same release-first fallback:
// a thrown post, an immediate terminal post status, and stop during the
// failed post (no fallback, no completion).
void test_immediate_post_failures() {
  {
    Harness harness;
    Script first = held_script("first");
    first.throw_on_post = true;
    first.retry_script =
        std::make_shared<Script>(retry_script(NIXL_SUCCESS, {0, 1}, "retry"));
    harness.driver.set(1, first);
    auto batch = make_batch(Op::BATCH_TILE_GET, 2);
    batch->per_key_results = {9, 9};
    harness.push(make_request(1, Op::BATCH_TILE_GET, 2, batch, 0));
    harness.start();
    CHECK(wait_until([&] { return harness.emit_count() == 1; }));
    harness.stop_and_join();
    CHECK(harness.driver.state(1)->completed.load());  // released first
    CHECK(harness.driver.trace_index("first:complete") <
          harness.driver.trace_index("fallback:first"));
    CHECK(harness.emitted(1, true));
    CHECK(batch->per_key_results == std::vector<uint8_t>({0, 1}));
    CHECK(harness.driver.fallback_calls() == 1);
  }
  {
    Harness harness;
    Script first = done_script({}, "first");
    first.post_status = NIXL_ERR_BACKEND;  // immediate terminal failure
    first.fallback_mode = kFallbackSoft;
    first.fallback_soft_results = {0, 0};
    harness.driver.set(1, first);
    auto batch = make_batch(Op::BATCH_TILE_GET, 2);
    batch->per_key_results = {9, 9};
    harness.push(make_request(1, Op::BATCH_TILE_GET, 2, batch, 0));
    harness.start();
    CHECK(wait_until([&] { return harness.emit_count() == 1; }));
    harness.stop_and_join();
    CHECK(harness.driver.state(1)->completed.load());
    CHECK(harness.driver.fallback_calls() == 1);
    CHECK(batch->per_key_results == std::vector<uint8_t>({0, 0}));
    CHECK(harness.emitted(1, true));
  }
  {
    Harness harness;
    Script first = held_script("first");
    first.throw_on_post = true;
    first.stop_on_post = true;  // stop observed inside the failed post
    first.retry_script =
        std::make_shared<Script>(retry_script(NIXL_SUCCESS, {0}, "retry"));
    harness.driver.set(1, first);
    harness.push(make_request(1, Op::BATCH_TILE_GET, 1,
                              make_batch(Op::BATCH_TILE_GET, 1), 0));
    harness.start();
    harness.join();
    CHECK(harness.driver.state(1)->completed.load());
    CHECK(harness.driver.fallback_calls() == 0);  // no query after stop
    CHECK(harness.driver.retry_state(1)->post_calls.load() == 0);
    CHECK(harness.emit_count() == 0);
  }
  std::printf("PASS immediate_post_failures\n");
}

// Admission-time failures keep the taxonomy: a generic throw fails the tile,
// a load prepare error completes it with the carried false bits and a log.
void test_prepare_hard_vs_soft() {
  Harness harness;
  Script hard;
  hard.prepare_mode = 1;
  Script soft;
  soft.prepare_mode = 2;
  soft.soft_results = {0, 0};
  harness.driver.set(1, hard);
  harness.driver.set(2, soft);
  auto batch = make_batch(Op::BATCH_TILE_GET, 2);
  batch->per_key_results = {7, 7};
  auto batch2 = make_batch(Op::BATCH_TILE_GET, 2);
  batch2->per_key_results = {7, 7};
  harness.push(make_request(1, Op::BATCH_TILE_GET, 2, batch, 0));
  harness.push(make_request(2, Op::BATCH_TILE_GET, 2, batch2, 0));
  harness.start();
  CHECK(wait_until([&] { return harness.emit_count() == 2; }));
  harness.stop_and_join();
  CHECK(harness.emitted(1, false, "fake hard prepare failure"));
  CHECK(batch->per_key_results == std::vector<uint8_t>({7, 7}));
  CHECK(harness.emitted(2, true));
  CHECK(batch2->per_key_results == std::vector<uint8_t>({0, 0}));
  CHECK(harness.driver.log_count("fake load prepare failure") == 1);
  CHECK(harness.driver.fallback_calls() == 0);
  std::printf("PASS prepare_hard_vs_soft\n");
}

// Store submissions never take the load fallback: a failure is hard.
void test_store_submit_failure_hard() {
  Harness harness;
  Script store = done_script({}, "store");
  store.throw_on_post = true;
  harness.driver.set(1, store);
  harness.push(make_request(1, Op::BATCH_TILE_SET, 1,
                            make_batch(Op::BATCH_TILE_SET, 1), 0));
  harness.start();
  CHECK(wait_until([&] { return harness.emit_count() == 1; }));
  harness.stop_and_join();
  CHECK(harness.emitted(1, false, "fake submission failure"));
  CHECK(harness.driver.fallback_calls() == 0);
  std::printf("PASS store_submit_failure_hard\n");
}

// Stop halts admission (a tile queued after stop is never prepared), drains
// every posted tile to terminal, and suppresses all completions.
void test_stop_drains_without_completion() {
  Harness harness;
  harness.driver.set(1, held_script());
  harness.driver.set(2, held_script());
  harness.driver.set(3, held_script());
  harness.push(make_request(1, Op::BATCH_TILE_SET, 1,
                            make_batch(Op::BATCH_TILE_SET, 1), 0));
  harness.push(make_request(2, Op::BATCH_TILE_SET, 1,
                            make_batch(Op::BATCH_TILE_SET, 1), 0));
  harness.start();
  CHECK(wait_until([&] { return harness.driver.state(2)->posted.load(); }));
  harness.stop.store(true);
  harness.queue_cv.notify_all();
  harness.push(  // queued after stop: admission halted
      make_request(3, Op::BATCH_TILE_SET, 1, make_batch(Op::BATCH_TILE_SET, 1),
                   0));
  CHECK(wait_until([&] { return harness.driver.state(1)->polls.load() > 0; }));
  harness.driver.state(1)->release.store(true);
  harness.driver.state(2)->release.store(true);
  harness.join();
  CHECK(harness.emit_count() == 0);  // shutdown suppresses completions
  CHECK(harness.driver.state(1)->completed.load());
  CHECK(harness.driver.state(2)->completed.load());
  const auto order = harness.driver.prepare_order();
  CHECK(std::find(order.begin(), order.end(), 3) == order.end());
  std::printf("PASS stop_drains_without_completion\n");
}

// A stop observed after prepare but before post must not post or emit.
void test_stop_before_post() {
  Harness harness;
  Script script;
  script.stop_during_prepare = true;
  harness.driver.set(1, script);
  harness.push(make_request(1, Op::BATCH_TILE_SET, 1,
                            make_batch(Op::BATCH_TILE_SET, 1), 0));
  harness.start();
  harness.join();  // stop raised inside prepare skips the post
  CHECK(harness.driver.state(1)->post_calls.load() == 0);
  CHECK(harness.emit_count() == 0);
  CHECK(harness.driver.state(1)->live.load() == 0);
  std::printf("PASS stop_before_post\n");
}

// Stop observed while releasing a failed first transfer skips the fallback
// query and repost entirely.
void test_stop_before_fallback() {
  Harness harness;
  Script first = done_script({}, "first");
  first.terminal_status = NIXL_ERR_NOT_FOUND;
  first.stop_in_complete = true;  // stop begins during first-op release
  first.retry_script =
      std::make_shared<Script>(retry_script(NIXL_SUCCESS, {0}, "retry"));
  harness.driver.set(1, first);
  harness.push(make_request(1, Op::BATCH_TILE_GET, 1,
                            make_batch(Op::BATCH_TILE_GET, 1), 0));
  harness.start();
  harness.join();  // the run exits on its own after the stop is observed
  CHECK(harness.driver.state(1)->completed.load());
  CHECK(harness.driver.fallback_calls() == 0);
  CHECK(harness.driver.retry_state(1)->post_calls.load() == 0);
  CHECK(harness.emit_count() == 0);
  std::printf("PASS stop_before_fallback\n");
}

// An escaping exception (here a completion-emit failure) drains every
// still-posted tile to terminal before propagating, emitting nothing.
void test_exception_path_drains_posted_work() {
  Harness harness;
  harness.emit_fail_after = 0;
  Script survivor = held_script("survivor");
  survivor.hold_polls = 2;  // stays posted past the throwing emit
  harness.driver.set(1, survivor);
  harness.driver.set(2, done_script({}, "thrower"));
  harness.push(make_request(1, Op::BATCH_TILE_GET, 1,
                            make_batch(Op::BATCH_TILE_GET, 1), 0));
  harness.push(make_request(2, Op::BATCH_TILE_GET, 1,
                            make_batch(Op::BATCH_TILE_GET, 1), 0));
  harness.start();
  CHECK(wait_until([&] { return harness.run_threw.load(); }));
  harness.join();
  CHECK(harness.driver.state(2)->completed.load());  // op released pre-emit
  CHECK(harness.driver.state(1)->completed.load());  // drained in the catch
  CHECK(harness.driver.state(1)->live.load() == 0);
  CHECK(harness.driver.state(2)->live.load() == 0);
  CHECK(harness.emit_count() == 0);
  std::printf("PASS exception_path_drains_posted_work\n");
}

// An emit that throws while a successfully completed (submission-terminal)
// retry is being finalized must not re-enter the retry's preparation
// handlers after the tile node is unnoded: exactly one completion attempt,
// no fallback-path second completion, and both operations freed.
void test_emit_throw_after_retry_completion() {
  Harness harness;
  Script first = done_script({}, "first");
  first.terminal_status = NIXL_ERR_NOT_FOUND;
  Script retry = retry_script(NIXL_SUCCESS, {0, 1}, "retry");
  retry.post_status = NIXL_SUCCESS;  // completes without ever staying posted
  first.retry_script = std::make_shared<Script>(retry);
  harness.driver.set(1, first);
  harness.push(make_request(1, Op::BATCH_TILE_GET, 2,
                            make_batch(Op::BATCH_TILE_GET, 2), 0));
  harness.fail_next_emit();
  harness.start();
  CHECK(wait_until([&] { return harness.run_threw.load(); }));
  harness.join();
  CHECK(harness.emit_attempts() == 1);
  CHECK(harness.emit_count() == 0);
  CHECK(harness.driver.state(1)->live.load() == 0);
  CHECK(harness.driver.retry_state(1)->live.load() == 0);
  CHECK(harness.driver.fallback_calls() == 1);
  std::printf("PASS emit_throw_after_retry_completion\n");
}

// Strategies without the pending capability keep the serial loop; inside
// the scheduler a non-posting driver runs every tile inline, never posts,
// and never reaches the prepare seam.
void test_unsupported_capability_serial_selection() {
  NixlStorageCapabilities plain_capabilities;  // FILE-style default
  CHECK(!plain_capabilities.pending_transfers);

  Harness harness;
  for (uint64_t id = 1; id <= 4; ++id) {
    Script sync;
    sync.posts = false;
    harness.driver.set(id, sync);
    harness.push(make_request(id, Op::BATCH_TILE_EXISTS, 1,
                              make_batch(Op::BATCH_TILE_EXISTS, 1), 0));
  }
  harness.start();
  CHECK(wait_until([&] { return harness.emit_count() == 4; }));
  harness.stop_and_join();
  CHECK(harness.driver.prepare_order().empty());  // nothing was ever posted
  CHECK(harness.driver.sync_calls() == 4);
  for (uint64_t id = 1; id <= 4; ++id) CHECK(harness.emitted(id, true));
  std::printf("PASS unsupported_capability_serial_selection\n");
}

}  // namespace

int main() {
  test_overlap_admission_and_exact_once();
  test_fast_load_success();
  test_fallback_taxonomy();
  test_retry_slot_and_release_order();
  test_immediate_post_failures();
  test_prepare_hard_vs_soft();
  test_store_submit_failure_hard();
  test_stop_drains_without_completion();
  test_stop_before_post();
  test_stop_before_fallback();
  test_exception_path_drains_posted_work();
  test_emit_throw_after_retry_completion();
  test_unsupported_capability_serial_selection();
  std::printf("all reactor tests passed\n");
  return 0;
}
