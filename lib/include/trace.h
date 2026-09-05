#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace picceler {

struct TraceEvent {
  uint64_t _timestampNs;
  const char *_name;
  uint32_t _opIndex;
  uint16_t _trackId;
  uint8_t _phase;
  uint8_t _pad;
};

static_assert(sizeof(TraceEvent) == 24, "TraceEvent layout changed, .bin format depends on it");
static_assert(alignof(TraceEvent) == 8, "TraceEvent alignment changed, .bin format depends on it");

/**
 * @brief Collects profiling events emitted by instrumented picceler binaries.
 *
 * Only used when the program was compiled with --profile. The instance is
 * created on first event and deliberately never destroyed: flushing runs from
 * an atexit handler, and the ordering between a static destructor and an
 * atexit handler registered later is unspecified.
 */
class TraceSession {
public:
  static TraceSession &getInstance();

  void addBeginEvent(const char *name, uint32_t opIndex, uint16_t trackId = 0);
  void addEndEvent(const char *name, uint32_t opIndex, uint16_t trackId = 0);

  void flushEvents(const std::string &filename = "picceler_profiling_trace.bin");

  TraceSession(const TraceSession &) = delete;
  TraceSession &operator=(const TraceSession &) = delete;

private:
  TraceSession();
  ~TraceSession() = default;

  void record(const char *name, uint32_t opIndex, uint16_t trackId, uint8_t phase);

  std::vector<TraceEvent> _events;
};

} // namespace picceler

#ifdef __cplusplus
extern "C" {
#endif

/**
 * C entry points for profiling, hooked by picceler compiler.
 * \{
 */

void piccelerTraceBegin(const char *name, uint32_t opIndex, uint16_t trackId);
void piccelerTraceEnd(const char *name, uint32_t opIndex, uint16_t trackId);

/**
 * \}
 */

#ifdef __cplusplus
}
#endif