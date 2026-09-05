#include "trace.h"

#include "spdlog/spdlog.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace picceler {

namespace {

constexpr size_t initialEventCapacity = 1u << 20;

constexpr uint32_t traceMagic = 0x50494354; // "PICT"
constexpr uint32_t traceVersion = 1;

struct TraceFileHeader {
  uint32_t _magic;
  uint32_t _version;
  uint64_t _eventCount;
  uint64_t _eventSize;
};

uint64_t nowNs() {
  const auto since = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(since).count());
}

} // namespace

TraceSession::TraceSession() { _events.reserve(initialEventCapacity); }

TraceSession &TraceSession::getInstance() {
  static TraceSession *instance = [] {
    auto *session = new TraceSession();
    std::atexit([] { getInstance().flushEvents(); });
    return session;
  }();
  return *instance;
}

void TraceSession::record(const char *name, uint32_t opIndex, uint16_t trackId, uint8_t phase) {
  _events.push_back(TraceEvent{nowNs(), name, opIndex, trackId, phase, 0});
}

void TraceSession::addBeginEvent(const char *name, uint32_t opIndex, uint16_t trackId) {
  record(name, opIndex, trackId, 'B');
}

void TraceSession::addEndEvent(const char *name, uint32_t opIndex, uint16_t trackId) {
  record(name, opIndex, trackId, 'E');
}

void TraceSession::flushEvents(const std::string &filename) {
  if (_events.empty()) {
    return;
  }

  std::FILE *file = std::fopen(filename.c_str(), "wb");
  if (file == nullptr) {
    spdlog::error("Failed to open trace file for writing: {}", filename);
    return;
  }

  // Names are pointers into this process's rodata, so they cannot be written
  // as-is. Emit a string table first, then rewrite each event's pointer field
  // as an offset into that table.
  std::vector<const char *> uniqueNames;
  std::vector<uint64_t> nameOffsets;
  std::string stringTable;

  for (const TraceEvent &event : _events) {
    bool found = false;
    for (size_t i = 0; i < uniqueNames.size(); ++i) {
      if (uniqueNames[i] == event._name || std::string(uniqueNames[i]) == event._name) {
        found = true;
        break;
      }
    }
    if (!found) {
      uniqueNames.push_back(event._name);
      nameOffsets.push_back(stringTable.size());
      stringTable.append(event._name);
      stringTable.push_back('\0');
    }
  }

  const TraceFileHeader header{traceMagic, traceVersion, static_cast<uint64_t>(_events.size()),
                               static_cast<uint64_t>(sizeof(TraceEvent))};
  std::fwrite(&header, sizeof(header), 1, file);

  const uint64_t tableSize = stringTable.size();
  std::fwrite(&tableSize, sizeof(tableSize), 1, file);
  std::fwrite(stringTable.data(), 1, stringTable.size(), file);

  for (const TraceEvent &event : _events) {
    uint64_t offset = 0;
    for (size_t i = 0; i < uniqueNames.size(); ++i) {
      if (uniqueNames[i] == event._name || std::string(uniqueNames[i]) == event._name) {
        offset = nameOffsets[i];
        break;
      }
    }

    std::fwrite(&event._timestampNs, sizeof(event._timestampNs), 1, file);
    std::fwrite(&offset, sizeof(offset), 1, file);
    std::fwrite(&event._opIndex, sizeof(event._opIndex), 1, file);
    std::fwrite(&event._trackId, sizeof(event._trackId), 1, file);
    std::fwrite(&event._phase, sizeof(event._phase), 1, file);
    std::fwrite(&event._pad, sizeof(event._pad), 1, file);
  }

  std::fclose(file);
  spdlog::debug("Wrote {} trace events to {}", _events.size(), filename);
}

} // namespace picceler

extern "C" {

void piccelerTraceBegin(const char *name, uint32_t opIndex, uint16_t trackId) {
  picceler::TraceSession::getInstance().addBeginEvent(name, opIndex, trackId);
}

void piccelerTraceEnd(const char *name, uint32_t opIndex, uint16_t trackId) {
  picceler::TraceSession::getInstance().addEndEvent(name, opIndex, trackId);
}

} // extern "C"