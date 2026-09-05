#include "trace.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>

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

template <typename T> void writeRaw(std::ofstream &file, const T &value) {
  file.write(reinterpret_cast<const char *>(&value), sizeof(value));
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

  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open trace file for writing: " << filename << "\n";
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
  writeRaw(file, header);

  const uint64_t tableSize = stringTable.size();
  writeRaw(file, tableSize);
  file.write(stringTable.data(), static_cast<std::streamsize>(stringTable.size()));

  for (const TraceEvent &event : _events) {
    uint64_t offset = 0;
    for (size_t i = 0; i < uniqueNames.size(); ++i) {
      if (uniqueNames[i] == event._name || std::string(uniqueNames[i]) == event._name) {
        offset = nameOffsets[i];
        break;
      }
    }

    writeRaw(file, event._timestampNs);
    writeRaw(file, offset);
    writeRaw(file, event._opIndex);
    writeRaw(file, event._trackId);
    writeRaw(file, event._phase);
    writeRaw(file, event._pad);
  }

  std::cerr << "Wrote " << _events.size() << " trace events to " << filename << "\n";
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
