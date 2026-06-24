// -----------------------------------------------------------------------------
// wal.cpp — Write-Ahead Log implementation
//
// On-disk format (per record, little-endian):
//   [crc32: 4B][payload_length: 4B][type: 1B][timestamp_us: 8B][payload: N B]
//
// CRC32 covers everything after the crc32 field:
//   payload_length + type + timestamp_us + payload
//
// LSN is implicit — it's the sequential index of the record in the file
// (first record = LSN 0, second = LSN 1, ...). On open, the file is scanned
// to find the last good record; any corrupt tail is truncated.
// -----------------------------------------------------------------------------
#include "wal.h"
#include "wal_record.h"
#include <cassert>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/uio.h>
#include <unistd.h>
#include <vector>

namespace vectordb {

// ---------------------------------------------------------------------------
// CRC32 (IEEE 802.3 / Ethernet polynomial, reflected form)
// ---------------------------------------------------------------------------

// Lookup table for CRC32 computation. Instead of doing polynomial division
// bit-by-bit for every byte of input, we precompute the CRC32 contribution
// of all 256 possible byte values (0x00–0xFF) at startup. Each byte of input
// then becomes a single array lookup + XOR instead of 8 bit operations.
// The table is 256 × 4 = 1024 bytes and is computed once on first use.
static uint32_t crc32_table[256];

static void init_crc32_table() {
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int j = 0; j < 8; ++j)
            c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        crc32_table[i] = c;
    }
}

// Returns CRC32 of `len` bytes starting at `data`.
static uint32_t crc32(const void* data, size_t len) {
    static bool inited = false;
    if (!inited) { init_crc32_table(); inited = true; }

    const uint8_t* p = reinterpret_cast<const uint8_t*>(data);
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i)
        c = crc32_table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

// CRC32 over the fields that follow crc32 in the header + the payload.
// The CRC field itself is excluded (it can't checksum itself).
static uint32_t record_crc(const WalRecordHeader& hdr,
                            const void* payload, uint32_t payload_len) {
    // Ensure table is initialized before use.
    static bool inited = false;
    if (!inited) { init_crc32_table(); inited = true; }

    // Feed the header fields after crc32: payload_length + type + timestamp_us
    const uint8_t* after_crc = reinterpret_cast<const uint8_t*>(&hdr)
                                + sizeof(hdr.crc32);
    size_t hdr_tail = sizeof(hdr) - sizeof(hdr.crc32);

    uint32_t c = 0xFFFFFFFFu;
    auto update = [&](const void* d, size_t n) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(d);
        for (size_t i = 0; i < n; ++i)
            c = crc32_table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    };
    update(after_crc, hdr_tail);
    update(payload, payload_len);
    return c ^ 0xFFFFFFFFu;
}

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct Wal::Impl {
    std::string path;
    int         fd      = -1;   // write fd (O_WRONLY | O_CREAT | O_APPEND)
    Lsn         next_lsn = 0;

    explicit Impl(const std::string& p) : path(p) {
        // Scan the existing file (if any) to find next_lsn and truncate any
        // corrupt tail record left by a previous crash.
        int rfd = ::open(path.c_str(), O_RDONLY);
        if (rfd >= 0) {
            off_t last_good = 0;  // file offset just after the last valid record
            while (true) {
                off_t record_start = ::lseek(rfd, 0, SEEK_CUR);

                WalRecordHeader hdr;
                ssize_t n = ::read(rfd, &hdr, sizeof(hdr));
                if (n == 0) break;                              // clean EOF
                if (n < static_cast<ssize_t>(sizeof(hdr))) break;  // truncated header

                std::vector<uint8_t> payload(hdr.payload_length);
                n = ::read(rfd, payload.data(), hdr.payload_length);
                if (n < static_cast<ssize_t>(hdr.payload_length)) break;  // truncated payload

                // Verify CRC — stop on mismatch (corrupt tail).
                uint32_t expected = record_crc(hdr, payload.data(), hdr.payload_length);
                if (hdr.crc32 != expected) break;

                ++next_lsn;
                last_good = ::lseek(rfd, 0, SEEK_CUR);
                (void)record_start;
            }
            ::close(rfd);

            // Truncate any incomplete tail record.
            if (::truncate(path.c_str(), last_good) != 0)
                throw std::runtime_error("Wal: truncate failed: " + path);
        }

        // Open for appending.
        fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (fd < 0)
            throw std::runtime_error("Wal: cannot open for writing: " + path);
    }

    ~Impl() {
        if (fd >= 0) ::close(fd);
    }

    static uint64_t now_us() {
        using namespace std::chrono;
        return static_cast<uint64_t>(
            duration_cast<microseconds>(
                system_clock::now().time_since_epoch()).count());
    }
};

// ---------------------------------------------------------------------------
// Wal public API
// ---------------------------------------------------------------------------

Wal::Wal(const std::string& path) : impl_(std::make_unique<Impl>(path)) {}
Wal::~Wal() = default;

// Appends one record and returns its assigned LSN.
// The record is written atomically (header + payload in one writev call)
// to avoid partial records on crash between the two writes.
Lsn Wal::append(WalRecordType type, const void* payload, uint32_t length) {
    auto& I = *impl_;

    WalRecordHeader hdr;
    hdr.payload_length = length;
    hdr.type           = type;
    hdr.timestamp_us   = Impl::now_us();
    hdr.crc32          = record_crc(hdr, payload, length);

    // Write header and payload together to minimize the crash window.
    struct iovec iov[2];
    iov[0].iov_base = &hdr;
    iov[0].iov_len  = sizeof(hdr);
    iov[1].iov_base = const_cast<void*>(payload);
    iov[1].iov_len  = length;

    ssize_t total = static_cast<ssize_t>(sizeof(hdr) + length);
    if (::writev(I.fd, iov, 2) != total)
        throw std::runtime_error("Wal: write failed");

    return I.next_lsn++;
}

// Calls fdatasync() to flush the write buffer to disk.
// On macOS, F_FULLFSYNC is used for stronger durability guarantees.
void Wal::sync() {
#ifdef __APPLE__
    if (::fcntl(impl_->fd, F_FULLFSYNC) != 0)
        throw std::runtime_error("Wal: fsync failed");
#else
    if (::fdatasync(impl_->fd) != 0)
        throw std::runtime_error("Wal: fdatasync failed");
#endif
}

// Iterates all records with LSN >= start_lsn, calling cb for each.
// Stops silently on a corrupt or truncated tail record.
void Wal::iterate(Lsn start_lsn,
                  std::function<void(Lsn, WalRecordType,
                                     const void*, uint32_t)> cb) const {
    int rfd = ::open(impl_->path.c_str(), O_RDONLY);
    if (rfd < 0) return;  // file doesn't exist yet — nothing to iterate

    Lsn lsn = 0;
    while (true) {
        WalRecordHeader hdr;
        ssize_t n = ::read(rfd, &hdr, sizeof(hdr));
        if (n == 0) break;
        if (n < static_cast<ssize_t>(sizeof(hdr))) break;

        std::vector<uint8_t> payload(hdr.payload_length);
        n = ::read(rfd, payload.data(), hdr.payload_length);
        if (n < static_cast<ssize_t>(hdr.payload_length)) break;

        uint32_t expected = record_crc(hdr, payload.data(), hdr.payload_length);
        if (hdr.crc32 != expected) break;  // corrupt tail — stop

        if (lsn >= start_lsn)
            cb(lsn, hdr.type, payload.data(), hdr.payload_length);
        ++lsn;
    }
    ::close(rfd);
}

// TODO Day 12: implement truncate_before (rewrite file keeping only records >= lsn).
void Wal::truncate_before(Lsn /*lsn*/) {}

Lsn Wal::current_lsn() const { return impl_->next_lsn; }

}  // namespace vectordb
