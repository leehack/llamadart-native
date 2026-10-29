#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

static void write_u16(std::ofstream &out, uint16_t value) {
    const char bytes[] = {static_cast<char>(value), static_cast<char>(value >> 8)};
    out.write(bytes, sizeof(bytes));
}

static void write_u32(std::ofstream &out, uint32_t value) {
    const char bytes[] = {
        static_cast<char>(value),
        static_cast<char>(value >> 8),
        static_cast<char>(value >> 16),
        static_cast<char>(value >> 24),
    };
    out.write(bytes, sizeof(bytes));
}

static bool write_wav(const std::string &path, const std::vector<float> &pcm, int32_t sample_rate) {
    if (pcm.size() > (std::numeric_limits<uint32_t>::max() - 44) / 2) {
        return false;
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    const uint32_t data_size = static_cast<uint32_t>(pcm.size() * 2);
    out.write("RIFF", 4);
    write_u32(out, 36 + data_size);
    out.write("WAVEfmt ", 8);
    write_u32(out, 16);
    write_u16(out, 1);
    write_u16(out, 1);
    write_u32(out, static_cast<uint32_t>(sample_rate));
    write_u32(out, static_cast<uint32_t>(sample_rate * 2));
    write_u16(out, 2);
    write_u16(out, 16);
    out.write("data", 4);
    write_u32(out, data_size);
    for (float sample : pcm) {
        const float clipped = std::max(-1.0f, std::min(1.0f, sample));
        const int16_t value = static_cast<int16_t>(std::lrintf(clipped * 32767.0f));
        write_u16(out, static_cast<uint16_t>(value));
    }
    return static_cast<bool>(out);
}
