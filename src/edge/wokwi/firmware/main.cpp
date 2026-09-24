#include <array>
#include <cstdint>
#include <string>

namespace mcu_sim {

struct TelemetryPacket {
    std::string node_id;
    std::uint32_t sequence_id;
    std::string timestamp;
    std::string state;
    float cpu_load;
    float memory_pressure;
    float network_quality;
    std::string participation_status;
    std::string source;
};

TelemetryPacket buildTelemetry(std::uint32_t sequence_id) {
    return {
        "Hospital-Edge-01",
        sequence_id,
        "2025-01-01T12:00:00.000Z",
        "AVAILABLE",
        28.0f,
        35.0f,
        92.0f,
        "READY",
        "mcu_simulation"
    };
}

}  // namespace mcu_sim

int main() {
    auto packet = mcu_sim::buildTelemetry(1u);
    (void)packet;
    return 0;
}
