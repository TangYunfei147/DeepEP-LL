// Appended after cb59_stream_kernel_operator_mock.h in the temporary include.
#include <atomic>
namespace AscendC {
enum PipeKind { PIPE_MTE3 };
template<PipeKind> inline void PipeBarrier() {}
template<class T> struct DataCopyPadExtParams { bool enabled; uint32_t left, right; T value; };
inline std::atomic<uint64_t> pairPrefixLoads{0};
template<class T> inline void DataCopy(GlobalTensor<T> target, LocalTensor<T> source, uint32_t elements) {
    assert(elements * sizeof(T) % 32U == 0U);
    DataCopy(target, source, DataCopyParams{1U, uint32_t(elements * sizeof(T) / 32U), 0U, 0U});
}
template<class T> inline void DataCopy(LocalTensor<T> target, GlobalTensor<T> source, uint32_t elements) {
    assert(elements * sizeof(T) % 32U == 0U);
    DataCopy(target, source, DataCopyParams{1U, uint32_t(elements * sizeof(T) / 32U), 0U, 0U});
}
template<class T> inline void DataCopyPad(LocalTensor<T> target, GlobalTensor<T> source,
                                         DataCopyExtParams p, DataCopyPadExtParams<T>) {
    assert(reinterpret_cast<uintptr_t>(source.pointer) % 32U == 0U);
    assert(p.blockCount == 1U && p.blockLen <= target.elements * sizeof(T));
    std::lock_guard<std::mutex> lock(gmMutex);
    std::memcpy(target.pointer, source.pointer, p.blockLen);
    ++pairPrefixLoads;
}
} // namespace AscendC
