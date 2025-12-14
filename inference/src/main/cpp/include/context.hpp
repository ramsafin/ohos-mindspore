#ifndef APP_CONTEXT_HPP
#define APP_CONTEXT_HPP

#include <mindspore/context.h> // OH_AI_ContextHandle

#include "backend.hpp" // Backend

namespace inference {

struct MindSporeContext final {
public:    
    MindSporeContext();
    ~MindSporeContext();
    
    // API
    void EnableBackend(Backend backend);
    bool IsCreated() const;
    
    // no copying
    MindSporeContext(const MindSporeContext&) = delete;
    MindSporeContext& operator=(const MindSporeContext&) = delete;

private:
    OH_AI_ContextHandle ctx{nullptr};
};

} // namespace inference

#endif //APP_CONTEXT_HPP
