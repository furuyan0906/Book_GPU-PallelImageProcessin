#ifndef  H__GRAPHICAL_RESOURCE_CONVERTER__H
#define  H__GRAPHICAL_RESOURCE_CONVERTER__H

#include <exception>
#include <sstream>
#include <cmath>
#include <cstdint>

namespace cpu
{
    class GraphicalResourceConverter
    {
        public:
            void ConvertToRGB(std::uint32_t* dst, float src, float src_max)
            {
                auto value = 255.0f - (src / src_max) * 255.0f;
                value = std::fmax(value, 0.0f);
                value = std::fmin(value, 255.0f);
            
                auto r = std::uint32_t(value);
                auto g = std::uint32_t(value);
                auto b = std::uint32_t(value);
                auto bgr = static_cast<std::uint32_t>((b << 16) | (g << 8) | (r << 0));
            
                *dst = bgr;
            }
    };
}

#endif  // H__GRAPHICAL_RESOURCE_CONVERTER__H

