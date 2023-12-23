#ifndef  H__BINARY_LOG__H
#define  H__BINARY_LOG__H


#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <exception>
#include <cstdint>

template<typename T>
void WriteIntoBinaryFile(const std::string& fileNamebase, T ptr, std::size_t size)
{
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << fileNamebase << "_" << std::put_time(localtime(&time), "%Y%m%d-%H%M%S") << ".bin";

    auto fileName = ss.str();

    std::ofstream ofs;
    ofs.open(fileName, std::ios::out | std::ios::binary);

    for (std::size_t i = 0; i < size; ++i)
    {
        auto data = ptr[i];
        ofs.write(reinterpret_cast<const char*>(&data), sizeof(data));
    }

    ofs.close();

    if (ofs.fail())
    {
        throw std::runtime_error("Fail to create binary file.");
    }
}


#endif  // H__BINARY_LOG__H

