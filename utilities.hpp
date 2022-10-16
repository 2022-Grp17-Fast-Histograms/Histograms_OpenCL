#pragma once

#include <chrono>
#include <string>

class TimeInterval
{
    public:
        TimeInterval()
            :   start(std::chrono::steady_clock::now())
            ,   type("seconds")
        {}
        TimeInterval(std::string type)
            :   start(std::chrono::steady_clock::now())
            ,   type(type)
        {}

        double Elapsed()
        {
            auto end = std::chrono::steady_clock::now();
            if (type == "milli") {
                return std::chrono::duration<double, std::milli>(end - start).count();
            }
            else if (type == "nano") {
                return std::chrono::duration<double, std::nano>(end - start).count();
            }
            else {
                return std::chrono::duration<double>(end - start).count();
            }
        }

    private:
        std::string type;
        std::chrono::steady_clock::time_point start;
};