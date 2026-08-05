#include <iostream>
#include <fstream>

int main(int argc, char *argv[]) {

    std::ofstream outf{ "OUTPUT_SAMPLE.csv" };

    if (!outf) {
        std::cerr << "Error opening file" << std::endl;
        return 1;
    }

    outf << "line 1\n";
    outf << "line 2";
    outf << " still line 2";

    outf.close();

    return 0;
}
