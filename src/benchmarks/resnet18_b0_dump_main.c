#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "resnet18_graph.h"
#include "resnet18_input.h"

void resnet18_run(void);
void resnet18_load_input(const int8_t *input, uint32_t elements);
void resnet18_copy_tensor(uint32_t tensor_id, int8_t *output, uint32_t elements);

int main(int argc, char **argv)
{
    uint32_t tensor_id;
    char path[256];

    if (argc != 2)
        return 2;
    resnet18_load_input(resnet18_input, sizeof(resnet18_input));
    resnet18_run();
    for (tensor_id = 0; tensor_id < resnet18_tensor_count; ++tensor_id) {
        FILE *file;
        int8_t *data;
        if (resnet18_tensors[tensor_id].constant_data)
            continue;
        if (snprintf(path, sizeof(path), "%s/tensor_%u.bin", argv[1],
                     tensor_id) >= (int)sizeof(path))
            return 3;
        file = fopen(path, "wb");
        if (!file)
            return 4;
        data = malloc(resnet18_tensors[tensor_id].elements);
        if (!data)
            return 5;
        resnet18_copy_tensor(tensor_id, data, resnet18_tensors[tensor_id].elements);
        if (fwrite(data, 1, resnet18_tensors[tensor_id].elements, file) !=
            resnet18_tensors[tensor_id].elements) {
            free(data);
            fclose(file);
            return 6;
        }
        free(data);
        fclose(file);
    }
    return 0;
}
