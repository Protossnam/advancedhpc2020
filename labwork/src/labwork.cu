#include <stdio.h>
#include <include/labwork.h>
#include <cuda_runtime_api.h>
#include <omp.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
        case 1:
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork1_OpenMP();
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
            labwork.labwork3_GPU();
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
            break;
        case 4:
            labwork.labwork4_GPU();
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
            break;
        case 5:
            labwork.labwork5_CPU();
            labwork.saveOutputImage("labwork5-cpu-out.jpg");
            labwork.labwork5_GPU(FALSE);
            labwork.saveOutputImage("labwork5-gpu-out.jpg");
            break;
        case 6:
            labwork.labwork6_GPU();
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
            break;
        case 7:
            labwork.labwork7_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
            printf("[ALGO ONLY] labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
    printf("labwork %d ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::saveOutputImage(std::string outputFileName) {
    if (outputFileName.find("labwork6") != std::string::npos) {
        jpegLoader.save("labwork6-gpu-binarization-out.jpg", outputImage, inputImage->width, inputImage->height, 90);
        jpegLoader.save("labwork6-gpu-brightness-out.jpg", outputImage1, inputImage->width, inputImage->height, 90);
        jpegLoader.save("labwork6-gpu-blending-out.jpg", outputImage2, inputImage->width, inputImage->height, 90);
    } else {
        jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
    }
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<unsigned char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (unsigned char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<unsigned char *>(malloc(pixelCount * 3));
    // do something here
        
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        # pragma omp parallel for
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (unsigned char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {
    int nDevices = 0;
    // get all devices
    cudaGetDeviceCount(&nDevices);
    printf("Number total of GPU : %d\n\n", nDevices);
    for (int i = 0; i < nDevices; i++){
        // get informations from individual device
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        // something more here
        printf("GPU info\n");
        printf("  Name:                    %s\n", prop.name);  
        printf("  Clock rate:              %d\n", prop.clockRate);
        printf("  CUDA cores:              %d\n", getSPcores(prop));
        printf("  Multiprocessors:         %d\n", prop.multiProcessorCount);
        printf("  Warp size:               %d\n", prop.warpSize);
        printf("  Memory Clock rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus width (bits): %d\n", prop.memoryBusWidth); 
    }

}

__global__ void grayscale(uchar3 *input, uchar3 *output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    output[tid].x = (unsigned char)(((int)input[tid].x + (int)input[tid].y + (int)input[tid].z) / 3);
    output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork3_GPU() {
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;
    unsigned char *hostInput = inputImage->buffer;
    outputImage = static_cast<unsigned char *>(malloc(pixelCount * 3));

    // Allocate CUDA memory    
    uchar3 *devInput;
    uchar3 *devOutput;
    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devOutput, pixelCount * 3);
    
    // Copy CUDA Memory from CPU to GPU
    cudaMemcpy(devInput, hostInput, pixelCount * 3, cudaMemcpyHostToDevice);

    // Processing
    int blockSize = 64;
    int numBlock = pixelCount / blockSize + 1;
    grayscale<<<numBlock, blockSize>>>(devInput, devOutput);

    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    // Cleaning
    free(hostInput);
    cudaFree(devInput);
    cudaFree(devOutput);
}

__global__ void grayscale_2d(uchar3 *input, uchar3 *output, int img_width, int img_height) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if (col >= img_width || row >= img_height) return;
    int tid = row * img_width + col;
    output[tid].x = (unsigned char)(((int)input[tid].x + (int)input[tid].y + (int)input[tid].z) / 3);
    output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork4_GPU() {
    // Calculate number of pixels
    int pixelCount = inputImage->width * inputImage->height;
    unsigned char *hostInput = inputImage->buffer;
    outputImage = static_cast<unsigned char *>(malloc(pixelCount * 3));

    // Allocate CUDA memory    
    uchar3 *devInput;
    uchar3 *devOutput;
    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devOutput, pixelCount * 3);
    
    // Copy CUDA Memory from CPU to GPU
    cudaMemcpy(devInput, hostInput, pixelCount * 3, cudaMemcpyHostToDevice);

    // Processing
    dim3 blockSize = dim3(32,32);
    dim3 gridSize = dim3((int)((inputImage->width + blockSize.x - 1) / blockSize.x), (int)((inputImage->height + blockSize.y - 1)/ blockSize.y));
    grayscale_2d<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height);

    // Copy CUDA Memory from GPU to CPU
    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    // Cleaning
    free(hostInput);
    cudaFree(devInput);
    cudaFree(devOutput);
}

void Labwork::labwork5_CPU() {
    int w = inputImage->width;
    int h = inputImage->height;

    labwork1_CPU();
    unsigned char *grayImage = outputImage;
    int pixelCount = w * h;
    outputImage = static_cast<unsigned char *>(malloc(pixelCount * 3));   
    memset(outputImage, 0, pixelCount * 3);

    float filter[]={
        0,   0,   1,   2,   1,   0,   0,
        0,   3,   13,  22,  13,  3,   0,
        1,   13,  59,  97,  59,  13,  1,
        2,   22,  97,  159, 97,  22,  2,
        1,   13,  59,  97,  59,  13,  1,
        0,   3,   13,  22,  13,  3,   0,
        0,   0,   1,   2,   1,   0,   0
    };

    for (int row = 3; row <= h - 3; ++row) {
        for (int col = 3; col <= w - 3; ++col) {
            int sum = 0;
            for (int j = -3; j <= 3; ++j) { 
                for (int i = -3; i <= 3; ++i) {
                    sum += grayImage[((row + j) * w + col + i) * 3] * filter[(j+3)*7+i+3];
                } 
            }
            sum /= 1003;
            outputImage[(row * w + col)*3] = sum;
            outputImage[(row * w + col)*3 + 1] = sum;
            outputImage[(row * w + col)*3 + 2] = sum;
        }
    }
}

__global__ void gaussian_no_shared(uchar3 *input, uchar3 *output, int img_width, int img_height) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    int w = img_width;
    int h = img_height;

    if (col >= w || row >= h) return;
    int tid = row * w + col;

    float filter[]={
        0,   0,   1,   2,   1,   0,   0,
        0,   3,   13,  22,  13,  3,   0,
        1,   13,  59,  97,  59,  13,  1,
        2,   22,  97,  159, 97,  22,  2,
        1,   13,  59,  97,  59,  13,  1,
        0,   3,   13,  22,  13,  3,   0,
        0,   0,   1,   2,   1,   0,   0
    };

    int sum = 0;
    for (int j = -3; j <= 3; ++j) { 
        for (int i = -3; i <= 3; ++i) {
            sum += input[(tid + j*w + i)].x * filter[(j+3)*7+i+3];
        } 
    }

    sum /= 1003;
    output[tid].x = output[tid].y = output[tid].z = sum;
}

__global__ void gaussian_shared(uchar3 *input, uchar3 *output, int img_width, int img_height) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    int w = img_width;
    int h = img_height;

    if (col >= w || row >= h) return;
    int tid = row * w + col;

    float filter[]={
        0,   0,   1,   2,   1,   0,   0,
        0,   3,   13,  22,  13,  3,   0,
        1,   13,  59,  97,  59,  13,  1,
        2,   22,  97,  159, 97,  22,  2,
        1,   13,  59,  97,  59,  13,  1,
        0,   3,   13,  22,  13,  3,   0,
        0,   0,   1,   2,   1,   0,   0
    };

    __shared__ float smem[7][7];

    smem[threadIdx.x][threadIdx.y] = filter[tid];

    __syncthreads();
    
    int sum = 0;
    for (int j = -3; j <= 3; ++j) { 
        for (int i = -3; i <= 3; ++i) {
            sum += input[(tid + j*w + i)].x * smem[threadIdx.x + i][threadIdx.y + j];
        } 
    }

    sum /= 1003;
    output[tid].x = output[tid].y = output[tid].z = sum;
}

void Labwork::labwork5_GPU(bool shared) {
    int w = inputImage->width;
    int h = inputImage->height;

    labwork4_GPU();
    unsigned char *grayImage = outputImage;
    int pixelCount = w * h;
    outputImage = static_cast<unsigned char *>(malloc(pixelCount * 3));   
    memset(outputImage, 0, pixelCount * 3);

    uchar3 *devInput;
    uchar3 *devOutput;
    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devOutput, pixelCount * 3);

    cudaMemcpy(devInput, grayImage, pixelCount * 3, cudaMemcpyHostToDevice);

    dim3 blockSize = dim3(7,7);
    dim3 gridSize = dim3((int)((w + blockSize.x - 1) / blockSize.x), (int)((h + blockSize.y - 1)/ blockSize.y));;
    
    if (shared == FALSE) {
        gaussian_no_shared<<<gridSize, blockSize>>>(devInput, devOutput, w, h);
    } else {
        gaussian_shared<<<gridSize, blockSize>>>(devInput, devOutput, w, h);
    }

    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    free(grayImage);
    cudaFree(devInput);
    cudaFree(devOutput);
}

__device__ int binarization(int input, int threshold) {
    if (input < threshold) 
        return 0; 
    else
        return 255;
}

__device__ int brightness(int input, int threshold) {
    int output = input + threshold;
    if (output < 0) output = 0;
    if (output > 255) output = 255;
    return output;
}

__device__ int blending(int input1, int input2, float coeff) {
    int output = (int)(coeff*input1 + (1-coeff)*input2);
    if (output > 255) output = 255;
    return output;
}

__global__ void six_a(uchar3 *input, uchar3 *output, int img_width, int img_height) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    int w = img_width;
    int h = img_height;

    if (col >= w || row >= h) return;
    int tid = row * w + col;

    int threshold = 127;

    output[tid].x = binarization(input[tid].x, threshold);
    output[tid].z = output[tid].y = output[tid].x;
}

__global__ void six_b(uchar3 *input, uchar3 *output, int img_width, int img_height) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    int w = img_width;
    int h = img_height;

    if (col >= w || row >= h) return;
    int tid = row * w + col;

    int threshold = 20;

    output[tid].x = brightness(input[tid].x, threshold);
    output[tid].y = brightness(input[tid].y, threshold);
    output[tid].z = brightness(input[tid].z, threshold);
}

__global__ void six_c(uchar3 *input1, uchar3 *input2, uchar3 *output, int img_width, int img_height) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    int w = img_width;
    int h = img_height;

    if (col >= w || row >= h) return;
    int tid = row * w + col;

    float coeff = 0.5;

    output[tid].x = blending(input1[tid].x, input2[tid].x, coeff);
    output[tid].y = blending(input1[tid].y, input2[tid].y, coeff);
    output[tid].z = blending(input1[tid].z, input2[tid].z, coeff);
}

void Labwork::labwork6_GPU() {
    int w = inputImage->width;
    int h = inputImage->height;
    int pixelCount = w * h;

    // Initialization
    uchar3 *devInput, *devInput1, *devInput2, *devInput3;
    uchar3 *devOutput, *devOutput1, *devOutput2;
    cudaMalloc(&devInput, pixelCount * 3);
    cudaMalloc(&devInput1, pixelCount * 3);
    cudaMalloc(&devInput2, pixelCount * 3);
    cudaMalloc(&devInput3, pixelCount * 3);
    cudaMalloc(&devOutput, pixelCount * 3);
    cudaMalloc(&devOutput1, pixelCount * 3);
    cudaMalloc(&devOutput2, pixelCount * 3);

    dim3 blockSize = dim3(32,32);
    dim3 gridSize = dim3((int)((w + blockSize.x - 1) / blockSize.x), (int)((h + blockSize.y - 1)/ blockSize.y));

    // Part a
    labwork4_GPU();
    unsigned char *grayImage = outputImage;
    outputImage = static_cast<unsigned char *>(malloc(pixelCount * 3));
    memset(outputImage, 0, pixelCount * 3);
    cudaMemcpy(devInput, grayImage, pixelCount * 3, cudaMemcpyHostToDevice);

    six_a<<<gridSize, blockSize>>>(devInput, devOutput, w, h);
    cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    // Part b
    outputImage1 = static_cast<unsigned char *>(malloc(pixelCount * 3));
    memset(outputImage1, 0, pixelCount * 3);
    cudaMemcpy(devInput1, inputImage, pixelCount * 3, cudaMemcpyHostToDevice);

    six_b<<<gridSize, blockSize>>>(devInput1, devOutput1, w, h);
    cudaMemcpy(outputImage1, devOutput1, pixelCount * 3, cudaMemcpyDeviceToHost);

    // Part c
    outputImage2 = static_cast<unsigned char *>(malloc(pixelCount * 3));
    memset(outputImage2, 0, pixelCount * 3);
    cudaMemcpy(devInput2, inputImage, pixelCount * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(devInput3, inputImage, pixelCount * 3, cudaMemcpyHostToDevice);

    six_c<<<gridSize, blockSize>>>(devInput2, devInput3, devOutput2, w, h);
    cudaMemcpy(outputImage2, devOutput2, pixelCount * 3, cudaMemcpyDeviceToHost);

    free(grayImage);
    cudaFree(devInput);
    cudaFree(devInput1);
    cudaFree(devInput2);
    cudaFree(devInput3);
    cudaFree(devOutput);
    cudaFree(devOutput1);
    cudaFree(devOutput2);
}

void Labwork::labwork7_GPU() {
}

void Labwork::labwork8_GPU() {
}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU(){
}


























