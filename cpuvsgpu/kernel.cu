#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>

#include "stb_image.h"
#include "stb_image_write.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>


#define BLOCK_SIZE 32
#define WINDOW_SIZE 3

#define R_INDEX 0
#define G_INDEX 1
#define B_INDEX 2


__device__ int median_calc(int arr[], int n, int k);
__global__ void median_filter_rgb(unsigned char* input_image, unsigned char* output_image, int width, int height);

void medianfilter_cpu(unsigned char* inputImage, unsigned char* outputImage, int width, int height);
void medianfilter_cpu_opencv(const std::string& filename);
void medianfilter_gpu_opencv(const std::string& filename, int width, int height);


int main()
{
	const std::string input_filename = "C:/Dechard/Görüntüler/yuksekpixel.jpg";
	const std::string gpu_output_filename = "C:/Dechard/gpu_median.jpg";
	const std::string cpu_output_filename = "C:/Dechard/cpu_median.jpg";
	int width, height, num_channels;

	unsigned char* input_image = stbi_load(input_filename.c_str(), &width, &height, &num_channels, 3);
	unsigned char* gpu_output_image = new unsigned char[width * height * 3];
	unsigned char* cpu_output_image = new unsigned char[width * height * 3];

	if (input_image == nullptr)
	{
		std::cerr << "Hata: Yuklenemedi" << std::endl;
	}


	//opencv_gpu
	auto start_time_gpu_opencv = std::chrono::high_resolution_clock::now();
	medianfilter_gpu_opencv(input_filename, width, height);
	auto end_time_gpu_opencv = std::chrono::high_resolution_clock::now();

	//opencv_cpu

	auto start_time_cpu_opencv = std::chrono::high_resolution_clock::now();
	medianfilter_cpu_opencv(input_filename);
	auto end_time_cpu_opencv = std::chrono::high_resolution_clock::now();

	//CPU

	auto start_time_cpu = std::chrono::high_resolution_clock::now();
	medianfilter_cpu(input_image, cpu_output_image, width, height);
	auto end_time_cpu = std::chrono::high_resolution_clock::now();

	// GPU
	unsigned char* median_input_image;
	unsigned char* median_output_image;

	cudaMalloc(&median_input_image, width * height * 3);
	cudaMalloc(&median_output_image, width * height * 3);

	dim3 block_size(BLOCK_SIZE * BLOCK_SIZE);
	dim3 grid_size((width * block_size.x - 1) / block_size.x, (height * block_size.y - 1) / block_size.y);

	cudaMemcpy(median_input_image, input_image, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

	auto start_time_gpu = std::chrono::high_resolution_clock::now();
	median_filter_rgb <<<grid_size, block_size >>> (median_input_image, median_output_image, width, height);
	cudaDeviceSynchronize();
	cudaMemcpy(gpu_output_image, median_output_image, width * height * 3, cudaMemcpyDeviceToHost);
	auto end_time_gpu = std::chrono::high_resolution_clock::now();
	

	std::cout << "GPU Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time_gpu - start_time_gpu).count() << " ms" << std::endl;
	std::cout << "CPU Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time_cpu - start_time_cpu).count() << " ms" << std::endl;
	std::cout << "OpenCV-CPU Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time_cpu_opencv - start_time_cpu_opencv).count() << " ms" << std::endl;
	std::cout << "OpenCV-GPU Execution Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time_gpu_opencv - start_time_gpu_opencv).count() << " ms" << std::endl;

	stbi_write_jpg(gpu_output_filename.c_str(), width, height, 3, gpu_output_image, 100);
	stbi_write_jpg(cpu_output_filename.c_str(), width, height, 3, cpu_output_image, 100);

	cudaFree(median_input_image);
	cudaFree(median_output_image);

	delete[] input_image;
	delete[] gpu_output_image;
	delete[] cpu_output_image;


	return EXIT_SUCCESS;
}

__global__ void median_filter_rgb(unsigned char* input_image, unsigned char* output_image, int width, int height)
{
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (col >= width || row >= height)
	{
		return;
	}

	int red[WINDOW_SIZE * WINDOW_SIZE];
	int green[WINDOW_SIZE * WINDOW_SIZE];
	int blue[WINDOW_SIZE * WINDOW_SIZE];

	int index = 0;

	for (int i = 0; i < WINDOW_SIZE; i++)
	{
		for (int j = 0; j < WINDOW_SIZE; j++)
		{

			int current_col = col - (WINDOW_SIZE / 2) + i;
			int current_row = row - (WINDOW_SIZE / 2) + j;

			if (current_row >= 0 && current_row < height && current_col >= 0 && current_col < width)
			{
				int current_index = ((current_row * width) + current_col) * 3;
				red[index] = input_image[current_index + R_INDEX];
				green[index] = input_image[current_index + G_INDEX];
				blue[index] = input_image[current_index + B_INDEX];

				index++;
			}
		}
	}


	int half_Size = WINDOW_SIZE * WINDOW_SIZE / 2;
	if (index > half_Size)
	{
		int red_median = median_calc(red, index, half_Size);
		int green_median = median_calc(green, index, half_Size);
		int blue_median = median_calc(blue, index, half_Size);

		output_image[(row * width + col) * 3 + R_INDEX] = red_median;
		output_image[(row * width + col) * 3 + G_INDEX] = green_median;
		output_image[(row * width + col) * 3 + B_INDEX] = blue_median;

	}


}
__device__ int median_calc(int arr[], int n, int k)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = i + 1; j < n; j++)
		{
			if (arr[i] > arr[j])
			{
				int temp = arr[i];
				arr[i] = arr[j];
				arr[j] = temp;
			}
		}
	}
	return arr[n / 2];
}

void medianfilter_cpu(unsigned char* inputImage, unsigned char* outputImage, int width, int height)
{
	int half_window_size = WINDOW_SIZE / 2;
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			std::vector<int> red_values;
			std::vector<int> green_values;
			std::vector<int> blue_values;

			for (int i = -half_window_size; i <= half_window_size; i++)
			{
				for (int j = -half_window_size; j <= half_window_size; j++)
				{
					int current_row = row + i;
					int current_col = col + j;
					if (current_row >= 0 && current_row < height && current_col >= 0 && current_col < width)
					{
						int current_index = (current_row * width + current_col) * 3;
						red_values.push_back(inputImage[current_index]);
						green_values.push_back(inputImage[current_index + 1]);
						blue_values.push_back(inputImage[current_index + 2]);
					}
				}
			}
			// Sort the values
			std::sort(red_values.begin(), red_values.end());
			std::sort(green_values.begin(), green_values.end());
			std::sort(blue_values.begin(), blue_values.end());

			// Get the median value
			int offset = red_values.size() / 2;
			outputImage[(row * width + col) * 3] = red_values[offset];
			outputImage[(row * width + col) * 3 + 1] = green_values[offset];
			outputImage[(row * width + col) * 3 + 2] = blue_values[offset];
		}
	}
}

void medianfilter_cpu_opencv(const std::string& filename)
{
	cv::Mat image = cv::imread(filename);
	cv::Mat filtered_image;
	cv::medianBlur(image, filtered_image, WINDOW_SIZE);

	cv::imwrite("C:/Dechard/opencv_cpu_median.jpg", filtered_image);

}

void medianfilter_gpu_opencv(const std::string& filename, int width, int height)
{
	cv::Mat inputImage = cv::imread(filename);
	cv::cuda::GpuMat gpu_input_image(inputImage);
	cv::cuda::GpuMat gpu_output_image;

	//cuda version

	cv::Ptr<cv::cuda::Filter> median_filter = cv::cuda::createMedianFilter(gpu_input_image.type(), WINDOW_SIZE);
	median_filter->apply(gpu_input_image, gpu_output_image);

	cv::Mat gpu_result;
	gpu_output_image.download(gpu_result);

	cv::imwrite("C:/Dechard/opencv_gpu_median.jpg", gpu_result);
}
