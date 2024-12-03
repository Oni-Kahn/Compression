//this file is the code that actually compresses using Discrete Cosine Transform (DCT)

//inclusions for the math processing
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
cout << "You have reached checkpoint 1"

//include necessary headers for image processing
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;
cout << "You have reached checkpoint 2"

//function to preform 2D DCT
vector<vector<double>> dctTransform(const vector<vector<double>>& matrix)
{
  int height = matrix.size(); //get height of the input matrix
  int width = matrix[0].size(); //get width of the input matrix
  vector<vector<double>> dct(height, vector<double>(width, 0.0)); //create 2D vector to store DCT coeff

  for (int u = 0; u < height; ++u)
  {
    for (int v = 0; v < width; ++v)
    {
      double cu = (u == 0) ? 1.0 / sqrt(2.0) : 1.0; //calc the normalization factor for height
      double cv = (v == 0) ? 1.0 / sqrt(2.0) : 1.0; //calc the normalization factor for width
      double sum = 0.0; //initialize the sum for DCT coeff

      for (int i = 0; i < height; ++i)
      {
        for (int j = 0; j < width; ++j) //calc DCT coeff for frequency (u,v)
        {
          sum += matrix[i][j] * cos((2 * i + 1) * u * M_PI / (2.0 * height)) * cos((2 * j + 1) * v * M_PI / (2.0 * width));
        }
      }
      dct[u][v] = (2.0 / sqrt(height * width)) * cu * cv * sum; //store calc DCT coeff
    }
  }
  return dct; //return the 2D vector of DCT coeffs
}
cout << "You have reached checkpoint 3"
// Function to perform inverse 2D DCT (IDCT)
vector<vector<double>> idctTransform(const vector<vector<double>>& dct) 
{
  int height = dct.size(); // Get the height of the DCT coefficient matrix
  int width = dct[0].size(); // Get the width of the DCT coefficient matrix
  vector<vector<double>> matrix(height, vector<double>(width, 0.0)); // Create a 2D vector to store the reconstructed image data

  for (int i = 0; i < height; ++i) 
  {
    for (int j = 0; j < width; ++j) 
    {
      double sum = 0.0; // Initialize the sum for the pixel value
      for (int u = 0; u < height; ++u) 
      {
        for (int v = 0; v < width; ++v) 
        {
          double cu = (u == 0) ? 1.0 / sqrt(2.0) : 1.0; // Calculate the normalization factor for u
          double cv = (v == 0) ? 1.0 / sqrt(2.0) : 1.0; // Calculate the normalization factor for v
          // Calculate the pixel value at (i, j)
          sum += cu * cv * dct[u][v] * cos((2 * i + 1) * u * M_PI / (2.0 * height)) * cos((2 * j + 1) * v * M_PI / (2.0 * width));
        }
      }
      matrix[i][j] = (2.0 / sqrt(height * width)) * sum; // Store the calculated pixel value
    }
  }
  return matrix; // Return the 2D vector of reconstructed image data
}
cout << "You have reached checkpoint 4"
// Function to quantize the DCT coefficients
vector<vector<double>> quantizeDCT(const vector<vector<double>>& dct, int quality) 
{
  // Define a quantization matrix (you can adjust these values)
  vector<vector<int>> quantizationMatrix = 
  {
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}
  };

  // Adjust quantization matrix based on quality (lower quality = more compression)
  double scalingFactor = (quality < 50) ? (5000.0 / quality) : (200.0 - 2 * quality);
  for (auto& row : quantizationMatrix) 
  {
    for (auto& val : row) 
    {
      val = static_cast<int>(val * scalingFactor / 100.0);
      if (val < 1) val = 1; // Prevent divide by zero
    }
  }

  int height = dct.size(); // Get the height of the DCT coefficient matrix
  int width = dct[0].size(); // Get the width of the DCT coefficient matrix
  vector<vector<double>> quantizedDCT(height, vector<double>(width, 0.0)); // Create a 2D vector to store the quantized DCT coefficients

  for (int u = 0; u < height; ++u) 
  {
    for (int v = 0; v < width; ++v) 
    {
      // Quantize the DCT coefficient by dividing it by the corresponding value in the quantization matrix
      quantizedDCT[u][v] = round(dct[u][v] / quantizationMatrix[u][v]); 
    }
  }

  return quantizedDCT; // Return the 2D vector of quantized DCT coefficients
}
cout << "You have reached checkpoint 5"
// Function to dequantize the DCT coefficients
vector<vector<double>> dequantizeDCT(const vector<vector<double>>& quantizedDCT, int quality) 
{
  // Define a quantization matrix (you can adjust these values) - Same as in quantizeDCT()
  vector<vector<int>> quantizationMatrix = 
  {
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}
  };

  // Adjust quantization matrix based on quality (lower quality = more compression) - Same as in quantizeDCT()
  double scalingFactor = (quality < 50) ? (5000.0 / quality) : (200.0 - 2 * quality);
  for (auto& row : quantizationMatrix) 
  {
    for (auto& val : row) 
    {
      val = static_cast<int>(val * scalingFactor / 100.0);
      if (val < 1) val = 1; // Prevent divide by zero
    }
  }

  int height = quantizedDCT.size(); // Get the height of the quantized DCT coefficient matrix
  int width = quantizedDCT[0].size(); // Get the width of the quantized DCT coefficient matrix
  vector<vector<double>> dct(height, vector<double>(width, 0.0)); // Create a 2D vector to store the dequantized DCT coefficients

  for (int u = 0; u < height; ++u) 
  {
    for (int v = 0; v < width; ++v) 
    {
      // Dequantize the DCT coefficient by multiplying it by the corresponding value in the quantization matrix
      dct[u][v] = round(quantizedDCT[u][v] * quantizationMatrix[u][v]); 
    }
  }

  return dct; // Return the 2D vector of dequantized DCT coefficients
}
cout << "You have reached checkpoint 6"
int main() 
{
  //step 1: call in image that the user uploaded
  string filename = "cat_test_256x256.jpg"; //test image 
  cout << "You have reached checkpoint 7"
  //step 2: load the image data
  int width, height, channels;
  unsigned char* imageData = stbi_load(filename.c_str(), &width, &height, &channels, 0);
  if (imageData == nullptr) 
  {
    cerr << "Error loading image: " << stbi_failure_reason() << endl;
    return 1;
  }
  cout << "You have reached checkpoint 8"
  //checks for image dimensions and data size
  size_t imageDataSize = width * height * channels * sizeof(unsigned char);
  if (width <= 0 || height <= 0 || channels <= 0 || (width * height * channels) > imageDataSize) 
  {
    cerr << "Error: Invalid image data." << endl;
    return 1;
  }
  cout << "You have reached checkpoint 9"
  //step 3: convert the image data to a 2D vector of doubles (grayscale)
  vector<vector<double>> imageMatrix(height, vector<double>(width));
  for (int i = 0; i < height; ++i) 
  {
    for (int j = 0; j < width; ++j) 
    {
      // Calculate the index into the imageData array
        int index = i * width * channels + j * channels; 

        // Check if the index is within the bounds of the array
        if (index >= 0 && index < imageDataSize) 
        {
            imageMatrix[i][j] = static_cast<double>(imageData[index]) / 255.0;
        } 
        else 
        {
            // Handle the error (e.g., print an error message, set a default value)
            cerr << "Error: Index out of bounds." << endl;
            imageMatrix[i][j] = 0.0; // Or another default value
        }
      // Assuming grayscale, take the average of R, G, B values if the image is not grayscale
      //imageMatrix[i][j] = static_cast<double>(imageData[i * width * channels + j * channels]) / 255.0; 
    }
  }
  stbi_image_free(imageData); //free the original image data
  cout << "You have reached checkpoint 10"
  //step 4: perform DCT
  vector<vector<double>> dctCoefficients = dctTransform(imageMatrix);
  cout << "You have reached checkpoint 11"
  //step 5: quantize DCT coefficients and adjust quality as needed
  int quality = 50; // You can get this from the user or another source
  //cout << "Enter desired quality (0-100): ";
  //cin >> quality;
  vector<vector<double>> quantizedDCT = quantizeDCT(dctCoefficients, quality);
  cout << "You have reached checkpoint 12"
  //step 6: quantize DCT coefficients
  vector<vector<double>> dequantizedDCT = dequantizeDCT(quantizedDCT, quality);
  cout << "You have reached checkpoint 13"
  //step 7: perform inverse DCT
  vector<vector<double>> reconstructedImage = idctTransform(dequantizedDCT);
  cout << "You have reached checkpoint 14"
  //step 8: convert back to an unsigned char format for saving
  vector<unsigned char> outputData(height * width);
  for (int i = 0; i < height; ++i) 
  {
    for (int j = 0; j < width; ++j) 
    {
      outputData[i * width + j] = static_cast<unsigned char>(reconstructedImage[i][j] * 255.0);
    }
  }
  cout << "You have reached checkpoint 15"
  //step 9: save the reconstructed image
  string outputFilename = "reconstructed_" + filename;
  if (stbi_write_png(outputFilename.c_str(), width, height, 1, outputData.data(), width) == 0) 
  {
    cerr << "Error saving image." << endl;
    return 1;
  }

  cout << "Reconstructed image saved as " << outputFilename << endl;
  return 0;
}
